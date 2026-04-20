use ndarray::Array2;
use rand::Rng;

use crate::{
    backend::{ExecutionStrategy, Operation, parallel_try_map_range_collect, resolve_strategy},
    domain::SampleOptions,
    errors::CopulaError,
    paircopula::{cond_first_given_second_batch_into, inverse_second_given_first_batch_into},
};

use super::{CompiledVineRuntime, VineCopula};

impl VineCopula {
    pub(crate) fn sample_internal<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        let strategy = resolve_strategy(options.exec, Operation::Sample, n)?;
        let runtime = self.compiled_runtime()?;
        let d = runtime.dim;
        let clip_eps = 1e-12;
        let values = match strategy {
            ExecutionStrategy::CpuSerial => sample_serial(&runtime, n, rng, clip_eps)?,
            ExecutionStrategy::CpuParallel => {
                let uniforms = (0..(n * d))
                    .map(|_| rng.random::<f64>().clamp(clip_eps, 1.0 - clip_eps))
                    .collect::<Vec<_>>();
                let chunk_size = sample_chunk_size(d, n);
                let chunk_count = n.div_ceil(chunk_size);
                let blocks = parallel_try_map_range_collect(chunk_count, strategy, |chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(n);
                    sample_block(
                        &runtime,
                        end - start,
                        &uniforms[(start * d)..(end * d)],
                        clip_eps,
                    )
                })?;
                let mut all = Vec::with_capacity(n * d);
                for block in blocks {
                    all.extend(block);
                }
                all
            }
            ExecutionStrategy::Cuda(_) | ExecutionStrategy::Metal => {
                unreachable!("sampling only supports CPU execution")
            }
        };
        Ok(Array2::from_shape_vec((n, d), values)
            .expect("sampled vine output should match the requested shape"))
    }
}

fn sample_serial<R: Rng + ?Sized>(
    runtime: &CompiledVineRuntime,
    n_rows: usize,
    rng: &mut R,
    clip_eps: f64,
) -> Result<Vec<f64>, CopulaError> {
    let d = runtime.dim;
    let mut out = vec![0.0; n_rows * d];
    let mut vdirect = vec![0.0; d * d];
    let mut vindirect = vec![0.0; d * d];

    for obs in 0..n_rows {
        vdirect.fill(0.0);
        vindirect.fill(0.0);
        for idx in 0..d {
            vdirect[single_workspace_index(idx, idx, d)] =
                rng.random::<f64>().clamp(clip_eps, 1.0 - clip_eps);
        }
        vindirect[0] = vdirect[0];

        for step in &runtime.sample_steps {
            let source_idx = single_workspace_index(step.row, step.label, d);
            let target_idx = single_workspace_index(step.row + 1, step.col, d);
            let out_idx = single_workspace_index(step.row, step.col, d);
            let source = if step.source_from_direct {
                vdirect[source_idx]
            } else {
                vindirect[source_idx]
            };
            let value = step
                .spec
                .inv_second_given_first(source, vdirect[target_idx], clip_eps)?;
            vdirect[out_idx] = value;
            if step.write_indirect {
                vindirect[single_workspace_index(step.row + 1, step.col, d)] =
                    step.spec.cond_first_given_second(source, value, clip_eps)?;
            }
        }

        for (idx, &var) in runtime.variable_order.iter().enumerate() {
            out[(obs * d) + var] = vdirect[single_workspace_index(0, idx, d)];
        }
    }

    Ok(out)
}

fn sample_block(
    runtime: &CompiledVineRuntime,
    n_rows: usize,
    uniforms: &[f64],
    clip_eps: f64,
) -> Result<Vec<f64>, CopulaError> {
    let d = runtime.dim;
    let mut vdirect = vec![0.0; n_rows * d * d];
    let mut vindirect = vec![0.0; n_rows * d * d];
    let mut sources = vec![0.0; n_rows];
    let mut targets = vec![0.0; n_rows];
    let mut direct_out = vec![0.0; n_rows];
    let mut indirect_out = vec![0.0; n_rows];

    for obs in 0..n_rows {
        for idx in 0..d {
            let diagonal = workspace_index(obs, idx, idx, d);
            vdirect[diagonal] = uniforms[(obs * d) + idx];
        }
        vindirect[workspace_index(obs, 0, 0, d)] = uniforms[obs * d];
    }

    for step in &runtime.sample_steps {
        for obs in 0..n_rows {
            let source_idx = workspace_index(obs, step.row, step.label, d);
            sources[obs] = if step.source_from_direct {
                vdirect[source_idx]
            } else {
                vindirect[source_idx]
            };
            targets[obs] = vdirect[workspace_index(obs, step.row + 1, step.col, d)];
        }
        inverse_second_given_first_batch_into(
            &step.spec,
            &sources,
            &targets,
            clip_eps,
            ExecutionStrategy::CpuSerial,
            &mut direct_out,
        )?;
        for obs in 0..n_rows {
            vdirect[workspace_index(obs, step.row, step.col, d)] = direct_out[obs];
        }
        if step.write_indirect {
            cond_first_given_second_batch_into(
                &step.spec,
                &sources,
                &direct_out,
                clip_eps,
                ExecutionStrategy::CpuSerial,
                &mut indirect_out,
            )?;
            for obs in 0..n_rows {
                vindirect[workspace_index(obs, step.row + 1, step.col, d)] = indirect_out[obs];
            }
        }
    }

    let mut out = vec![0.0; n_rows * d];
    for obs in 0..n_rows {
        for (idx, &var) in runtime.variable_order.iter().enumerate() {
            out[(obs * d) + var] = vdirect[workspace_index(obs, 0, idx, d)];
        }
    }
    Ok(out)
}

#[inline]
fn workspace_index(obs: usize, row: usize, col: usize, dim: usize) -> usize {
    (obs * dim * dim) + (row * dim) + col
}

#[inline]
fn single_workspace_index(row: usize, col: usize, dim: usize) -> usize {
    (row * dim) + col
}

fn sample_chunk_size(dim: usize, n_rows: usize) -> usize {
    let target_rows = (4096 / dim.max(1)).max(128);
    target_rows.min(n_rows.max(1))
}
