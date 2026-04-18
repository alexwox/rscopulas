use crate::{
    backend::{ExecutionStrategy, Operation, parallel_try_map_range_collect, resolve_strategy},
    data::PseudoObs,
    domain::{Device, ExecPolicy},
    errors::{BackendError, CopulaError, FitError},
    paircopula::{PairCopulaFamily, PairCopulaParams, Rotation, evaluate_pair_batch_into},
};

use super::{CompiledVineRuntime, VineCopula};

struct VineEvalContext<'a> {
    runtime: &'a CompiledVineRuntime,
    view: ndarray::ArrayView2<'a, f64>,
    clip_eps: f64,
}

impl VineCopula {
    pub(crate) fn log_pdf_internal(
        &self,
        data: &PseudoObs,
        exec: ExecPolicy,
        clip_eps: f64,
    ) -> Result<Vec<f64>, CopulaError> {
        if data.dim() != self.dim {
            return Err(FitError::Failed {
                reason: "input dimension does not match vine dimension",
            }
            .into());
        }

        let runtime = self.compiled_runtime();
        let view = data.as_view();
        let strategy = resolve_strategy(exec, Operation::VineLogPdf, data.n_obs())?;
        let ctx = VineEvalContext {
            runtime: &runtime,
            view,
            clip_eps,
        };

        match strategy {
            ExecutionStrategy::CpuSerial | ExecutionStrategy::CpuParallel => {
                evaluate_vine_cpu(&ctx, strategy)
            }
            ExecutionStrategy::Cuda(ordinal) => evaluate_vine_with_gpu_pair_batches(
                &ctx,
                rscopulas_accel::Device::Cuda(ordinal),
                "cuda",
            ),
            ExecutionStrategy::Metal => {
                evaluate_vine_with_gpu_pair_batches(&ctx, rscopulas_accel::Device::Metal, "metal")
            }
        }
    }
}

fn evaluate_vine_cpu(
    ctx: &VineEvalContext<'_>,
    strategy: ExecutionStrategy,
) -> Result<Vec<f64>, CopulaError> {
    match strategy {
        ExecutionStrategy::CpuSerial => {
            evaluate_vine_block(ctx.runtime, &ctx.view, 0, ctx.view.nrows(), ctx.clip_eps)
        }
        ExecutionStrategy::CpuParallel => {
            let chunk_size = eval_chunk_size(ctx.runtime.dim, ctx.view.nrows());
            let chunk_count = ctx.view.nrows().div_ceil(chunk_size);
            let blocks = parallel_try_map_range_collect(chunk_count, strategy, |chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(ctx.view.nrows());
                evaluate_vine_block(ctx.runtime, &ctx.view, start, end, ctx.clip_eps)
            })?;
            let mut totals = Vec::with_capacity(ctx.view.nrows());
            for block in blocks {
                totals.extend(block);
            }
            Ok(totals)
        }
        ExecutionStrategy::Cuda(_) | ExecutionStrategy::Metal => {
            unreachable!("CPU vine evaluation expects a CPU strategy")
        }
    }
}

fn evaluate_vine_with_gpu_pair_batches(
    ctx: &VineEvalContext<'_>,
    device: rscopulas_accel::Device,
    backend: &'static str,
) -> Result<Vec<f64>, CopulaError> {
    if !ctx.runtime.all_gaussian {
        let cpu_strategy = if ctx.view.nrows() >= 128 {
            ExecutionStrategy::CpuParallel
        } else {
            ExecutionStrategy::CpuSerial
        };
        return evaluate_vine_cpu(ctx, cpu_strategy);
    }

    let n_obs = ctx.view.nrows();
    let d = ctx.runtime.dim;
    let mut totals = vec![0.0; n_obs];
    let mut vdirect = vec![0.0; n_obs * d * d];
    let mut vindirect = vec![0.0; n_obs * d * d];

    for obs in 0..n_obs {
        for (idx, &var) in ctx.runtime.variable_order.iter().enumerate() {
            vdirect[workspace_index(obs, 0, idx, d)] =
                ctx.view[(obs, var)].clamp(ctx.clip_eps, 1.0 - ctx.clip_eps);
        }
        vindirect[workspace_index(obs, 0, 0, d)] = vdirect[workspace_index(obs, 0, 0, d)];
    }

    for step in &ctx.runtime.eval_steps {
        let rho = gaussian_rho(&step.spec).ok_or(FitError::Failed {
            reason: "gaussian batch evaluation expected only gaussian vine steps",
        })?;
        let mut sources = vec![0.0; n_obs];
        let mut targets = vec![0.0; n_obs];
        for obs in 0..n_obs {
            let source_idx = workspace_index(obs, step.row, step.label, d);
            sources[obs] = if step.source_from_direct {
                vdirect[source_idx]
            } else {
                vindirect[source_idx]
            };
            targets[obs] = vdirect[workspace_index(obs, step.row, step.col, d)];
        }

        let batch = rscopulas_accel::evaluate_gaussian_pair_batch(
            device,
            rscopulas_accel::GaussianPairBatchRequest {
                u1: &sources,
                u2: &targets,
                rho,
                clip_eps: ctx.clip_eps,
            },
        )
        .map_err(|err| BackendError::Failed {
            backend,
            reason: err.to_string(),
        })?;

        for obs in 0..n_obs {
            totals[obs] += batch.log_pdf[obs];
            vdirect[workspace_index(obs, step.row + 1, step.col, d)] = batch.cond_on_second[obs];
            if step.write_indirect {
                vindirect[workspace_index(obs, step.row + 1, step.col, d)] =
                    batch.cond_on_first[obs];
            }
        }
    }

    Ok(totals)
}

fn evaluate_vine_block(
    runtime: &CompiledVineRuntime,
    view: &ndarray::ArrayView2<'_, f64>,
    start: usize,
    end: usize,
    clip_eps: f64,
) -> Result<Vec<f64>, CopulaError> {
    let n_rows = end.saturating_sub(start);
    let d = runtime.dim;
    let mut totals = vec![0.0; n_rows];
    let mut vdirect = vec![0.0; n_rows * d * d];
    let mut vindirect = vec![0.0; n_rows * d * d];
    let mut sources = vec![0.0; n_rows];
    let mut targets = vec![0.0; n_rows];
    let mut log_pdf = vec![0.0; n_rows];
    let mut cond_on_first = vec![0.0; n_rows];
    let mut cond_on_second = vec![0.0; n_rows];

    for local_obs in 0..n_rows {
        let obs = start + local_obs;
        for (idx, &var) in runtime.variable_order.iter().enumerate() {
            vdirect[workspace_index(local_obs, 0, idx, d)] =
                view[(obs, var)].clamp(clip_eps, 1.0 - clip_eps);
        }
        vindirect[workspace_index(local_obs, 0, 0, d)] =
            vdirect[workspace_index(local_obs, 0, 0, d)];
    }

    for step in &runtime.eval_steps {
        for local_obs in 0..n_rows {
            let source_idx = workspace_index(local_obs, step.row, step.label, d);
            sources[local_obs] = if step.source_from_direct {
                vdirect[source_idx]
            } else {
                vindirect[source_idx]
            };
            targets[local_obs] = vdirect[workspace_index(local_obs, step.row, step.col, d)];
        }
        evaluate_pair_batch_into(
            &step.spec,
            &sources,
            &targets,
            clip_eps,
            ExecPolicy::Force(Device::Cpu),
            &mut log_pdf,
            &mut cond_on_first,
            &mut cond_on_second,
        )?;
        for local_obs in 0..n_rows {
            totals[local_obs] += log_pdf[local_obs];
            vdirect[workspace_index(local_obs, step.row + 1, step.col, d)] =
                cond_on_second[local_obs];
            if step.write_indirect {
                vindirect[workspace_index(local_obs, step.row + 1, step.col, d)] =
                    cond_on_first[local_obs];
            }
        }
    }

    Ok(totals)
}

fn gaussian_rho(spec: &crate::paircopula::PairCopulaSpec) -> Option<f64> {
    match (spec.family, spec.rotation, &spec.params) {
        (PairCopulaFamily::Gaussian, Rotation::R0, PairCopulaParams::One(rho)) => Some(*rho),
        _ => None,
    }
}

#[inline]
fn workspace_index(obs: usize, row: usize, col: usize, dim: usize) -> usize {
    (obs * dim * dim) + (row * dim) + col
}

fn eval_chunk_size(dim: usize, n_rows: usize) -> usize {
    let target_rows = (8192 / dim.max(1)).max(128);
    target_rows.min(n_rows.max(1))
}
