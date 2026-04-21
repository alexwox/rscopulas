use ndarray::{Array2, ArrayView2};

use crate::{
    backend::{ExecutionStrategy, Operation, parallel_try_map_range_collect, resolve_strategy},
    domain::SampleOptions,
    errors::{CopulaError, FitError, InputError},
    paircopula::{
        cond_first_given_second_batch_into, cond_second_given_first_batch_into,
        inverse_second_given_first_batch_into,
    },
};

use super::{CompiledVineRuntime, VineCopula};

/// Probability clipping used by both Rosenblatt directions. Matches the
/// historical value used by `sample_internal` so the refactored sampler
/// produces bit-identical draws for a fixed seed.
pub(crate) const DEFAULT_CLIP_EPS: f64 = 1e-12;

impl VineCopula {
    /// Applies the inverse Rosenblatt transform `V = F^{-1}(U)` row-by-row.
    ///
    /// `u` is indexed by the original variable label: `u[(obs, var)]` is the
    /// independent uniform associated with variable `var`. The output `v` is
    /// indexed the same way, so `v[(obs, var)]` has the marginal distribution
    /// induced by the fitted vine on variable `var`.
    ///
    /// Internally `u[(obs, variable_order[idx])]` is placed into the `idx`-th
    /// diagonal cell of the workspace and the sampling recursion is run.
    pub fn inverse_rosenblatt(
        &self,
        u: ArrayView2<f64>,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        validate_matrix_shape(u, self.dim)?;
        let runtime = self.compiled_runtime()?;
        let n = u.nrows();
        let d = runtime.dim;
        let strategy = resolve_strategy(options.exec, Operation::Sample, n)?;
        let values = run_in_row_chunks(n, d, strategy, |start, end| {
            inverse_rosenblatt_block(&runtime, u, start, end, DEFAULT_CLIP_EPS)
        })?;
        Ok(Array2::from_shape_vec((n, d), values)
            .expect("inverse_rosenblatt output should match the requested shape"))
    }

    /// Applies the Rosenblatt transform `U = F(V)` row-by-row.
    ///
    /// `v` and the returned `u` are both indexed by the original variable
    /// label. For any `v`, the identities
    ///
    /// ```text
    /// inverse_rosenblatt(rosenblatt(v)) == v
    /// rosenblatt(inverse_rosenblatt(u)) == u
    /// ```
    ///
    /// hold up to the `clip_eps` boundary clamp.
    pub fn rosenblatt(
        &self,
        v: ArrayView2<f64>,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        let dim = self.dim;
        self.rosenblatt_with_limit(v, dim, options).map(|flat| {
            let n = v.nrows();
            to_variable_label_order(&flat, n, dim, self.variable_order())
        })
    }

    /// Applies the Rosenblatt transform restricted to the first `col_limit`
    /// positions in `variable_order`.
    ///
    /// The output has shape `(n, col_limit)` and is indexed **by diagonal
    /// position**, not by variable label: column `idx` of the output is the
    /// Rosenblatt uniform for variable `variable_order()[idx]`. This matches
    /// the natural ordering needed by `sample_conditional` when splicing U
    /// prefixes back together.
    ///
    /// Columns of `v` that do not correspond to `variable_order()[0..col_limit]`
    /// are never read, because the forward recursion short-circuits once
    /// `step.col >= col_limit`.
    pub fn rosenblatt_prefix(
        &self,
        v: ArrayView2<f64>,
        col_limit: usize,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        if col_limit == 0 || col_limit > self.dim {
            return Err(FitError::Failed {
                reason: "rosenblatt_prefix col_limit must be in 1..=dim",
            }
            .into());
        }
        let flat = self.rosenblatt_with_limit(v, col_limit, options)?;
        let n = v.nrows();
        Ok(Array2::from_shape_vec((n, col_limit), flat)
            .expect("rosenblatt_prefix output should match the requested shape"))
    }

    /// Returns the diagonal order in which the Rosenblatt transform emits
    /// variables. `variable_order()[0]` is the unconditional anchor: its
    /// input uniform passes through unchanged.
    pub fn variable_order(&self) -> &[usize] {
        &self.variable_order
    }

    fn rosenblatt_with_limit(
        &self,
        v: ArrayView2<f64>,
        col_limit: usize,
        options: &SampleOptions,
    ) -> Result<Vec<f64>, CopulaError> {
        validate_matrix_shape(v, self.dim)?;
        let runtime = self.compiled_runtime()?;
        let n = v.nrows();
        let strategy = resolve_strategy(options.exec, Operation::Sample, n)?;
        run_in_row_chunks(n, col_limit, strategy, |start, end| {
            rosenblatt_block(&runtime, v, start, end, col_limit, DEFAULT_CLIP_EPS)
        })
    }
}

/// Generates `n_rows * d` uniforms in positional `(obs, idx)` order, stores
/// them at the variable-label column implied by `variable_order`, and returns
/// the matrix ready to be consumed by `inverse_rosenblatt`.
pub(crate) fn draw_uniform_matrix<R: rand::Rng + ?Sized>(
    rng: &mut R,
    n: usize,
    variable_order: &[usize],
    clip_eps: f64,
) -> Array2<f64> {
    let d = variable_order.len();
    let mut u = Array2::zeros((n, d));
    for obs in 0..n {
        for &var in variable_order.iter() {
            u[(obs, var)] = rng.random::<f64>().clamp(clip_eps, 1.0 - clip_eps);
        }
    }
    u
}

pub(crate) fn inverse_rosenblatt_block(
    runtime: &CompiledVineRuntime,
    u: ArrayView2<f64>,
    start: usize,
    end: usize,
    clip_eps: f64,
) -> Result<Vec<f64>, CopulaError> {
    let n_rows = end.saturating_sub(start);
    let d = runtime.dim;
    if n_rows == 0 {
        return Ok(Vec::new());
    }

    let mut vdirect = vec![0.0; n_rows * d * d];
    let mut vindirect = vec![0.0; n_rows * d * d];
    let mut sources = vec![0.0; n_rows];
    let mut targets = vec![0.0; n_rows];
    let mut direct_out = vec![0.0; n_rows];
    let mut indirect_out = vec![0.0; n_rows];

    for local_obs in 0..n_rows {
        let obs = start + local_obs;
        for (idx, &var) in runtime.variable_order.iter().enumerate() {
            vdirect[workspace_index(local_obs, idx, idx, d)] =
                u[(obs, var)].clamp(clip_eps, 1.0 - clip_eps);
        }
        // `vindirect[(0, 0)]` doubles as the source when `source_from_direct`
        // is false at the first tree level. Keep it in sync with the anchor
        // uniform — this mirrors the historical sampler and the evaluator.
        vindirect[workspace_index(local_obs, 0, 0, d)] =
            vdirect[workspace_index(local_obs, 0, 0, d)];
    }

    for step in &runtime.sample_steps {
        for local_obs in 0..n_rows {
            let source_idx = workspace_index(local_obs, step.row, step.label, d);
            sources[local_obs] = if step.source_from_direct {
                vdirect[source_idx]
            } else {
                vindirect[source_idx]
            };
            targets[local_obs] = vdirect[workspace_index(local_obs, step.row + 1, step.col, d)];
        }
        inverse_second_given_first_batch_into(
            &step.spec,
            &sources,
            &targets,
            clip_eps,
            ExecutionStrategy::CpuSerial,
            &mut direct_out,
        )?;
        for local_obs in 0..n_rows {
            vdirect[workspace_index(local_obs, step.row, step.col, d)] = direct_out[local_obs];
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
            for local_obs in 0..n_rows {
                vindirect[workspace_index(local_obs, step.row + 1, step.col, d)] =
                    indirect_out[local_obs];
            }
        }
    }

    let mut out = vec![0.0; n_rows * d];
    for local_obs in 0..n_rows {
        for (idx, &var) in runtime.variable_order.iter().enumerate() {
            out[(local_obs * d) + var] = vdirect[workspace_index(local_obs, 0, idx, d)];
        }
    }
    Ok(out)
}

pub(crate) fn rosenblatt_block(
    runtime: &CompiledVineRuntime,
    v: ArrayView2<f64>,
    start: usize,
    end: usize,
    col_limit: usize,
    clip_eps: f64,
) -> Result<Vec<f64>, CopulaError> {
    let n_rows = end.saturating_sub(start);
    let d = runtime.dim;
    if n_rows == 0 {
        return Ok(Vec::new());
    }

    let mut vdirect = vec![0.0; n_rows * d * d];
    let mut vindirect = vec![0.0; n_rows * d * d];
    let mut sources = vec![0.0; n_rows];
    let mut targets = vec![0.0; n_rows];
    let mut direct_out = vec![0.0; n_rows];
    let mut indirect_out = vec![0.0; n_rows];

    for local_obs in 0..n_rows {
        let obs = start + local_obs;
        for (idx, &var) in runtime.variable_order.iter().enumerate() {
            vdirect[workspace_index(local_obs, 0, idx, d)] =
                v[(obs, var)].clamp(clip_eps, 1.0 - clip_eps);
        }
        vindirect[workspace_index(local_obs, 0, 0, d)] =
            vdirect[workspace_index(local_obs, 0, 0, d)];
    }

    for step in &runtime.eval_steps {
        if step.col >= col_limit {
            break;
        }
        for local_obs in 0..n_rows {
            let source_idx = workspace_index(local_obs, step.row, step.label, d);
            sources[local_obs] = if step.source_from_direct {
                vdirect[source_idx]
            } else {
                vindirect[source_idx]
            };
            targets[local_obs] = vdirect[workspace_index(local_obs, step.row, step.col, d)];
        }
        cond_second_given_first_batch_into(
            &step.spec,
            &sources,
            &targets,
            clip_eps,
            ExecutionStrategy::CpuSerial,
            &mut direct_out,
        )?;
        for local_obs in 0..n_rows {
            vdirect[workspace_index(local_obs, step.row + 1, step.col, d)] = direct_out[local_obs];
        }
        if step.write_indirect {
            cond_first_given_second_batch_into(
                &step.spec,
                &sources,
                &targets,
                clip_eps,
                ExecutionStrategy::CpuSerial,
                &mut indirect_out,
            )?;
            for local_obs in 0..n_rows {
                vindirect[workspace_index(local_obs, step.row + 1, step.col, d)] =
                    indirect_out[local_obs];
            }
        }
    }

    // Output is indexed by diagonal position: out[(obs, idx)] = U[variable_order[idx]].
    // Callers that want variable-label indexing (the public `rosenblatt`) permute
    // afterwards; `rosenblatt_prefix` leaves the positional layout as is.
    let mut out = vec![0.0; n_rows * col_limit];
    for local_obs in 0..n_rows {
        for idx in 0..col_limit {
            out[(local_obs * col_limit) + idx] =
                vdirect[workspace_index(local_obs, idx, idx, d)];
        }
    }
    Ok(out)
}

fn run_in_row_chunks<F>(
    n: usize,
    d: usize,
    strategy: ExecutionStrategy,
    block: F,
) -> Result<Vec<f64>, CopulaError>
where
    F: Fn(usize, usize) -> Result<Vec<f64>, CopulaError> + Sync + Send,
{
    match strategy {
        ExecutionStrategy::CpuSerial => block(0, n),
        ExecutionStrategy::CpuParallel => {
            let chunk = chunk_size_for(d, n);
            let chunk_count = n.div_ceil(chunk.max(1));
            let blocks = parallel_try_map_range_collect(chunk_count, strategy, |chunk_idx| {
                let start = chunk_idx * chunk;
                let end = (start + chunk).min(n);
                block(start, end)
            })?;
            let mut all = Vec::with_capacity(n * d);
            for piece in blocks {
                all.extend(piece);
            }
            Ok(all)
        }
        ExecutionStrategy::Cuda(_) | ExecutionStrategy::Metal => {
            unreachable!("rosenblatt transforms only support CPU execution")
        }
    }
}

fn to_variable_label_order(
    positional: &[f64],
    n: usize,
    d: usize,
    variable_order: &[usize],
) -> Array2<f64> {
    let mut out = Array2::zeros((n, d));
    for obs in 0..n {
        for (idx, &var) in variable_order.iter().enumerate() {
            out[(obs, var)] = positional[(obs * d) + idx];
        }
    }
    out
}

fn validate_matrix_shape(data: ArrayView2<f64>, dim: usize) -> Result<(), CopulaError> {
    if data.ncols() != dim {
        return Err(FitError::Failed {
            reason: "rosenblatt input has a different dimension than the vine",
        }
        .into());
    }
    if data.nrows() == 0 {
        return Err(InputError::EmptyObservations.into());
    }
    Ok(())
}

#[inline]
fn workspace_index(obs: usize, row: usize, col: usize, dim: usize) -> usize {
    (obs * dim * dim) + (row * dim) + col
}

fn chunk_size_for(dim: usize, n: usize) -> usize {
    let target_rows = (4096 / dim.max(1)).max(128);
    target_rows.min(n.max(1))
}
