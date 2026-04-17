use ndarray::Array2;

use crate::{
    backend::{ExecutionStrategy, Operation, parallel_try_map_range_collect, resolve_strategy},
    data::PseudoObs,
    domain::ExecPolicy,
    errors::{BackendError, CopulaError, FitError},
    paircopula::{PairCopulaFamily, PairCopulaParams, Rotation},
};

use super::{
    VineCopula,
    structure::{revert_matrix, revert_pair_matrix},
};

struct VineEvalContext<'a> {
    variable_order: &'a [usize],
    mat: &'a Array2<usize>,
    maxmat: &'a Array2<usize>,
    cindirect: &'a Array2<bool>,
    specs: &'a Array2<Option<crate::paircopula::PairCopulaSpec>>,
    view: ndarray::ArrayView2<'a, f64>,
    d: usize,
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

        let d = self.dim;
        let mat = revert_matrix(&self.normalized_matrix);
        let maxmat = revert_matrix(&self.max_matrix);
        let cindirect = revert_matrix(&self.cond_indirect);
        let specs = revert_pair_matrix(&self.pair_matrix);
        let view = data.as_view();
        let strategy = resolve_strategy(exec, Operation::VineLogPdf, data.n_obs())?;
        let ctx = VineEvalContext {
            variable_order: &self.variable_order,
            mat: &mat,
            maxmat: &maxmat,
            cindirect: &cindirect,
            specs: &specs,
            view,
            d,
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
    parallel_try_map_range_collect(ctx.view.nrows(), strategy, |obs| {
        let mut vdirect = Array2::zeros((ctx.d, ctx.d));
        let mut vindirect = Array2::zeros((ctx.d, ctx.d));
        let mut total = 0.0;

        for (idx, var) in ctx.variable_order.iter().copied().enumerate() {
            vdirect[(0, idx)] = ctx.view[(obs, var)].clamp(ctx.clip_eps, 1.0 - ctx.clip_eps);
        }

        vindirect[(0, 0)] = vdirect[(0, 0)];

        for i in 1..ctx.d {
            for k in 0..i {
                let label = ctx.maxmat[(k, i)];
                let source = if ctx.mat[(k, i)] == label {
                    vdirect[(k, label)]
                } else {
                    vindirect[(k, label)]
                };
                let target = vdirect[(k, i)];
                let spec = ctx.specs[(k, i)].ok_or(FitError::Failed {
                    reason: "missing pair-copula specification for vine evaluation",
                })?;

                total += spec.log_pdf(source, target, ctx.clip_eps)?;
                vdirect[(k + 1, i)] = spec.cond_second_given_first(source, target, ctx.clip_eps)?;
                if i + 1 < ctx.d && ctx.cindirect[(k + 1, i)] {
                    vindirect[(k + 1, i)] =
                        spec.cond_first_given_second(source, target, ctx.clip_eps)?;
                }
            }
        }
        Ok(total)
    })
}

fn evaluate_vine_with_gpu_pair_batches(
    ctx: &VineEvalContext<'_>,
    device: rscopulas_accel::Device,
    backend: &'static str,
) -> Result<Vec<f64>, CopulaError> {
    if !all_gaussian_specs(ctx.specs) {
        let cpu_strategy = if ctx.view.nrows() >= 128 {
            ExecutionStrategy::CpuParallel
        } else {
            ExecutionStrategy::CpuSerial
        };
        return evaluate_vine_cpu(ctx, cpu_strategy);
    }

    let n_obs = ctx.view.nrows();
    let mut totals = vec![0.0; n_obs];
    let mut vdirect = (0..n_obs)
        .map(|obs| {
            let mut direct = Array2::zeros((ctx.d, ctx.d));
            for (idx, var) in ctx.variable_order.iter().copied().enumerate() {
                direct[(0, idx)] = ctx.view[(obs, var)].clamp(ctx.clip_eps, 1.0 - ctx.clip_eps);
            }
            direct
        })
        .collect::<Vec<_>>();
    let mut vindirect = (0..n_obs)
        .map(|obs| {
            let mut indirect = Array2::zeros((ctx.d, ctx.d));
            indirect[(0, 0)] =
                ctx.view[(obs, ctx.variable_order[0])].clamp(ctx.clip_eps, 1.0 - ctx.clip_eps);
            indirect
        })
        .collect::<Vec<_>>();

    for i in 1..ctx.d {
        for k in 0..i {
            let spec = ctx.specs[(k, i)].ok_or(FitError::Failed {
                reason: "missing pair-copula specification for vine evaluation",
            })?;
            let label = ctx.maxmat[(k, i)];
            let mut sources = Vec::with_capacity(n_obs);
            let mut targets = Vec::with_capacity(n_obs);
            for obs in 0..n_obs {
                let source = if ctx.mat[(k, i)] == label {
                    vdirect[obs][(k, label)]
                } else {
                    vindirect[obs][(k, label)]
                };
                let target = vdirect[obs][(k, i)];
                sources.push(source);
                targets.push(target);
            }

            if let Some(rho) = gaussian_rho(spec) {
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
                    vdirect[obs][(k + 1, i)] = batch.cond_on_second[obs];
                    if i + 1 < ctx.d && ctx.cindirect[(k + 1, i)] {
                        vindirect[obs][(k + 1, i)] = batch.cond_on_first[obs];
                    }
                }
            } else {
                for obs in 0..n_obs {
                    totals[obs] += spec.log_pdf(sources[obs], targets[obs], ctx.clip_eps)?;
                    vdirect[obs][(k + 1, i)] =
                        spec.cond_second_given_first(sources[obs], targets[obs], ctx.clip_eps)?;
                    if i + 1 < ctx.d && ctx.cindirect[(k + 1, i)] {
                        vindirect[obs][(k + 1, i)] =
                            spec.cond_first_given_second(sources[obs], targets[obs], ctx.clip_eps)?;
                    }
                }
            }
        }
    }

    Ok(totals)
}

fn gaussian_rho(spec: crate::paircopula::PairCopulaSpec) -> Option<f64> {
    match (spec.family, spec.rotation, spec.params) {
        (PairCopulaFamily::Gaussian, Rotation::R0, PairCopulaParams::One(rho)) => Some(rho),
        _ => None,
    }
}

fn all_gaussian_specs(specs: &Array2<Option<crate::paircopula::PairCopulaSpec>>) -> bool {
    specs
        .iter()
        .flatten()
        .all(|spec| gaussian_rho(*spec).is_some())
}
