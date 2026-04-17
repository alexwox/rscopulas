use ndarray::Array2;

use crate::{
    data::PseudoObs,
    errors::{CopulaError, FitError},
};

use super::{
    VineCopula,
    structure::{revert_matrix, revert_pair_matrix},
};

impl VineCopula {
    pub(crate) fn log_pdf_internal(
        &self,
        data: &PseudoObs,
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
        let mut totals = vec![0.0; data.n_obs()];

        for obs in 0..data.n_obs() {
            let mut vdirect = Array2::zeros((d, d));
            let mut vindirect = Array2::zeros((d, d));

            for (idx, var) in self.variable_order.iter().copied().enumerate() {
                vdirect[(0, idx)] = view[(obs, var)].clamp(clip_eps, 1.0 - clip_eps);
            }

            vindirect[(0, 0)] = vdirect[(0, 0)];

            for i in 1..d {
                for k in 0..i {
                    let label = maxmat[(k, i)];
                    let source = if mat[(k, i)] == label {
                        vdirect[(k, label)]
                    } else {
                        vindirect[(k, label)]
                    };
                    let target = vdirect[(k, i)];
                    let spec = specs[(k, i)].ok_or(FitError::Failed {
                        reason: "missing pair-copula specification for vine evaluation",
                    })?;

                    totals[obs] += spec.log_pdf(source, target, clip_eps)?;
                    vdirect[(k + 1, i)] =
                        spec.cond_second_given_first(source, target, clip_eps)?;
                    if i + 1 < d && cindirect[(k + 1, i)] {
                        vindirect[(k + 1, i)] =
                            spec.cond_first_given_second(source, target, clip_eps)?;
                    }
                }
            }
        }

        Ok(totals)
    }
}
