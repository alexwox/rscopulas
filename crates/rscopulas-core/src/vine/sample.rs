use ndarray::Array2;
use rand::Rng;

use crate::errors::{CopulaError, FitError};

use super::{VineCopula, structure::{revert_matrix, revert_pair_matrix}};

impl VineCopula {
    pub(crate) fn sample_internal<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        clip_eps: f64,
    ) -> Result<Array2<f64>, CopulaError> {
        let d = self.dim;
        let mat = revert_matrix(&self.normalized_matrix);
        let maxmat = revert_matrix(&self.max_matrix);
        let cindirect = revert_matrix(&self.cond_indirect);
        let specs = revert_pair_matrix(&self.pair_matrix);

        let mut out = Array2::zeros((n, d));
        let simulated_order = self.variable_order.clone();

        for obs in 0..n {
            let mut vdirect = Array2::zeros((d, d));
            let mut vindirect = Array2::zeros((d, d));
            for idx in 0..d {
                vdirect[(idx, idx)] = rng.random::<f64>().clamp(clip_eps, 1.0 - clip_eps);
            }
            vindirect[(0, 0)] = vdirect[(0, 0)];

            for i in 1..d {
                for k in (0..i).rev() {
                    let label = maxmat[(k, i)];
                    let source = if mat[(k, i)] == label {
                        vdirect[(k, label)]
                    } else {
                        vindirect[(k, label)]
                    };
                    let spec = specs[(k, i)].ok_or(FitError::Failed {
                        reason: "missing pair-copula specification for vine simulation",
                    })?;
                    vdirect[(k, i)] =
                        spec.inv_second_given_first(source, vdirect[(k + 1, i)], clip_eps)?;
                    if i + 1 < d && cindirect[(k + 1, i)] {
                        vindirect[(k + 1, i)] =
                            spec.cond_first_given_second(source, vdirect[(k, i)], clip_eps)?;
                    }
                }
            }

            for (idx, var) in simulated_order.iter().copied().enumerate() {
                out[(obs, var)] = vdirect[(0, idx)];
            }
        }

        Ok(out)
    }
}
