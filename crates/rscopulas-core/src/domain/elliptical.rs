use ndarray::Array2;
use rand::Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::{
    data::PseudoObs,
    errors::{CopulaError, FitError},
    fit::FitResult,
    math::{
        cholesky, log_determinant_from_cholesky, quadratic_form_from_cholesky,
        validate_correlation_matrix,
    },
    stats::kendall_tau_matrix,
};

use super::{CopulaFamily, CopulaModel, EvalOptions, FitOptions, SampleOptions};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianCopula {
    dim: usize,
    correlation: Array2<f64>,
    #[serde(skip)]
    cholesky: Array2<f64>,
    log_det: f64,
}

impl GaussianCopula {
    pub fn new(correlation: Array2<f64>) -> Result<Self, CopulaError> {
        validate_correlation_matrix(&correlation)?;
        let cholesky = cholesky(&correlation)?;
        let log_det = log_determinant_from_cholesky(&cholesky);
        let dim = correlation.ncols();
        Ok(Self {
            dim,
            correlation,
            cholesky,
            log_det,
        })
    }

    pub fn fit(data: &PseudoObs, _options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        let tau = kendall_tau_matrix(data);
        let correlation = tau.mapv(|value| (std::f64::consts::FRAC_PI_2 * value).sin());
        let model = Self::new(correlation)?;
        let loglik = model
            .log_pdf(data, &EvalOptions::default())?
            .into_iter()
            .sum::<f64>();
        let parameter_count = (model.dim * (model.dim - 1) / 2) as f64;
        let n_obs = data.n_obs() as f64;
        let diagnostics = super::FitDiagnostics {
            loglik,
            aic: 2.0 * parameter_count - 2.0 * loglik,
            bic: parameter_count * n_obs.ln() - 2.0 * loglik,
            converged: true,
            n_iter: 0,
        };

        Ok(FitResult { model, diagnostics })
    }

    pub fn correlation(&self) -> &Array2<f64> {
        &self.correlation
    }

    fn standard_normal() -> Normal {
        Normal::new(0.0, 1.0).expect("standard normal parameters should be valid")
    }
}

impl CopulaModel for GaussianCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::Gaussian
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn log_pdf(&self, data: &PseudoObs, options: &EvalOptions) -> Result<Vec<f64>, CopulaError> {
        if data.dim() != self.dim {
            return Err(FitError::Failed {
                reason: "input dimension does not match model dimension",
            }
            .into());
        }

        let standard_normal = Self::standard_normal();
        let mut values = Vec::with_capacity(data.n_obs());
        for row in data.as_view().rows() {
            let z = row
                .iter()
                .map(|value| {
                    let clipped = value.clamp(options.clip_eps, 1.0 - options.clip_eps);
                    standard_normal.inverse_cdf(clipped)
                })
                .collect::<Vec<_>>();
            let quadratic = quadratic_form_from_cholesky(&self.cholesky, &z)?;
            let euclidean = z.iter().map(|value| value * value).sum::<f64>();
            values.push(-0.5 * self.log_det - 0.5 * (quadratic - euclidean));
        }

        Ok(values)
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        _options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        let standard_normal = Self::standard_normal();
        let mut samples = Array2::zeros((n, self.dim));

        for row_idx in 0..n {
            let epsilon = (0..self.dim)
                .map(|_| rng.sample::<f64, _>(StandardNormal))
                .collect::<Vec<_>>();
            let mut correlated = vec![0.0; self.dim];
            for i in 0..self.dim {
                let mut value = 0.0;
                for j in 0..=i {
                    value += self.cholesky[(i, j)] * epsilon[j];
                }
                correlated[i] = value;
            }

            for col_idx in 0..self.dim {
                samples[(row_idx, col_idx)] = standard_normal.cdf(correlated[col_idx]);
            }
        }

        Ok(samples)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentTCopula {
    dim: usize,
    correlation: Array2<f64>,
    degrees_of_freedom: f64,
}

impl StudentTCopula {
    pub fn new(correlation: Array2<f64>, degrees_of_freedom: f64) -> Self {
        let dim = correlation.ncols();
        Self {
            dim,
            correlation,
            degrees_of_freedom,
        }
    }

    pub fn fit(_data: &PseudoObs, _options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        Err(FitError::NotImplemented.into())
    }

    pub fn correlation(&self) -> &Array2<f64> {
        &self.correlation
    }

    pub fn degrees_of_freedom(&self) -> f64 {
        self.degrees_of_freedom
    }
}

impl CopulaModel for StudentTCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::StudentT
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn log_pdf(&self, _data: &PseudoObs, _options: &EvalOptions) -> Result<Vec<f64>, CopulaError> {
        Err(FitError::NotImplemented.into())
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        _n: usize,
        _rng: &mut R,
        _options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        Err(FitError::NotImplemented.into())
    }
}
