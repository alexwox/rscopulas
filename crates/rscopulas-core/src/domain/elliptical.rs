use ndarray::Array2;
use rand::Rng;
use rand_distr::{ChiSquared, StandardNormal};
use serde::{Deserialize, Deserializer, Serialize};
use statrs::{
    distribution::{Continuous, ContinuousCDF, Normal, StudentsT},
    function::gamma::ln_gamma,
};

use crate::{
    backend::{Operation, parallel_try_map_range_collect, resolve_strategy},
    data::PseudoObs,
    errors::{CopulaError, FitError},
    fit::FitResult,
    math::{
        cholesky, log_determinant_from_cholesky, make_spd_correlation,
        quadratic_form_from_cholesky, validate_correlation_matrix,
    },
    stats::try_kendall_tau_matrix,
};

use super::{CopulaFamily, CopulaModel, EvalOptions, FitOptions, SampleOptions};

/// Gaussian copula parameterized by a correlation matrix.
#[derive(Debug, Clone, Serialize)]
pub struct GaussianCopula {
    dim: usize,
    correlation: Array2<f64>,
    #[serde(skip)]
    cholesky: Array2<f64>,
    log_det: f64,
}

impl GaussianCopula {
    /// Constructs a Gaussian copula from a valid correlation matrix.
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

    /// Fits a Gaussian copula by inverting the Kendall tau matrix.
    pub fn fit(data: &PseudoObs, options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        let tau = try_kendall_tau_matrix(data, options.exec)?;
        let correlation =
            make_spd_correlation(&tau.mapv(|value| (std::f64::consts::FRAC_PI_2 * value).sin()))?;
        let model = Self::new(correlation)?;
        let loglik = model
            .log_pdf(
                data,
                &EvalOptions {
                    exec: options.exec,
                    clip_eps: options.clip_eps,
                },
            )?
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

    /// Returns the fitted correlation matrix.
    pub fn correlation(&self) -> &Array2<f64> {
        &self.correlation
    }

    fn standard_normal() -> Normal {
        Normal::new(0.0, 1.0).expect("standard normal parameters should be valid")
    }
}

#[derive(Deserialize)]
struct GaussianCopulaRepr {
    #[serde(default)]
    dim: Option<usize>,
    correlation: Array2<f64>,
    #[serde(default)]
    log_det: Option<f64>,
}

impl<'de> Deserialize<'de> for GaussianCopula {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = GaussianCopulaRepr::deserialize(deserializer)?;
        let model = Self::new(repr.correlation).map_err(serde::de::Error::custom)?;
        if repr.dim.is_some_and(|dim| dim != model.dim) {
            return Err(serde::de::Error::custom(
                "serialized GaussianCopula dim does not match correlation",
            ));
        }
        if repr.log_det.is_some_and(|log_det| !log_det.is_finite()) {
            return Err(serde::de::Error::custom(
                "serialized GaussianCopula log_det must be finite",
            ));
        }
        Ok(model)
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
        let strategy = resolve_strategy(options.exec, Operation::DensityEval, data.n_obs())?;
        let view = data.as_view();
        parallel_try_map_range_collect(data.n_obs(), strategy, |row_idx| {
            let row = view.row(row_idx);
            let z = row
                .iter()
                .map(|value| {
                    let clipped = value.clamp(options.clip_eps, 1.0 - options.clip_eps);
                    standard_normal.inverse_cdf(clipped)
                })
                .collect::<Vec<_>>();
            let quadratic = quadratic_form_from_cholesky(&self.cholesky, &z)?;
            let euclidean = z.iter().map(|value| value * value).sum::<f64>();
            Ok(-0.5 * self.log_det - 0.5 * (quadratic - euclidean))
        })
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        resolve_strategy(options.exec, Operation::Sample, n)?;
        let standard_normal = Self::standard_normal();
        let mut samples = Array2::zeros((n, self.dim));

        for row_idx in 0..n {
            let epsilon = (0..self.dim)
                .map(|_| rng.sample::<f64, _>(StandardNormal))
                .collect::<Vec<_>>();
            let mut correlated = vec![0.0; self.dim];
            for (i, correlated_value) in correlated.iter_mut().enumerate() {
                let mut value = 0.0;
                for (j, epsilon_value) in epsilon.iter().enumerate().take(i + 1) {
                    value += self.cholesky[(i, j)] * epsilon_value;
                }
                *correlated_value = value;
            }

            for col_idx in 0..self.dim {
                samples[(row_idx, col_idx)] = standard_normal.cdf(correlated[col_idx]);
            }
        }

        Ok(samples)
    }
}

/// Student t copula parameterized by a correlation matrix and degrees of freedom.
#[derive(Debug, Clone, Serialize)]
pub struct StudentTCopula {
    dim: usize,
    correlation: Array2<f64>,
    degrees_of_freedom: f64,
    #[serde(skip)]
    cholesky: Array2<f64>,
    log_det: f64,
}

impl StudentTCopula {
    /// Constructs a Student t copula from a valid correlation matrix and `nu > 0`.
    pub fn new(correlation: Array2<f64>, degrees_of_freedom: f64) -> Result<Self, CopulaError> {
        if !degrees_of_freedom.is_finite() || degrees_of_freedom <= 0.0 {
            return Err(FitError::Failed {
                reason: "degrees of freedom must be positive",
            }
            .into());
        }

        validate_correlation_matrix(&correlation)?;
        let cholesky = cholesky(&correlation)?;
        let log_det = log_determinant_from_cholesky(&cholesky);
        let dim = correlation.ncols();
        Ok(Self {
            dim,
            correlation,
            degrees_of_freedom,
            cholesky,
            log_det,
        })
    }

    /// Fits a Student t copula by Kendall tau inversion plus a grid search over `nu`.
    pub fn fit(data: &PseudoObs, options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        let tau = try_kendall_tau_matrix(data, options.exec)?;
        let correlation =
            make_spd_correlation(&tau.mapv(|value| (std::f64::consts::FRAC_PI_2 * value).sin()))?;

        let mut best_model = None;
        let mut best_loglik = f64::NEG_INFINITY;
        let mut iterations = 0usize;
        for degrees_of_freedom in candidate_degrees_of_freedom() {
            iterations += 1;
            let candidate = Self::new(correlation.clone(), degrees_of_freedom)?;
            let loglik = candidate
                .log_pdf(
                    data,
                    &EvalOptions {
                        exec: options.exec,
                        clip_eps: options.clip_eps,
                    },
                )?
                .into_iter()
                .sum::<f64>();

            if loglik > best_loglik {
                best_loglik = loglik;
                best_model = Some(candidate);
            }
        }

        let model = best_model.ok_or(FitError::Failed {
            reason: "failed to select degrees of freedom",
        })?;
        let parameter_count = (model.dim * (model.dim - 1) / 2 + 1) as f64;
        let n_obs = data.n_obs() as f64;
        let diagnostics = super::FitDiagnostics {
            loglik: best_loglik,
            aic: 2.0 * parameter_count - 2.0 * best_loglik,
            bic: parameter_count * n_obs.ln() - 2.0 * best_loglik,
            converged: true,
            n_iter: iterations,
        };

        Ok(FitResult { model, diagnostics })
    }

    /// Returns the fitted correlation matrix.
    pub fn correlation(&self) -> &Array2<f64> {
        &self.correlation
    }

    /// Returns the fitted degrees of freedom.
    pub fn degrees_of_freedom(&self) -> f64 {
        self.degrees_of_freedom
    }

    fn univariate_t(&self) -> StudentsT {
        StudentsT::new(0.0, 1.0, self.degrees_of_freedom)
            .expect("validated degrees of freedom should construct a t distribution")
    }
}

#[derive(Deserialize)]
struct StudentTCopulaRepr {
    #[serde(default)]
    dim: Option<usize>,
    correlation: Array2<f64>,
    degrees_of_freedom: f64,
    #[serde(default)]
    log_det: Option<f64>,
}

impl<'de> Deserialize<'de> for StudentTCopula {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = StudentTCopulaRepr::deserialize(deserializer)?;
        let model = Self::new(repr.correlation, repr.degrees_of_freedom)
            .map_err(serde::de::Error::custom)?;
        if repr.dim.is_some_and(|dim| dim != model.dim) {
            return Err(serde::de::Error::custom(
                "serialized StudentTCopula dim does not match correlation",
            ));
        }
        if repr.log_det.is_some_and(|log_det| !log_det.is_finite()) {
            return Err(serde::de::Error::custom(
                "serialized StudentTCopula log_det must be finite",
            ));
        }
        Ok(model)
    }
}

impl CopulaModel for StudentTCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::StudentT
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

        let t_distribution = self.univariate_t();
        let dim = self.dim as f64;
        let nu = self.degrees_of_freedom;
        let constant = ln_gamma((nu + dim) / 2.0)
            - ln_gamma(nu / 2.0)
            - 0.5 * self.log_det
            - 0.5 * dim * (nu * std::f64::consts::PI).ln();

        let strategy = resolve_strategy(options.exec, Operation::DensityEval, data.n_obs())?;
        let view = data.as_view();
        parallel_try_map_range_collect(data.n_obs(), strategy, |row_idx| {
            let row = view.row(row_idx);
            let z = row
                .iter()
                .map(|value| {
                    let clipped = value.clamp(options.clip_eps, 1.0 - options.clip_eps);
                    t_distribution.inverse_cdf(clipped)
                })
                .collect::<Vec<_>>();
            let quadratic = quadratic_form_from_cholesky(&self.cholesky, &z)?;
            let mv_log_pdf = constant - 0.5 * (nu + dim) * (1.0 + quadratic / nu).ln();
            let marginal_log_pdf = z
                .iter()
                .map(|value| t_distribution.ln_pdf(*value))
                .sum::<f64>();
            Ok(mv_log_pdf - marginal_log_pdf)
        })
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        resolve_strategy(options.exec, Operation::Sample, n)?;
        let t_distribution = self.univariate_t();
        let chi_squared =
            ChiSquared::new(self.degrees_of_freedom).map_err(|_| FitError::Failed {
                reason: "degrees of freedom must be positive",
            })?;
        let mut samples = Array2::zeros((n, self.dim));

        for row_idx in 0..n {
            let epsilon = (0..self.dim)
                .map(|_| rng.sample::<f64, _>(StandardNormal))
                .collect::<Vec<_>>();
            let scale = (self.degrees_of_freedom / rng.sample::<f64, _>(chi_squared)).sqrt();
            let mut correlated = vec![0.0; self.dim];
            for (i, correlated_value) in correlated.iter_mut().enumerate() {
                let mut value = 0.0;
                for (j, epsilon_value) in epsilon.iter().enumerate().take(i + 1) {
                    value += self.cholesky[(i, j)] * epsilon_value;
                }
                *correlated_value = value * scale;
            }

            for col_idx in 0..self.dim {
                samples[(row_idx, col_idx)] = t_distribution.cdf(correlated[col_idx]);
            }
        }

        Ok(samples)
    }
}

fn candidate_degrees_of_freedom() -> Vec<f64> {
    let min = 2.1_f64.ln();
    let max = 50.0_f64.ln();
    let steps = 40usize;

    (0..steps)
        .map(|idx| {
            let fraction = idx as f64 / (steps - 1) as f64;
            (min + (max - min) * fraction).exp()
        })
        .collect()
}
