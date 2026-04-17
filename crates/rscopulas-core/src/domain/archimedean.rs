use ndarray::Array2;
use rand::Rng;
use rand_distr::{Exp1, Gamma};
use serde::{Deserialize, Serialize};

use crate::{
    archimedean_math::{frank, gumbel},
    backend::{Operation, parallel_try_map_range_collect, resolve_strategy},
    data::PseudoObs,
    errors::{CopulaError, FitError},
    fit::FitResult,
    stats::{mean_off_diagonal, try_kendall_tau_matrix},
};

use super::{CopulaFamily, CopulaModel, EvalOptions, FitOptions, SampleOptions};

/// Clayton copula with `theta > 0`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaytonCopula {
    dim: usize,
    theta: f64,
}

impl ClaytonCopula {
    /// Constructs a Clayton copula of dimension `dim` with `theta > 0`.
    pub fn new(dim: usize, theta: f64) -> Result<Self, CopulaError> {
        validate_archimedean_parameters("Clayton", dim, theta, 0.0, false)?;
        Ok(Self { dim, theta })
    }

    /// Fits a Clayton copula from the mean pairwise Kendall tau.
    pub fn fit(data: &PseudoObs, options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        let mean_tau = fit_mean_tau("Clayton", data, options)?;
        let theta = 2.0 * mean_tau / (1.0 - mean_tau);
        let model = Self::new(data.dim(), theta)?;
        fit_result(model, data, options)
    }

    /// Returns the fitted `theta` parameter.
    pub fn theta(&self) -> f64 {
        self.theta
    }
}

impl CopulaModel for ClaytonCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::Clayton
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn log_pdf(&self, data: &PseudoObs, options: &EvalOptions) -> Result<Vec<f64>, CopulaError> {
        validate_input_dim(self.dim, data)?;

        let dim = self.dim as f64;
        let theta = self.theta;
        let log_coeff = (0..self.dim)
            .map(|idx| (1.0 + idx as f64 * theta).ln())
            .sum::<f64>();
        evaluate_density_rows(data, options, move |row| {
            let clipped = clipped_row(row, options.clip_eps);
            let sum_power = clipped.iter().map(|value| value.powf(-theta)).sum::<f64>() - dim + 1.0;
            let sum_logs = clipped.iter().map(|value| value.ln()).sum::<f64>();
            Ok(log_coeff - (1.0 + theta) * sum_logs - (dim + 1.0 / theta) * sum_power.ln())
        })
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        resolve_strategy(options.exec, Operation::Sample, n)?;
        let frailty = Gamma::new(1.0 / self.theta, 1.0).map_err(|_| FitError::Failed {
            reason: "failed to construct Clayton frailty sampler",
        })?;
        let mut samples = Array2::zeros((n, self.dim));

        for row_idx in 0..n {
            let shared = rng.sample::<f64, _>(frailty);
            for col_idx in 0..self.dim {
                let exponential = rng.sample::<f64, _>(Exp1);
                samples[(row_idx, col_idx)] = (1.0 + exponential / shared).powf(-1.0 / self.theta);
            }
        }

        Ok(samples)
    }
}

/// Frank copula with `theta > 0`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrankCopula {
    dim: usize,
    theta: f64,
}

impl FrankCopula {
    /// Constructs a Frank copula of dimension `dim` with `theta > 0`.
    pub fn new(dim: usize, theta: f64) -> Result<Self, CopulaError> {
        validate_archimedean_parameters("Frank", dim, theta, 0.0, false)?;
        Ok(Self { dim, theta })
    }

    /// Fits a Frank copula from the mean pairwise Kendall tau.
    pub fn fit(data: &PseudoObs, options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        let target_tau = fit_mean_tau("Frank", data, options)?;
        let theta = frank::invert_tau(target_tau, "Frank tau inversion failed to bracket root")?;
        let model = Self::new(data.dim(), theta)?;
        fit_result(model, data, options)
    }

    /// Returns the fitted `theta` parameter.
    pub fn theta(&self) -> f64 {
        self.theta
    }
}

impl CopulaModel for FrankCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::Frank
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn log_pdf(&self, data: &PseudoObs, options: &EvalOptions) -> Result<Vec<f64>, CopulaError> {
        validate_input_dim(self.dim, data)?;

        let dim = self.dim;
        let theta = self.theta;
        let a = 1.0 - (-theta).exp();
        evaluate_density_rows(data, options, move |row| {
            let clipped = clipped_row(row, options.clip_eps);
            let t = clipped
                .iter()
                .map(|value| {
                    let numerator = (-theta * value).exp_m1().abs();
                    let denominator = (-theta).exp_m1().abs();
                    -(numerator / denominator).ln()
                })
                .sum::<f64>();
            let q = a * (-t).exp();
            let log_abs_derivative = frank::log_abs_generator_derivative(dim, q, theta);
            let log_phi = clipped
                .iter()
                .map(|value| theta.ln() - (theta * value).exp_m1().ln())
                .sum::<f64>();
            Ok(log_abs_derivative + log_phi)
        })
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        resolve_strategy(options.exec, Operation::Sample, n)?;
        let p = 1.0 - (-self.theta).exp();
        let mut samples = Array2::zeros((n, self.dim));

        for row_idx in 0..n {
            let frailty = sample_log_series(rng, p);
            for col_idx in 0..self.dim {
                let exponential = rng.sample::<f64, _>(Exp1);
                samples[(row_idx, col_idx)] =
                    frank::generator(exponential / frailty as f64, self.theta);
            }
        }

        Ok(samples)
    }
}

/// Gumbel-Hougaard copula with `theta >= 1`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GumbelHougaardCopula {
    dim: usize,
    theta: f64,
}

impl GumbelHougaardCopula {
    /// Constructs a Gumbel-Hougaard copula of dimension `dim` with `theta >= 1`.
    pub fn new(dim: usize, theta: f64) -> Result<Self, CopulaError> {
        validate_archimedean_parameters("Gumbel", dim, theta, 1.0, true)?;
        Ok(Self { dim, theta })
    }

    /// Fits a Gumbel-Hougaard copula from the mean pairwise Kendall tau.
    pub fn fit(data: &PseudoObs, options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        let mean_tau = fit_mean_tau("Gumbel", data, options)?;
        let theta = 1.0 / (1.0 - mean_tau);
        let model = Self::new(data.dim(), theta)?;
        fit_result(model, data, options)
    }

    /// Returns the fitted `theta` parameter.
    pub fn theta(&self) -> f64 {
        self.theta
    }
}

impl CopulaModel for GumbelHougaardCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::Gumbel
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn log_pdf(&self, data: &PseudoObs, options: &EvalOptions) -> Result<Vec<f64>, CopulaError> {
        validate_input_dim(self.dim, data)?;

        let alpha = 1.0 / self.theta;
        let theta = self.theta;
        let dim = self.dim;
        evaluate_density_rows(data, options, move |row| {
            let clipped = clipped_row(row, options.clip_eps);
            let t = clipped
                .iter()
                .map(|value| (-value.ln()).powf(theta))
                .sum::<f64>();
            let log_abs_derivative = gumbel::log_abs_generator_derivative(dim, t, alpha);
            let log_phi = clipped
                .iter()
                .map(|value| theta.ln() + (theta - 1.0) * (-value.ln()).ln() - value.ln())
                .sum::<f64>();
            Ok(log_abs_derivative + log_phi)
        })
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        resolve_strategy(options.exec, Operation::Sample, n)?;
        let alpha = 1.0 / self.theta;
        let mut samples = Array2::zeros((n, self.dim));

        for row_idx in 0..n {
            let frailty = sample_positive_stable(rng, alpha);
            for col_idx in 0..self.dim {
                let exponential = rng.sample::<f64, _>(Exp1);
                samples[(row_idx, col_idx)] = (-(exponential / frailty).powf(alpha)).exp();
            }
        }

        Ok(samples)
    }
}

fn validate_archimedean_parameters(
    family: &'static str,
    dim: usize,
    theta: f64,
    min_theta: f64,
    inclusive_lower_bound: bool,
) -> Result<(), CopulaError> {
    if dim < 2 {
        return Err(FitError::UnsupportedDimension { family, dim }.into());
    }

    let invalid = if inclusive_lower_bound {
        !theta.is_finite() || theta < min_theta
    } else {
        !theta.is_finite() || theta <= min_theta
    };

    if invalid {
        return Err(FitError::Failed {
            reason: match family {
                "Clayton" => "Clayton theta must be positive",
                "Frank" => "Frank theta must be positive",
                "Gumbel" => "Gumbel theta must be at least 1",
                _ => "invalid Archimedean parameter",
            },
        }
        .into());
    }

    Ok(())
}

fn fit_result<T>(
    model: T,
    data: &PseudoObs,
    options: &FitOptions,
) -> Result<FitResult<T>, CopulaError>
where
    T: CopulaModel,
{
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
    let n_obs = data.n_obs() as f64;
    let diagnostics = super::FitDiagnostics {
        loglik,
        aic: 2.0 - 2.0 * loglik,
        bic: n_obs.ln() - 2.0 * loglik,
        converged: true,
        n_iter: 0,
    };

    Ok(FitResult { model, diagnostics })
}

fn fit_mean_tau(
    family: &'static str,
    data: &PseudoObs,
    options: &FitOptions,
) -> Result<f64, CopulaError> {
    let tau = try_kendall_tau_matrix(data, options.exec)?;
    let mean_tau = mean_off_diagonal(&tau);
    if mean_tau <= 0.0 || mean_tau >= 1.0 {
        return Err(FitError::Failed {
            reason: match family {
                "Clayton" => "Clayton fit requires mean Kendall tau strictly between 0 and 1",
                "Frank" => "Frank fit requires mean Kendall tau strictly between 0 and 1",
                "Gumbel" => "Gumbel fit requires mean Kendall tau strictly between 0 and 1",
                _ => "invalid Kendall tau for Archimedean fit",
            },
        }
        .into());
    }

    Ok(mean_tau)
}

fn validate_input_dim(expected_dim: usize, data: &PseudoObs) -> Result<(), CopulaError> {
    if data.dim() != expected_dim {
        return Err(FitError::Failed {
            reason: "input dimension does not match model dimension",
        }
        .into());
    }

    Ok(())
}

fn clipped_row(row: &[f64], clip_eps: f64) -> Vec<f64> {
    row.iter()
        .map(|value| value.clamp(clip_eps, 1.0 - clip_eps))
        .collect()
}

fn evaluate_density_rows<F>(
    data: &PseudoObs,
    options: &EvalOptions,
    evaluator: F,
) -> Result<Vec<f64>, CopulaError>
where
    F: Fn(&[f64]) -> Result<f64, CopulaError> + Sync + Send,
{
    let strategy = resolve_strategy(options.exec, Operation::DensityEval, data.n_obs())?;
    let view = data.as_view();
    parallel_try_map_range_collect(data.n_obs(), strategy, |row_idx| {
        let row = view.row(row_idx);
        let values = row.iter().copied().collect::<Vec<_>>();
        evaluator(&values)
    })
}

fn sample_log_series<R: Rng + ?Sized>(rng: &mut R, p: f64) -> usize {
    let normalizer = -1.0 / (1.0 - p).ln();
    let threshold: f64 = rng.random();
    let mut cumulative = 0.0;
    let mut probability = normalizer * p;
    let mut k = 1usize;

    loop {
        cumulative += probability;
        if threshold <= cumulative {
            return k;
        }

        k += 1;
        probability *= p * (k as f64 - 1.0) / k as f64;
    }
}

fn sample_positive_stable<R: Rng + ?Sized>(rng: &mut R, alpha: f64) -> f64 {
    if (alpha - 1.0).abs() < 1e-12 {
        return 1.0;
    }

    let uniform: f64 = rng.random::<f64>() * std::f64::consts::PI;
    let exponential = rng.sample::<f64, _>(Exp1);
    let left = (alpha * uniform).sin() / uniform.sin().powf(1.0 / alpha);
    let right = (((1.0 - alpha) * uniform).sin() / exponential).powf((1.0 - alpha) / alpha);
    left * right
}
