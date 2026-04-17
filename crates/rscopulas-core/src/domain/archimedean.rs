use ndarray::Array2;
use rand::Rng;
use rand_distr::{Exp1, Gamma};
use serde::{Deserialize, Serialize};

use crate::{
    data::PseudoObs,
    errors::{CopulaError, FitError},
    fit::FitResult,
    stats::{kendall_tau_matrix, mean_off_diagonal},
};

use super::{CopulaFamily, CopulaModel, EvalOptions, FitOptions, SampleOptions};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaytonCopula {
    dim: usize,
    theta: f64,
}

impl ClaytonCopula {
    pub fn new(dim: usize, theta: f64) -> Result<Self, CopulaError> {
        validate_archimedean_parameters("Clayton", dim, theta, 0.0, false)?;
        Ok(Self { dim, theta })
    }

    pub fn fit(data: &PseudoObs, _options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        let mean_tau = fit_mean_tau("Clayton", data)?;
        let theta = 2.0 * mean_tau / (1.0 - mean_tau);
        let model = Self::new(data.dim(), theta)?;
        fit_result(model, data)
    }

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
        let mut values = Vec::with_capacity(data.n_obs());
        for row in data.as_view().rows() {
            let clipped = clipped_row(&row.to_vec(), options.clip_eps);
            let sum_power = clipped.iter().map(|value| value.powf(-theta)).sum::<f64>() - dim + 1.0;
            let sum_logs = clipped.iter().map(|value| value.ln()).sum::<f64>();
            let log_pdf =
                log_coeff - (1.0 + theta) * sum_logs - (dim + 1.0 / theta) * sum_power.ln();
            values.push(log_pdf);
        }

        Ok(values)
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        _options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrankCopula {
    dim: usize,
    theta: f64,
}

impl FrankCopula {
    pub fn new(dim: usize, theta: f64) -> Result<Self, CopulaError> {
        validate_archimedean_parameters("Frank", dim, theta, 0.0, false)?;
        Ok(Self { dim, theta })
    }

    pub fn fit(data: &PseudoObs, _options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        let target_tau = fit_mean_tau("Frank", data)?;
        let theta = invert_frank_tau(target_tau)?;
        let model = Self::new(data.dim(), theta)?;
        fit_result(model, data)
    }

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
        let mut values = Vec::with_capacity(data.n_obs());
        for row in data.as_view().rows() {
            let clipped = clipped_row(&row.to_vec(), options.clip_eps);
            let t = clipped
                .iter()
                .map(|value| {
                    let numerator = (-theta * value).exp_m1().abs();
                    let denominator = (-theta).exp_m1().abs();
                    -(numerator / denominator).ln()
                })
                .sum::<f64>();
            let q = a * (-t).exp();
            let log_abs_derivative = frank_log_abs_generator_derivative(dim, q, theta);
            let log_phi = clipped
                .iter()
                .map(|value| theta.ln() - (theta * value).exp_m1().ln())
                .sum::<f64>();
            values.push(log_abs_derivative + log_phi);
        }

        Ok(values)
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        _options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        let p = 1.0 - (-self.theta).exp();
        let mut samples = Array2::zeros((n, self.dim));

        for row_idx in 0..n {
            let frailty = sample_log_series(rng, p);
            for col_idx in 0..self.dim {
                let exponential = rng.sample::<f64, _>(Exp1);
                samples[(row_idx, col_idx)] =
                    frank_generator(exponential / frailty as f64, self.theta);
            }
        }

        Ok(samples)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GumbelHougaardCopula {
    dim: usize,
    theta: f64,
}

impl GumbelHougaardCopula {
    pub fn new(dim: usize, theta: f64) -> Result<Self, CopulaError> {
        validate_archimedean_parameters("Gumbel", dim, theta, 1.0, true)?;
        Ok(Self { dim, theta })
    }

    pub fn fit(data: &PseudoObs, _options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        let mean_tau = fit_mean_tau("Gumbel", data)?;
        let theta = 1.0 / (1.0 - mean_tau);
        let model = Self::new(data.dim(), theta)?;
        fit_result(model, data)
    }

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
        let mut values = Vec::with_capacity(data.n_obs());
        for row in data.as_view().rows() {
            let clipped = clipped_row(&row.to_vec(), options.clip_eps);
            let t = clipped
                .iter()
                .map(|value| (-value.ln()).powf(self.theta))
                .sum::<f64>();
            let log_abs_derivative = gumbel_log_abs_generator_derivative(self.dim, t, alpha);
            let log_phi = clipped
                .iter()
                .map(|value| self.theta.ln() + (self.theta - 1.0) * (-value.ln()).ln() - value.ln())
                .sum::<f64>();
            values.push(log_abs_derivative + log_phi);
        }

        Ok(values)
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        _options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
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

fn fit_result<T>(model: T, data: &PseudoObs) -> Result<FitResult<T>, CopulaError>
where
    T: CopulaModel,
{
    let loglik = model
        .log_pdf(data, &EvalOptions::default())?
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

fn fit_mean_tau(family: &'static str, data: &PseudoObs) -> Result<f64, CopulaError> {
    let tau = kendall_tau_matrix(data);
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

fn frank_generator(t: f64, theta: f64) -> f64 {
    let scale = 1.0 - (-theta).exp();
    -(1.0 - scale * (-t).exp()).ln() / theta
}

fn frank_log_abs_generator_derivative(dim: usize, q: f64, theta: f64) -> f64 {
    let numerator_poly = frank_derivative_polynomial(dim, q);
    q.ln() + numerator_poly.abs().ln() - (dim as f64) * (1.0 - q).ln() - theta.ln()
}

fn frank_derivative_polynomial(dim: usize, q: f64) -> f64 {
    let mut coeffs = vec![1.0];
    for n in 1..dim {
        let mut deriv = vec![0.0; coeffs.len().saturating_sub(1)];
        for idx in 1..coeffs.len() {
            deriv[idx - 1] = idx as f64 * coeffs[idx];
        }

        let mut next = vec![0.0; coeffs.len() + 1];
        for (idx, coeff) in coeffs.iter().enumerate() {
            next[idx] += coeff;
            next[idx + 1] += (n as f64 - 1.0) * coeff;
        }
        for (idx, coeff) in deriv.iter().enumerate() {
            next[idx + 1] += coeff;
            next[idx + 2] -= coeff;
        }
        coeffs = next;
    }

    coeffs
        .iter()
        .enumerate()
        .map(|(idx, coeff)| coeff * q.powi(idx as i32))
        .sum()
}

fn debye_1(theta: f64) -> f64 {
    if theta.abs() < 1e-6 {
        return 1.0 - theta / 4.0 + theta * theta / 36.0;
    }

    let intervals = 1024usize;
    let step = theta / intervals as f64;
    let integrand = |x: f64| -> f64 { if x == 0.0 { 1.0 } else { x / x.exp_m1() } };

    let mut total = integrand(0.0) + integrand(theta);
    for idx in 1..intervals {
        let x = idx as f64 * step;
        total += if idx % 2 == 0 {
            2.0 * integrand(x)
        } else {
            4.0 * integrand(x)
        };
    }

    (step / 3.0) * total / theta
}

fn frank_tau(theta: f64) -> f64 {
    1.0 - 4.0 / theta + 4.0 * debye_1(theta) / theta
}

fn invert_frank_tau(target_tau: f64) -> Result<f64, CopulaError> {
    let mut low = 1e-6;
    let mut high = 1.0;
    while frank_tau(high) < target_tau && high < 1e6 {
        high *= 2.0;
    }

    if frank_tau(high) < target_tau {
        return Err(FitError::Failed {
            reason: "Frank tau inversion failed to bracket root",
        }
        .into());
    }

    for _ in 0..80 {
        let mid = 0.5 * (low + high);
        if frank_tau(mid) < target_tau {
            low = mid;
        } else {
            high = mid;
        }
    }

    Ok(0.5 * (low + high))
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

fn gumbel_log_abs_generator_derivative(dim: usize, t: f64, alpha: f64) -> f64 {
    let derivatives = (1..=dim)
        .map(|order| gumbel_g_derivative(order, t, alpha))
        .collect::<Vec<_>>();
    let bell = complete_bell_polynomial(&derivatives);
    -t.powf(alpha) + bell.abs().ln()
}

fn gumbel_g_derivative(order: usize, t: f64, alpha: f64) -> f64 {
    let falling = (0..order).fold(1.0, |acc, idx| acc * (alpha - idx as f64));
    -falling * t.powf(alpha - order as f64)
}

fn complete_bell_polynomial(derivatives: &[f64]) -> f64 {
    let n = derivatives.len();
    let mut bell = vec![0.0; n + 1];
    bell[0] = 1.0;

    for order in 1..=n {
        let mut total = 0.0;
        for k in 1..=order {
            total += binomial(order - 1, k - 1) * derivatives[k - 1] * bell[order - k];
        }
        bell[order] = total;
    }

    bell[n]
}

fn binomial(n: usize, k: usize) -> f64 {
    if k == 0 || k == n {
        return 1.0;
    }

    let k = k.min(n - k);
    let mut value = 1.0;
    for idx in 0..k {
        value *= (n - idx) as f64 / (idx + 1) as f64;
    }
    value
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
