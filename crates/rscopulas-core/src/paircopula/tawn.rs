use crate::errors::{CopulaError, FitError};

// Tawn extreme-value bivariate copula with asymmetric Pickands dependence
//   A(t) = (1 - α)(1 - t) + (1 - β) t + [(α(1-t))^θ + (β t)^θ]^(1/θ).
//
// This module exposes the *general* 3-parameter form; the dispatch layer
// wires the two named families VineCopula supports (Tawn1 and Tawn2) by
// fixing β = 1 or α = 1 respectively:
//
//   Tawn1 (VineCopula family 104/114/124/134): β = 1, par = θ, par2 = α.
//   Tawn2 (VineCopula family 204/214/224/234): α = 1, par = θ, par2 = β.
//
// Extreme-value density derivation (Schepsmeier 2014, Gudendorf & Segers):
//   Let z = -ln(u v) > 0 and t = -ln(v) / z so t, 1-t ∈ (0, 1).
//   C(u, v) = exp(-z · A(t)).
//   ∂C/∂u = C / u · (A - t · A').
//   ∂C/∂v = C / v · (A + (1-t) · A').
//   c(u, v) = C / (u v) · [(A - t A')(A + (1-t) A') + t(1-t) A'' / z].
//
// All 6 Archimedean helpers below share this derivation; the `(α, β)` pair is
// kept explicit so we can validate both named families against VineCopula with
// the same code path.

fn t_from_logs(u1: f64, u2: f64) -> (f64, f64) {
    // Returns (z, t) with z = -ln(u1·u2) and t = -ln(u2) / z.
    let z = -(u1.ln() + u2.ln());
    let t = -u2.ln() / z;
    (z, t)
}

pub(super) fn pickands_a(t: f64, theta: f64, alpha: f64, beta: f64) -> f64 {
    let p = (alpha * (1.0 - t)).powf(theta) + (beta * t).powf(theta);
    (1.0 - alpha) * (1.0 - t) + (1.0 - beta) * t + p.powf(1.0 / theta)
}

pub(super) fn pickands_a_prime(t: f64, theta: f64, alpha: f64, beta: f64) -> f64 {
    // A'(t) = (α - β) + p^(1/θ - 1) · [β^θ t^(θ-1) - α^θ (1-t)^(θ-1)]
    // where p = (α(1-t))^θ + (β t)^θ.
    let p = (alpha * (1.0 - t)).powf(theta) + (beta * t).powf(theta);
    let q = beta.powf(theta) * t.powf(theta - 1.0)
        - alpha.powf(theta) * (1.0 - t).powf(theta - 1.0);
    (alpha - beta) + p.powf(1.0 / theta - 1.0) * q
}

pub(super) fn pickands_a_double_prime(t: f64, theta: f64, alpha: f64, beta: f64) -> f64 {
    // A''(t) = (θ-1) p^(1/θ - 2) · [p · R - Q²],
    //   R = β^θ t^(θ-2) + α^θ (1-t)^(θ-2)
    //   Q = β^θ t^(θ-1) - α^θ (1-t)^(θ-1)
    let p = (alpha * (1.0 - t)).powf(theta) + (beta * t).powf(theta);
    let q = beta.powf(theta) * t.powf(theta - 1.0)
        - alpha.powf(theta) * (1.0 - t).powf(theta - 1.0);
    let r = beta.powf(theta) * t.powf(theta - 2.0)
        + alpha.powf(theta) * (1.0 - t).powf(theta - 2.0);
    (theta - 1.0) * p.powf(1.0 / theta - 2.0) * (p * r - q * q)
}

fn validate_params(theta: f64, alpha: f64, beta: f64) -> Result<(), CopulaError> {
    if !theta.is_finite() || theta < 1.0 {
        return Err(FitError::Failed {
            reason: "tawn pair theta must be at least 1",
        }
        .into());
    }
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(FitError::Failed {
            reason: "tawn pair alpha must lie in [0, 1]",
        }
        .into());
    }
    if !beta.is_finite() || !(0.0..=1.0).contains(&beta) {
        return Err(FitError::Failed {
            reason: "tawn pair beta must lie in [0, 1]",
        }
        .into());
    }
    Ok(())
}

pub fn log_pdf(
    u1: f64,
    u2: f64,
    theta: f64,
    alpha: f64,
    beta: f64,
) -> Result<f64, CopulaError> {
    validate_params(theta, alpha, beta)?;
    let (z, t) = t_from_logs(u1, u2);
    let a = pickands_a(t, theta, alpha, beta);
    let ap = pickands_a_prime(t, theta, alpha, beta);
    let app = pickands_a_double_prime(t, theta, alpha, beta);
    let bracket = (a - t * ap) * (a + (1.0 - t) * ap) + t * (1.0 - t) * app / z;
    // log c = log C - log u1 - log u2 + log bracket = -z·A - log u1 - log u2 + log bracket.
    Ok(-z * a - u1.ln() - u2.ln() + bracket.ln())
}

pub fn cond_first_given_second(
    u1: f64,
    u2: f64,
    theta: f64,
    alpha: f64,
    beta: f64,
) -> Result<f64, CopulaError> {
    validate_params(theta, alpha, beta)?;
    let (z, t) = t_from_logs(u1, u2);
    let a = pickands_a(t, theta, alpha, beta);
    let ap = pickands_a_prime(t, theta, alpha, beta);
    let cdf_val = (-z * a).exp();
    Ok(cdf_val / u2 * (a + (1.0 - t) * ap))
}

pub fn cond_second_given_first(
    u1: f64,
    u2: f64,
    theta: f64,
    alpha: f64,
    beta: f64,
) -> Result<f64, CopulaError> {
    validate_params(theta, alpha, beta)?;
    let (z, t) = t_from_logs(u1, u2);
    let a = pickands_a(t, theta, alpha, beta);
    let ap = pickands_a_prime(t, theta, alpha, beta);
    let cdf_val = (-z * a).exp();
    Ok(cdf_val / u1 * (a - t * ap))
}

pub fn inv_first_given_second(
    p: f64,
    u2: f64,
    theta: f64,
    alpha: f64,
    beta: f64,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    // h_{1|2}(u|v) is strictly increasing in u (EV copulas are positively
    // ordered), so bisection converges cleanly. 90 iterations = ~double
    // precision over the [clip_eps, 1 - clip_eps] bracket.
    let mut low = clip_eps;
    let mut high = 1.0 - clip_eps;
    for _ in 0..90 {
        let mid = 0.5 * (low + high);
        if cond_first_given_second(mid, u2, theta, alpha, beta)? < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(0.5 * (low + high))
}

pub fn inv_second_given_first(
    u1: f64,
    p: f64,
    theta: f64,
    alpha: f64,
    beta: f64,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let mut low = clip_eps;
    let mut high = 1.0 - clip_eps;
    for _ in 0..90 {
        let mid = 0.5 * (low + high);
        if cond_second_given_first(u1, mid, theta, alpha, beta)? < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(0.5 * (low + high))
}

pub fn cdf(
    u1: f64,
    u2: f64,
    theta: f64,
    alpha: f64,
    beta: f64,
) -> Result<f64, CopulaError> {
    validate_params(theta, alpha, beta)?;
    let (z, t) = t_from_logs(u1, u2);
    let a = pickands_a(t, theta, alpha, beta);
    Ok((-z * a).exp().clamp(0.0, 1.0))
}
