use crate::archimedean_math::joe;
use crate::errors::{CopulaError, FitError};

// Joe bivariate copula with parameter θ ∈ [1, ∞).
//
// Notation used below:
//   a = (1 - u1)^θ,   b = (1 - u2)^θ,   s = a + b - a·b = 1 - (1-a)(1-b).
// The CDF is `C(u1, u2) = 1 - s^(1/θ)` and the density is derived by
// differentiating C twice:
//   c(u1, u2) = (1-u1)^(θ-1) · (1-u2)^(θ-1) · s^(1/θ - 2) · (s + θ - 1).
// For θ ≥ 1 and (u1, u2) ∈ (0, 1)² all quantities are finite and positive, so
// the log-density is computed directly without the stabilising tricks Frank
// needs (Joe's density does not suffer the same catastrophic cancellation).

pub fn theta_from_tau(tau: f64) -> Result<f64, CopulaError> {
    if !tau.is_finite() || tau <= 0.0 || tau >= 1.0 {
        return Err(FitError::Failed {
            reason: "joe pair fit requires tau in (0, 1)",
        }
        .into());
    }
    joe::invert_tau(tau, "joe tau inversion failed to bracket root")
}

pub fn log_pdf(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    if !theta.is_finite() || theta < 1.0 {
        return Err(FitError::Failed {
            reason: "joe pair theta must be at least 1",
        }
        .into());
    }
    let a = (1.0 - u1).powf(theta);
    let b = (1.0 - u2).powf(theta);
    let s = a + b - a * b;
    Ok((theta - 1.0) * (1.0 - u1).ln()
        + (theta - 1.0) * (1.0 - u2).ln()
        + (1.0 / theta - 2.0) * s.ln()
        + (s + theta - 1.0).ln())
}

pub fn cond_first_given_second(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    // h_{1|2}(u1 | u2) = ∂C/∂u2 = s^(1/θ - 1) · (1 - a) · (1-u2)^(θ-1)
    let a = (1.0 - u1).powf(theta);
    let b = (1.0 - u2).powf(theta);
    let s = a + b - a * b;
    Ok(s.powf(1.0 / theta - 1.0) * (1.0 - a) * (1.0 - u2).powf(theta - 1.0))
}

pub fn cond_second_given_first(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    // h_{2|1}(u2 | u1) = ∂C/∂u1 = s^(1/θ - 1) · (1 - b) · (1-u1)^(θ-1)
    let a = (1.0 - u1).powf(theta);
    let b = (1.0 - u2).powf(theta);
    let s = a + b - a * b;
    Ok(s.powf(1.0 / theta - 1.0) * (1.0 - b) * (1.0 - u1).powf(theta - 1.0))
}

pub fn inv_first_given_second(
    p: f64,
    u2: f64,
    theta: f64,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    // Bisection mirrors the Gumbel pattern — Joe's inverse h has no closed
    // form (its equation is transcendental in (1-u1)^θ) so numerical root
    // finding is the standard treatment here.
    let mut low = clip_eps;
    let mut high = 1.0 - clip_eps;
    for _ in 0..90 {
        let mid = 0.5 * (low + high);
        if cond_first_given_second(mid, u2, theta)? < p {
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
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let mut low = clip_eps;
    let mut high = 1.0 - clip_eps;
    for _ in 0..90 {
        let mid = 0.5 * (low + high);
        if cond_second_given_first(u1, mid, theta)? < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(0.5 * (low + high))
}
