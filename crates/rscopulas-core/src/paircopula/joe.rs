use crate::archimedean_math::joe;
use crate::errors::{CopulaError, FitError};
use crate::math::{log1mexp, logaddexp};

fn terms(u: f64, v: f64, theta: f64) -> (f64, f64, f64) {
    let x = (-u).ln_1p();
    let y = (-v).ln_1p();
    (x, y, log_sum_survival(x, y, theta))
}

fn log_sum_survival(x: f64, y: f64, theta: f64) -> f64 {
    let hi = theta * x.max(y);
    let lo = theta * x.min(y);
    hi + ((lo - hi).exp() * (-hi.exp_m1())).ln_1p()
}

pub fn cdf(u: f64, v: f64, theta: f64) -> Result<f64, CopulaError> {
    Ok(-(terms(u, v, theta).2 / theta).exp_m1())
}

// Joe bivariate copula with parameter θ ∈ [1, ∞).
//
// Notation used below:
//   a = (1 - u1)^θ,   b = (1 - u2)^θ,   s = a + b - a·b = 1 - (1-a)(1-b).
// The CDF is `C(u1, u2) = 1 - s^(1/θ)` and the density is derived by
// differentiating C twice:
//   c(u1, u2) = (1-u1)^(θ-1) · (1-u2)^(θ-1) · s^(1/θ - 2) · (s + θ - 1).
// For θ ≥ 1 and (u1, u2) ∈ (0, 1)² all quantities are finite and positive, so
// we evaluate s and the bracket in log space to avoid upper-tail underflow.

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
    if theta == 1.0 {
        return Ok(0.0);
    }
    Ok(log_pdf_from_log_survival(
        (-u1).ln_1p(),
        (-u2).ln_1p(),
        theta,
    ))
}

pub fn cond_first_given_second(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    // h_{1|2}(u1 | u2) = ∂C/∂u2 = s^(1/θ - 1) · (1 - a) · (1-u2)^(θ-1)
    let (x, y, s) = terms(u1, u2, theta);
    Ok(((1.0 / theta - 1.0) * s + log1mexp(theta * x) + (theta - 1.0) * y).exp())
}

pub fn cond_second_given_first(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    // h_{2|1}(u2 | u1) = ∂C/∂u1 = s^(1/θ - 1) · (1 - b) · (1-u1)^(θ-1)
    cond_first_given_second(u2, u1, theta)
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

pub(super) fn log_pdf_from_log_survival(x: f64, y: f64, theta: f64) -> f64 {
    if theta == 1.0 {
        return 0.0;
    }
    let s = log_sum_survival(x, y, theta);
    (theta - 1.0) * (x + y) + (1.0 / theta - 2.0) * s + logaddexp(s, (theta - 1.0).ln())
}
