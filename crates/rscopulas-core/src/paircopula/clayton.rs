use crate::errors::{CopulaError, FitError};
use crate::math::softplus;

// log(u^-theta + v^-theta - 1), scaling before exponentiating.
fn log_sum_logs(u: f64, v: f64, theta: f64) -> f64 {
    let a = -theta * u;
    let b = -theta * v;
    let hi = a.max(b);
    hi + ((a.min(b) - hi).exp() * (-(-a.min(b)).exp_m1())).ln_1p()
}

pub fn cdf(u: f64, v: f64, theta: f64) -> Result<f64, CopulaError> {
    Ok((-log_sum_logs(u.ln(), v.ln(), theta) / theta).exp())
}

pub fn theta_from_tau(tau: f64) -> Result<f64, CopulaError> {
    if !tau.is_finite() || tau <= 0.0 || tau >= 1.0 {
        return Err(FitError::Failed {
            reason: "clayton pair fit requires tau in (0, 1)",
        }
        .into());
    }
    Ok(2.0 * tau / (1.0 - tau))
}

pub fn log_pdf(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    if !theta.is_finite() || theta <= 0.0 {
        return Err(FitError::Failed {
            reason: "clayton pair theta must be positive",
        }
        .into());
    }
    Ok(log_pdf_from_logs(u1.ln(), u2.ln(), theta))
}

pub fn cond_first_given_second(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    Ok(
        ((-1.0 / theta - 1.0) * log_sum_logs(u1.ln(), u2.ln(), theta) - (theta + 1.0) * u2.ln())
            .exp(),
    )
}

pub fn cond_second_given_first(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    cond_first_given_second(u2, u1, theta)
}

pub fn inv_first_given_second(p: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    let t = -theta / (1.0 + theta) * p.ln();
    let log_expm1 = t + crate::math::log1mexp(-t);
    Ok((-softplus(-theta * u2.ln() + log_expm1) / theta).exp())
}

pub fn inv_second_given_first(u1: f64, p: f64, theta: f64) -> Result<f64, CopulaError> {
    inv_first_given_second(p, u1, theta)
}

pub(super) fn log_pdf_from_logs(u: f64, v: f64, theta: f64) -> f64 {
    (1.0 + theta).ln() - (1.0 + theta) * (u + v) - (2.0 + 1.0 / theta) * log_sum_logs(u, v, theta)
}
