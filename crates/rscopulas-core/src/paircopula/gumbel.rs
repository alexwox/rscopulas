use crate::archimedean_math::gumbel;
use crate::errors::{CopulaError, FitError};

pub fn theta_from_tau(tau: f64) -> Result<f64, CopulaError> {
    if !tau.is_finite() || tau <= 0.0 || tau >= 1.0 {
        return Err(FitError::Failed {
            reason: "gumbel pair fit requires tau in (0, 1)",
        }
        .into());
    }
    Ok(1.0 / (1.0 - tau))
}

pub fn log_pdf(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    if !theta.is_finite() || theta < 1.0 {
        return Err(FitError::Failed {
            reason: "gumbel pair theta must be at least 1",
        }
        .into());
    }
    let alpha = 1.0 / theta;
    let t = (-u1.ln()).powf(theta) + (-u2.ln()).powf(theta);
    let log_abs_derivative = gumbel::log_abs_generator_derivative(2, t, alpha);
    let log_phi = theta.ln() + (theta - 1.0) * (-u1.ln()).ln() - u1.ln()
        + theta.ln()
        + (theta - 1.0) * (-u2.ln()).ln()
        - u2.ln();
    Ok(log_abs_derivative + log_phi)
}

pub fn cond_first_given_second(
    u1: f64,
    u2: f64,
    theta: f64,
    _clip_eps: f64,
) -> Result<f64, CopulaError> {
    let x = (-u1.ln()).powf(theta);
    let y = (-u2.ln()).powf(theta);
    let t = (x + y).powf(1.0 / theta);
    Ok((-t).exp() * (x + y).powf(1.0 / theta - 1.0) * (-u2.ln()).powf(theta - 1.0) / u2)
}

pub fn cond_second_given_first(
    u1: f64,
    u2: f64,
    theta: f64,
    _clip_eps: f64,
) -> Result<f64, CopulaError> {
    let x = (-u1.ln()).powf(theta);
    let y = (-u2.ln()).powf(theta);
    let t = (x + y).powf(1.0 / theta);
    Ok((-t).exp() * (x + y).powf(1.0 / theta - 1.0) * (-u1.ln()).powf(theta - 1.0) / u1)
}

pub fn inv_first_given_second(
    p: f64,
    u2: f64,
    theta: f64,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let mut low = clip_eps;
    let mut high = 1.0 - clip_eps;
    for _ in 0..90 {
        let mid = 0.5 * (low + high);
        if cond_first_given_second(mid, u2, theta, clip_eps)? < p {
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
        if cond_second_given_first(u1, mid, theta, clip_eps)? < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(0.5 * (low + high))
}
