use crate::archimedean_math::frank;
use crate::errors::{CopulaError, FitError};

pub fn theta_from_tau(tau: f64) -> Result<f64, CopulaError> {
    if !tau.is_finite() || tau <= 0.0 || tau >= 1.0 {
        return Err(FitError::Failed {
            reason: "frank pair fit requires tau in (0, 1)",
        }
        .into());
    }
    frank::invert_tau(tau, "frank tau inversion failed to bracket root")
}

pub fn log_pdf(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    if !theta.is_finite() || theta <= 0.0 {
        return Err(FitError::Failed {
            reason: "frank pair theta must be positive",
        }
        .into());
    }
    let d = (-theta).exp_m1();
    let a = (-theta * u1).exp_m1();
    let b = (-theta * u2).exp_m1();
    let den = d + a * b;
    Ok(theta.ln() + (-d).ln() - theta * (u1 + u2) - 2.0 * den.abs().ln())
}

pub fn cond_first_given_second(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    let a = (-theta * u1).exp() - 1.0;
    let b = (-theta * u2).exp() - 1.0;
    let d = (-theta).exp() - 1.0;
    Ok((-theta * u2).exp() * a / (d + a * b))
}

pub fn cond_second_given_first(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    let a = (-theta * u1).exp() - 1.0;
    let b = (-theta * u2).exp() - 1.0;
    let d = (-theta).exp() - 1.0;
    Ok((-theta * u1).exp() * b / (d + a * b))
}

pub fn inv_first_given_second(p: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    let d = (-theta).exp() - 1.0;
    let e2 = (-theta * u2).exp();
    let a = p * d / ((1.0 - p) * e2 + p);
    Ok(-(1.0 + a).ln() / theta)
}

pub fn inv_second_given_first(u1: f64, p: f64, theta: f64) -> Result<f64, CopulaError> {
    let d = (-theta).exp() - 1.0;
    let e1 = (-theta * u1).exp();
    let b = p * d / ((1.0 - p) * e1 + p);
    Ok(-(1.0 + b).ln() / theta)
}
