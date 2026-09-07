use statrs::distribution::{ContinuousCDF, Normal};

use crate::errors::{CopulaError, FitError};

fn standard_normal() -> Normal {
    Normal::new(0.0, 1.0).expect("standard normal parameters should be valid")
}

pub fn tau_to_rho(tau: f64) -> f64 {
    (std::f64::consts::FRAC_PI_2 * tau).sin()
}

pub fn cdf(u: f64, v: f64, rho: f64) -> Result<f64, CopulaError> {
    // Plackett's correlation integral, with rho=sin(t) to remove the
    // endpoint square-root singularity. The integrand is symmetric in u,v.
    let normal = standard_normal();
    let x = normal.inverse_cdf(u);
    let y = normal.inverse_cdf(v);
    let correction = super::common::integrate_1d(
        &|t| {
            let r = t.sin();
            Ok((-(x * x + y * y - 2.0 * x * y * r) / (2.0 * t.cos().powi(2))).exp())
        },
        0.0,
        rho.asin(),
        1e-12,
        18,
    )? / (2.0 * std::f64::consts::PI);
    Ok((u * v + correction).clamp((u + v - 1.0).max(0.0), u.min(v)))
}

pub fn log_pdf(u1: f64, u2: f64, rho: f64) -> Result<f64, CopulaError> {
    if !rho.is_finite() || rho.abs() >= 1.0 {
        return Err(FitError::Failed {
            reason: "gaussian pair rho must lie inside (-1, 1)",
        }
        .into());
    }

    let normal = standard_normal();
    let z1 = normal.inverse_cdf(u1);
    let z2 = normal.inverse_cdf(u2);
    let one_minus = 1.0 - rho * rho;
    Ok(-0.5 * one_minus.ln()
        - (rho * rho * (z1 * z1 + z2 * z2) - 2.0 * rho * z1 * z2) / (2.0 * one_minus))
}

pub fn cond_first_given_second(u1: f64, u2: f64, rho: f64) -> Result<f64, CopulaError> {
    let normal = standard_normal();
    let z1 = normal.inverse_cdf(u1);
    let z2 = normal.inverse_cdf(u2);
    let scale = (1.0 - rho * rho).sqrt();
    Ok(normal.cdf((z1 - rho * z2) / scale))
}

pub fn cond_second_given_first(u1: f64, u2: f64, rho: f64) -> Result<f64, CopulaError> {
    let normal = standard_normal();
    let z1 = normal.inverse_cdf(u1);
    let z2 = normal.inverse_cdf(u2);
    let scale = (1.0 - rho * rho).sqrt();
    Ok(normal.cdf((z2 - rho * z1) / scale))
}

pub fn inv_first_given_second(p: f64, u2: f64, rho: f64) -> Result<f64, CopulaError> {
    let normal = standard_normal();
    let z2 = normal.inverse_cdf(u2);
    let q = normal.inverse_cdf(p);
    let scale = (1.0 - rho * rho).sqrt();
    Ok(normal.cdf(scale * q + rho * z2))
}

pub fn inv_second_given_first(u1: f64, p: f64, rho: f64) -> Result<f64, CopulaError> {
    let normal = standard_normal();
    let z1 = normal.inverse_cdf(u1);
    let q = normal.inverse_cdf(p);
    let scale = (1.0 - rho * rho).sqrt();
    Ok(normal.cdf(scale * q + rho * z1))
}
