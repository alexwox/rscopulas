use statrs::{
    distribution::{Continuous, ContinuousCDF, StudentsT},
    function::gamma::ln_gamma,
};

use crate::errors::{CopulaError, FitError};

fn t_dist(nu: f64) -> Result<StudentsT, CopulaError> {
    StudentsT::new(0.0, 1.0, nu).map_err(|_| {
        FitError::Failed {
            reason: "student t pair nu must be positive",
        }
        .into()
    })
}

pub fn candidate_nus() -> Vec<f64> {
    let min = 2.05_f64.ln();
    let max = 50.0_f64.ln();
    let steps = 24usize;
    (0..steps)
        .map(|idx| {
            let fraction = idx as f64 / (steps - 1) as f64;
            (min + (max - min) * fraction).exp()
        })
        .collect()
}

pub fn log_pdf(u1: f64, u2: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    if !rho.is_finite() || rho.abs() >= 1.0 || !nu.is_finite() || nu <= 0.0 {
        return Err(FitError::Failed {
            reason: "student t pair parameters are invalid",
        }
        .into());
    }

    let dist = t_dist(nu)?;
    let x = dist.inverse_cdf(u1);
    let y = dist.inverse_cdf(u2);
    let one_minus = 1.0 - rho * rho;
    let quad = (x * x - 2.0 * rho * x * y + y * y) / one_minus;
    let mv_log_pdf = ln_gamma((nu + 2.0) / 2.0)
        - ln_gamma(nu / 2.0)
        - 0.5 * one_minus.ln()
        - (nu * std::f64::consts::PI).ln()
        - 0.5 * (nu + 2.0) * (1.0 + quad / nu).ln();
    Ok(mv_log_pdf - dist.ln_pdf(x) - dist.ln_pdf(y))
}

pub fn cond_first_given_second(u1: f64, u2: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    let dist = t_dist(nu)?;
    let cond = t_dist(nu + 1.0)?;
    let x = dist.inverse_cdf(u1);
    let y = dist.inverse_cdf(u2);
    let scale = (((nu + y * y) * (1.0 - rho * rho)) / (nu + 1.0)).sqrt();
    Ok(cond.cdf((x - rho * y) / scale))
}

pub fn cond_second_given_first(u1: f64, u2: f64, rho: f64, nu: f64) -> Result<f64, CopulaError> {
    let dist = t_dist(nu)?;
    let cond = t_dist(nu + 1.0)?;
    let x = dist.inverse_cdf(u1);
    let y = dist.inverse_cdf(u2);
    let scale = (((nu + x * x) * (1.0 - rho * rho)) / (nu + 1.0)).sqrt();
    Ok(cond.cdf((y - rho * x) / scale))
}

pub fn inv_first_given_second(
    p: f64,
    u2: f64,
    rho: f64,
    nu: f64,
) -> Result<f64, CopulaError> {
    let dist = t_dist(nu)?;
    let cond = t_dist(nu + 1.0)?;
    let y = dist.inverse_cdf(u2);
    let q = cond.inverse_cdf(p);
    let scale = (((nu + y * y) * (1.0 - rho * rho)) / (nu + 1.0)).sqrt();
    Ok(dist.cdf(rho * y + scale * q))
}

pub fn inv_second_given_first(
    u1: f64,
    p: f64,
    rho: f64,
    nu: f64,
) -> Result<f64, CopulaError> {
    let dist = t_dist(nu)?;
    let cond = t_dist(nu + 1.0)?;
    let x = dist.inverse_cdf(u1);
    let q = cond.inverse_cdf(p);
    let scale = (((nu + x * x) * (1.0 - rho * rho)) / (nu + 1.0)).sqrt();
    Ok(dist.cdf(rho * x + scale * q))
}
