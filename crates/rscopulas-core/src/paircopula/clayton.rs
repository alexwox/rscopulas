use crate::errors::{CopulaError, FitError};

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
    let sum_power = u1.powf(-theta) + u2.powf(-theta) - 1.0;
    Ok(
        (1.0 + theta).ln()
            - (1.0 + theta) * (u1.ln() + u2.ln())
            - (2.0 + 1.0 / theta) * sum_power.ln(),
    )
}

pub fn cond_first_given_second(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    let s = u1.powf(-theta) + u2.powf(-theta) - 1.0;
    Ok(s.powf(-1.0 / theta - 1.0) * u2.powf(-theta - 1.0))
}

pub fn cond_second_given_first(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    let s = u1.powf(-theta) + u2.powf(-theta) - 1.0;
    Ok(s.powf(-1.0 / theta - 1.0) * u1.powf(-theta - 1.0))
}

pub fn inv_first_given_second(p: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    Ok((1.0 + u2.powf(-theta) * (p.powf(-theta / (1.0 + theta)) - 1.0)).powf(-1.0 / theta))
}

pub fn inv_second_given_first(u1: f64, p: f64, theta: f64) -> Result<f64, CopulaError> {
    Ok((1.0 + u1.powf(-theta) * (p.powf(-theta / (1.0 + theta)) - 1.0)).powf(-1.0 / theta))
}
