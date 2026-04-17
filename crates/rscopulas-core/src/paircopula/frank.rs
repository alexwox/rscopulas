use crate::errors::{CopulaError, FitError};

pub fn theta_from_tau(tau: f64) -> Result<f64, CopulaError> {
    if !tau.is_finite() || tau <= 0.0 || tau >= 1.0 {
        return Err(FitError::Failed {
            reason: "frank pair fit requires tau in (0, 1)",
        }
        .into());
    }
    invert_frank_tau(tau)
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
            reason: "frank tau inversion failed to bracket root",
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
