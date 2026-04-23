use crate::errors::{CopulaError, FitError};

// BB7 bivariate copula (Joe–Clayton blend).
// Parameters: θ ≥ 1, δ > 0.
//
// Notation:
//   x_u = 1 - (1-u)^θ,
//   t   = x_u^(-δ) + x_v^(-δ) - 1,
//   a   = t^(-1/δ),  A = 1 - a.
// CDF:  C(u, v) = 1 - A^(1/θ).
// Density is derived from ψ''·φ'·φ' with the Archimedean generator
//   φ(u) = x_u^(-δ) - 1. Inverse h-functions have no closed form and fall
// back to bisection (mirrors the BB1/BB6/Joe pattern).

fn prep(u1: f64, u2: f64, theta: f64) -> (f64, f64) {
    (
        1.0 - (1.0 - u1).powf(theta),
        1.0 - (1.0 - u2).powf(theta),
    )
}

pub fn log_pdf(u1: f64, u2: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    if !theta.is_finite() || theta < 1.0 {
        return Err(FitError::Failed {
            reason: "bb7 pair theta must be at least 1",
        }
        .into());
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err(FitError::Failed {
            reason: "bb7 pair delta must be positive",
        }
        .into());
    }
    let (x_u, x_v) = prep(u1, u2, theta);
    let t = x_u.powf(-delta) + x_v.powf(-delta) - 1.0;
    let a = t.powf(-1.0 / delta);
    let aa = 1.0 - a;
    let bracket = theta * (1.0 + delta) - a * (1.0 + theta * delta);
    Ok((-1.0 / delta - 2.0) * t.ln()
        + (1.0 / theta - 2.0) * aa.ln()
        + bracket.ln()
        + (-delta - 1.0) * (x_u.ln() + x_v.ln())
        + (theta - 1.0) * ((1.0 - u1).ln() + (1.0 - u2).ln()))
}

pub fn cond_first_given_second(
    u1: f64,
    u2: f64,
    theta: f64,
    delta: f64,
) -> Result<f64, CopulaError> {
    let (x_u, x_v) = prep(u1, u2, theta);
    let t = x_u.powf(-delta) + x_v.powf(-delta) - 1.0;
    let a = t.powf(-1.0 / delta);
    let aa = 1.0 - a;
    Ok(aa.powf(1.0 / theta - 1.0)
        * t.powf(-1.0 / delta - 1.0)
        * x_v.powf(-delta - 1.0)
        * (1.0 - u2).powf(theta - 1.0))
}

pub fn cond_second_given_first(
    u1: f64,
    u2: f64,
    theta: f64,
    delta: f64,
) -> Result<f64, CopulaError> {
    let (x_u, x_v) = prep(u1, u2, theta);
    let t = x_u.powf(-delta) + x_v.powf(-delta) - 1.0;
    let a = t.powf(-1.0 / delta);
    let aa = 1.0 - a;
    Ok(aa.powf(1.0 / theta - 1.0)
        * t.powf(-1.0 / delta - 1.0)
        * x_u.powf(-delta - 1.0)
        * (1.0 - u1).powf(theta - 1.0))
}

pub fn inv_first_given_second(
    p: f64,
    u2: f64,
    theta: f64,
    delta: f64,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let mut low = clip_eps;
    let mut high = 1.0 - clip_eps;
    for _ in 0..90 {
        let mid = 0.5 * (low + high);
        if cond_first_given_second(mid, u2, theta, delta)? < p {
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
    delta: f64,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let mut low = clip_eps;
    let mut high = 1.0 - clip_eps;
    for _ in 0..90 {
        let mid = 0.5 * (low + high);
        if cond_second_given_first(u1, mid, theta, delta)? < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(0.5 * (low + high))
}

pub fn cdf(u1: f64, u2: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    let (x_u, x_v) = prep(u1, u2, theta);
    let t = x_u.powf(-delta) + x_v.powf(-delta) - 1.0;
    let a = t.powf(-1.0 / delta);
    let aa = 1.0 - a;
    Ok((1.0 - aa.max(0.0).powf(1.0 / theta)).clamp(0.0, 1.0))
}
