use crate::errors::{CopulaError, FitError};

// BB8 bivariate copula (Joe–Frank blend).
// Parameters: θ ≥ 1, δ ∈ (0, 1].
//
// Notation:
//   β   = 1 - (1 - δ)^θ          (constant across (u, v)),
//   x_u = 1 - (1 - δ u)^θ,
//   D   = 1 - x_u · x_v / β.
// CDF:  C(u, v) = (1 - D^(1/θ)) / δ.
// h-functions derived by differentiating the CDF. The density assembles
// cleanly: ∂²C/∂u∂v = θδ · (1-δu)^(θ-1) · (1-δv)^(θ-1) · D^(1/θ-2)
//                     · (β - x_u x_v / θ) / β².
// Inverse h-functions fall back to bisection.

fn prep(u1: f64, u2: f64, theta: f64, delta: f64) -> (f64, f64, f64) {
    let beta = 1.0 - (1.0 - delta).powf(theta);
    let x_u = 1.0 - (1.0 - delta * u1).powf(theta);
    let x_v = 1.0 - (1.0 - delta * u2).powf(theta);
    (beta, x_u, x_v)
}

pub fn log_pdf(u1: f64, u2: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    if !theta.is_finite() || theta < 1.0 {
        return Err(FitError::Failed {
            reason: "bb8 pair theta must be at least 1",
        }
        .into());
    }
    if !delta.is_finite() || delta <= 0.0 || delta > 1.0 {
        return Err(FitError::Failed {
            reason: "bb8 pair delta must lie in (0, 1]",
        }
        .into());
    }
    let (beta, x_u, x_v) = prep(u1, u2, theta, delta);
    let d_val = 1.0 - x_u * x_v / beta;
    let numer = beta - x_u * x_v / theta;
    Ok(theta.ln()
        + delta.ln()
        + (theta - 1.0) * ((1.0 - delta * u1).ln() + (1.0 - delta * u2).ln())
        + (1.0 / theta - 2.0) * d_val.ln()
        + numer.ln()
        - 2.0 * beta.ln())
}

pub fn cond_first_given_second(
    u1: f64,
    u2: f64,
    theta: f64,
    delta: f64,
) -> Result<f64, CopulaError> {
    let (beta, x_u, x_v) = prep(u1, u2, theta, delta);
    let d_val = 1.0 - x_u * x_v / beta;
    Ok(x_u * (1.0 - delta * u2).powf(theta - 1.0) * d_val.powf(1.0 / theta - 1.0) / beta)
}

pub fn cond_second_given_first(
    u1: f64,
    u2: f64,
    theta: f64,
    delta: f64,
) -> Result<f64, CopulaError> {
    let (beta, x_u, x_v) = prep(u1, u2, theta, delta);
    let d_val = 1.0 - x_u * x_v / beta;
    Ok(x_v * (1.0 - delta * u1).powf(theta - 1.0) * d_val.powf(1.0 / theta - 1.0) / beta)
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
    let (beta, x_u, x_v) = prep(u1, u2, theta, delta);
    let d_val = 1.0 - x_u * x_v / beta;
    Ok(((1.0 - d_val.max(0.0).powf(1.0 / theta)) / delta).clamp(0.0, 1.0))
}
