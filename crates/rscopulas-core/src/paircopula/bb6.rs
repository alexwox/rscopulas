use crate::errors::{CopulaError, FitError};

// BB6 bivariate copula (Joe–Gumbel blend).
// Parameters: θ ≥ 1, δ ≥ 1.
//
// Notation:
//   x_u = 1 - (1-u)^θ,           w_u = -ln(x_u)  (so w_u > 0),
//   s   = w_u^δ + w_v^δ,         q = s^(1/δ),    e = exp(-q).
// CDF:   C(u, v) = 1 - (1 - e)^(1/θ).
// h-functions are closed-form via the Archimedean derivative; inverse
// h-functions have no closed form and fall back to bisection.
// τ(θ, δ) has no closed form either — the fitter uses a grid-over-δ with an
// inner MLE on θ, as for BB1.

fn prep(u1: f64, u2: f64, theta: f64) -> (f64, f64, f64, f64) {
    let x_u = 1.0 - (1.0 - u1).powf(theta);
    let x_v = 1.0 - (1.0 - u2).powf(theta);
    let w_u = -x_u.ln();
    let w_v = -x_v.ln();
    (x_u, x_v, w_u, w_v)
}

pub fn log_pdf(u1: f64, u2: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    if !theta.is_finite() || theta < 1.0 {
        return Err(FitError::Failed {
            reason: "bb6 pair theta must be at least 1",
        }
        .into());
    }
    if !delta.is_finite() || delta < 1.0 {
        return Err(FitError::Failed {
            reason: "bb6 pair delta must be at least 1",
        }
        .into());
    }
    let (x_u, x_v, w_u, w_v) = prep(u1, u2, theta);
    let s = w_u.powf(delta) + w_v.powf(delta);
    let q = s.powf(1.0 / delta);
    let e = (-q).exp();
    let bracket = q * (1.0 - e / theta) + (delta - 1.0) * (1.0 - e);
    Ok(theta.ln() - q
        + (1.0 / delta - 2.0) * s.ln()
        + (1.0 / theta - 2.0) * (1.0 - e).ln()
        + bracket.ln()
        + (delta - 1.0) * (w_u.ln() + w_v.ln())
        + (theta - 1.0) * ((1.0 - u1).ln() + (1.0 - u2).ln())
        - (x_u.ln() + x_v.ln()))
}

pub fn cond_first_given_second(
    u1: f64,
    u2: f64,
    theta: f64,
    delta: f64,
) -> Result<f64, CopulaError> {
    let (_, x_v, w_u, w_v) = prep(u1, u2, theta);
    let s = w_u.powf(delta) + w_v.powf(delta);
    let q = s.powf(1.0 / delta);
    let e = (-q).exp();
    Ok(e * (1.0 - e).powf(1.0 / theta - 1.0)
        * s.powf(1.0 / delta - 1.0)
        * w_v.powf(delta - 1.0)
        * (1.0 - u2).powf(theta - 1.0)
        / x_v)
}

pub fn cond_second_given_first(
    u1: f64,
    u2: f64,
    theta: f64,
    delta: f64,
) -> Result<f64, CopulaError> {
    let (x_u, _, w_u, w_v) = prep(u1, u2, theta);
    let s = w_u.powf(delta) + w_v.powf(delta);
    let q = s.powf(1.0 / delta);
    let e = (-q).exp();
    Ok(e * (1.0 - e).powf(1.0 / theta - 1.0)
        * s.powf(1.0 / delta - 1.0)
        * w_u.powf(delta - 1.0)
        * (1.0 - u1).powf(theta - 1.0)
        / x_u)
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
    let (_, _, w_u, w_v) = prep(u1, u2, theta);
    let s = w_u.powf(delta) + w_v.powf(delta);
    let q = s.powf(1.0 / delta);
    let e = (-q).exp();
    Ok((1.0 - (1.0 - e).powf(1.0 / theta)).clamp(0.0, 1.0))
}
