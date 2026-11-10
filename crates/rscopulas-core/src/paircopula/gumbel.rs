use crate::errors::{CopulaError, FitError};
use crate::math::logaddexp;

fn terms(u: f64, v: f64, theta: f64) -> (f64, f64, f64, f64) {
    let x = (-u.ln()).ln();
    let y = (-v.ln()).ln();
    let s = logaddexp(theta * x, theta * y);
    (x, y, s, (s / theta).exp())
}

pub fn cdf(u: f64, v: f64, theta: f64) -> Result<f64, CopulaError> {
    Ok((-terms(u, v, theta).3).exp())
}

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
    if theta == 1.0 {
        return Ok(0.0);
    }
    Ok(log_pdf_from_logs(u1.ln(), u2.ln(), theta))
}

pub fn cond_first_given_second(
    u1: f64,
    u2: f64,
    theta: f64,
    _clip_eps: f64,
) -> Result<f64, CopulaError> {
    let (_, y, s, t) = terms(u1, u2, theta);
    Ok((-t + (1.0 / theta - 1.0) * s + (theta - 1.0) * y - u2.ln()).exp())
}

pub fn cond_second_given_first(
    u1: f64,
    u2: f64,
    theta: f64,
    _clip_eps: f64,
) -> Result<f64, CopulaError> {
    cond_first_given_second(u2, u1, theta, _clip_eps)
}

pub fn inv_first_given_second(
    p: f64,
    u2: f64,
    theta: f64,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    super::invert::invert_increasing(
        |mid| cond_first_given_second(mid, u2, theta, clip_eps),
        p,
        clip_eps,
    )
}

pub fn inv_second_given_first(
    u1: f64,
    p: f64,
    theta: f64,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    super::invert::invert_increasing(
        |mid| cond_second_given_first(u1, mid, theta, clip_eps),
        p,
        clip_eps,
    )
}

#[inline]
pub(super) fn log_pdf_from_logs(u: f64, v: f64, theta: f64) -> f64 {
    if theta == 1.0 {
        return 0.0;
    }
    let (x, y) = ((-u).ln(), (-v).ln());
    let s = logaddexp(theta * x, theta * y);
    let t = (s / theta).exp();
    -t - u - v
        + (theta - 1.0) * (x + y)
        + (2.0 / theta - 2.0) * s
        + logaddexp(0.0, (theta - 1.0).ln() - s / theta)
}

/// `(log c, h_{1|2}, h_{2|1})` sharing the Gumbel terms. Each component is
/// bit-identical to its stand-alone function.
#[inline]
pub(super) fn eval3(u1: f64, u2: f64, theta: f64) -> (f64, f64, f64) {
    let (lu, lv) = (u1.ln(), u2.ln());
    let (x, y) = ((-lu).ln(), (-lv).ln());
    let s = logaddexp(theta * x, theta * y);
    let t = (s / theta).exp();
    let density = if theta == 1.0 {
        0.0
    } else {
        -t - lu - lv
            + (theta - 1.0) * (x + y)
            + (2.0 / theta - 2.0) * s
            + logaddexp(0.0, (theta - 1.0).ln() - s / theta)
    };
    let shared = -t + (1.0 / theta - 1.0) * s;
    (
        density,
        (shared + (theta - 1.0) * y - lv).exp(),
        (shared + (theta - 1.0) * x - lu).exp(),
    )
}

/// `(h_{1|2}, h_{2|1})`, bit-identical to the stand-alone functions.
#[inline]
pub(super) fn cond_both(u1: f64, u2: f64, theta: f64) -> (f64, f64) {
    let (lu, lv) = (u1.ln(), u2.ln());
    let (x, y) = ((-lu).ln(), (-lv).ln());
    let s = logaddexp(theta * x, theta * y);
    let shared = -(s / theta).exp() + (1.0 / theta - 1.0) * s;
    (
        (shared + (theta - 1.0) * y - lv).exp(),
        (shared + (theta - 1.0) * x - lu).exp(),
    )
}
