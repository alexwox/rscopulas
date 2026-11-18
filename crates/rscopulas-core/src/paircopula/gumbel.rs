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
    _clip_eps: f64,
) -> Result<f64, CopulaError> {
    Ok(inverse_conditional(u2, p, theta))
}

pub fn inv_second_given_first(
    u1: f64,
    p: f64,
    theta: f64,
    _clip_eps: f64,
) -> Result<f64, CopulaError> {
    Ok(inverse_conditional(u1, p, theta))
}

/// Solves `h(v | u) = p` for `v`, where `u` is the conditioning variable.
///
/// With `x = −ln u`, `y = −ln v` and `z = (x^θ + y^θ)^{1/θ}` the conditional
/// distribution is `exp(x − z) (x / z)^{θ−1}`. In `ρ = ln(z / x) ≥ 0` the
/// equation becomes
///
/// ```text
/// x·expm1(ρ) + (θ − 1)·ρ = −ln p
/// ```
///
/// whose left side is increasing and convex, so Newton's method started to
/// the right of the root decreases monotonically onto it. Each of the two
/// terms alone bounds `ρ` from above, which gives the starting point.
/// Working in `ρ` keeps full relative accuracy when `z` is close to `x`
/// (`v` close to 1).
#[inline]
fn inverse_conditional(u: f64, p: f64, theta: f64) -> f64 {
    let level = -p.ln();
    if level <= 0.0 || level.is_nan() {
        return if level.is_nan() { f64::NAN } else { 1.0 };
    }
    let x = (-u.ln()).max(f64::MIN_POSITIVE);
    let k = theta - 1.0;
    let mut rho = (level / x).ln_1p();
    if k > 0.0 {
        rho = rho.min(level / k);
    }
    for _ in 0..64 {
        let e = rho.exp_m1();
        let step = (x * e + k * rho - level) / (x * (e + 1.0) + k);
        rho -= step;
        if step.abs() <= 4.0 * f64::EPSILON * rho {
            break;
        }
    }
    // y^θ = z^θ − x^θ = x^θ·expm1(θρ).
    let t = theta * rho;
    let log_expm1 = if t > 40.0 { t } else { t.exp_m1().ln() };
    (-(x.ln() + log_expm1 / theta).exp()).exp()
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
