use crate::archimedean_math::joe;
use crate::errors::{CopulaError, FitError};
use crate::math::{log1mexp, logaddexp};

fn terms(u: f64, v: f64, theta: f64) -> (f64, f64, f64) {
    let x = (-u).ln_1p();
    let y = (-v).ln_1p();
    (x, y, log_sum_survival(x, y, theta))
}

/// `ln s` with `s = a + b − a·b`, `a = e^{θx}`, `b = e^{θy}`.
#[inline]
fn log_sum_survival(x: f64, y: f64, theta: f64) -> f64 {
    let hi = theta * x.max(y);
    let lo = theta * x.min(y);
    if hi > -700.0 {
        // s = a + b·(1 − a) with a the larger term: a sum of positive
        // numbers, and `1 − a` taken through `expm1` where it would cancel.
        let a = hi.exp();
        let one_minus_a = if a <= 0.5 { 1.0 - a } else { -hi.exp_m1() };
        return (a + lo.exp() * one_minus_a).ln();
    }
    hi + ((lo - hi).exp() * (-hi.exp_m1())).ln_1p()
}

pub fn cdf(u: f64, v: f64, theta: f64) -> Result<f64, CopulaError> {
    Ok(-(terms(u, v, theta).2 / theta).exp_m1())
}

// Joe bivariate copula with parameter θ ∈ [1, ∞).
//
// Notation used below:
//   a = (1 - u1)^θ,   b = (1 - u2)^θ,   s = a + b - a·b = 1 - (1-a)(1-b).
// The CDF is `C(u1, u2) = 1 - s^(1/θ)` and the density is derived by
// differentiating C twice:
//   c(u1, u2) = (1-u1)^(θ-1) · (1-u2)^(θ-1) · s^(1/θ - 2) · (s + θ - 1).
// For θ ≥ 1 and (u1, u2) ∈ (0, 1)² all quantities are finite and positive, so
// we evaluate s and the bracket in log space to avoid upper-tail underflow.

pub fn theta_from_tau(tau: f64) -> Result<f64, CopulaError> {
    if !tau.is_finite() || tau <= 0.0 || tau >= 1.0 {
        return Err(FitError::Failed {
            reason: "joe pair fit requires tau in (0, 1)",
        }
        .into());
    }
    joe::invert_tau(tau, "joe tau inversion failed to bracket root")
}

pub fn log_pdf(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    if !theta.is_finite() || theta < 1.0 {
        return Err(FitError::Failed {
            reason: "joe pair theta must be at least 1",
        }
        .into());
    }
    if theta == 1.0 {
        return Ok(0.0);
    }
    Ok(log_pdf_from_log_survival(
        (-u1).ln_1p(),
        (-u2).ln_1p(),
        theta,
    ))
}

pub fn cond_first_given_second(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    // h_{1|2}(u1 | u2) = ∂C/∂u2 = s^(1/θ - 1) · (1 - a) · (1-u2)^(θ-1)
    let (x, y, s) = terms(u1, u2, theta);
    Ok(((1.0 / theta - 1.0) * s + log1mexp(theta * x) + (theta - 1.0) * y).exp())
}

pub fn cond_second_given_first(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    // h_{2|1}(u2 | u1) = ∂C/∂u1 = s^(1/θ - 1) · (1 - b) · (1-u1)^(θ-1)
    cond_first_given_second(u2, u1, theta)
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
/// Write `a = (1 − u)^θ`, `m = 1 − a`, `b = (1 − v)^θ` and `κ = 1 − 1/θ`.
/// The conditional distribution is
///
/// ```text
/// h = (1 − q)^{−κ} · q · (1 − u)^{θ−1} / m,      q = (1 − b)·m.
/// ```
///
/// In the logit `λ = ln(q / (1 − q))` the equation reads
///
/// ```text
/// −softplus(−λ) + κ·softplus(λ) = ln R,      R = p·m / (1 − u)^{θ−1},
/// ```
///
/// whose left side is increasing and concave with slope between `κ` and 1,
/// so Newton's method converges from either asymptote in a few steps and
/// returns both `q` and `1 − q` to full relative accuracy. `b·m` is then
/// `(1 − q) − a`; where that difference cancels (`v` much closer to 1 than
/// `u`), `b` is polished by a safeguarded Newton iteration on the same
/// equation written in `b`.
#[inline]
fn inverse_conditional(u: f64, p: f64, theta: f64) -> f64 {
    if theta == 1.0 {
        return p;
    }
    let la = (-u).ln_1p();
    let a = (theta * la).exp();
    let m = -(theta * la).exp_m1();
    let kappa = 1.0 - 1.0 / theta;
    let log_p = p.ln();
    let log_r = log_p + m.ln() - (theta - 1.0) * la;
    if log_r.is_nan() {
        return f64::NAN;
    }

    let mut lambda = if log_r < 0.0 { log_r } else { log_r / kappa };
    for _ in 0..100 {
        let e = (-lambda.abs()).exp();
        let tail = e.ln_1p();
        let (softplus_pos, softplus_neg, q) = if lambda >= 0.0 {
            (lambda + tail, tail, 1.0 / (1.0 + e))
        } else {
            (tail, tail - lambda, e / (1.0 + e))
        };
        let step = (kappa * softplus_pos - softplus_neg - log_r) / ((1.0 - q) + kappa * q);
        lambda -= step;
        // Quadratic convergence: the curvature term |ψ''/(2ψ')| is at most
        // 1/2, so after a step of size s the error is at most s²/2.
        if step * step <= 2.0 * f64::EPSILON * lambda.abs().max(1.0) {
            break;
        }
    }
    let e = (-lambda.abs()).exp();
    let (q, one_minus_q) = if lambda >= 0.0 {
        (1.0 / (1.0 + e), e / (1.0 + e))
    } else {
        (e / (1.0 + e), 1.0 / (1.0 + e))
    };
    let s = q / m;
    if s <= 0.5 {
        // v = 1 − (1 − s)^{1/θ}
        return -((-s).ln_1p() / theta).exp_m1();
    }
    let scaled = one_minus_q - a;
    if scaled <= 0.0 {
        return 1.0;
    }
    if scaled >= 0.25 * one_minus_q {
        return -((scaled / m).ln() / theta).exp_m1();
    }

    // F(b) = −κ·ln(a + b·m) + ln(1 − b) + (θ − 1)·ln(1 − u) − ln p, decreasing,
    // with F(0) ≥ 0 and F(1) = −∞.
    let offset = (theta - 1.0) * la - log_p;
    let (mut low, mut high) = (0.0_f64, 1.0_f64);
    let mut b = (scaled / m).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    for _ in 0..200 {
        let inner = a + b * m;
        let value = -kappa * inner.ln() + (-b).ln_1p() + offset;
        if value > 0.0 {
            low = b;
        } else {
            high = b;
        }
        let slope = -kappa * m / inner - 1.0 / (1.0 - b);
        let mut next = b - value / slope;
        if !(next > low && next < high) {
            next = 0.5 * (low + high);
        }
        let moved = (next - b).abs();
        b = next;
        // The start is already close; a relative step of 1e-8 leaves a
        // relative error near 1e-16.
        if moved <= 1e-8 * b {
            break;
        }
    }
    // v = 1 − b^{1/θ}
    -(b.ln() / theta).exp_m1()
}

pub(super) fn log_pdf_from_log_survival(x: f64, y: f64, theta: f64) -> f64 {
    if theta == 1.0 {
        return 0.0;
    }
    let s = log_sum_survival(x, y, theta);
    (theta - 1.0) * (x + y) + (1.0 / theta - 2.0) * s + log_bracket(s, theta)
}

/// `ln(s + θ − 1)` from `ln s`.
#[inline]
fn log_bracket(log_s: f64, theta: f64) -> f64 {
    if log_s > -700.0 {
        (log_s.exp() + (theta - 1.0)).ln()
    } else {
        logaddexp(log_s, (theta - 1.0).ln())
    }
}

/// `(log c, h_{1|2}, h_{2|1})` sharing the Joe terms. Each component is
/// bit-identical to its stand-alone function.
#[inline]
pub(super) fn eval3(u1: f64, u2: f64, theta: f64) -> (f64, f64, f64) {
    let (x, y, s) = terms(u1, u2, theta);
    let density = if theta == 1.0 {
        0.0
    } else {
        (theta - 1.0) * (x + y) + (1.0 / theta - 2.0) * s + log_bracket(s, theta)
    };
    let shared = (1.0 / theta - 1.0) * s;
    (
        density,
        (shared + log1mexp(theta * x) + (theta - 1.0) * y).exp(),
        (shared + log1mexp(theta * y) + (theta - 1.0) * x).exp(),
    )
}

/// `(h_{1|2}, h_{2|1})`, bit-identical to the stand-alone functions.
#[inline]
pub(super) fn cond_both(u1: f64, u2: f64, theta: f64) -> (f64, f64) {
    let (x, y, s) = terms(u1, u2, theta);
    let shared = (1.0 / theta - 1.0) * s;
    (
        (shared + log1mexp(theta * x) + (theta - 1.0) * y).exp(),
        (shared + log1mexp(theta * y) + (theta - 1.0) * x).exp(),
    )
}
