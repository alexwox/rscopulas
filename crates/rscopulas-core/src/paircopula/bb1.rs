use crate::errors::{CopulaError, FitError};
use crate::math::{log1mexp, logaddexp, softplus};

fn terms(u: f64, v: f64, theta: f64, delta: f64) -> (f64, f64, f64, f64) {
    terms_from_logs(u.ln(), v.ln(), theta, delta)
}

fn terms_from_logs(u: f64, v: f64, theta: f64, delta: f64) -> (f64, f64, f64, f64) {
    let a = -theta * u;
    let b = -theta * v;
    let x = a + log1mexp(-a);
    let y = b + log1mexp(-b);
    let s = logaddexp(delta * x, delta * y);
    (x, y, s, softplus(s / delta))
}

// BB1 bivariate copula (Clayton–Gumbel blend).
// Parameters: θ > 0, δ ≥ 1.
//
// Notation:
//   x = u1^(-θ) - 1,   y = u2^(-θ) - 1,
//   s = x^δ + y^δ,      a = s^(1/δ).
// CDF:  C(u, v) = (1 + a)^(-1/θ).
// Density obtained by differentiating the Archimedean form
//   ψ'' · φ'(u) · φ'(v) with ψ = φ^{-1}:
//   c(u, v) = s^(1/δ - 2) · (1 + a)^(-1/θ - 2)
//           · [θ(δ - 1) + a(θδ + 1)]
//           · u1^(-θ - 1) · u2^(-θ - 1)
//           · x^(δ - 1) · y^(δ - 1).
// Kendall's τ is closed-form: τ = 1 − 2 / [δ(θ + 2)].
// The h-functions have clean closed forms (below); their inverses do not,
// so they fall back to bisection mirroring Gumbel/Joe.

pub fn params_from_tau(tau: f64, delta: f64) -> Result<f64, CopulaError> {
    // Given δ ≥ 1, solve τ = 1 - 2/(δ(θ+2)) for θ. Used as a warm start for
    // the outer grid-over-δ fitter; the caller supplies the fixed δ.
    if !tau.is_finite() || tau <= 0.0 || tau >= 1.0 {
        return Err(FitError::Failed {
            reason: "bb1 pair fit requires tau in (0, 1)",
        }
        .into());
    }
    if !delta.is_finite() || delta < 1.0 {
        return Err(FitError::Failed {
            reason: "bb1 pair fit requires delta >= 1",
        }
        .into());
    }
    Ok((2.0 / (delta * (1.0 - tau)) - 2.0).max(1e-6))
}

pub fn log_pdf(u1: f64, u2: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    if !theta.is_finite() || theta <= 0.0 {
        return Err(FitError::Failed {
            reason: "bb1 pair theta must be positive",
        }
        .into());
    }
    if !delta.is_finite() || delta < 1.0 {
        return Err(FitError::Failed {
            reason: "bb1 pair delta must be at least 1",
        }
        .into());
    }
    Ok(log_pdf_from_logs(u1.ln(), u2.ln(), theta, delta))
}

pub fn cond_first_given_second(
    u1: f64,
    u2: f64,
    theta: f64,
    delta: f64,
) -> Result<f64, CopulaError> {
    // h_{1|2}(u1|u2) = ∂C/∂u2
    //   = (1 + a)^(-1/θ - 1) · s^(1/δ - 1) · u2^(-θ-1) · y^(δ-1)
    let (_, y, s, log1pa) = terms(u1, u2, theta, delta);
    Ok(
        ((-1.0 / theta - 1.0) * log1pa + (1.0 / delta - 1.0) * s - (theta + 1.0) * u2.ln()
            + (delta - 1.0) * y)
            .exp(),
    )
}

pub fn cond_second_given_first(
    u1: f64,
    u2: f64,
    theta: f64,
    delta: f64,
) -> Result<f64, CopulaError> {
    cond_first_given_second(u2, u1, theta, delta)
}

pub fn inv_first_given_second(
    p: f64,
    u2: f64,
    theta: f64,
    delta: f64,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    // No closed form: bisect over u1 ∈ [clip_eps, 1 - clip_eps]. The h-function
    // is strictly increasing in u1 (Joe 2014, §5.3) so the 90-iter bisection
    // recovers ~double-precision in the bracket.
    super::invert::invert_increasing(
        |mid| cond_first_given_second(mid, u2, theta, delta),
        p,
        clip_eps,
    )
}

pub fn inv_second_given_first(
    u1: f64,
    p: f64,
    theta: f64,
    delta: f64,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    super::invert::invert_increasing(
        |mid| cond_second_given_first(u1, mid, theta, delta),
        p,
        clip_eps,
    )
}

pub fn cdf(u1: f64, u2: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    let (_, _, _, log1pa) = terms(u1, u2, theta, delta);
    Ok((-log1pa / theta).exp())
}

/// BB1 log-density with its parameter-only terms evaluated once, for callers
/// that evaluate many points at fixed parameters (a likelihood pass).
pub(super) struct Density {
    theta: f64,
    delta: f64,
    /// `ln θ + ln(δ − 1)`.
    log_theta_delta_minus_one: f64,
    /// `ln(1 + θδ)`.
    log1p_theta_delta: f64,
}

impl Density {
    pub(super) fn new(theta: f64, delta: f64) -> Self {
        Self {
            theta,
            delta,
            log_theta_delta_minus_one: theta.ln() + (delta - 1.0).ln(),
            log1p_theta_delta: (theta * delta).ln_1p(),
        }
    }

    /// Log-density at `(ln u₁, ln u₂)`.
    #[inline]
    pub(super) fn log_pdf_from_logs(&self, u: f64, v: f64) -> f64 {
        let (theta, delta) = (self.theta, self.delta);
        let (x, y, s, log1pa) = terms_from_logs(u, v, theta, delta);
        let bracket = logaddexp(
            self.log_theta_delta_minus_one,
            s / delta + self.log1p_theta_delta,
        );
        (1.0 / delta - 2.0) * s + (-1.0 / theta - 2.0) * log1pa + bracket - (theta + 1.0) * (u + v)
            + (delta - 1.0) * (x + y)
    }
}

pub(super) fn log_pdf_from_logs(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    Density::new(theta, delta).log_pdf_from_logs(u, v)
}

impl Density {
    /// `(h_{1|2}, h_{2|1})`, bit-identical to the stand-alone functions.
    #[inline]
    pub(super) fn cond_both(&self, u1: f64, u2: f64) -> (f64, f64) {
        let (theta, delta) = (self.theta, self.delta);
        let (lu, lv) = (u1.ln(), u2.ln());
        let (x, y, s, log1pa) = terms_from_logs(lu, lv, theta, delta);
        let shared = (-1.0 / theta - 1.0) * log1pa + (1.0 / delta - 1.0) * s;
        (
            (shared - (theta + 1.0) * lv + (delta - 1.0) * y).exp(),
            (shared - (theta + 1.0) * lu + (delta - 1.0) * x).exp(),
        )
    }

    /// `(log c, h_{1|2}, h_{2|1})` sharing the BB1 terms. Each component is
    /// bit-identical to its stand-alone function.
    #[inline]
    pub(super) fn eval3(&self, u1: f64, u2: f64) -> (f64, f64, f64) {
        let (theta, delta) = (self.theta, self.delta);
        let (lu, lv) = (u1.ln(), u2.ln());
        let (x, y, s, log1pa) = terms_from_logs(lu, lv, theta, delta);
        let bracket = logaddexp(
            self.log_theta_delta_minus_one,
            s / delta + self.log1p_theta_delta,
        );
        let density = (1.0 / delta - 2.0) * s + (-1.0 / theta - 2.0) * log1pa + bracket
            - (theta + 1.0) * (lu + lv)
            + (delta - 1.0) * (x + y);
        let shared = (-1.0 / theta - 1.0) * log1pa + (1.0 / delta - 1.0) * s;
        (
            density,
            (shared - (theta + 1.0) * lv + (delta - 1.0) * y).exp(),
            (shared - (theta + 1.0) * lu + (delta - 1.0) * x).exp(),
        )
    }
}
