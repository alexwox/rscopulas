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
    let (_, _, _, log1pa) = terms(u1, u2, theta, delta);
    Ok((-log1pa / theta).exp())
}

pub(super) fn log_pdf_from_logs(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
    let (x, y, s, log1pa) = terms_from_logs(u, v, theta, delta);
    let bracket = logaddexp(
        theta.ln() + (delta - 1.0).ln(),
        s / delta + (theta * delta).ln_1p(),
    );
    (1.0 / delta - 2.0) * s + (-1.0 / theta - 2.0) * log1pa + bracket - (theta + 1.0) * (u + v)
        + (delta - 1.0) * (x + y)
}
