use crate::{
    errors::{CopulaError, FitError},
    special::{norm_cdf, norm_ppf},
};

pub fn tau_to_rho(tau: f64) -> f64 {
    (std::f64::consts::FRAC_PI_2 * tau).sin()
}

pub fn cdf(u: f64, v: f64, rho: f64) -> Result<f64, CopulaError> {
    // Plackett's correlation integral, with rho=sin(t) to remove the
    // endpoint square-root singularity. The integrand is symmetric in u,v.
    let x = norm_ppf(u);
    let y = norm_ppf(v);
    let correction = super::common::integrate_1d(
        &|t| {
            let r = t.sin();
            Ok((-(x * x + y * y - 2.0 * x * y * r) / (2.0 * t.cos().powi(2))).exp())
        },
        0.0,
        rho.asin(),
        1e-12,
        18,
    )? / (2.0 * std::f64::consts::PI);
    Ok((u * v + correction).clamp((u + v - 1.0).max(0.0), u.min(v)))
}

/// Log-density at normal scores `(z1, z2)`.
#[inline]
pub(super) fn log_pdf_from_scores(z1: f64, z2: f64, rho: f64) -> f64 {
    let one_minus = 1.0 - rho * rho;
    -0.5 * one_minus.ln()
        - (rho * rho * (z1 * z1 + z2 * z2) - 2.0 * rho * z1 * z2) / (2.0 * one_minus)
}

pub fn log_pdf(u1: f64, u2: f64, rho: f64) -> Result<f64, CopulaError> {
    if !rho.is_finite() || rho.abs() >= 1.0 {
        return Err(FitError::Failed {
            reason: "gaussian pair rho must lie inside (-1, 1)",
        }
        .into());
    }
    Ok(log_pdf_from_scores(norm_ppf(u1), norm_ppf(u2), rho))
}

/// Log-density and both h-functions from one pair of quantile evaluations.
/// Returns `(log c, h_{1|2}, h_{2|1})`, each identical to the corresponding
/// stand-alone function.
pub(super) fn eval3(u1: f64, u2: f64, rho: f64) -> Result<(f64, f64, f64), CopulaError> {
    if !rho.is_finite() || rho.abs() >= 1.0 {
        return Err(FitError::Failed {
            reason: "gaussian pair rho must lie inside (-1, 1)",
        }
        .into());
    }
    let z1 = norm_ppf(u1);
    let z2 = norm_ppf(u2);
    let scale = (1.0 - rho * rho).sqrt();
    Ok((
        log_pdf_from_scores(z1, z2, rho),
        norm_cdf((z1 - rho * z2) / scale),
        norm_cdf((z2 - rho * z1) / scale),
    ))
}

pub fn cond_first_given_second(u1: f64, u2: f64, rho: f64) -> Result<f64, CopulaError> {
    let z1 = norm_ppf(u1);
    let z2 = norm_ppf(u2);
    let scale = (1.0 - rho * rho).sqrt();
    Ok(norm_cdf((z1 - rho * z2) / scale))
}

pub fn cond_second_given_first(u1: f64, u2: f64, rho: f64) -> Result<f64, CopulaError> {
    let z1 = norm_ppf(u1);
    let z2 = norm_ppf(u2);
    let scale = (1.0 - rho * rho).sqrt();
    Ok(norm_cdf((z2 - rho * z1) / scale))
}

pub fn inv_first_given_second(p: f64, u2: f64, rho: f64) -> Result<f64, CopulaError> {
    let z2 = norm_ppf(u2);
    let q = norm_ppf(p);
    let scale = (1.0 - rho * rho).sqrt();
    Ok(norm_cdf(scale * q + rho * z2))
}

pub fn inv_second_given_first(u1: f64, p: f64, rho: f64) -> Result<f64, CopulaError> {
    let z1 = norm_ppf(u1);
    let q = norm_ppf(p);
    let scale = (1.0 - rho * rho).sqrt();
    Ok(norm_cdf(scale * q + rho * z1))
}

/// `(h_{1|2}, h_{2|1})` from one pair of quantile evaluations.
#[inline]
pub(super) fn cond_both(u1: f64, u2: f64, rho: f64) -> (f64, f64) {
    let z1 = norm_ppf(u1);
    let z2 = norm_ppf(u2);
    let scale = (1.0 - rho * rho).sqrt();
    (
        norm_cdf((z1 - rho * z2) / scale),
        norm_cdf((z2 - rho * z1) / scale),
    )
}

/// `h_{2|1}⁻¹(p | u1)` followed by `h_{1|2}(u1 | ·)` at the result clamped to
/// `[clip, 1 − clip]`. The second step reuses the score of the first unless
/// the clamp moved it.
#[inline]
pub(super) fn inverse_then_cond(u1: f64, p: f64, rho: f64, clip: f64) -> (f64, f64) {
    let z1 = norm_ppf(u1);
    let scale = (1.0 - rho * rho).sqrt();
    let score = scale * norm_ppf(p) + rho * z1;
    let solved = norm_cdf(score);
    let clamped = solved.clamp(clip, 1.0 - clip);
    let score = if clamped == solved {
        score
    } else {
        norm_ppf(clamped)
    };
    (solved, norm_cdf((z1 - rho * score) / scale))
}
