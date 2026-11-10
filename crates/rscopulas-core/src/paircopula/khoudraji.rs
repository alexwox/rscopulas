use crate::errors::CopulaError;

use super::common::KhoudrajiParams;

#[derive(Debug)]
struct KhoudrajiParts {
    cdf_first: f64,
    cdf_second: f64,
    h12_first: f64,
    h21_first: f64,
    h12_second: f64,
    h21_second: f64,
    log_pdf_first: f64,
    log_pdf_second: f64,
    du_first: f64,
    dv_first: f64,
    du_second: f64,
    dv_second: f64,
}

pub fn cdf(u: f64, v: f64, params: &KhoudrajiParams, clip_eps: f64) -> Result<f64, CopulaError> {
    Ok(params.first.cdf(
        u.powf(1.0 - params.shape_first),
        v.powf(1.0 - params.shape_second),
        clip_eps,
    )? * params.second.cdf(
        u.powf(params.shape_first),
        v.powf(params.shape_second),
        clip_eps,
    )?)
}

pub fn log_pdf(
    u1: f64,
    u2: f64,
    params: &KhoudrajiParams,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let parts = evaluate_parts(u1, u2, params, clip_eps)?;
    let term_logs = [
        ln_if_positive(parts.du_first)
            + ln_if_positive(parts.dv_first)
            + parts.log_pdf_first
            + ln_if_positive(parts.cdf_second),
        ln_if_positive(parts.du_first)
            + ln_if_positive(parts.dv_second)
            + ln_if_positive(parts.h21_first)
            + ln_if_positive(parts.h12_second),
        ln_if_positive(parts.du_second)
            + ln_if_positive(parts.dv_first)
            + ln_if_positive(parts.h21_second)
            + ln_if_positive(parts.h12_first),
        ln_if_positive(parts.du_second)
            + ln_if_positive(parts.dv_second)
            + ln_if_positive(parts.cdf_first)
            + parts.log_pdf_second,
    ];
    Ok(logsumexp(&term_logs))
}

pub fn cond_first_given_second(
    u1: f64,
    u2: f64,
    params: &KhoudrajiParams,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    // ∂C/∂u₂ needs each base copula's CDF and its h-function in the second
    // argument only; the densities and the other pair of h-functions that
    // the log-density uses are not evaluated.
    let m = Margins::new(u1, u2, params, clip_eps);
    let cdf_first = params.first.cdf(m.x1, m.y1, clip_eps)?;
    let cdf_second = params.second.cdf(m.x2, m.y2, clip_eps)?;
    let h12_first = params.first.cond_first_given_second(m.x1, m.y1, clip_eps)?;
    let h12_second = params
        .second
        .cond_first_given_second(m.x2, m.y2, clip_eps)?;
    Ok(
        (m.dv_first * h12_first * cdf_second + cdf_first * m.dv_second * h12_second)
            .clamp(clip_eps, 1.0 - clip_eps),
    )
}

pub fn cond_second_given_first(
    u1: f64,
    u2: f64,
    params: &KhoudrajiParams,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let m = Margins::new(u1, u2, params, clip_eps);
    let cdf_first = params.first.cdf(m.x1, m.y1, clip_eps)?;
    let cdf_second = params.second.cdf(m.x2, m.y2, clip_eps)?;
    let h21_first = params.first.cond_second_given_first(m.x1, m.y1, clip_eps)?;
    let h21_second = params
        .second
        .cond_second_given_first(m.x2, m.y2, clip_eps)?;
    Ok(
        (m.du_first * h21_first * cdf_second + cdf_first * m.du_second * h21_second)
            .clamp(clip_eps, 1.0 - clip_eps),
    )
}

pub fn inv_first_given_second(
    p: f64,
    u2: f64,
    params: &KhoudrajiParams,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    invert_monotone(
        |u1| cond_first_given_second(u1, u2, params, clip_eps),
        p,
        clip_eps,
    )
}

pub fn inv_second_given_first(
    u1: f64,
    p: f64,
    params: &KhoudrajiParams,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    invert_monotone(
        |u2| cond_second_given_first(u1, u2, params, clip_eps),
        p,
        clip_eps,
    )
}

/// Power-transformed margins `u^(1−a), v^(1−b)` (first base copula) and
/// `u^a, v^b` (second), with the derivatives of those powers.
struct Margins {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    du_first: f64,
    dv_first: f64,
    du_second: f64,
    dv_second: f64,
}

impl Margins {
    fn new(u1: f64, u2: f64, params: &KhoudrajiParams, clip_eps: f64) -> Self {
        let u1 = u1.clamp(clip_eps, 1.0 - clip_eps);
        let u2 = u2.clamp(clip_eps, 1.0 - clip_eps);
        let alpha = 1.0 - params.shape_first;
        let beta = 1.0 - params.shape_second;
        let a = params.shape_first;
        let b = params.shape_second;
        Self {
            x1: power_margin(u1, alpha, clip_eps),
            y1: power_margin(u2, beta, clip_eps),
            x2: power_margin(u1, a, clip_eps),
            y2: power_margin(u2, b, clip_eps),
            du_first: power_derivative(u1, alpha),
            dv_first: power_derivative(u2, beta),
            du_second: power_derivative(u1, a),
            dv_second: power_derivative(u2, b),
        }
    }
}

fn evaluate_parts(
    u1: f64,
    u2: f64,
    params: &KhoudrajiParams,
    clip_eps: f64,
) -> Result<KhoudrajiParts, CopulaError> {
    let m = Margins::new(u1, u2, params, clip_eps);
    Ok(KhoudrajiParts {
        cdf_first: params.first.cdf(m.x1, m.y1, clip_eps)?,
        cdf_second: params.second.cdf(m.x2, m.y2, clip_eps)?,
        h12_first: params.first.cond_first_given_second(m.x1, m.y1, clip_eps)?,
        h21_first: params.first.cond_second_given_first(m.x1, m.y1, clip_eps)?,
        h12_second: params
            .second
            .cond_first_given_second(m.x2, m.y2, clip_eps)?,
        h21_second: params
            .second
            .cond_second_given_first(m.x2, m.y2, clip_eps)?,
        log_pdf_first: params.first.log_pdf(m.x1, m.y1, clip_eps)?,
        log_pdf_second: params.second.log_pdf(m.x2, m.y2, clip_eps)?,
        du_first: m.du_first,
        dv_first: m.dv_first,
        du_second: m.du_second,
        dv_second: m.dv_second,
    })
}

fn power_margin(u: f64, exponent: f64, clip_eps: f64) -> f64 {
    u.powf(exponent).clamp(clip_eps, 1.0 - clip_eps)
}

fn power_derivative(u: f64, exponent: f64) -> f64 {
    if exponent <= 0.0 {
        0.0
    } else {
        exponent * u.powf(exponent - 1.0)
    }
}

fn invert_monotone<F>(f: F, target: f64, clip_eps: f64) -> Result<f64, CopulaError>
where
    F: Fn(f64) -> Result<f64, CopulaError>,
{
    super::invert::invert_increasing(f, target.clamp(clip_eps, 1.0 - clip_eps), clip_eps)
}

fn ln_if_positive(value: f64) -> f64 {
    if value > 0.0 {
        value.ln()
    } else {
        f64::NEG_INFINITY
    }
}

fn logsumexp(values: &[f64]) -> f64 {
    let max = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |left, right| left.max(right));
    if !max.is_finite() {
        return f64::NEG_INFINITY;
    }
    let sum = values.iter().map(|value| (*value - max).exp()).sum::<f64>();
    max + sum.ln()
}
