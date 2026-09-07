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

struct Terms {
    lx: [f64; 2],
    lu: [f64; 2],
    lt: f64,
    la: f64,
}
fn prep(u: f64, v: f64, theta: f64, delta: f64) -> Terms {
    use crate::math::{log_neg_log1mexp, log1mexp, log1mexp_neg_exp, logaddexp, softplus};
    let lu = [(-u).ln_1p(), (-v).ln_1p()];
    let lx = [log1mexp(theta * lu[0]), log1mexp(theta * lu[1])];
    let log_phi = |l: f64| {
        let log_a = delta.ln() + log_neg_log1mexp(theta * l);
        if log_a < -36.0 {
            log_a
        } else {
            let a = log_a.exp();
            a + log1mexp(-a)
        }
    };
    let ls = logaddexp(log_phi(lu[0]), log_phi(lu[1]));
    let lt = softplus(ls);
    let log_lt = if ls < -36.0 { ls } else { lt.ln() };
    Terms {
        lx,
        lu,
        lt,
        la: log1mexp_neg_exp(log_lt - delta.ln()),
    }
}

pub fn log_pdf(u: f64, v: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    if !theta.is_finite() || theta < 1.0 || !delta.is_finite() || delta <= 0.0 {
        return Err(FitError::Failed {
            reason: "bb7 requires theta >= 1 and delta > 0",
        }
        .into());
    }
    let t = prep(u, v, theta, delta);
    let bracket = crate::math::logaddexp((theta - 1.0).ln(), (theta * delta).ln_1p() + t.la);
    Ok((-1.0 / delta - 2.0) * t.lt
        + (1.0 / theta - 2.0) * t.la
        + bracket
        + (-delta - 1.0) * (t.lx[0] + t.lx[1])
        + (theta - 1.0) * (t.lu[0] + t.lu[1]))
}

pub fn cond_first_given_second(u: f64, v: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    let t = prep(u, v, theta, delta);
    Ok(((1.0 / theta - 1.0) * t.la
        + (-1.0 / delta - 1.0) * t.lt
        + (-delta - 1.0) * t.lx[1]
        + (theta - 1.0) * t.lu[1])
        .exp())
}

pub fn cond_second_given_first(u: f64, v: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    cond_first_given_second(v, u, theta, delta)
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

pub fn cdf(u: f64, v: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    Ok(-(prep(u, v, theta, delta).la / theta).exp_m1())
}
