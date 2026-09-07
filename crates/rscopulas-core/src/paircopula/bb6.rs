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

// All small complements stay in log space, including 1 - exp(-q).
struct Terms {
    lx: [f64; 2],
    lw: [f64; 2],
    lu: [f64; 2],
    ls: f64,
    lq: f64,
    q: f64,
    la: f64,
}
fn prep(u: f64, v: f64, theta: f64, delta: f64) -> Terms {
    use crate::math::{log_neg_log1mexp, log1mexp, log1mexp_neg_exp, logaddexp};
    let lu = [(-u).ln_1p(), (-v).ln_1p()];
    let lx = [log1mexp(theta * lu[0]), log1mexp(theta * lu[1])];
    let lw = [
        log_neg_log1mexp(theta * lu[0]),
        log_neg_log1mexp(theta * lu[1]),
    ];
    let ls = logaddexp(delta * lw[0], delta * lw[1]);
    let lq = ls / delta;
    Terms {
        lx,
        lw,
        lu,
        ls,
        lq,
        q: lq.exp(),
        la: log1mexp_neg_exp(lq),
    }
}

pub fn log_pdf(u: f64, v: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    if !theta.is_finite() || theta < 1.0 || !delta.is_finite() || delta < 1.0 {
        return Err(FitError::Failed {
            reason: "bb6 requires theta >= 1 and delta >= 1",
        }
        .into());
    }
    let t = prep(u, v, theta, delta);
    let first = if theta == 1.0 {
        t.la
    } else {
        crate::math::log1mexp(-t.q - theta.ln())
    };
    let bracket = crate::math::logaddexp(t.lq + first, (delta - 1.0).ln() + t.la);
    Ok(theta.ln() - t.q
        + (1.0 / delta - 2.0) * t.ls
        + (1.0 / theta - 2.0) * t.la
        + bracket
        + (delta - 1.0) * (t.lw[0] + t.lw[1])
        + (theta - 1.0) * (t.lu[0] + t.lu[1])
        - t.lx[0]
        - t.lx[1])
}

pub fn cond_first_given_second(u: f64, v: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    let t = prep(u, v, theta, delta);
    Ok((-t.q
        + (1.0 / theta - 1.0) * t.la
        + (1.0 / delta - 1.0) * t.ls
        + (delta - 1.0) * t.lw[1]
        + (theta - 1.0) * t.lu[1]
        - t.lx[1])
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
