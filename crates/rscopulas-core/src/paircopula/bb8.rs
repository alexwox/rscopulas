use crate::errors::{CopulaError, FitError};

// BB8 bivariate copula (Joe–Frank blend).
// Parameters: θ ≥ 1, δ ∈ (0, 1].
//
// Notation:
//   β   = 1 - (1 - δ)^θ          (constant across (u, v)),
//   x_u = 1 - (1 - δ u)^θ,
//   D   = 1 - x_u · x_v / β.
// CDF:  C(u, v) = (1 - D^(1/θ)) / δ.
// h-functions derived by differentiating the CDF. The density assembles
// cleanly: ∂²C/∂u∂v = θδ · (1-δu)^(θ-1) · (1-δv)^(θ-1) · D^(1/θ-2)
//                     · (β - x_u x_v / θ) / β².
// Inverse h-functions fall back to bisection.

struct Terms {
    lb: f64,
    lx: [f64; 2],
    lu: [f64; 2],
    ld: f64,
}
fn prep(u: f64, v: f64, theta: f64, delta: f64) -> Terms {
    use crate::math::{log1mexp, logaddexp};
    let lu = [(-delta * u).ln_1p(), (-delta * v).ln_1p()];
    let log_b_tail = theta * (-delta).ln_1p();
    let lb = log1mexp(log_b_tail);
    let lx = [log1mexp(theta * lu[0]), log1mexp(theta * lu[1])];
    // beta - x_u*x_v = (1-x_u)*x_v + ((1-delta*v)^theta - (1-delta)^theta).
    let ld = logaddexp(
        theta * lu[0] + lx[1],
        theta * lu[1] + log1mexp(log_b_tail - theta * lu[1]),
    ) - lb;
    Terms { lb, lx, lu, ld }
}

pub fn log_pdf(u: f64, v: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    if !theta.is_finite() || theta < 1.0 || !delta.is_finite() || delta <= 0.0 || delta > 1.0 {
        return Err(FitError::Failed {
            reason: "bb8 requires theta >= 1 and delta in (0, 1]",
        }
        .into());
    }
    let t = prep(u, v, theta, delta);
    let lnumer =
        t.lb + crate::math::logaddexp(crate::math::log1mexp(-theta.ln()), t.ld - theta.ln());
    Ok(theta.ln()
        + delta.ln()
        + (theta - 1.0) * (t.lu[0] + t.lu[1])
        + (1.0 / theta - 2.0) * t.ld
        + lnumer
        - 2.0 * t.lb)
}

pub fn cond_first_given_second(u: f64, v: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    let t = prep(u, v, theta, delta);
    Ok((t.lx[0] + (theta - 1.0) * t.lu[1] + (1.0 / theta - 1.0) * t.ld - t.lb).exp())
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
    Ok(-(prep(u, v, theta, delta).ld / theta).exp_m1() / delta)
}
