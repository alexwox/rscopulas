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
/// `1 − e^{x}` for `x < 0` without cancellation, given `e^{x}`.
#[inline]
fn one_minus(exp_x: f64, x: f64) -> f64 {
    if exp_x <= 0.5 {
        1.0 - exp_x
    } else {
        -x.exp_m1()
    }
}

fn prep(u: f64, v: f64, theta: f64, delta: f64) -> Terms {
    use crate::math::{log1mexp, logaddexp};
    let lu = [(-delta * u).ln_1p(), (-delta * v).ln_1p()];
    let log_b_tail = theta * (-delta).ln_1p();
    let scaled = [theta * lu[0], theta * lu[1]];
    if scaled[0] > -600.0 && scaled[1] > -600.0 {
        // Plain evaluation. With p_u = (1 − δu)^θ and p_β = (1 − δ)^θ,
        //   β − x_u·x_v = p_u·x_v + (p_v − p_β),
        // a sum of two non-negative terms; each `1 − p` and the difference
        // `p_v − p_β` go through `expm1` where they would cancel.
        let power = [scaled[0].exp(), scaled[1].exp()];
        let x = [
            one_minus(power[0], scaled[0]),
            one_minus(power[1], scaled[1]),
        ];
        let floor = log_b_tail.exp();
        let lb = one_minus(floor, log_b_tail).ln();
        let gap = log_b_tail - scaled[1];
        let above_floor = if gap < -std::f64::consts::LN_2 {
            power[1] - floor
        } else {
            -power[1] * gap.exp_m1()
        };
        let numerator = power[0] * x[1] + above_floor;
        if numerator > 0.0 {
            return Terms {
                lb,
                lx: [x[0].ln(), x[1].ln()],
                lu,
                ld: numerator.ln() - lb,
            };
        }
    }
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
    // ln(β·(1 − 1/θ + D/θ)); θ·(that) = β·(θ − 1 + D).
    let lnumer = if t.ld > -700.0 {
        t.lb + (theta - 1.0 + t.ld.exp()).ln() - theta.ln()
    } else {
        t.lb + crate::math::logaddexp(crate::math::log1mexp(-theta.ln()), t.ld - theta.ln())
    };
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

pub fn cdf(u: f64, v: f64, theta: f64, delta: f64) -> Result<f64, CopulaError> {
    Ok(-(prep(u, v, theta, delta).ld / theta).exp_m1() / delta)
}
