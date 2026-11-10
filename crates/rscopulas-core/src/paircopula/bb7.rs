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
    Ok(Density::new(theta, delta).log_pdf_from_log_survival((-u).ln_1p(), (-v).ln_1p()))
}

/// BB7 log-density with its parameter-only terms evaluated once, for callers
/// that evaluate many points at fixed parameters (a likelihood pass).
///
/// Evaluates exactly the expression [`prep`] builds, but computes
/// `ln(1 − (1−u)^θ)` once per margin where `prep` needs it twice, and takes
/// the three parameter logarithms from the struct.
pub(super) struct Density {
    theta: f64,
    delta: f64,
    log_delta: f64,
    /// `ln(θ − 1)`.
    log_theta_minus_one: f64,
    /// `ln(1 + θδ)`.
    log1p_theta_delta: f64,
}

impl Density {
    pub(super) fn new(theta: f64, delta: f64) -> Self {
        Self {
            theta,
            delta,
            log_delta: delta.ln(),
            log_theta_minus_one: (theta - 1.0).ln(),
            log1p_theta_delta: (theta * delta).ln_1p(),
        }
    }

    /// Log-density at `(ln(1 − u₁), ln(1 − u₂))`.
    #[inline]
    pub(super) fn log_pdf_from_log_survival(&self, lu0: f64, lu1: f64) -> f64 {
        use crate::math::{log1mexp, log1mexp_neg_exp, logaddexp, softplus};
        let (theta, delta) = (self.theta, self.delta);
        let scaled = [theta * lu0, theta * lu1];
        let lx = [log1mexp(scaled[0]), log1mexp(scaled[1])];
        // ln φ(u) for the generator φ(u) = x_u^(−δ) − 1, with
        // ln(−ln x_u) taken from `lx` (see `math::log_neg_log1mexp`).
        let log_phi = |scaled: f64, lx: f64| {
            let log_neg_log_x = if scaled < -36.0 { scaled } else { (-lx).ln() };
            let log_a = self.log_delta + log_neg_log_x;
            if log_a < -36.0 {
                log_a
            } else {
                let a = log_a.exp();
                a + log1mexp(-a)
            }
        };
        let ls = logaddexp(log_phi(scaled[0], lx[0]), log_phi(scaled[1], lx[1]));
        let lt = softplus(ls);
        let log_lt = if ls < -36.0 { ls } else { lt.ln() };
        let la = log1mexp_neg_exp(log_lt - self.log_delta);
        let bracket = logaddexp(self.log_theta_minus_one, self.log1p_theta_delta + la);
        (-1.0 / delta - 2.0) * lt
            + (1.0 / theta - 2.0) * la
            + bracket
            + (-delta - 1.0) * (lx[0] + lx[1])
            + (theta - 1.0) * (lu0 + lu1)
    }
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
    Ok(-(prep(u, v, theta, delta).la / theta).exp_m1())
}

impl Density {
    /// `(h_{1|2}, h_{2|1})`, bit-identical to the stand-alone functions.
    #[inline]
    pub(super) fn cond_both(&self, u1: f64, u2: f64) -> (f64, f64) {
        let (theta, delta) = (self.theta, self.delta);
        let t = prep(u1, u2, theta, delta);
        let shared = (1.0 / theta - 1.0) * t.la + (-1.0 / delta - 1.0) * t.lt;
        (
            (shared + (-delta - 1.0) * t.lx[1] + (theta - 1.0) * t.lu[1]).exp(),
            (shared + (-delta - 1.0) * t.lx[0] + (theta - 1.0) * t.lu[0]).exp(),
        )
    }

    /// `(log c, h_{1|2}, h_{2|1})` sharing the BB7 terms. Each component is
    /// bit-identical to its stand-alone function.
    #[inline]
    pub(super) fn eval3(&self, u1: f64, u2: f64) -> (f64, f64, f64) {
        let (theta, delta) = (self.theta, self.delta);
        let t = prep(u1, u2, theta, delta);
        let bracket =
            crate::math::logaddexp(self.log_theta_minus_one, self.log1p_theta_delta + t.la);
        let density = (-1.0 / delta - 2.0) * t.lt
            + (1.0 / theta - 2.0) * t.la
            + bracket
            + (-delta - 1.0) * (t.lx[0] + t.lx[1])
            + (theta - 1.0) * (t.lu[0] + t.lu[1]);
        let shared = (1.0 / theta - 1.0) * t.la + (-1.0 / delta - 1.0) * t.lt;
        (
            density,
            (shared + (-delta - 1.0) * t.lx[1] + (theta - 1.0) * t.lu[1]).exp(),
            (shared + (-delta - 1.0) * t.lx[0] + (theta - 1.0) * t.lu[0]).exp(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{Density, prep};

    /// The log-density written directly in terms of [`prep`], as it was
    /// before `Density` shared sub-expressions.
    fn log_pdf_via_prep(u: f64, v: f64, theta: f64, delta: f64) -> f64 {
        let t = prep(u, v, theta, delta);
        let bracket = crate::math::logaddexp((theta - 1.0).ln(), (theta * delta).ln_1p() + t.la);
        (-1.0 / delta - 2.0) * t.lt
            + (1.0 / theta - 2.0) * t.la
            + bracket
            + (-delta - 1.0) * (t.lx[0] + t.lx[1])
            + (theta - 1.0) * (t.lu[0] + t.lu[1])
    }

    #[test]
    fn density_is_bit_identical_to_the_prep_based_formula() {
        let grid: [f64; 10] = [
            1e-12,
            1e-6,
            1e-3,
            0.05,
            0.3,
            0.5,
            0.8,
            0.97,
            1.0 - 1e-6,
            1.0 - 1e-12,
        ];
        for (theta, delta) in [
            (1.0, 0.5),
            (1.4, 0.6),
            (3.0, 3.0),
            (12.0, 0.01),
            (1.001, 40.0),
        ] {
            let density = Density::new(theta, delta);
            for &u in &grid {
                for &v in &grid {
                    let fast = density.log_pdf_from_log_survival((-u).ln_1p(), (-v).ln_1p());
                    let reference = log_pdf_via_prep(u, v, theta, delta);
                    assert!(
                        fast.to_bits() == reference.to_bits()
                            || (fast.is_nan() && reference.is_nan()),
                        "theta {theta} delta {delta} ({u}, {v}): {fast} vs {reference}"
                    );
                }
            }
        }
    }
}
