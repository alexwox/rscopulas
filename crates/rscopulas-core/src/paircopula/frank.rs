use crate::archimedean_math::frank;
use crate::errors::{CopulaError, FitError};

// Frank is the one Archimedean family in this crate whose parameter runs over
// the whole real line: θ > 0 gives positive dependence, θ < 0 negative
// dependence, and θ → 0 is independence. The negative branch is the 90°
// reflection of the positive one,
//
//   C_{-θ}(u, v) = v − C_θ(1 − u, v),
//   c_{-θ}(u, v) = c_θ(1 − u, v),
//   h_{2|1}^{-θ}(v | u) = h_{2|1}^{θ}(v | 1 − u),
//   h_{1|2}^{-θ}(u | v) = 1 − h_{1|2}^{θ}(1 − u | v),
//
// so every function below maps θ < 0 onto the numerically stable θ > 0
// formulas through that reflection instead of re-deriving the log-space
// algebra for a negative parameter. θ = 0 itself is rejected: it is the
// independence copula and has its own family.

fn validate_theta(theta: f64) -> Result<(), CopulaError> {
    if !theta.is_finite() || theta == 0.0 {
        return Err(FitError::Failed {
            reason: "frank pair theta must be finite and non-zero",
        }
        .into());
    }
    Ok(())
}

/// Inverts Kendall's τ for Frank. τ and θ share their sign, so τ ∈ (−1, 0)
/// is handled through the positive branch; τ = 0 has no finite non-zero θ.
pub fn theta_from_tau(tau: f64) -> Result<f64, CopulaError> {
    if !tau.is_finite() || tau == 0.0 || tau <= -1.0 || tau >= 1.0 {
        return Err(FitError::Failed {
            reason: "frank pair fit requires non-zero tau in (-1, 1)",
        }
        .into());
    }
    let magnitude = frank::invert_tau(tau.abs(), "frank tau inversion failed to bracket root")?;
    Ok(if tau < 0.0 { -magnitude } else { magnitude })
}

// Stable log(1 - exp(-x)) for x > 0.
//
// For small x the subtraction 1 - exp(-x) loses precision, and for large x
// exp(-x) underflows to 0 so the naive log(1.0 - (-x).exp()) saturates to 0.
// The piecewise form below keeps ~machine precision across the full range.
fn log_one_minus_exp_neg(x: f64) -> f64 {
    // Reject NaN or non-positive inputs explicitly: at x = 0, 1 - exp(-x) = 0
    // and the caller should avoid this; for NaN we propagate a sentinel
    // `NEG_INFINITY` so downstream `logsumexp2` passes through correctly.
    if x.is_nan() || x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x > std::f64::consts::LN_2 {
        // 1 - exp(-x) is close to 1, use log1p(-exp(-x)).
        (-((-x).exp())).ln_1p()
    } else {
        // For small x, -expm1(-x) = 1 - exp(-x) keeps precision.
        (-((-x).exp_m1())).ln()
    }
}

fn logsumexp2(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

pub fn log_pdf(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    validate_theta(theta)?;
    if theta < 0.0 {
        return log_pdf(1.0 - u1, u2, -theta);
    }

    // The Frank bivariate density is
    //   c(u, v) = θ (1 - e^{-θ}) e^{-θ(u+v)} / [(1 - e^{-θ}) - (1 - e^{-θu})(1 - e^{-θv})]^2
    // Expanding the bracketed term produces the identity
    //   (1 - e^{-θ}) - (1 - e^{-θu})(1 - e^{-θv})
    //     = e^{-θu}(1 - e^{-θv}) + e^{-θv}(1 - e^{-θ(1-v)})
    //     = T1 + T2
    // where T1, T2 are both strictly positive for u, v, θ > 0. Computing the
    // denominator via logsumexp of log T1 and log T2 avoids the catastrophic
    // cancellation that wrecks the naive `exp_m1`-based form for large θ
    // (e.g. θ ≈ 37 with u, v well away from 0 makes the raw subtraction
    // evaluate to exactly zero and produces a spurious +∞ log density).
    let log_t1 = -theta * u1 + log_one_minus_exp_neg(theta * u2);
    let log_t2 = -theta * u2 + log_one_minus_exp_neg(theta * (1.0 - u2));
    let log_den = logsumexp2(log_t1, log_t2);
    let log_d = log_one_minus_exp_neg(theta);
    Ok(theta.ln() + log_d - theta * (u1 + u2) - 2.0 * log_den)
}

pub fn cond_first_given_second(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    validate_theta(theta)?;
    if theta < 0.0 {
        return Ok(1.0 - cond_first_given_second(1.0 - u1, u2, -theta)?);
    }
    // h_{1|2}(u1 | u2) = e^{-θ u2} (1 - e^{-θ u1}) / [T1 + T2]
    // The log form is stable even when the denominator underflows.
    let log_num = -theta * u2 + log_one_minus_exp_neg(theta * u1);
    let log_t1 = -theta * u1 + log_one_minus_exp_neg(theta * u2);
    let log_t2 = -theta * u2 + log_one_minus_exp_neg(theta * (1.0 - u2));
    let log_den = logsumexp2(log_t1, log_t2);
    Ok((log_num - log_den).exp())
}

pub fn cond_second_given_first(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    // Frank is symmetric in (u1, u2); reuse the h_{1|2} formula with swapped
    // decomposition so the log-denominator is computed in the cheaper basis.
    validate_theta(theta)?;
    if theta < 0.0 {
        return cond_second_given_first(1.0 - u1, u2, -theta);
    }
    let log_num = -theta * u1 + log_one_minus_exp_neg(theta * u2);
    let log_t1 = -theta * u2 + log_one_minus_exp_neg(theta * u1);
    let log_t2 = -theta * u1 + log_one_minus_exp_neg(theta * (1.0 - u1));
    let log_den = logsumexp2(log_t1, log_t2);
    Ok((log_num - log_den).exp())
}

// Solve h_{1|2}(u1 | u2) = p for u1.
//
// Starting from the closed form
//   a := p d / ((1 - p) e^{-θ u2} + p)   with d = e^{-θ} - 1
//   u1 = -ln(1 + a) / θ
// the naive evaluation in the old code computed 1 + a ≈ 0 for large θ (since
// d ≈ -1 and e^{-θ u2} underflows to 0), overflowing the logarithm. We rewrite
//   exp(-θ u1) = ((1 - p) e^{-θ u2} + p e^{-θ}) / ((1 - p) e^{-θ u2} + p)
// and compute each sum via logsumexp so the result is well defined for any θ.
pub fn inv_first_given_second(p: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    validate_theta(theta)?;
    if theta < 0.0 {
        // 1 − h^{θ'}_{1|2}(1 − u1 | u2) = p  ⇔  1 − u1 = (h^{θ'}_{1|2})^{-1}(1 − p | u2).
        return Ok(1.0 - inv_first_given_second(1.0 - p, u2, -theta)?);
    }
    let log_p = p.ln();
    let log_one_minus_p = (1.0 - p).ln();
    let log_num = logsumexp2(log_one_minus_p - theta * u2, log_p - theta);
    let log_den = logsumexp2(log_one_minus_p - theta * u2, log_p);
    Ok((log_den - log_num) / theta)
}

pub fn inv_second_given_first(u1: f64, p: f64, theta: f64) -> Result<f64, CopulaError> {
    // Symmetric counterpart of inv_first_given_second.
    validate_theta(theta)?;
    if theta < 0.0 {
        return inv_second_given_first(1.0 - u1, p, -theta);
    }
    let log_p = p.ln();
    let log_one_minus_p = (1.0 - p).ln();
    let log_num = logsumexp2(log_one_minus_p - theta * u1, log_p - theta);
    let log_den = logsumexp2(log_one_minus_p - theta * u1, log_p);
    Ok((log_den - log_num) / theta)
}

/// Bivariate Frank CDF, stable for any non-zero θ. For θ > 0 the log of
/// `1 + (e^{-θu} − 1)(e^{-θv} − 1)/(e^{-θ} − 1)` is rewritten in terms of the
/// strictly positive terms `T1 + T2` used by the density, which avoids the
/// cancellation that makes the naive form collapse for large θ. θ < 0 goes
/// through the reflection `C_{-θ}(u, v) = v − C_θ(1 − u, v)`.
pub fn cdf(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    validate_theta(theta)?;
    if theta < 0.0 {
        return Ok((u2 - cdf(1.0 - u1, u2, -theta)?).clamp(0.0, 1.0));
    }
    let log_t1 = -theta * u1 + log_one_minus_exp_neg(theta * u2);
    let log_t2 = -theta * u2 + log_one_minus_exp_neg(theta * (1.0 - u2));
    let log_den = logsumexp2(log_t1, log_t2);
    let log_d = log_one_minus_exp_neg(theta);
    Ok(((log_d - log_den) / theta).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference implementation using the naive algebraic form. Only used to
    // cross-check the stable form for moderate |θ| where both agree. The
    // formula is valid for negative θ as written, which makes it an
    // independent check of the reflection used for θ < 0.
    fn reference_log_pdf(u1: f64, u2: f64, theta: f64) -> f64 {
        // c(u, v) = θ (1 − e^{−θ}) e^{−θ(u+v)} / [(1 − e^{−θ}) − (1 − e^{−θu})(1 − e^{−θv})]²
        // with d = e^{−θ} − 1 < 0 for θ > 0 and d > 0 for θ < 0, so the
        // prefactor θ (1 − e^{−θ}) = −θ d is positive for either sign.
        let d = (-theta).exp_m1();
        let a = (-theta * u1).exp_m1();
        let b = (-theta * u2).exp_m1();
        let den = d + a * b;
        (-theta * d).ln() - theta * (u1 + u2) - 2.0 * den.abs().ln()
    }

    fn reference_h12(u1: f64, u2: f64, theta: f64) -> f64 {
        let a = (-theta * u1).exp() - 1.0;
        let b = (-theta * u2).exp() - 1.0;
        let d = (-theta).exp() - 1.0;
        (-theta * u2).exp() * a / (d + a * b)
    }

    fn reference_cdf(u1: f64, u2: f64, theta: f64) -> f64 {
        let d = (-theta).exp_m1();
        let a = (-theta * u1).exp_m1();
        let b = (-theta * u2).exp_m1();
        -(1.0 + a * b / d).ln() / theta
    }

    #[test]
    fn stable_log_pdf_matches_naive_for_moderate_theta() {
        for &theta in &[0.5_f64, 1.0, 3.0, 7.0, 12.0] {
            for u1 in [0.05_f64, 0.2, 0.5, 0.8, 0.95] {
                for u2 in [0.05_f64, 0.2, 0.5, 0.8, 0.95] {
                    let stable = log_pdf(u1, u2, theta).expect("stable log_pdf ok");
                    let naive = reference_log_pdf(u1, u2, theta);
                    assert!(
                        (stable - naive).abs() < 1e-9,
                        "θ={theta} u1={u1} u2={u2}: stable={stable}, naive={naive}"
                    );
                }
            }
        }
    }

    #[test]
    fn stable_log_pdf_is_finite_for_large_theta() {
        for &theta in &[30.0_f64, 37.45, 60.0, 120.0] {
            for u1 in [1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-6] {
                for u2 in [1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-6] {
                    let value = log_pdf(u1, u2, theta).expect("stable log_pdf ok");
                    assert!(
                        value.is_finite(),
                        "stable log_pdf must stay finite (θ={theta}, u1={u1}, u2={u2}, value={value})"
                    );
                }
            }
        }
    }

    #[test]
    fn stable_cond_first_given_second_matches_naive_for_moderate_theta() {
        for &theta in &[0.5_f64, 1.0, 3.0, 7.0, 12.0] {
            for u1 in [0.05_f64, 0.2, 0.5, 0.8, 0.95] {
                for u2 in [0.05_f64, 0.2, 0.5, 0.8, 0.95] {
                    let stable = cond_first_given_second(u1, u2, theta).expect("ok");
                    let naive = reference_h12(u1, u2, theta);
                    assert!(
                        (stable - naive).abs() < 1e-9,
                        "θ={theta} u1={u1} u2={u2}: stable={stable}, naive={naive}"
                    );
                }
            }
        }
    }

    #[test]
    fn inverse_is_inverse_of_conditional_for_moderate_theta() {
        // Round-trip precision of h ∘ h^{-1} is bounded by the precision of
        // the forward h-function itself, which saturates to 1.0 in f64 once
        // θ · u is large enough. For θ up to ~15 both directions have
        // several decimals of slack, which is more than enough for sampling
        // where we drive h^{-1} with fresh uniforms rather than with h
        // output, so we only check round-trip at moderate θ here.
        for &theta in &[5.0_f64, 15.0] {
            for u1 in [0.05_f64, 0.2, 0.5, 0.8, 0.95] {
                for u2 in [0.05_f64, 0.2, 0.5, 0.8, 0.95] {
                    let p = cond_second_given_first(u1, u2, theta).expect("cond ok");
                    assert!((0.0..=1.0).contains(&p), "h must be in [0,1]: {p}");
                    let u2_hat = inv_second_given_first(u1, p, theta).expect("inv ok");
                    assert!(
                        (u2 - u2_hat).abs() < 1e-6,
                        "θ={theta} u1={u1} u2={u2}: recovered u2_hat={u2_hat}"
                    );
                }
            }
        }
    }

    #[test]
    fn inverse_stays_strictly_inside_unit_interval_for_large_theta() {
        // This is the property that matters for sampling: for any uniform p,
        // h^{-1} must return a value strictly inside (0, 1). The legacy
        // closed-form code would overflow and get clamped to the boundary,
        // producing the pile-up of samples at u ≈ 1 reported by callers.
        for &theta in &[15.0_f64, 37.45, 60.0, 120.0] {
            for &p in &[1e-6_f64, 1e-3, 0.5, 1.0 - 1e-3, 1.0 - 1e-6] {
                for &u_cond in &[1e-6_f64, 1e-3, 0.5, 1.0 - 1e-3, 1.0 - 1e-6] {
                    let v = inv_second_given_first(u_cond, p, theta).expect("inv ok");
                    assert!(
                        v.is_finite() && v > 0.0 && v < 1.0,
                        "θ={theta}: inv_second_given_first({u_cond}, {p}) = {v} escaped (0,1)"
                    );
                }
            }
        }
    }

    #[test]
    fn inverse_stays_in_unit_interval_for_pathological_inputs() {
        // Large θ with near-uniform conditioning variable used to overflow in
        // the old closed form and produce values at the clamp boundary.
        for p in [1e-8, 1e-4, 0.5, 1.0 - 1e-4, 1.0 - 1e-8] {
            for u2 in [1e-8, 0.01, 0.5, 0.99, 1.0 - 1e-8] {
                let value = inv_first_given_second(p, u2, 37.45).expect("ok");
                assert!(
                    value.is_finite() && (0.0..=1.0).contains(&value),
                    "inv_first_given_second(p={p}, u2={u2}, θ=37.45) = {value}"
                );
            }
        }
    }

    #[test]
    fn zero_theta_is_rejected_everywhere() {
        assert!(log_pdf(0.3, 0.4, 0.0).is_err());
        assert!(cond_first_given_second(0.3, 0.4, 0.0).is_err());
        assert!(cond_second_given_first(0.3, 0.4, 0.0).is_err());
        assert!(inv_first_given_second(0.3, 0.4, 0.0).is_err());
        assert!(inv_second_given_first(0.3, 0.4, 0.0).is_err());
        assert!(cdf(0.3, 0.4, 0.0).is_err());
        assert!(theta_from_tau(0.0).is_err());
        assert!(log_pdf(0.3, 0.4, f64::NAN).is_err());
    }

    #[test]
    fn negative_theta_density_is_the_reflection_of_the_positive_one() {
        for &theta in &[0.5_f64, 2.0, 5.0, 12.0, 40.0] {
            for u1 in [0.02_f64, 0.2, 0.5, 0.8, 0.98] {
                for u2 in [0.02_f64, 0.2, 0.5, 0.8, 0.98] {
                    let negative = log_pdf(u1, u2, -theta).expect("negative theta ok");
                    let reflected = log_pdf(1.0 - u1, u2, theta).expect("positive theta ok");
                    assert!(
                        (negative - reflected).abs() < 1e-12,
                        "θ={theta} u1={u1} u2={u2}: {negative} vs {reflected}"
                    );
                    // Frank is exchangeable, so reflecting the other axis
                    // must give the same value as well.
                    let other = log_pdf(u1, 1.0 - u2, theta).expect("positive theta ok");
                    assert!((negative - other).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn negative_theta_matches_the_naive_closed_forms() {
        // The naive algebra is valid for θ < 0 as written, which makes it an
        // independent check that the reflection is the right one.
        for &theta in &[-0.5_f64, -1.0, -3.0, -7.0] {
            for u1 in [0.05_f64, 0.2, 0.5, 0.8, 0.95] {
                for u2 in [0.05_f64, 0.2, 0.5, 0.8, 0.95] {
                    let stable = log_pdf(u1, u2, theta).expect("ok");
                    let naive = reference_log_pdf(u1, u2, theta);
                    assert!(
                        (stable - naive).abs() < 1e-9,
                        "θ={theta} u1={u1} u2={u2}: stable={stable}, naive={naive}"
                    );
                    let h12 = cond_first_given_second(u1, u2, theta).expect("ok");
                    assert!(
                        (h12 - reference_h12(u1, u2, theta)).abs() < 1e-9,
                        "θ={theta} u1={u1} u2={u2}: h12={h12}"
                    );
                    let h21 = cond_second_given_first(u1, u2, theta).expect("ok");
                    assert!(
                        (h21 - reference_h12(u2, u1, theta)).abs() < 1e-9,
                        "θ={theta} u1={u1} u2={u2}: h21={h21}"
                    );
                    let c = cdf(u1, u2, theta).expect("ok");
                    assert!(
                        (c - reference_cdf(u1, u2, theta)).abs() < 1e-9,
                        "θ={theta} u1={u1} u2={u2}: cdf={c}"
                    );
                }
            }
        }
    }

    #[test]
    fn negative_theta_density_integrates_to_one_with_uniform_margins() {
        let (nodes, weights) = crate::math::gauss_legendre_01(64);
        for &theta in &[-0.7_f64, -3.0, -8.0, -20.0] {
            let mut mass = 0.0;
            for (&u, &wu) in nodes.iter().zip(&weights) {
                let mut inner = 0.0;
                for (&v, &wv) in nodes.iter().zip(&weights) {
                    inner += wv * log_pdf(u, v, theta).expect("ok").exp();
                }
                // Integrating over v at fixed u must give the uniform margin.
                assert!(
                    (inner - 1.0).abs() < 2e-4,
                    "θ={theta}: margin at u={u} integrates to {inner}"
                );
                mass += wu * inner;
            }
            assert!((mass - 1.0).abs() < 2e-4, "θ={theta}: total mass {mass}");
        }
    }

    #[test]
    fn negative_theta_h_functions_are_partial_derivatives_of_the_cdf() {
        let step = 1e-6;
        for &theta in &[-0.7_f64, -3.0, -8.0] {
            for u1 in [0.1_f64, 0.4, 0.7] {
                for u2 in [0.2_f64, 0.5, 0.9] {
                    let numeric_h12 = (cdf(u1, u2 + step, theta).unwrap()
                        - cdf(u1, u2 - step, theta).unwrap())
                        / (2.0 * step);
                    let numeric_h21 = (cdf(u1 + step, u2, theta).unwrap()
                        - cdf(u1 - step, u2, theta).unwrap())
                        / (2.0 * step);
                    assert!(
                        (numeric_h12 - cond_first_given_second(u1, u2, theta).unwrap()).abs()
                            < 1e-6
                    );
                    assert!(
                        (numeric_h21 - cond_second_given_first(u1, u2, theta).unwrap()).abs()
                            < 1e-6
                    );
                }
            }
        }
    }

    #[test]
    fn negative_theta_inverses_round_trip() {
        for &theta in &[-0.5_f64, -2.0, -6.0, -15.0, -40.0] {
            for u in [0.03_f64, 0.2, 0.5, 0.8, 0.97] {
                for p in [0.02_f64, 0.25, 0.5, 0.75, 0.98] {
                    let v = inv_second_given_first(u, p, theta).expect("inv ok");
                    assert!(v > 0.0 && v < 1.0, "θ={theta}: inv_2|1({u}, {p}) = {v}");
                    let back = cond_second_given_first(u, v, theta).expect("cond ok");
                    assert!(
                        (back - p).abs() < 1e-8,
                        "θ={theta} u={u} p={p}: h(inv) = {back}"
                    );

                    let w = inv_first_given_second(p, u, theta).expect("inv ok");
                    assert!(w > 0.0 && w < 1.0, "θ={theta}: inv_1|2({p}, {u}) = {w}");
                    let back = cond_first_given_second(w, u, theta).expect("cond ok");
                    assert!(
                        (back - p).abs() < 1e-8,
                        "θ={theta} u={u} p={p}: h(inv) = {back}"
                    );
                }
            }
        }
    }

    #[test]
    fn theta_from_tau_is_odd_in_tau() {
        for tau in [0.05_f64, 0.3, 0.6, 0.9] {
            let positive = theta_from_tau(tau).expect("positive tau");
            let negative = theta_from_tau(-tau).expect("negative tau");
            assert!(positive > 0.0);
            assert!((positive + negative).abs() < 1e-12);
        }
        assert!(theta_from_tau(1.0).is_err());
        assert!(theta_from_tau(-1.0).is_err());
    }
}
