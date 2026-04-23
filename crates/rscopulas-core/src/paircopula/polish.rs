//! Unconstrained-space reparametrisations of pair-copula parameters, used by
//! the factor-copula joint-MLE polish stage.
//!
//! The polish runs a coordinate-ascent loop over the factor log-likelihood in
//! a Euclidean parameter space, so every family's natural-parameter bound
//! (`ρ ∈ (-1, 1)`, `θ > 0`, `θ ≥ 1`, `δ ∈ (0, 1]`, …) is mapped to R through
//! a smooth bijection. The conventions here follow Joe's `CopulaModel` package
//! with a few modernisations:
//!
//! * Gaussian ρ uses `atanh` (Joe leaves ρ unreparametrised; `atanh` is cleaner
//!   and unifies the treatment with Student-t's ρ block).
//! * `θ ≥ 1` parameters use `ln(θ − 1)` rather than Joe's `1 + p²` squared-
//!   shift. Both give a smooth bijection onto R; the logarithm's Jacobian is
//!   trivial to write down for delta-method standard errors.
//! * `δ ∈ (0, 1]` parameters use `logit`.
//!
//! Skipped families (Independence, TLL, Khoudraji, and Student-t's ν block)
//! return no polishable entries — their parameters stay at their sequential-
//! MLE values through the entire polish stage.

use super::common::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec};

/// Flattens the polishable parameters of `spec` into an unconstrained vector
/// for the coordinate-ascent polish. Returns an empty vector when the family
/// has no polishable parameters (Independence / TLL / Khoudraji).
pub(crate) fn encode_params(spec: &PairCopulaSpec) -> Vec<f64> {
    match (spec.family, &spec.params) {
        (PairCopulaFamily::Gaussian, PairCopulaParams::One(rho)) => vec![rho.atanh()],
        (PairCopulaFamily::StudentT, PairCopulaParams::Two(rho, _nu)) => vec![rho.atanh()],
        (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => vec![theta.ln()],
        (PairCopulaFamily::Frank, PairCopulaParams::One(theta)) => vec![*theta],
        (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => vec![(theta - 1.0).ln()],
        (PairCopulaFamily::Joe, PairCopulaParams::One(theta)) => vec![(theta - 1.0).ln()],
        (PairCopulaFamily::Bb1, PairCopulaParams::Two(theta, delta)) => {
            vec![theta.ln(), (delta - 1.0).ln()]
        }
        (PairCopulaFamily::Bb6, PairCopulaParams::Two(theta, delta)) => {
            vec![(theta - 1.0).ln(), (delta - 1.0).ln()]
        }
        (PairCopulaFamily::Bb7, PairCopulaParams::Two(theta, delta)) => {
            vec![(theta - 1.0).ln(), delta.ln()]
        }
        (PairCopulaFamily::Bb8, PairCopulaParams::Two(theta, delta)) => {
            vec![(theta - 1.0).ln(), logit(*delta)]
        }
        (PairCopulaFamily::Tawn1, PairCopulaParams::Two(theta, alpha)) => {
            vec![(theta - 1.0).ln(), logit(*alpha)]
        }
        (PairCopulaFamily::Tawn2, PairCopulaParams::Two(theta, beta)) => {
            vec![(theta - 1.0).ln(), logit(*beta)]
        }
        // Skipped: polish leaves these parameters fixed.
        (PairCopulaFamily::Independence, _)
        | (PairCopulaFamily::Tll, _)
        | (PairCopulaFamily::Khoudraji, _) => Vec::new(),
        // Mismatched family/param combinations should never appear in practice
        // — upstream constructors guarantee the invariant.
        _ => Vec::new(),
    }
}

/// Rebuilds a `PairCopulaSpec` from a slice of unconstrained parameters.
/// `template` supplies the family, rotation, and (for Student-t) the held-
/// fixed ν. Panics if `values.len()` does not match `encode_params(template).len()`
/// — callers are responsible for slicing appropriately.
pub(crate) fn decode_params(template: &PairCopulaSpec, values: &[f64]) -> PairCopulaSpec {
    let params = match (template.family, &template.params) {
        (PairCopulaFamily::Gaussian, _) => {
            assert_eq!(values.len(), 1, "Gaussian expects one unconstrained value");
            PairCopulaParams::One(values[0].tanh().clamp(-0.999_999, 0.999_999))
        }
        (PairCopulaFamily::StudentT, PairCopulaParams::Two(_, nu)) => {
            assert_eq!(values.len(), 1, "StudentT expects one unconstrained value (ρ only)");
            PairCopulaParams::Two(values[0].tanh().clamp(-0.999_999, 0.999_999), *nu)
        }
        (PairCopulaFamily::Clayton, _) => {
            assert_eq!(values.len(), 1);
            PairCopulaParams::One(values[0].exp().max(1e-12))
        }
        (PairCopulaFamily::Frank, _) => {
            assert_eq!(values.len(), 1);
            PairCopulaParams::One(values[0])
        }
        (PairCopulaFamily::Gumbel, _) => {
            assert_eq!(values.len(), 1);
            PairCopulaParams::One(1.0 + values[0].exp().max(1e-12))
        }
        (PairCopulaFamily::Joe, _) => {
            assert_eq!(values.len(), 1);
            PairCopulaParams::One(1.0 + values[0].exp().max(1e-12))
        }
        (PairCopulaFamily::Bb1, _) => {
            assert_eq!(values.len(), 2);
            PairCopulaParams::Two(values[0].exp().max(1e-12), 1.0 + values[1].exp().max(1e-12))
        }
        (PairCopulaFamily::Bb6, _) => {
            assert_eq!(values.len(), 2);
            PairCopulaParams::Two(
                1.0 + values[0].exp().max(1e-12),
                1.0 + values[1].exp().max(1e-12),
            )
        }
        (PairCopulaFamily::Bb7, _) => {
            assert_eq!(values.len(), 2);
            PairCopulaParams::Two(1.0 + values[0].exp().max(1e-12), values[1].exp().max(1e-12))
        }
        (PairCopulaFamily::Bb8, _) => {
            assert_eq!(values.len(), 2);
            let delta = inv_logit(values[1]).clamp(1e-6, 1.0 - 1e-6);
            PairCopulaParams::Two(1.0 + values[0].exp().max(1e-12), delta)
        }
        (PairCopulaFamily::Tawn1, _) => {
            assert_eq!(values.len(), 2);
            PairCopulaParams::Two(
                1.0 + values[0].exp().max(1e-12),
                inv_logit(values[1]).clamp(1e-6, 1.0 - 1e-6),
            )
        }
        (PairCopulaFamily::Tawn2, _) => {
            assert_eq!(values.len(), 2);
            PairCopulaParams::Two(
                1.0 + values[0].exp().max(1e-12),
                inv_logit(values[1]).clamp(1e-6, 1.0 - 1e-6),
            )
        }
        // Skipped families — return the template params unchanged and ignore
        // `values` (which should have been empty if callers respected
        // `encode_params`).
        _ => template.params.clone(),
    };

    PairCopulaSpec {
        family: template.family,
        rotation: template.rotation,
        params,
    }
}

/// Brackets (in unconstrained space) for each polishable parameter of `family`.
/// The order matches `encode_params` so callers can zip the two directly.
pub(crate) fn encode_brackets(family: PairCopulaFamily) -> Vec<(f64, f64)> {
    match family {
        // atanh((-1, 1)) ≈ (-∞, ∞); we bound at ±4 ⇔ |ρ| ≤ tanh(4) ≈ 0.9993,
        // which matches the clamp inside `decode_params` and leaves enough
        // slack for polishing toward the boundary from a warm start.
        PairCopulaFamily::Gaussian | PairCopulaFamily::StudentT => vec![(-4.0, 4.0)],
        // log θ for θ > 0: (e⁻¹⁰, e⁶) ≈ (4.5e-5, 403).
        PairCopulaFamily::Clayton => vec![(-10.0, 6.0)],
        // Frank θ ∈ R; the package's sequential fitter currently only emits
        // positive θ (it works on the τ-implied sign), but the polish can
        // still search over the full line.
        PairCopulaFamily::Frank => vec![(-30.0, 30.0)],
        // log(θ - 1) for θ ≥ 1: (1 + 4.5e-5, 1 + 54.6).
        PairCopulaFamily::Gumbel | PairCopulaFamily::Joe => vec![(-10.0, 4.0)],
        PairCopulaFamily::Bb1 => vec![(-10.0, 6.0), (-10.0, 4.0)],
        PairCopulaFamily::Bb6 => vec![(-10.0, 4.0), (-10.0, 4.0)],
        PairCopulaFamily::Bb7 => vec![(-10.0, 4.0), (-10.0, 6.0)],
        PairCopulaFamily::Bb8 => vec![(-10.0, 4.0), (-10.0, 10.0)],
        PairCopulaFamily::Tawn1 | PairCopulaFamily::Tawn2 => vec![(-10.0, 4.0), (-10.0, 10.0)],
        PairCopulaFamily::Independence
        | PairCopulaFamily::Tll
        | PairCopulaFamily::Khoudraji => Vec::new(),
    }
}

/// Jacobian |dθ_k / dx_k| at the current `spec`, matching the ordering of
/// `encode_params`. Used by the delta method to convert unconstrained-space
/// standard errors back to natural-parameter standard errors:
///
/// ```text
///     SE(θ_k) = |dθ_k / dx_k| · sqrt([H⁻¹]_{kk})
/// ```
pub(crate) fn encode_jacobian(spec: &PairCopulaSpec) -> Vec<f64> {
    match (spec.family, &spec.params) {
        // d(tanh x)/dx = 1 - tanh²(x) = 1 - ρ².
        (PairCopulaFamily::Gaussian, PairCopulaParams::One(rho)) => vec![1.0 - rho * rho],
        (PairCopulaFamily::StudentT, PairCopulaParams::Two(rho, _)) => vec![1.0 - rho * rho],
        // d(exp x)/dx = exp(x) = θ.
        (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => vec![*theta],
        (PairCopulaFamily::Frank, PairCopulaParams::One(_)) => vec![1.0],
        // θ = 1 + exp(x); dθ/dx = exp(x) = θ - 1.
        (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => vec![theta - 1.0],
        (PairCopulaFamily::Joe, PairCopulaParams::One(theta)) => vec![theta - 1.0],
        (PairCopulaFamily::Bb1, PairCopulaParams::Two(theta, delta)) => {
            vec![*theta, delta - 1.0]
        }
        (PairCopulaFamily::Bb6, PairCopulaParams::Two(theta, delta)) => {
            vec![theta - 1.0, delta - 1.0]
        }
        (PairCopulaFamily::Bb7, PairCopulaParams::Two(theta, delta)) => {
            vec![theta - 1.0, *delta]
        }
        (PairCopulaFamily::Bb8, PairCopulaParams::Two(theta, delta)) => {
            // d(inv_logit y)/dy = δ(1 − δ).
            vec![theta - 1.0, delta * (1.0 - delta)]
        }
        (PairCopulaFamily::Tawn1, PairCopulaParams::Two(theta, alpha)) => {
            vec![theta - 1.0, alpha * (1.0 - alpha)]
        }
        (PairCopulaFamily::Tawn2, PairCopulaParams::Two(theta, beta)) => {
            vec![theta - 1.0, beta * (1.0 - beta)]
        }
        _ => Vec::new(),
    }
}

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    (p / (1.0 - p)).ln()
}

fn inv_logit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::super::common::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation};
    use super::*;

    fn spec_one(family: PairCopulaFamily, param: f64) -> PairCopulaSpec {
        PairCopulaSpec {
            family,
            rotation: Rotation::R0,
            params: PairCopulaParams::One(param),
        }
    }

    fn spec_two(family: PairCopulaFamily, first: f64, second: f64) -> PairCopulaSpec {
        PairCopulaSpec {
            family,
            rotation: Rotation::R0,
            params: PairCopulaParams::Two(first, second),
        }
    }

    fn assert_round_trip(spec: &PairCopulaSpec) {
        let encoded = encode_params(spec);
        assert_eq!(encoded.len(), encode_brackets(spec.family).len());
        assert_eq!(encoded.len(), encode_jacobian(spec).len());
        let restored = decode_params(spec, &encoded);
        assert_eq!(restored.family, spec.family);
        assert_eq!(restored.rotation, spec.rotation);
        match (&restored.params, &spec.params) {
            (PairCopulaParams::One(a), PairCopulaParams::One(b)) => assert!((a - b).abs() < 1e-9),
            (PairCopulaParams::Two(a1, a2), PairCopulaParams::Two(b1, b2)) => {
                assert!((a1 - b1).abs() < 1e-9);
                assert!((a2 - b2).abs() < 1e-9);
            }
            _ => panic!("mismatched params after round-trip"),
        }
    }

    #[test]
    fn gaussian_round_trip() {
        for rho in [-0.8, -0.1, 0.0, 0.25, 0.7, 0.95] {
            assert_round_trip(&spec_one(PairCopulaFamily::Gaussian, rho));
        }
    }

    #[test]
    fn clayton_round_trip() {
        for theta in [1e-3, 0.5, 2.0, 8.0] {
            assert_round_trip(&spec_one(PairCopulaFamily::Clayton, theta));
        }
    }

    #[test]
    fn frank_round_trip_through_zero() {
        for theta in [-5.0, -0.01, 0.01, 3.5, 12.0] {
            assert_round_trip(&spec_one(PairCopulaFamily::Frank, theta));
        }
    }

    #[test]
    fn gumbel_round_trip() {
        for theta in [1.0 + 1e-4, 1.5, 4.0, 12.0] {
            assert_round_trip(&spec_one(PairCopulaFamily::Gumbel, theta));
        }
    }

    #[test]
    fn bb1_round_trip() {
        assert_round_trip(&spec_two(PairCopulaFamily::Bb1, 0.8, 1.6));
        assert_round_trip(&spec_two(PairCopulaFamily::Bb1, 2.5, 3.0));
    }

    #[test]
    fn tawn_round_trip_alpha_bounded() {
        assert_round_trip(&spec_two(PairCopulaFamily::Tawn1, 1.5, 0.3));
        assert_round_trip(&spec_two(PairCopulaFamily::Tawn2, 2.0, 0.9));
    }

    #[test]
    fn skipped_family_returns_empty_encoding() {
        let spec = PairCopulaSpec {
            family: PairCopulaFamily::Independence,
            rotation: Rotation::R0,
            params: PairCopulaParams::None,
        };
        assert!(encode_params(&spec).is_empty());
        assert!(encode_brackets(spec.family).is_empty());
        assert!(encode_jacobian(&spec).is_empty());
    }

    #[test]
    fn jacobian_signs_are_positive() {
        // All transforms are monotone, so the Jacobian magnitude is positive
        // everywhere on the bounded domain.
        for spec in [
            spec_one(PairCopulaFamily::Gaussian, 0.6),
            spec_one(PairCopulaFamily::Clayton, 1.2),
            spec_one(PairCopulaFamily::Frank, 4.0),
            spec_one(PairCopulaFamily::Gumbel, 2.3),
            spec_two(PairCopulaFamily::Bb1, 0.7, 2.1),
            spec_two(PairCopulaFamily::Tawn1, 1.8, 0.4),
        ] {
            for value in encode_jacobian(&spec) {
                assert!(value > 0.0, "jacobian entry {value} non-positive");
            }
        }
    }
}
