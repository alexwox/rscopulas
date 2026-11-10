//! Shared numerical inverse for h-functions without a closed-form inverse.
//!
//! An h-function `x ↦ h(x | v)` is a conditional CDF: increasing from 0 to 1
//! on the unit interval. Its inverse at level `p` is found after putting
//! *both* axes on the logit scale, i.e. as the root of
//!
//! ```text
//! r(z) = logit(h(expit(z))) − logit(p),      z = logit(x).
//! ```
//!
//! On the input side this stretches the tails, so a root at `1e-9` is as easy
//! to resolve as one at `0.5` and comes back with full *relative* accuracy.
//! On the output side it straightens the sigmoid: for most copulas `r` is
//! close to a straight line, which interpolation-based root finders converge
//! on in very few evaluations.
//!
//! The search brackets the root by stepping from the independence guess
//! `x = p` by the size of the residual (exact for a unit-slope `r`), then runs
//! Chandrupatla's method (inverse quadratic interpolation with a bisection
//! safeguard). A typical inverse costs 5–8 evaluations of `h`, against the 90
//! of a full-width bisection.

use crate::errors::CopulaError;

/// Absolute tolerance on the logit-scale root. `dx = x (1 − x) dz`, so this
/// pins `x` to ~1e-14 relative in both tails.
const LOGIT_TOL: f64 = 1e-14;
/// Residual (on the output logit scale) below which the root is accepted
/// without shrinking the bracket further: `h` then matches `p` to ~13
/// significant digits.
const RESIDUAL_TOL: f64 = 1e-13;
/// Bounds on the first bracketing step, in logit units.
const MIN_STEP: f64 = 0.05;
const MAX_STEP: f64 = 8.0;
/// Hard cap on refinement steps; never reached for a monotone `h` (pure
/// bisection would need ~60 to exhaust the bracket).
const MAX_REFINEMENTS: usize = 200;

#[inline]
fn logit(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}

/// `logit` for a computed probability that may have rounded to exactly 0 or
/// 1: clamped so the result stays finite and monotone.
#[inline]
fn logit_saturating(y: f64) -> f64 {
    logit(y.clamp(1e-300, 1.0 - f64::EPSILON / 2.0))
}

#[inline]
fn expit(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Solves `h(x) = p` for `x` in `[clip_eps, 1 − clip_eps]`, with `h`
/// non-decreasing.
///
/// When `p` lies outside the range `h` attains on the clipped interval the
/// nearest end point is returned. A NaN from `h` is treated as "not below
/// `p`", so a kernel that loses precision in a far tail steers the search
/// away from that tail rather than poisoning it.
pub(super) fn invert_increasing<F>(h: F, p: f64, clip_eps: f64) -> Result<f64, CopulaError>
where
    F: Fn(f64) -> Result<f64, CopulaError>,
{
    let low = clip_eps;
    let high = 1.0 - clip_eps;
    let z_low = logit(low);
    let z_high = logit(high);
    // Maps a logit-scale point back to the clipped unit interval, making the
    // bracket ends exactly the clip bounds.
    let to_unit = |z: f64| {
        if z <= z_low {
            low
        } else if z >= z_high {
            high
        } else {
            expit(z).clamp(low, high)
        }
    };
    let target = logit_saturating(p);
    let residual = |z: f64| -> Result<f64, CopulaError> {
        let value = h(to_unit(z))?;
        Ok(if value.is_nan() {
            1.0
        } else if value == p {
            0.0
        } else {
            logit_saturating(value) - target
        })
    };

    // Bracketing: step from the independence guess by the residual itself
    // (plus a margin so a near-unit slope still brackets), doubling until
    // the sign flips.
    let z_start = logit(p.clamp(low, high)).clamp(z_low, z_high);
    let f_start = residual(z_start)?;
    if f_start.abs() <= RESIDUAL_TOL {
        return Ok(to_unit(z_start));
    }
    let direction = if f_start < 0.0 { 1.0 } else { -1.0 };
    let (mut z_near, mut f_near) = (z_start, f_start);
    let mut step = (1.25 * f_start.abs()).clamp(MIN_STEP, MAX_STEP);
    let (z_far, f_far) = loop {
        let candidate = (z_near + direction * step).clamp(z_low, z_high);
        let f_candidate = residual(candidate)?;
        if f_candidate.abs() <= RESIDUAL_TOL {
            return Ok(to_unit(candidate));
        }
        if (f_candidate < 0.0) != (f_start < 0.0) {
            break (candidate, f_candidate);
        }
        if candidate <= z_low || candidate >= z_high {
            // `p` is outside the attainable range: the root sits at the clip
            // bound in the search direction.
            return Ok(to_unit(candidate));
        }
        // Extrapolate along the secant through the last two points when it
        // is usable; otherwise keep doubling.
        let slope = (f_candidate - f_near) / (candidate - z_near);
        let secant = if slope.is_finite() && slope > 0.0 {
            1.5 * f_candidate.abs() / slope
        } else {
            0.0
        };
        step = secant.max(2.0 * step).min(4.0 * step.max(MAX_STEP));
        z_near = candidate;
        f_near = f_candidate;
    };

    // Chandrupatla's method on the bracket [z_near, z_far], started from the
    // secant point rather than the midpoint.
    let (mut a, mut fa) = (z_far, f_far);
    let (mut b, mut fb) = (z_near, f_near);
    let (mut c, mut fc);
    let mut t = (fa / (fa - fb)).clamp(0.05, 0.95);
    for _ in 0..MAX_REFINEMENTS {
        let zt = a + t * (b - a);
        let ft = residual(zt)?;
        if (ft < 0.0) == (fa < 0.0) {
            c = a;
            fc = fa;
        } else {
            c = b;
            fc = fb;
            b = a;
            fb = fa;
        }
        a = zt;
        fa = ft;
        let (zm, fm) = if fa.abs() < fb.abs() {
            (a, fa)
        } else {
            (b, fb)
        };
        if fm.abs() <= RESIDUAL_TOL {
            return Ok(to_unit(zm));
        }
        // Resolution floor: `x` moves in steps of one ulp (~eps), which is
        // eps / (x (1 − x)) ≈ eps · e^|z| on the logit scale. Below that the
        // residual is a staircase and no further progress is possible.
        let tol = 2.0 * f64::EPSILON * (zm.abs() + zm.abs().exp()) + LOGIT_TOL;
        let t_lim = tol / (b - c).abs();
        if t_lim > 0.5 {
            return Ok(to_unit(zm));
        }
        let xi = (a - b) / (c - b);
        let phi = (fa - fb) / (fc - fb);
        t = if phi * phi < xi && (1.0 - phi) * (1.0 - phi) < 1.0 - xi {
            fa / (fb - fa) * fc / (fb - fc) + (c - a) / (b - a) * fa / (fc - fa) * fb / (fc - fb)
        } else {
            0.5
        };
        t = t.clamp(t_lim, 1.0 - t_lim);
    }
    Ok(to_unit(if fa.abs() < fb.abs() { a } else { b }))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::invert_increasing;

    const CLIP: f64 = 1e-12;

    /// Logistic conditional CDF with location `m` and scale `s` on the logit
    /// scale — the same sigmoid shape real h-functions have.
    fn sigmoid_h(m: f64, s: f64) -> impl Fn(f64) -> f64 {
        move |x: f64| {
            let z = (x / (1.0 - x)).ln();
            1.0 / (1.0 + (-(z - m) / s).exp())
        }
    }

    #[test]
    fn recovers_roots_across_the_unit_interval_including_tails() {
        for (m, s) in [
            (0.0, 1.0),
            (3.0, 0.4),
            (-6.0, 2.5),
            (12.0, 0.2),
            (-15.0, 1.0),
        ] {
            let h = sigmoid_h(m, s);
            for p in [1e-9, 1e-4, 0.03, 0.5, 0.97, 1.0 - 1e-6] {
                let x = invert_increasing(|x| Ok(h(x)), p, CLIP).unwrap();
                if x > CLIP && x < 1.0 - CLIP {
                    // `x` itself is only representable to 1 ulp, which caps
                    // the attainable residual where `h` is steep near 1.
                    assert!(
                        (h(x) - p).abs() <= 1e-10,
                        "m {m} s {s} p {p}: x {x} h {}",
                        h(x)
                    );
                }
            }
        }
    }

    #[test]
    fn stays_far_below_the_cost_of_a_full_bisection() {
        let mut worst = 0usize;
        for (m, s) in [(0.0, 1.0), (3.0, 0.4), (-6.0, 2.5), (8.0, 0.3)] {
            let h = sigmoid_h(m, s);
            for p in [1e-6, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0 - 1e-6] {
                let calls = Cell::new(0usize);
                invert_increasing(
                    |x| {
                        calls.set(calls.get() + 1);
                        Ok(h(x))
                    },
                    p,
                    CLIP,
                )
                .unwrap();
                worst = worst.max(calls.get());
            }
        }
        assert!(worst <= 30, "worst case used {worst} evaluations");
    }

    #[test]
    fn returns_the_clip_bound_when_the_level_is_unattainable() {
        // h only spans [0.2, 0.8] on the clipped interval.
        let h = |x: f64| 0.2 + 0.6 * x;
        assert_eq!(invert_increasing(|x| Ok(h(x)), 0.05, CLIP).unwrap(), CLIP);
        assert_eq!(
            invert_increasing(|x| Ok(h(x)), 0.95, CLIP).unwrap(),
            1.0 - CLIP
        );
    }

    #[test]
    fn treats_nan_as_not_below_the_target() {
        // NaN above 0.9: the search must settle below it.
        let h = |x: f64| if x > 0.9 { f64::NAN } else { x };
        let x = invert_increasing(|x| Ok(h(x)), 0.4, CLIP).unwrap();
        assert!((x - 0.4).abs() < 1e-12, "x = {x}");
        let x = invert_increasing(|x| Ok(h(x)), 0.95, CLIP).unwrap();
        assert!(x <= 0.9 + 1e-9, "x = {x}");
    }

    #[test]
    fn handles_step_like_conditionals() {
        // Near-comonotone pair: h jumps from 0 to 1 around x = 0.3.
        let h = |x: f64| 1.0 / (1.0 + (-(x - 0.3) * 1e7).exp());
        let x = invert_increasing(|x| Ok(h(x)), 0.5, CLIP).unwrap();
        assert!((x - 0.3).abs() < 1e-9, "x = {x}");
    }
}
