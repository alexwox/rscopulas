//! Kernel-level invariants for every pair-copula family and rotation.
//!
//! These tests need no reference implementation: they check mathematical
//! identities every bivariate copula kernel has to satisfy. Each test walks
//! all cases, collects every violation and fails once with the full list so a
//! single run shows the complete picture.
//!
//! * **h-inverse round trips** — for every family x rotation x parameter set
//!   and every point of a 9 x 9 grid (plus the corner values `1e-8` and
//!   `1 - 1e-8`): `inv(cond(u | v) | v) == u` to `1e-6`, in both conditioning
//!   directions. The forward identity `cond(inv(p | v) | v) == p` is enforced
//!   at every point to `1e-7` (the inverse resolves `u` to f64 precision, so
//!   the residual is `ulp(u) * c(u, v)`, which reaches ~1e-8 for Joe
//!   `theta = 5` at the corners). The round trip itself is only well posed
//!   while the conditional probability `p` is at least `1e-9` away from 0 and
//!   1: the inverse's error is `ulp(p) / c(u, v)` and at the corners the
//!   density vanishes while `1 - p` keeps only ~7 significant digits, so
//!   there (and where `p` is clamped to `[clip_eps, 1 - clip_eps]`) only the
//!   forward identity is checked.
//! * **Density normalisation** — `exp(log_pdf)` integrated on a 400 x 400
//!   midpoint grid over `(0, 1)^2` is within `2e-3` of one. The midpoint rule
//!   cannot resolve the corner singularity of strongly tail-dependent
//!   parameter sets (e.g. Clayton `theta = 6` gives 1.0058), so those sets
//!   are flagged `strong_tail` and skip the 2-D integral; they remain covered
//!   by the one-dimensional identities below, which only use interior `u`.
//! * **Uniform margins of the density** — for fixed interior `u` the
//!   one-dimensional midpoint integral over `v` is within `1e-3` of one and
//!   its running sum reproduces `h_{2|1}(v | u)` to `2e-3` at every grid
//!   boundary, tying the density kernel to the h-function kernel. The same
//!   holds with the roles of `u` and `v` swapped.

use rscopulas::{KhoudrajiParams, PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation};

const CLIP_EPS: f64 = 1e-12;
const ROUND_TRIP_TOL: f64 = 1e-6;
const FORWARD_TOL: f64 = 1e-7;
/// Distance from 0/1 below which the inverse h-function is ill-conditioned.
const WELL_POSED_BAND: f64 = 1e-9;
const GRID: usize = 400;
const NORMALISATION_TOL_2D: f64 = 2e-3;
const NORMALISATION_TOL_1D: f64 = 1e-3;
const CUMULATIVE_TOL: f64 = 2e-3;

struct Case {
    name: String,
    spec: PairCopulaSpec,
    strong_tail: bool,
}

struct ParameterSet {
    family: PairCopulaFamily,
    label: &'static str,
    params: PairCopulaParams,
    /// Kendall tau well above 0.6 with a tail singularity of the density.
    strong_tail: bool,
}

fn rotations() -> [Rotation; 4] {
    [Rotation::R0, Rotation::R90, Rotation::R180, Rotation::R270]
}

fn one(value: f64) -> PairCopulaParams {
    PairCopulaParams::One(value)
}

fn two(first: f64, second: f64) -> PairCopulaParams {
    PairCopulaParams::Two(first, second)
}

fn base(family: PairCopulaFamily, params: PairCopulaParams) -> PairCopulaSpec {
    PairCopulaSpec {
        family,
        rotation: Rotation::R0,
        params,
    }
}

fn khoudraji(first: PairCopulaSpec, second: PairCopulaSpec, a: f64, b: f64) -> PairCopulaParams {
    PairCopulaParams::Khoudraji(KhoudrajiParams::new(first, second, a, b).expect("valid khoudraji"))
}

fn set(
    family: PairCopulaFamily,
    label: &'static str,
    params: PairCopulaParams,
    strong_tail: bool,
) -> ParameterSet {
    ParameterSet {
        family,
        label,
        params,
        strong_tail,
    }
}

/// Parameter sets per family: weak/moderate/strong dependence for the
/// one-parameter families, moderate/strong for the two-parameter ones.
/// Khoudraji bases use closed-form Archimedean copulas so its numerically
/// inverted h-functions stay cheap.
fn parameter_sets() -> Vec<ParameterSet> {
    use PairCopulaFamily as F;
    vec![
        set(F::Independence, "none", PairCopulaParams::None, false),
        set(F::Gaussian, "rho=0.2", one(0.2), false),
        set(F::Gaussian, "rho=0.6", one(0.6), false),
        set(F::Gaussian, "rho=0.9", one(0.9), true),
        set(F::StudentT, "rho=0.3,nu=3", two(0.3, 3.0), false),
        set(F::StudentT, "rho=0.7,nu=8", two(0.7, 8.0), false),
        set(F::Clayton, "theta=0.5", one(0.5), false),
        set(F::Clayton, "theta=2", one(2.0), false),
        set(F::Clayton, "theta=6", one(6.0), true),
        set(F::Frank, "theta=1", one(1.0), false),
        set(F::Frank, "theta=4", one(4.0), false),
        set(F::Frank, "theta=12", one(12.0), false),
        set(F::Gumbel, "theta=1.2", one(1.2), false),
        set(F::Gumbel, "theta=2", one(2.0), false),
        set(F::Gumbel, "theta=4", one(4.0), true),
        set(F::Joe, "theta=1.3", one(1.3), false),
        set(F::Joe, "theta=2.5", one(2.5), false),
        set(F::Joe, "theta=5", one(5.0), true),
        set(F::Bb1, "theta=0.5,delta=1.3", two(0.5, 1.3), false),
        set(F::Bb1, "theta=1.5,delta=2", two(1.5, 2.0), true),
        set(F::Bb6, "theta=1.5,delta=1.5", two(1.5, 1.5), false),
        set(F::Bb6, "theta=2.5,delta=1.2", two(2.5, 1.2), false),
        set(F::Bb7, "theta=1.5,delta=0.8", two(1.5, 0.8), false),
        set(F::Bb7, "theta=2.5,delta=2", two(2.5, 2.0), true),
        set(F::Bb8, "theta=2,delta=0.6", two(2.0, 0.6), false),
        set(F::Bb8, "theta=4,delta=0.9", two(4.0, 0.9), false),
        set(F::Tawn1, "theta=2,alpha=0.6", two(2.0, 0.6), false),
        set(F::Tawn1, "theta=4,alpha=0.3", two(4.0, 0.3), false),
        set(F::Tawn2, "theta=2,beta=0.6", two(2.0, 0.6), false),
        set(F::Tawn2, "theta=4,beta=0.3", two(4.0, 0.3), false),
        set(
            F::Khoudraji,
            "clayton2/gumbel1.8,0.6,0.4",
            khoudraji(
                base(F::Clayton, one(2.0)),
                base(F::Gumbel, one(1.8)),
                0.6,
                0.4,
            ),
            false,
        ),
        set(
            F::Khoudraji,
            "indep/clayton2.5,0.45,0.7",
            khoudraji(
                PairCopulaSpec::independence(),
                base(F::Clayton, one(2.5)),
                0.45,
                0.7,
            ),
            false,
        ),
    ]
}

fn all_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for parameter_set in parameter_sets() {
        for rotation in rotations() {
            let spec = PairCopulaSpec {
                family: parameter_set.family,
                rotation,
                params: parameter_set.params.clone(),
            };
            spec.validate().expect("case spec should be valid");
            cases.push(Case {
                name: format!(
                    "{:?}[{}] {rotation:?}",
                    parameter_set.family, parameter_set.label
                ),
                spec,
                strong_tail: parameter_set.strong_tail,
            });
        }
    }
    cases
}

fn grid_points() -> Vec<f64> {
    let mut points = vec![1e-8];
    points.extend((1..=9).map(|idx| idx as f64 / 10.0));
    points.push(1.0 - 1e-8);
    points
}

fn well_posed(p: f64) -> bool {
    p > WELL_POSED_BAND && p < 1.0 - WELL_POSED_BAND
}

fn fail_if_any(what: &str, violations: Vec<String>) {
    assert!(
        violations.is_empty(),
        "{} {what} violation(s):\n{}",
        violations.len(),
        violations.join("\n")
    );
}

#[test]
fn inverse_h_functions_round_trip_on_grid_and_corners() {
    let grid = grid_points();
    let mut violations = Vec::new();
    for case in all_cases() {
        let spec = &case.spec;
        let mut worst_forward = 0.0_f64;
        let mut worst_round_trip = 0.0_f64;
        let mut worst_point = (0.0, 0.0);
        for &u in &grid {
            for &v in &grid {
                // Direction 1: p = h_{1|2}(u | v), then invert for u.
                let p = spec
                    .cond_first_given_second(u, v, CLIP_EPS)
                    .unwrap_or_else(|err| panic!("{}: h12({u},{v}) failed: {err}", case.name));
                let u_back = spec
                    .inv_first_given_second(p, v, CLIP_EPS)
                    .unwrap_or_else(|err| panic!("{}: hinv12({p},{v}) failed: {err}", case.name));
                let p_back = spec
                    .cond_first_given_second(u_back, v, CLIP_EPS)
                    .expect("h12 should evaluate at the inverse");
                worst_forward = worst_forward.max((p_back - p).abs());
                if well_posed(p) {
                    let err = (u_back - u).abs();
                    if err > worst_round_trip {
                        worst_round_trip = err;
                        worst_point = (u, v);
                    }
                }

                // Direction 2: p = h_{2|1}(v | u), then invert for v.
                let p = spec
                    .cond_second_given_first(u, v, CLIP_EPS)
                    .unwrap_or_else(|err| panic!("{}: h21({u},{v}) failed: {err}", case.name));
                let v_back = spec
                    .inv_second_given_first(u, p, CLIP_EPS)
                    .unwrap_or_else(|err| panic!("{}: hinv21({u},{p}) failed: {err}", case.name));
                let p_back = spec
                    .cond_second_given_first(u, v_back, CLIP_EPS)
                    .expect("h21 should evaluate at the inverse");
                worst_forward = worst_forward.max((p_back - p).abs());
                if well_posed(p) {
                    let err = (v_back - v).abs();
                    if err > worst_round_trip {
                        worst_round_trip = err;
                        worst_point = (u, v);
                    }
                }
            }
        }
        if worst_forward >= FORWARD_TOL {
            violations.push(format!(
                "{}: forward identity h(hinv(p)) = p violated by {worst_forward:e}",
                case.name
            ));
        }
        if worst_round_trip >= ROUND_TRIP_TOL {
            violations.push(format!(
                "{}: round trip hinv(h(x)) = x violated by {worst_round_trip:e} at (u, v) = {worst_point:?}",
                case.name
            ));
        }
    }
    fail_if_any("h-inverse", violations);
}

fn midpoints() -> Vec<f64> {
    (0..GRID)
        .map(|idx| (idx as f64 + 0.5) / GRID as f64)
        .collect()
}

fn density(spec: &PairCopulaSpec, u: f64, v: f64, name: &str) -> f64 {
    let value = spec
        .log_pdf(u, v, CLIP_EPS)
        .unwrap_or_else(|err| panic!("{name}: log_pdf({u},{v}) failed: {err}"))
        .exp();
    assert!(
        value.is_finite() && value >= 0.0,
        "{name}: density at ({u},{v}) is {value}"
    );
    value
}

#[test]
fn densities_integrate_to_one_on_midpoint_grid() {
    let nodes = midpoints();
    let weight = 1.0 / (GRID * GRID) as f64;
    let mut violations = Vec::new();
    for case in all_cases().into_iter().filter(|case| !case.strong_tail) {
        let total: f64 = nodes
            .iter()
            .map(|&u| {
                nodes
                    .iter()
                    .map(|&v| density(&case.spec, u, v, &case.name))
                    .sum::<f64>()
            })
            .sum::<f64>()
            * weight;
        if (total - 1.0).abs() >= NORMALISATION_TOL_2D {
            violations.push(format!(
                "{}: midpoint integral over (0,1)^2 is {total} (err {:e})",
                case.name,
                total - 1.0
            ));
        }
    }
    fail_if_any("2-D normalisation", violations);
}

/// For fixed `u`, `∫_0^1 c(u, v) dv = 1` and `∫_0^v c(u, t) dt = h_{2|1}(v | u)`;
/// symmetrically for fixed `v`.
#[test]
fn density_has_uniform_margins_and_integrates_to_h_functions() {
    let nodes = midpoints();
    let step = 1.0 / GRID as f64;
    let fixed = [0.1, 0.3, 0.5, 0.7, 0.9];
    let mut violations = Vec::new();
    for case in all_cases() {
        let spec = &case.spec;
        let mut worst_margin = 0.0_f64;
        let mut worst_cumulative = 0.0_f64;
        for &u in &fixed {
            // Integrate over v for fixed u.
            let mut running = 0.0;
            for (idx, &v) in nodes.iter().enumerate() {
                running += density(spec, u, v, &case.name) * step;
                let boundary = (idx + 1) as f64 * step;
                if idx + 1 < GRID {
                    let h = spec
                        .cond_second_given_first(u, boundary, CLIP_EPS)
                        .expect("h21 should evaluate");
                    worst_cumulative = worst_cumulative.max((running - h).abs());
                }
            }
            worst_margin = worst_margin.max((running - 1.0).abs());

            // Integrate over u for fixed v (= the same fixed value).
            let v = u;
            let mut running = 0.0;
            for (idx, &s) in nodes.iter().enumerate() {
                running += density(spec, s, v, &case.name) * step;
                let boundary = (idx + 1) as f64 * step;
                if idx + 1 < GRID {
                    let h = spec
                        .cond_first_given_second(boundary, v, CLIP_EPS)
                        .expect("h12 should evaluate");
                    worst_cumulative = worst_cumulative.max((running - h).abs());
                }
            }
            worst_margin = worst_margin.max((running - 1.0).abs());
        }
        if worst_margin >= NORMALISATION_TOL_1D {
            violations.push(format!(
                "{}: ∫ c(u, ·) for fixed u in {fixed:?} deviates from 1 by {worst_margin:e}",
                case.name
            ));
        }
        if worst_cumulative >= CUMULATIVE_TOL {
            violations.push(format!(
                "{}: running integral of c deviates from the h-function by {worst_cumulative:e}",
                case.name
            ));
        }
    }
    fail_if_any("1-D normalisation / h-consistency", violations);
}
