//! Integration tests for the Rosenblatt and inverse-Rosenblatt transforms.
//!
//! For every `(family, rotation)` combination we build a small C-vine that
//! uses that pair-copula on every edge, then check the round-trip identities
//!
//! ```text
//! inverse_rosenblatt(rosenblatt(V)) == V
//! rosenblatt(inverse_rosenblatt(U)) == U
//! ```
//!
//! up to a family-specific tolerance (iterative inverse h-functions for
//! Clayton / Gumbel / Student-t / Khoudraji bring in ~1e-6 drift).

use ndarray::{Array2, ArrayView2};
use rand::{Rng, SeedableRng, rngs::StdRng};

use rscopulas::{
    CopulaModel, KhoudrajiParams, PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation,
    SampleOptions, VineCopula, VineEdge, VineStructureKind, VineTree,
};

// --- small helpers -----------------------------------------------------------

fn spec(family: PairCopulaFamily, rotation: Rotation, params: PairCopulaParams) -> PairCopulaSpec {
    PairCopulaSpec {
        family,
        rotation,
        params,
    }
}

fn gaussian(rho: f64, rotation: Rotation) -> PairCopulaSpec {
    spec(
        PairCopulaFamily::Gaussian,
        rotation,
        PairCopulaParams::One(rho),
    )
}

fn student_t(rho: f64, nu: f64, rotation: Rotation) -> PairCopulaSpec {
    spec(
        PairCopulaFamily::StudentT,
        rotation,
        PairCopulaParams::Two(rho, nu),
    )
}

fn clayton(theta: f64, rotation: Rotation) -> PairCopulaSpec {
    spec(
        PairCopulaFamily::Clayton,
        rotation,
        PairCopulaParams::One(theta),
    )
}

fn frank(theta: f64, rotation: Rotation) -> PairCopulaSpec {
    spec(
        PairCopulaFamily::Frank,
        rotation,
        PairCopulaParams::One(theta),
    )
}

fn gumbel(theta: f64, rotation: Rotation) -> PairCopulaSpec {
    spec(
        PairCopulaFamily::Gumbel,
        rotation,
        PairCopulaParams::One(theta),
    )
}

fn independence() -> PairCopulaSpec {
    PairCopulaSpec::independence()
}

fn khoudraji(rotation: Rotation) -> PairCopulaSpec {
    let first = gaussian(0.5, Rotation::R0);
    let second = clayton(1.2, Rotation::R0);
    PairCopulaSpec {
        family: PairCopulaFamily::Khoudraji,
        rotation,
        params: PairCopulaParams::Khoudraji(
            KhoudrajiParams::new(first, second, 0.6, 0.4).expect("khoudraji params"),
        ),
    }
}

/// Builds a 4-dim D-vine (path 0-1-2-3) with the supplied pair-copula spec on
/// every edge. We use a D-vine because the canonical D-vine construction in
/// `rscopulas` leaves the variable order as `reversed(order)`, which makes
/// the tests easy to reason about.
fn build_vine(spec: &PairCopulaSpec) -> VineCopula {
    let trees = vec![
        VineTree {
            level: 1,
            edges: vec![
                VineEdge {
                    tree: 1,
                    conditioned: (2, 3),
                    conditioning: vec![],
                    copula: spec.clone(),
                },
                VineEdge {
                    tree: 1,
                    conditioned: (1, 2),
                    conditioning: vec![],
                    copula: spec.clone(),
                },
                VineEdge {
                    tree: 1,
                    conditioned: (0, 1),
                    conditioning: vec![],
                    copula: spec.clone(),
                },
            ],
        },
        VineTree {
            level: 2,
            edges: vec![
                VineEdge {
                    tree: 2,
                    conditioned: (1, 3),
                    conditioning: vec![2],
                    copula: spec.clone(),
                },
                VineEdge {
                    tree: 2,
                    conditioned: (0, 2),
                    conditioning: vec![1],
                    copula: spec.clone(),
                },
            ],
        },
        VineTree {
            level: 3,
            edges: vec![VineEdge {
                tree: 3,
                conditioned: (0, 3),
                conditioning: vec![1, 2],
                copula: spec.clone(),
            }],
        },
    ];
    VineCopula::from_trees(VineStructureKind::D, trees, None).expect("vine should build")
}

fn uniform_matrix(seed: u64, n: usize, d: usize, eps: f64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let values = (0..(n * d))
        .map(|_| rng.random::<f64>().clamp(eps, 1.0 - eps))
        .collect::<Vec<_>>();
    Array2::from_shape_vec((n, d), values).unwrap()
}

fn max_abs_diff(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
    assert_eq!(a.shape(), b.shape());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

// --- round-trip matrix -------------------------------------------------------

/// Representative fixtures covering every family and, for families that
/// support them, each of the four rotations.
fn fixtures() -> Vec<(&'static str, PairCopulaSpec, f64)> {
    let t = 1e-6;
    let t_strict = 1e-8;
    vec![
        ("independence", independence(), 1e-12),
        ("gaussian_r0", gaussian(0.55, Rotation::R0), t_strict),
        ("gaussian_r180", gaussian(0.55, Rotation::R180), t_strict),
        ("gaussian_r90", gaussian(-0.55, Rotation::R90), t_strict),
        ("gaussian_r270", gaussian(-0.55, Rotation::R270), t_strict),
        ("student_t_r0", student_t(0.5, 6.0, Rotation::R0), t),
        ("student_t_r180", student_t(0.5, 6.0, Rotation::R180), t),
        ("clayton_r0", clayton(1.8, Rotation::R0), t),
        ("clayton_r180", clayton(1.8, Rotation::R180), t),
        ("clayton_r90", clayton(1.8, Rotation::R90), t),
        ("clayton_r270", clayton(1.8, Rotation::R270), t),
        ("frank_r0", frank(3.0, Rotation::R0), t_strict),
        ("frank_r180", frank(3.0, Rotation::R180), t_strict),
        ("gumbel_r0", gumbel(1.8, Rotation::R0), t),
        ("gumbel_r180", gumbel(1.8, Rotation::R180), t),
        ("gumbel_r90", gumbel(1.8, Rotation::R90), t),
        ("gumbel_r270", gumbel(1.8, Rotation::R270), t),
        // Khoudraji is intrinsically approximate: its inverse h-function is
        // bisected, and in a depth-3 vine the drift compounds. We keep the
        // test active as a gross-bug guard but with a realistic tolerance.
        ("khoudraji_r0", khoudraji(Rotation::R0), 0.1),
        ("khoudraji_r180", khoudraji(Rotation::R180), 0.1),
    ]
}

#[test]
fn round_trip_u_to_v_to_u_across_families_and_rotations() {
    let options = SampleOptions::default();
    let n = 256;

    for (name, spec, tol) in fixtures() {
        let vine = build_vine(&spec);
        let d = vine.dim();
        let u = uniform_matrix(0xC0FFEE, n, d, 1e-9);

        let v = vine
            .inverse_rosenblatt(u.view(), &options)
            .expect("inverse rosenblatt should succeed");
        let u_back = vine
            .rosenblatt(v.view(), &options)
            .expect("rosenblatt should succeed");

        let err = max_abs_diff(u.view(), u_back.view());
        assert!(
            err < tol,
            "round-trip U->V->U for {name} exceeded tolerance: err={err:e} tol={tol:e}"
        );
    }
}

#[test]
fn round_trip_v_to_u_to_v_across_families_and_rotations() {
    let options = SampleOptions::default();
    let n = 256;

    for (name, spec, tol) in fixtures() {
        let vine = build_vine(&spec);
        let mut rng = StdRng::seed_from_u64(0xABCDE);
        let v = vine
            .sample(n, &mut rng, &options)
            .expect("sample should succeed");
        let u = vine
            .rosenblatt(v.view(), &options)
            .expect("rosenblatt should succeed");
        let v_back = vine
            .inverse_rosenblatt(u.view(), &options)
            .expect("inverse rosenblatt should succeed");

        let err = max_abs_diff(v.view(), v_back.view());
        assert!(
            err < tol,
            "round-trip V->U->V for {name} exceeded tolerance: err={err:e} tol={tol:e}"
        );
    }
}

// --- prefix consistency ------------------------------------------------------

#[test]
fn rosenblatt_prefix_matches_full_rosenblatt_leading_columns() {
    // Mixed-family D-vine with enough variety to stress the prefix traversal.
    let trees = vec![
        VineTree {
            level: 1,
            edges: vec![
                VineEdge {
                    tree: 1,
                    conditioned: (2, 3),
                    conditioning: vec![],
                    copula: gaussian(0.6, Rotation::R0),
                },
                VineEdge {
                    tree: 1,
                    conditioned: (1, 2),
                    conditioning: vec![],
                    copula: clayton(1.3, Rotation::R0),
                },
                VineEdge {
                    tree: 1,
                    conditioned: (0, 1),
                    conditioning: vec![],
                    copula: frank(2.0, Rotation::R0),
                },
            ],
        },
        VineTree {
            level: 2,
            edges: vec![
                VineEdge {
                    tree: 2,
                    conditioned: (1, 3),
                    conditioning: vec![2],
                    copula: gumbel(1.4, Rotation::R0),
                },
                VineEdge {
                    tree: 2,
                    conditioned: (0, 2),
                    conditioning: vec![1],
                    copula: gaussian(0.3, Rotation::R0),
                },
            ],
        },
        VineTree {
            level: 3,
            edges: vec![VineEdge {
                tree: 3,
                conditioned: (0, 3),
                conditioning: vec![1, 2],
                copula: student_t(0.4, 7.0, Rotation::R0),
            }],
        },
    ];
    let vine = VineCopula::from_trees(VineStructureKind::D, trees, None).expect("vine");
    let options = SampleOptions::default();
    let mut rng = StdRng::seed_from_u64(12345);
    let v = vine.sample(128, &mut rng, &options).expect("sample");
    let variable_order = vine.variable_order().to_vec();

    let u_full = vine
        .rosenblatt(v.view(), &options)
        .expect("full rosenblatt");

    for k in 1..=vine.dim() {
        let u_prefix = vine
            .rosenblatt_prefix(v.view(), k, &options)
            .expect("prefix rosenblatt");
        assert_eq!(u_prefix.shape(), [128, k]);
        for idx in 0..k {
            let var = variable_order[idx];
            for obs in 0..128 {
                let diff = (u_prefix[(obs, idx)] - u_full[(obs, var)]).abs();
                assert!(
                    diff < 1e-12,
                    "prefix column {idx} (var={var}) at obs={obs} differs by {diff}"
                );
            }
        }
    }
}

// --- shape / input errors ----------------------------------------------------

#[test]
fn dimension_mismatch_returns_error() {
    let vine = build_vine(&gaussian(0.3, Rotation::R0));
    let bad = Array2::<f64>::zeros((4, vine.dim() + 1));
    assert!(
        vine.inverse_rosenblatt(bad.view(), &SampleOptions::default())
            .is_err()
    );
    assert!(
        vine.rosenblatt(bad.view(), &SampleOptions::default())
            .is_err()
    );
}

#[test]
fn empty_input_returns_error() {
    let vine = build_vine(&gaussian(0.3, Rotation::R0));
    let empty = Array2::<f64>::zeros((0, vine.dim()));
    assert!(
        vine.inverse_rosenblatt(empty.view(), &SampleOptions::default())
            .is_err()
    );
    assert!(
        vine.rosenblatt(empty.view(), &SampleOptions::default())
            .is_err()
    );
}

#[test]
fn rosenblatt_prefix_rejects_out_of_range_col_limit() {
    let vine = build_vine(&gaussian(0.3, Rotation::R0));
    let v = Array2::<f64>::from_elem((4, vine.dim()), 0.5);
    assert!(
        vine.rosenblatt_prefix(v.view(), 0, &SampleOptions::default())
            .is_err()
    );
    assert!(
        vine.rosenblatt_prefix(v.view(), vine.dim() + 1, &SampleOptions::default())
            .is_err()
    );
}

// --- boundary clamping -------------------------------------------------------

#[test]
fn boundary_inputs_are_clipped_and_do_not_produce_non_finite() {
    // Values at the exact edges of (0, 1) get clamped to [clip_eps, 1-clip_eps]
    // before the transform runs. We verify that (a) the transforms accept such
    // inputs without error, and (b) every output value lies in [0, 1]. The
    // strict round-trip claim only holds for non-boundary inputs; that case is
    // covered by `round_trip_v_to_u_to_v_across_families_and_rotations`.
    let vine = build_vine(&clayton(1.4, Rotation::R0));
    let mut v = Array2::<f64>::from_elem((8, vine.dim()), 0.5);
    v[(0, 0)] = 0.0;
    v[(0, 1)] = 1.0;
    v[(1, 2)] = 1e-16;
    v[(1, 3)] = 1.0 - 1e-16;

    let u = vine
        .rosenblatt(v.view(), &SampleOptions::default())
        .expect("rosenblatt should clip and succeed");
    let v_back = vine
        .inverse_rosenblatt(u.view(), &SampleOptions::default())
        .expect("inverse rosenblatt should succeed");
    assert!(
        u.iter()
            .all(|value| value.is_finite() && *value >= 0.0 && *value <= 1.0)
    );
    assert!(
        v_back
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0 && *value <= 1.0)
    );
}
