//! Invariants of the R-, C- and D-vine fitters under the **default**
//! `VineFitOptions` (full default family set including Student t, BB1, BB7
//! and Khoudraji, rotations enabled, AIC selection).
//!
//! Data sets (all simulated in-crate with fixed seeds):
//! 1. negative dependence — a d = 3 Gaussian C-vine with negative
//!    correlations;
//! 2. asymmetric dependence — a d = 3 vine with Tawn and Khoudraji edges;
//! 3. d = 2 — a single Gaussian pair (one tree, the smallest legal vine);
//! 4. d = 6 — a mixed-family D-vine.
//!
//! For every (fitter, data set) the fitted model must satisfy:
//! * `diagnostics.loglik` equals `sum(model.log_pdf(data))` to `1e-9`
//!   (the fitter's bookkeeping and the evaluator agree on the same object);
//! * sampling and then applying the Rosenblatt transform yields independent
//!   uniforms: every pairwise Kendall tau satisfies `|tau| < 0.05` on
//!   `n = 5000` draws (SE of tau under independence is 0.0094, so 0.05 is
//!   5.3 SE and a false alarm is negligible while a wrong h-function moves
//!   tau by 0.1 or more) and every column passes a KS test at the 0.1% level;
//! * `inverse_rosenblatt(rosenblatt(x)) == x` to `1e-8` on the training data.

use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};

use rscopulas::{
    CopulaModel, EvalOptions, KhoudrajiParams, PairCopulaFamily, PairCopulaParams, PairCopulaSpec,
    PseudoObs, Rotation, SampleOptions, VineCopula, VineEdge, VineFitOptions, VineStructureKind,
    VineTree, fit::FitResult,
};

const N_SAMPLE: usize = 5000;
const LOGLIK_TOL: f64 = 1e-9;
const TAU_LIMIT: f64 = 0.05;
const ROUND_TRIP_TOL: f64 = 1e-8;

fn spec(family: PairCopulaFamily, rotation: Rotation, params: PairCopulaParams) -> PairCopulaSpec {
    PairCopulaSpec {
        family,
        rotation,
        params,
    }
}

fn one(family: PairCopulaFamily, rotation: Rotation, value: f64) -> PairCopulaSpec {
    spec(family, rotation, PairCopulaParams::One(value))
}

/// D-vine trees along `order` with one spec per edge (tree by tree).
fn d_vine_trees(order: &[usize], specs: &[PairCopulaSpec]) -> Vec<VineTree> {
    let dim = order.len();
    let mut trees = Vec::new();
    let mut offset = 0;
    for gap in 1..dim {
        let mut edges = Vec::new();
        for start in (0..(dim - gap)).rev() {
            edges.push(VineEdge {
                tree: gap,
                conditioned: (order[start], order[start + gap]),
                conditioning: order[(start + 1)..(start + gap)].to_vec(),
                copula: specs[offset].clone(),
            });
            offset += 1;
        }
        trees.push(VineTree { level: gap, edges });
    }
    assert_eq!(offset, specs.len());
    trees
}

fn simulate<M: CopulaModel>(model: &M, n: usize, seed: u64) -> PseudoObs {
    let mut rng = StdRng::seed_from_u64(seed);
    let samples = model
        .sample(n, &mut rng, &SampleOptions::default())
        .expect("sampling should succeed");
    PseudoObs::new(samples).expect("samples should be pseudo-observations")
}

fn negative_dependence_data() -> PseudoObs {
    let corr = array![[1.0, -0.6, -0.4], [-0.6, 1.0, 0.3], [-0.4, 0.3, 1.0]];
    let model = VineCopula::gaussian_c_vine(vec![0, 1, 2], corr).expect("valid gaussian c-vine");
    simulate(&model, 300, 11)
}

fn asymmetric_data() -> PseudoObs {
    let khoudraji = spec(
        PairCopulaFamily::Khoudraji,
        Rotation::R0,
        PairCopulaParams::Khoudraji(
            KhoudrajiParams::new(
                one(PairCopulaFamily::Clayton, Rotation::R0, 2.5),
                one(PairCopulaFamily::Gumbel, Rotation::R0, 2.0),
                0.7,
                0.3,
            )
            .expect("valid khoudraji"),
        ),
    );
    let trees = vec![
        VineTree {
            level: 1,
            edges: vec![
                VineEdge {
                    tree: 1,
                    conditioned: (0, 2),
                    conditioning: vec![],
                    copula: khoudraji,
                },
                VineEdge {
                    tree: 1,
                    conditioned: (0, 1),
                    conditioning: vec![],
                    copula: spec(
                        PairCopulaFamily::Tawn1,
                        Rotation::R0,
                        PairCopulaParams::Two(3.0, 0.5),
                    ),
                },
            ],
        },
        VineTree {
            level: 2,
            edges: vec![VineEdge {
                tree: 2,
                conditioned: (1, 2),
                conditioning: vec![0],
                copula: spec(
                    PairCopulaFamily::Tawn2,
                    Rotation::R180,
                    PairCopulaParams::Two(2.0, 0.7),
                ),
            }],
        },
    ];
    let model = VineCopula::from_trees(VineStructureKind::C, trees, None).expect("valid vine");
    simulate(&model, 300, 12)
}

fn pair_data() -> PseudoObs {
    let trees = vec![VineTree {
        level: 1,
        edges: vec![VineEdge {
            tree: 1,
            conditioned: (0, 1),
            conditioning: vec![],
            copula: one(PairCopulaFamily::Gaussian, Rotation::R0, 0.5),
        }],
    }];
    let model = VineCopula::from_trees(VineStructureKind::R, trees, None).expect("valid vine");
    simulate(&model, 300, 13)
}

fn six_dimensional_data() -> PseudoObs {
    use PairCopulaFamily as F;
    use Rotation as R;
    let specs = vec![
        // tree 1 (5 edges)
        one(F::Clayton, R::R0, 2.0),
        one(F::Gumbel, R::R180, 1.8),
        one(F::Frank, R::R0, 4.0),
        one(F::Gaussian, R::R0, -0.5),
        one(F::Joe, R::R90, 2.0),
        // tree 2 (4 edges)
        spec(F::StudentT, R::R0, PairCopulaParams::Two(0.4, 5.0)),
        one(F::Clayton, R::R270, 1.2),
        one(F::Gaussian, R::R0, 0.3),
        one(F::Gumbel, R::R0, 1.4),
        // tree 3 (3 edges)
        one(F::Frank, R::R0, 2.0),
        PairCopulaSpec::independence(),
        one(F::Gaussian, R::R0, -0.25),
        // tree 4 (2 edges)
        one(F::Clayton, R::R0, 0.6),
        PairCopulaSpec::independence(),
        // tree 5 (1 edge)
        one(F::Gaussian, R::R0, 0.2),
    ];
    let trees = d_vine_trees(&[0, 1, 2, 3, 4, 5], &specs);
    let model = VineCopula::from_trees(VineStructureKind::D, trees, None).expect("valid vine");
    simulate(&model, 200, 14)
}

#[derive(Clone, Copy, Debug)]
enum Fitter {
    R,
    C,
    D,
}

fn fit(fitter: Fitter, data: &PseudoObs) -> FitResult<VineCopula> {
    let options = VineFitOptions::default();
    match fitter {
        Fitter::R => VineCopula::fit_r_vine(data, &options),
        Fitter::C => VineCopula::fit_c_vine(data, &options),
        Fitter::D => VineCopula::fit_d_vine(data, &options),
    }
    .unwrap_or_else(|err| panic!("{fitter:?}-vine fit should succeed: {err}"))
}

fn ks_statistic(column: &[f64]) -> f64 {
    let mut sorted = column.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len() as f64;
    sorted
        .iter()
        .enumerate()
        .map(|(idx, &x)| ((idx + 1) as f64 / n - x).max(x - idx as f64 / n))
        .fold(0.0, f64::max)
}

fn check_invariants(name: &str, fitter: Fitter, data: &PseudoObs) {
    let fitted = fit(fitter, data);
    let model = &fitted.model;
    let families = model
        .trees()
        .iter()
        .flat_map(|tree| tree.edges.iter())
        .map(|edge| format!("{:?}/{:?}", edge.copula.family, edge.copula.rotation))
        .collect::<Vec<_>>();
    let label = format!("{name} {fitter:?}-vine [{}]", families.join(", "));

    // 1. diagnostics.loglik == sum(log_pdf(data))
    let recomputed: f64 = model
        .log_pdf(data, &EvalOptions::default())
        .expect("log pdf should evaluate")
        .iter()
        .sum();
    assert!(
        (recomputed - fitted.diagnostics.loglik).abs() < LOGLIK_TOL,
        "{label}: diagnostics.loglik {} != sum(log_pdf) {recomputed} (diff {:e})",
        fitted.diagnostics.loglik,
        recomputed - fitted.diagnostics.loglik
    );

    // 2. sample -> rosenblatt gives independent uniforms.
    let mut rng = StdRng::seed_from_u64(0xF17_7E5);
    let samples = model
        .sample(N_SAMPLE, &mut rng, &SampleOptions::default())
        .expect("sampling should succeed");
    let uniforms = model
        .rosenblatt(samples.view(), &SampleOptions::default())
        .expect("rosenblatt should succeed");
    let uniform_obs = PseudoObs::new(uniforms.clone()).expect("rosenblatt output should be valid");
    let tau = rscopulas::stats::kendall_tau_matrix(&uniform_obs);
    let critical = ((2.0_f64 / 0.001).ln() / (2.0 * N_SAMPLE as f64)).sqrt();
    for i in 0..model.dim() {
        let ks = ks_statistic(&uniforms.column(i).to_vec());
        assert!(
            ks < critical,
            "{label}: rosenblatt column {i} KS statistic {ks} exceeds {critical}"
        );
        for j in (i + 1)..model.dim() {
            assert!(
                tau[(i, j)].abs() < TAU_LIMIT,
                "{label}: rosenblatt output tau({i},{j}) = {} is not near zero",
                tau[(i, j)]
            );
        }
    }

    // 3. inverse_rosenblatt(rosenblatt(x)) == x on the training data.
    let forward = model
        .rosenblatt(data.as_view(), &SampleOptions::default())
        .expect("rosenblatt should succeed");
    let back = model
        .inverse_rosenblatt(forward.view(), &SampleOptions::default())
        .expect("inverse rosenblatt should succeed");
    let max_err = data
        .as_view()
        .iter()
        .zip(back.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(
        max_err < ROUND_TRIP_TOL,
        "{label}: inverse_rosenblatt(rosenblatt(x)) deviates from x by {max_err:e}"
    );
}

macro_rules! fitter_invariant_tests {
    ($( $(#[$meta:meta])* $name:ident => ($data:expr, $fitter:expr); )+) => {
        $(
            $(#[$meta])*
            #[test]
            fn $name() {
                let data = $data;
                check_invariants(stringify!($name), $fitter, &data);
            }
        )+
    };
}

fitter_invariant_tests! {
    negative_dependence_r_vine => (negative_dependence_data(), Fitter::R);
    negative_dependence_c_vine => (negative_dependence_data(), Fitter::C);
    negative_dependence_d_vine => (negative_dependence_data(), Fitter::D);
    asymmetric_r_vine => (asymmetric_data(), Fitter::R);
    asymmetric_c_vine => (asymmetric_data(), Fitter::C);
    asymmetric_d_vine => (asymmetric_data(), Fitter::D);
    pair_d2_r_vine => (pair_data(), Fitter::R);
    pair_d2_c_vine => (pair_data(), Fitter::C);
    pair_d2_d_vine => (pair_data(), Fitter::D);
    six_dimensional_r_vine => (six_dimensional_data(), Fitter::R);
    six_dimensional_c_vine => (six_dimensional_data(), Fitter::C);
    six_dimensional_d_vine => (six_dimensional_data(), Fitter::D);
}
