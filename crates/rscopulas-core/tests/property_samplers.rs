//! Distributional invariants of every sampler in the crate.
//!
//! For each model we draw `n = 5000` rows with a fixed seed and check
//!
//! * **uniform margins** — the Kolmogorov-Smirnov statistic of every column
//!   against `U(0, 1)` stays below the asymptotic 0.1% critical value
//!   `sqrt(ln(2 / 0.001) / (2 n))` (≈ 0.0276 for n = 5000);
//! * **Kendall's tau** — wherever the pair has a closed-form tau (Gaussian /
//!   Student t `2 asin(rho) / pi`, Clayton `theta / (theta + 2)`, Gumbel
//!   `1 - 1 / theta`, Frank via the Debye function, BB1
//!   `1 - 2 / (delta (theta + 2))`, and sign flips for 90/270 rotations) the
//!   sample tau is within `3 SE` of it, with `SE = sqrt(2 (2n + 5) / (9 n (n - 1)))`
//!   (≈ 0.0094 for n = 5000). That is the variance of tau under independence;
//!   for the positively/negatively ordered families used here the variance
//!   under dependence is smaller, so this is a conservative bound.
//!
//! Models covered: the single-family copulas at d = 3 and d = 5, a d = 3
//! C-vine built with `from_trees` for every pair family and every rotation,
//! nested same-family HACs (Clayton, Frank, Gumbel) and a Basic1F factor
//! copula with mixed links.

use ndarray::{Array2, array};
use rand::{SeedableRng, rngs::StdRng};

use rscopulas::{
    ClaytonCopula, CopulaModel, FactorCopula, FrankCopula, GaussianCopula, GumbelHougaardCopula,
    HacFamily, HacNode, HacTree, HierarchicalArchimedeanCopula, KhoudrajiParams, PairCopulaFamily,
    PairCopulaParams, PairCopulaSpec, PseudoObs, Rotation, SampleOptions, StudentTCopula,
    VineCopula, VineEdge, VineStructureKind, VineTree,
};

const N: usize = 5000;
const SEED: u64 = 0x5A11_7E57;

/// Asymptotic Kolmogorov critical value at level `alpha`.
fn ks_critical(n: usize, alpha: f64) -> f64 {
    ((2.0 / alpha).ln() / (2.0 * n as f64)).sqrt()
}

fn ks_statistic(column: &[f64]) -> f64 {
    let mut sorted = column.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len() as f64;
    sorted
        .iter()
        .enumerate()
        .map(|(idx, &x)| {
            let upper = (idx + 1) as f64 / n - x;
            let lower = x - idx as f64 / n;
            upper.max(lower)
        })
        .fold(0.0, f64::max)
}

/// Standard error of Kendall's tau under independence.
fn tau_null_se(n: usize) -> f64 {
    let n = n as f64;
    (2.0 * (2.0 * n + 5.0) / (9.0 * n * (n - 1.0))).sqrt()
}

fn tau_gaussian(rho: f64) -> f64 {
    2.0 * rho.asin() / std::f64::consts::PI
}

fn tau_clayton(theta: f64) -> f64 {
    theta / (theta + 2.0)
}

fn tau_gumbel(theta: f64) -> f64 {
    1.0 - 1.0 / theta
}

/// Debye function `D_1(x) = (1 / x) ∫_0^x t / (e^t - 1) dt` via composite
/// Simpson (the integrand is smooth with limit 1 at t = 0).
fn debye_1(x: f64) -> f64 {
    let m = 4000usize;
    let h = x / m as f64;
    let f = |t: f64| if t < 1e-12 { 1.0 } else { t / t.exp_m1() };
    let mut sum = f(0.0) + f(x);
    for idx in 1..m {
        let weight = if idx % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * f(idx as f64 * h);
    }
    sum * h / 3.0 / x
}

fn tau_frank(theta: f64) -> f64 {
    1.0 - 4.0 / theta * (1.0 - debye_1(theta))
}

fn tau_bb1(theta: f64, delta: f64) -> f64 {
    1.0 - 2.0 / (delta * (theta + 2.0))
}

/// Closed-form Kendall tau of a pair-copula spec, if one exists.
fn closed_form_tau(spec: &PairCopulaSpec) -> Option<f64> {
    use PairCopulaFamily as F;
    use PairCopulaParams as P;
    let base = match (spec.family, &spec.params) {
        (F::Independence, P::None) => 0.0,
        (F::Gaussian, P::One(rho)) => tau_gaussian(*rho),
        (F::StudentT, P::Two(rho, _)) => tau_gaussian(*rho),
        (F::Clayton, P::One(theta)) => tau_clayton(*theta),
        (F::Frank, P::One(theta)) => tau_frank(*theta),
        (F::Gumbel, P::One(theta)) => tau_gumbel(*theta),
        (F::Bb1, P::Two(theta, delta)) => tau_bb1(*theta, *delta),
        _ => return None,
    };
    Some(match spec.rotation {
        Rotation::R0 | Rotation::R180 => base,
        Rotation::R90 | Rotation::R270 => -base,
    })
}

fn assert_uniform_margins(name: &str, samples: &Array2<f64>) {
    let critical = ks_critical(samples.nrows(), 0.001);
    for col in 0..samples.ncols() {
        let column = samples.column(col).to_vec();
        assert!(
            column
                .iter()
                .all(|value| value.is_finite() && *value > 0.0 && *value < 1.0),
            "{name}: column {col} left (0, 1)"
        );
        let ks = ks_statistic(&column);
        assert!(
            ks < critical,
            "{name}: column {col} KS statistic {ks} exceeds the 0.1% critical value {critical}"
        );
    }
}

fn assert_tau(name: &str, tau: &Array2<f64>, i: usize, j: usize, expected: f64) {
    let tolerance = 3.0 * tau_null_se(N);
    let actual = tau[(i, j)];
    assert!(
        (actual - expected).abs() < tolerance,
        "{name}: Kendall tau({i},{j}) = {actual}, expected {expected} (|diff| {} > 3 SE = {tolerance})",
        (actual - expected).abs()
    );
}

fn sample<M: CopulaModel>(model: &M, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    model
        .sample(N, &mut rng, &SampleOptions::default())
        .expect("sampling should succeed")
}

fn tau_matrix(samples: &Array2<f64>) -> Array2<f64> {
    let obs = PseudoObs::new(samples.clone()).expect("samples should be pseudo-observations");
    rscopulas::stats::kendall_tau_matrix(&obs)
}

fn correlation(dim: usize) -> Array2<f64> {
    match dim {
        3 => array![[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]],
        _ => Array2::from_shape_fn((dim, dim), |(i, j)| {
            0.6_f64.powi((i as i64 - j as i64).unsigned_abs() as i32)
        }),
    }
}

// --- single-family copulas ---------------------------------------------------

#[test]
fn gaussian_and_student_t_samplers_have_uniform_margins_and_correct_tau() {
    for dim in [3usize, 5] {
        let corr = correlation(dim);
        let gaussian = GaussianCopula::new(corr.clone()).expect("valid correlation");
        let student = StudentTCopula::new(corr.clone(), 5.0).expect("valid correlation");
        for (name, samples) in [
            (
                format!("Gaussian d={dim}"),
                sample(&gaussian, SEED + dim as u64),
            ),
            (
                format!("StudentT d={dim}"),
                sample(&student, SEED + 10 + dim as u64),
            ),
        ] {
            assert_uniform_margins(&name, &samples);
            let tau = tau_matrix(&samples);
            for i in 0..dim {
                for j in (i + 1)..dim {
                    assert_tau(&name, &tau, i, j, tau_gaussian(corr[(i, j)]));
                }
            }
        }
    }
}

#[test]
fn archimedean_samplers_have_uniform_margins_and_correct_tau() {
    for dim in [3usize, 5] {
        let clayton = ClaytonCopula::new(dim, 1.5).expect("valid clayton");
        let frank = FrankCopula::new(dim, 3.0).expect("valid frank");
        let gumbel = GumbelHougaardCopula::new(dim, 1.8).expect("valid gumbel");
        let cases = [
            (
                format!("Clayton d={dim}"),
                sample(&clayton, SEED + 20 + dim as u64),
                tau_clayton(1.5),
            ),
            (
                format!("Frank d={dim}"),
                sample(&frank, SEED + 30 + dim as u64),
                tau_frank(3.0),
            ),
            (
                format!("Gumbel d={dim}"),
                sample(&gumbel, SEED + 40 + dim as u64),
                tau_gumbel(1.8),
            ),
        ];
        for (name, samples, expected) in cases {
            assert_uniform_margins(&name, &samples);
            let tau = tau_matrix(&samples);
            for i in 0..dim {
                for j in (i + 1)..dim {
                    assert_tau(&name, &tau, i, j, expected);
                }
            }
        }
    }
}

// --- vines built from explicit trees -----------------------------------------

fn spec(family: PairCopulaFamily, rotation: Rotation, params: PairCopulaParams) -> PairCopulaSpec {
    PairCopulaSpec {
        family,
        rotation,
        params,
    }
}

fn pair_specs_for_vines() -> Vec<PairCopulaSpec> {
    use PairCopulaFamily as F;
    use PairCopulaParams as P;
    let khoudraji = P::Khoudraji(
        KhoudrajiParams::new(
            spec(F::Clayton, Rotation::R0, P::One(2.0)),
            spec(F::Gumbel, Rotation::R0, P::One(1.8)),
            0.6,
            0.4,
        )
        .expect("valid khoudraji"),
    );
    let bases = vec![
        (F::Independence, P::None),
        (F::Gaussian, P::One(0.6)),
        (F::StudentT, P::Two(0.6, 5.0)),
        (F::Clayton, P::One(2.0)),
        (F::Frank, P::One(4.0)),
        (F::Gumbel, P::One(1.8)),
        (F::Joe, P::One(2.5)),
        (F::Bb1, P::Two(1.0, 1.8)),
        (F::Bb6, P::Two(1.8, 1.6)),
        (F::Bb7, P::Two(1.5, 1.5)),
        (F::Bb8, P::Two(2.0, 0.8)),
        (F::Tawn1, P::Two(2.0, 0.6)),
        (F::Tawn2, P::Two(2.0, 0.6)),
        (F::Khoudraji, khoudraji),
    ];
    let mut specs = Vec::new();
    for (family, params) in bases {
        for rotation in [Rotation::R0, Rotation::R90, Rotation::R180, Rotation::R270] {
            specs.push(spec(family, rotation, params.clone()));
        }
    }
    specs
}

/// d = 3 C-vine rooted at variable 0 with the same pair-copula on every edge.
fn c_vine_d3(pair: &PairCopulaSpec) -> VineCopula {
    let trees = vec![
        VineTree {
            level: 1,
            edges: vec![
                VineEdge {
                    tree: 1,
                    conditioned: (0, 2),
                    conditioning: vec![],
                    copula: pair.clone(),
                },
                VineEdge {
                    tree: 1,
                    conditioned: (0, 1),
                    conditioning: vec![],
                    copula: pair.clone(),
                },
            ],
        },
        VineTree {
            level: 2,
            edges: vec![VineEdge {
                tree: 2,
                conditioned: (1, 2),
                conditioning: vec![0],
                copula: pair.clone(),
            }],
        },
    ];
    VineCopula::from_trees(VineStructureKind::C, trees, None).expect("vine should build")
}

#[test]
fn vine_sampler_has_uniform_margins_and_correct_tau_for_every_pair_family_and_rotation() {
    for (idx, pair) in pair_specs_for_vines().iter().enumerate() {
        let name = format!(
            "C-vine d=3 {:?} {:?} {:?}",
            pair.family, pair.rotation, pair.params
        );
        let vine = c_vine_d3(pair);
        let samples = sample(&vine, SEED + 100 + idx as u64);
        assert_uniform_margins(&name, &samples);
        if let Some(expected) = closed_form_tau(pair) {
            let tau = tau_matrix(&samples);
            // Tree-1 edges (0,1) and (0,2) carry the pair copula directly.
            assert_tau(&name, &tau, 0, 1, expected);
            assert_tau(&name, &tau, 0, 2, expected);
        }
    }
}

// --- hierarchical Archimedean copulas ----------------------------------------

fn nested_hac(family: HacFamily, outer: f64, inner: f64) -> HierarchicalArchimedeanCopula {
    let tree = HacTree::Node(HacNode::new(
        family,
        outer,
        vec![
            HacTree::Leaf(0),
            HacTree::Node(HacNode::new(
                family,
                inner,
                vec![HacTree::Leaf(1), HacTree::Leaf(2)],
            )),
        ],
    ));
    HierarchicalArchimedeanCopula::new(tree).expect("valid HAC tree")
}

fn check_nested_hac(name: &str, family: HacFamily, outer: f64, inner: f64, seed: u64) {
    let model = nested_hac(family, outer, inner);
    let samples = sample(&model, seed);
    assert_uniform_margins(name, &samples);
    let tau = tau_matrix(&samples);
    let tau_of = |theta: f64| match family {
        HacFamily::Clayton => tau_clayton(theta),
        HacFamily::Frank => tau_frank(theta),
        HacFamily::Gumbel => tau_gumbel(theta),
    };
    assert_tau(name, &tau, 1, 2, tau_of(inner));
    assert_tau(name, &tau, 0, 1, tau_of(outer));
    assert_tau(name, &tau, 0, 2, tau_of(outer));
}

#[test]
fn nested_gumbel_hac_sampler_has_uniform_margins_and_correct_tau() {
    check_nested_hac("nested Gumbel HAC", HacFamily::Gumbel, 1.4, 3.0, SEED + 200);
}

#[test]
fn nested_clayton_hac_sampler_has_uniform_margins_and_correct_tau() {
    check_nested_hac(
        "nested Clayton HAC",
        HacFamily::Clayton,
        0.8,
        2.5,
        SEED + 201,
    );
}

#[test]
fn nested_frank_hac_sampler_has_uniform_margins_and_correct_tau() {
    check_nested_hac("nested Frank HAC", HacFamily::Frank, 1.5, 5.0, SEED + 202);
}

// --- factor copula -------------------------------------------------------------

#[test]
fn factor_copula_sampler_with_mixed_links_has_uniform_margins() {
    use PairCopulaFamily as F;
    use PairCopulaParams as P;
    let links = vec![
        spec(F::Gaussian, Rotation::R0, P::One(0.7)),
        spec(F::Clayton, Rotation::R0, P::One(2.0)),
        spec(F::Gumbel, Rotation::R180, P::One(2.0)),
        spec(F::Frank, Rotation::R0, P::One(5.0)),
        spec(F::Clayton, Rotation::R90, P::One(1.5)),
    ];
    let model = FactorCopula::basic_1f(links, 32).expect("valid factor copula");
    let samples = sample(&model, SEED + 300);
    assert_uniform_margins("Basic1F factor copula", &samples);

    // The latent factor induces positive association between links with
    // positive dependence and negative association with the 90-degree link;
    // no closed form exists, so only the signs are checked.
    let tau = tau_matrix(&samples);
    for (i, j) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        assert!(tau[(i, j)] > 0.1, "factor tau({i},{j}) = {}", tau[(i, j)]);
    }
    for j in 0..4 {
        assert!(tau[(j, 4)] < -0.1, "factor tau({j},4) = {}", tau[(j, 4)]);
    }
}
