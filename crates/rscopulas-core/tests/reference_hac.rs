use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};

use rscopulas::{
    CopulaModel, HacFamily, HacFitMethod, HacFitOptions, HacNode, HacStructureMethod, HacTree,
    HierarchicalArchimedeanCopula, PseudoObs,
};

fn gumbel_tree() -> HacTree {
    HacTree::Node(HacNode::new(
        HacFamily::Gumbel,
        1.2,
        vec![
            HacTree::Leaf(0),
            HacTree::Leaf(1),
            HacTree::Node(HacNode::new(
                HacFamily::Gumbel,
                2.0,
                vec![HacTree::Leaf(2), HacTree::Leaf(3)],
            )),
        ],
    ))
}

#[test]
fn hac_rejects_duplicate_leaf_indices() {
    let tree = HacTree::Node(HacNode::new(
        HacFamily::Gumbel,
        1.2,
        vec![HacTree::Leaf(0), HacTree::Leaf(0)],
    ));
    let error = HierarchicalArchimedeanCopula::new(tree).expect_err("duplicate leaves must fail");
    assert!(error.to_string().contains("duplicate leaf"));
}

#[test]
fn nested_gumbel_sampling_recovers_hierarchical_tau_blocks() {
    let model = HierarchicalArchimedeanCopula::new(gumbel_tree()).expect("tree should be valid");
    let mut rng = StdRng::seed_from_u64(7);
    let sample = model
        .sample(1500, &mut rng, &Default::default())
        .expect("sampling should succeed");
    let pobs = PseudoObs::new(sample).expect("sample must stay inside (0,1)");
    let tau = rscopulas::stats::kendall_tau_matrix(&pobs);

    assert!((tau[(2, 3)] - 0.5).abs() < 0.1);
    assert!((tau[(0, 1)] - (1.0 - 1.0 / 1.2)).abs() < 0.1);
    assert!(tau[(2, 3)] > tau[(0, 2)]);
    assert!(tau[(2, 3)] > tau[(0, 1)]);
}

#[test]
fn hac_fit_with_fixed_tree_recovers_gumbel_parameters() {
    let model = HierarchicalArchimedeanCopula::new(gumbel_tree()).expect("tree should be valid");
    let mut rng = StdRng::seed_from_u64(13);
    let sample = model
        .sample(1200, &mut rng, &Default::default())
        .expect("sampling should succeed");
    let data = PseudoObs::new(sample).expect("sample should remain valid");

    let options = HacFitOptions {
        family_set: vec![HacFamily::Gumbel],
        structure_method: HacStructureMethod::GivenTree,
        fit_method: HacFitMethod::RecursiveMle,
        ..HacFitOptions::default()
    };
    let fit = HierarchicalArchimedeanCopula::fit_with_tree(&data, gumbel_tree(), &options)
        .expect("fit should succeed");
    let params = fit.model.parameters();

    assert_eq!(
        fit.model.families(),
        vec![HacFamily::Gumbel, HacFamily::Gumbel]
    );
    assert_eq!(fit.model.leaf_order(), vec![0, 1, 2, 3]);
    assert!((params[0] - 1.2).abs() < 0.35);
    assert!((params[1] - 2.0).abs() < 0.5);
}

#[test]
fn hac_structure_fit_recovers_tightest_cluster() {
    let model = HierarchicalArchimedeanCopula::new(gumbel_tree()).expect("tree should be valid");
    let mut rng = StdRng::seed_from_u64(17);
    let sample = model
        .sample(1000, &mut rng, &Default::default())
        .expect("sampling should succeed");
    let data = PseudoObs::new(sample).expect("sample should remain valid");

    let options = HacFitOptions {
        family_set: vec![HacFamily::Gumbel],
        structure_method: HacStructureMethod::AgglomerativeTauThenCollapse,
        fit_method: HacFitMethod::RecursiveMle,
        ..HacFitOptions::default()
    };
    let fit = HierarchicalArchimedeanCopula::fit(&data, &options).expect("fit should succeed");
    let tree = fit.model.tree();

    let nested_pair = match tree {
        HacTree::Node(root) => root
            .children
            .iter()
            .find_map(|child| match child {
                HacTree::Node(node) => Some(node),
                HacTree::Leaf(_) => None,
            })
            .expect("expected one nested child"),
        HacTree::Leaf(_) => panic!("fit should not return a leaf root"),
    };
    let leaves = nested_pair
        .children
        .iter()
        .map(|child| match child {
            HacTree::Leaf(index) => *index,
            HacTree::Node(_) => usize::MAX,
        })
        .collect::<Vec<_>>();

    assert_eq!(leaves, vec![2, 3]);
}

#[test]
fn mixed_family_hac_is_marked_experimental_and_still_samples() {
    let tree = HacTree::Node(HacNode::new(
        HacFamily::Clayton,
        0.8,
        vec![
            HacTree::Leaf(0),
            HacTree::Node(HacNode::new(
                HacFamily::Gumbel,
                1.6,
                vec![HacTree::Leaf(1), HacTree::Leaf(2)],
            )),
        ],
    ));
    let model = HierarchicalArchimedeanCopula::new(tree).expect("mixed tree should construct");
    let mut rng = StdRng::seed_from_u64(23);
    let sample = model
        .sample(64, &mut rng, &Default::default())
        .expect("experimental sampling should succeed");

    assert!(!model.is_exact());
    assert_eq!(sample.dim(), (64, 3));
}

#[test]
fn exchangeable_hac_matches_exact_exchangeable_log_pdf() {
    let tree = HacTree::Node(HacNode::new(
        HacFamily::Clayton,
        1.4,
        vec![HacTree::Leaf(0), HacTree::Leaf(1), HacTree::Leaf(2)],
    ));
    let model = HierarchicalArchimedeanCopula::new(tree).expect("tree should be valid");
    let data = PseudoObs::new(array![
        [0.22, 0.28, 0.31],
        [0.44, 0.47, 0.51],
        [0.63, 0.59, 0.67],
    ])
    .expect("data should be valid");

    let actual = model
        .log_pdf(&data, &Default::default())
        .expect("log pdf should evaluate");
    let exact = rscopulas::ClaytonCopula::new(3, 1.4)
        .expect("theta should be valid")
        .log_pdf(&data, &Default::default())
        .expect("exact log pdf should evaluate");

    for (left, right) in actual.iter().zip(exact.iter()) {
        assert!((left - right).abs() < 1e-10);
    }
}

// --- Nested samplers against the closed-form nested distribution function ---
//
// A nested Archimedean copula has the distribution function
// `C(u) = ψ(Σ_children ψ⁻¹(C_child(u)))`, which involves no frailty at all,
// so it checks the frailty-based sampler independently: the fraction of a
// sample falling below a point must match `C` there to within binomial noise.

fn generator(family: HacFamily, theta: f64, t: f64) -> f64 {
    match family {
        HacFamily::Gumbel => (-t.powf(1.0 / theta)).exp(),
        HacFamily::Clayton => (1.0 + t).powf(-1.0 / theta),
        HacFamily::Frank => -(-(-t).exp() * (-theta).exp_m1()).ln_1p() / theta,
    }
}

fn generator_inverse(family: HacFamily, theta: f64, u: f64) -> f64 {
    match family {
        HacFamily::Gumbel => (-u.ln()).powf(theta),
        HacFamily::Clayton => u.powf(-theta) - 1.0,
        HacFamily::Frank => -((-theta * u).exp_m1() / (-theta).exp_m1()).ln(),
    }
}

fn nested_cdf(tree: &HacTree, point: &[f64]) -> f64 {
    match tree {
        HacTree::Leaf(index) => point[*index],
        HacTree::Node(node) => generator(
            node.family,
            node.theta,
            node.children
                .iter()
                .map(|child| generator_inverse(node.family, node.theta, nested_cdf(child, point)))
                .sum(),
        ),
    }
}

fn node(family: HacFamily, theta: f64, children: Vec<HacTree>) -> HacTree {
    HacTree::Node(HacNode::new(family, theta, children))
}

/// Largest standardised gap between the empirical and exact distribution
/// function over a fixed set of points, and the largest standardised
/// marginal Kolmogorov–Smirnov distance.
fn sampler_discrepancy(tree: HacTree, n: usize, seed: u64) -> (f64, f64) {
    use rand::Rng;
    let model = HierarchicalArchimedeanCopula::new(tree.clone()).expect("valid tree");
    let dim = model.dim();
    let mut rng = StdRng::seed_from_u64(seed);
    let sample = model
        .sample(n, &mut rng, &Default::default())
        .expect("sampling should succeed");

    let mut points = StdRng::seed_from_u64(seed ^ 0x9e37_79b9);
    let mut worst_z = 0.0_f64;
    for idx in 0..120 {
        // A third each: anywhere, near the diagonal, in the upper corner.
        let anchor: f64 = points.random_range(0.05..0.95);
        let point: Vec<f64> = (0..dim)
            .map(|_| match idx % 3 {
                0 => points.random_range(0.03..0.97),
                1 => (anchor + points.random_range(-0.04..0.04)).clamp(0.01, 0.99),
                _ => points.random_range(0.9..0.999),
            })
            .collect();
        let exact = nested_cdf(&tree, &point);
        let below = sample
            .rows()
            .into_iter()
            .filter(|row| row.iter().zip(&point).all(|(x, p)| x <= p))
            .count();
        let standard_error = (exact * (1.0 - exact) / n as f64).sqrt();
        worst_z = worst_z.max((below as f64 / n as f64 - exact).abs() / standard_error);
    }

    let mut worst_ks = 0.0_f64;
    for col in 0..dim {
        let mut column = sample.column(col).to_vec();
        column.sort_by(f64::total_cmp);
        let ks = column
            .iter()
            .enumerate()
            .map(|(idx, x)| (x - (idx as f64 + 0.5) / n as f64).abs())
            .fold(0.0, f64::max);
        worst_ks = worst_ks.max(ks * (n as f64).sqrt());
    }
    (worst_z, worst_ks)
}

#[test]
fn nested_samplers_follow_the_nested_distribution_function() {
    use HacFamily::{Clayton, Frank, Gumbel};
    let leaf = HacTree::Leaf;
    let two_level = |family, outer, inner| {
        node(
            family,
            outer,
            vec![leaf(0), node(family, inner, vec![leaf(1), leaf(2)])],
        )
    };
    let three_level = |family, a, b, c| {
        node(
            family,
            a,
            vec![
                leaf(0),
                node(
                    family,
                    b,
                    vec![leaf(1), node(family, c, vec![leaf(2), leaf(3)])],
                ),
                leaf(4),
            ],
        )
    };
    let cases: Vec<(&str, HacTree)> = vec![
        ("gumbel 1.3 > 2.5", two_level(Gumbel, 1.3, 2.5)),
        ("gumbel 1.3 > 2.5 > 6", three_level(Gumbel, 1.3, 2.5, 6.0)),
        ("clayton 0.4 > 2", two_level(Clayton, 0.4, 2.0)),
        ("clayton 0.4 > 2 > 7", three_level(Clayton, 0.4, 2.0, 7.0)),
        ("clayton 0.05 > 12", two_level(Clayton, 0.05, 12.0)),
        ("clayton 15 > 19", two_level(Clayton, 15.0, 19.0)),
        ("frank 1.5 > 6", two_level(Frank, 1.5, 6.0)),
        ("frank 1.5 > 6 > 18", three_level(Frank, 1.5, 6.0, 18.0)),
        ("frank 0.2 > 0.3", two_level(Frank, 0.2, 0.3)),
        // Parent frailties around and far above the stable-limit threshold.
        ("frank 9.2 > 20", two_level(Frank, 9.2, 20.0)),
        ("frank 12 > 18", two_level(Frank, 12.0, 18.0)),
    ];
    let n = 200_000;
    for (label, tree) in cases {
        let (z, ks) = sampler_discrepancy(tree, n, 2024);
        // 120 points: |z| above 4.5 has probability below 1e-3 under the
        // null; the samplers these replaced scored 13 to 600 at n = 2e6.
        assert!(
            z < 4.5,
            "{label}: distribution function off by {z:.1} standard errors"
        );
        // Kolmogorov: 2.0 is the 99.93 % point.
        assert!(ks < 2.0, "{label}: marginal KS statistic {ks:.2}");
    }
}
