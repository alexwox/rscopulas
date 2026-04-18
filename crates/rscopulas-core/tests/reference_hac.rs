use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};

use rscopulas_core::{
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
    let tau = rscopulas_core::stats::kendall_tau_matrix(&pobs);

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
    let exact = rscopulas_core::ClaytonCopula::new(3, 1.4)
        .expect("theta should be valid")
        .log_pdf(&data, &Default::default())
        .expect("exact log pdf should evaluate");

    for (left, right) in actual.iter().zip(exact.iter()) {
        assert!((left - right).abs() < 1e-10);
    }
}
