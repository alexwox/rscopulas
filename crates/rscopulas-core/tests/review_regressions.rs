use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};
use rscopulas::{
    CopulaModel, FactorCopula, GaussianCopula, HacFamily, HacNode, HacTree,
    HierarchicalArchimedeanCopula, PairCopulaFamily as F, PairCopulaParams as P, PairCopulaSpec,
    PseudoObs, Rotation, VineCopula, VineFitOptions,
};

fn pair(family: F, params: P) -> PairCopulaSpec {
    PairCopulaSpec {
        family,
        params,
        rotation: Rotation::R0,
    }
}

#[test]
fn nested_clayton_and_frank_have_uniform_margins_and_hierarchical_dependence() {
    // Frank taus independently evaluated with R copula::tau(frankCopula(theta)).
    for (family, outer, inner, outer_tau, inner_tau) in [
        (HacFamily::Clayton, 1.0, 2.5, 1.0 / 3.0, 2.5 / 4.5),
        (
            HacFamily::Frank,
            2.0,
            5.0,
            0.213894569219620,
            0.456700958160117,
        ),
    ] {
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
        let model = HierarchicalArchimedeanCopula::new(tree).unwrap();
        let mut rng = StdRng::seed_from_u64(72);
        let sample = model.sample(6000, &mut rng, &Default::default()).unwrap();
        for col in sample.columns() {
            assert!(
                (col.mean().unwrap() - 0.5).abs() < 0.025,
                "{family:?}: mean {}",
                col.mean().unwrap()
            );
            let mut sorted = col.to_vec();
            sorted.sort_by(f64::total_cmp);
            for (i, &u) in sorted.iter().enumerate() {
                assert!(u.is_finite() && u > 0.0 && u < 1.0);
                assert!(
                    (u - (i as f64 + 0.5) / sorted.len() as f64).abs() < 0.035,
                    "{family:?}: margin CDF"
                );
            }
        }
        let tau = rscopulas::stats::kendall_tau_matrix(&PseudoObs::new(sample).unwrap());
        assert!(
            (tau[(1, 2)] - inner_tau).abs() < 0.035,
            "{family:?}: inner {}",
            tau[(1, 2)]
        );
        for j in [1, 2] {
            assert!(
                (tau[(0, j)] - outer_tau).abs() < 0.035,
                "{family:?}: cross {}",
                tau[(0, j)]
            );
        }
    }
}

#[test]
fn elliptical_khoudraji_transposes_preserve_density() {
    for base in [
        pair(F::Gaussian, P::One(0.9)),
        pair(F::StudentT, P::Two(0.8, 4.0)),
    ] {
        for rotation in [Rotation::R0, Rotation::R90, Rotation::R180, Rotation::R270] {
            let mut base = base.clone();
            base.rotation = rotation;
            let model =
                PairCopulaSpec::khoudraji(PairCopulaSpec::independence(), base, 0.3, 0.8).unwrap();
            for (u, v) in [(0.1, 0.7), (0.9, 0.2), (0.3, 0.45), (1e-6, 0.3)] {
                let a = model.log_pdf(u, v, 1e-12).unwrap();
                let b = model.clone().swap_axes().log_pdf(v, u, 1e-12).unwrap();
                assert!((a - b).abs() < 1e-10, "{model:?}: {a} != {b}");
            }
        }
    }
}

#[test]
fn gaussian_pair_fit_preserves_near_unit_dependence() {
    let model = GaussianCopula::new(array![[1.0, 0.999], [0.999, 1.0]]).unwrap();
    let mut rng = StdRng::seed_from_u64(24);
    let data = PseudoObs::new(model.sample(500, &mut rng, &Default::default()).unwrap()).unwrap();
    let fit = VineCopula::fit_r_vine(
        &data,
        &VineFitOptions {
            family_set: vec![F::Gaussian],
            ..Default::default()
        },
    )
    .unwrap();
    assert!(fit.model.pair_parameters()[0] > 0.995);
}

#[test]
fn strong_archimedean_pair_and_factor_densities_remain_finite() {
    for theta in [26.0, 40.0, 1000.0] {
        let clayton = pair(F::Clayton, P::One(theta));
        let bb1 = pair(F::Bb1, P::Two(theta, 1.0));
        for (u, v) in [
            (1e-15, 0.2),
            (1e-15, 1e-15),
            (0.2, 0.7),
            (1.0 - 1e-12, 1.0 - 1e-12),
        ] {
            let a = clayton.log_pdf(u, v, 1e-16).unwrap();
            let b = bb1.log_pdf(u, v, 1e-16).unwrap();
            assert!(a.is_finite() && b.is_finite());
            assert!((a - b).abs() < 1e-8);
            for spec in [&clayton, &bb1] {
                assert!(
                    spec.cond_first_given_second(u, v, 1e-16)
                        .unwrap()
                        .is_finite()
                );
            }
        }
    }
    for family in [F::Gumbel, F::Joe] {
        let spec = pair(family, P::One(40.0));
        for (u, v) in [(1e-15, 0.2), (1.0 - 1e-12, 1.0 - 1e-12)] {
            assert!(spec.log_pdf(u, v, 1e-16).unwrap().is_finite());
            assert!(
                spec.cond_first_given_second(u, v, 1e-16)
                    .unwrap()
                    .is_finite()
            );
        }
    }
    let model = FactorCopula::basic_1f(vec![pair(F::Clayton, P::One(30.0)); 3], 25).unwrap();
    let data = PseudoObs::new(array![[0.2, 0.3, 0.4]]).unwrap();
    assert!(model.log_pdf(&data, &Default::default()).unwrap()[0].is_finite());
    for spec in [
        pair(F::Gumbel, P::One(10.0)),
        pair(F::Joe, P::One(40.0)),
        pair(F::Bb1, P::Two(30.0, 2.0)),
    ] {
        let model = FactorCopula::basic_1f(vec![spec; 3], 25).unwrap();
        let tails =
            PseudoObs::new(array![[1e-12; 3], [1.0 - 1e-10; 3], [1e-12, 0.2, 0.9]]).unwrap();
        assert!(
            model
                .log_pdf(&tails, &Default::default())
                .unwrap()
                .iter()
                .all(|x| x.is_finite())
        );
    }
}

#[test]
fn factor_budget_fixed_mode_and_serialization_are_explicit() {
    use rscopulas::FactorQuadrature;
    let model = FactorCopula::basic_1f(vec![pair(F::Clayton, P::One(3.0)); 3], 25).unwrap();
    let data = PseudoObs::new(array![[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]]).unwrap();
    let constrained = model
        .clone()
        .with_quadrature(FactorQuadrature {
            max_nodes: 48,
            ..Default::default()
        })
        .unwrap();
    assert!(
        constrained
            .log_pdf(&data, &Default::default())
            .unwrap_err()
            .to_string()
            .contains("node budget")
    );
    let policy = FactorQuadrature {
        adaptive: false,
        max_nodes: 48,
        rel_tol: 1e-6,
    };
    let fixed = model.with_quadrature(policy).unwrap();
    let values = fixed.log_pdf(&data, &Default::default()).unwrap();
    assert!(values.iter().all(|x| x.is_finite()));
    let restored: FactorCopula =
        serde_json::from_str(&serde_json::to_string(&fixed).unwrap()).unwrap();
    assert_eq!(restored.quadrature(), policy);
    assert_eq!(
        restored.log_pdf(&data, &Default::default()).unwrap(),
        values
    );
    assert!(
        fixed
            .with_quadrature(FactorQuadrature {
                rel_tol: f64::NAN,
                ..Default::default()
            })
            .is_err()
    );
}

#[test]
fn adaptive_factor_matches_independent_r_integration() {
    // R copula::dCopula for N(.7), Clayton(3), Gumbel(2), integrated with
    // integrate() on normal latent scale [-8,8], rel.tol=1e-11.
    let model = FactorCopula::basic_1f(
        vec![
            pair(F::Gaussian, P::One(0.7)),
            pair(F::Clayton, P::One(3.0)),
            pair(F::Gumbel, P::One(2.0)),
        ],
        25,
    )
    .unwrap();
    let data = PseudoObs::new(array![
        [0.2, 0.3, 0.4],
        [0.01, 0.02, 0.03],
        [0.99, 0.98, 0.97],
        [0.001, 0.5, 0.99]
    ])
    .unwrap();
    let expected = [
        0.636770498294356,
        3.826090271636655,
        2.999659278903385,
        -7.184716164832886,
    ];
    for (actual, reference) in model
        .log_pdf(&data, &Default::default())
        .unwrap()
        .iter()
        .zip(expected)
    {
        assert!((actual - reference).abs() < 1e-8, "{actual} != {reference}");
    }
}

#[test]
fn legacy_vine_state_requires_an_explicit_migration() {
    let model = VineCopula::gaussian_c_vine(vec![0, 1], array![[1.0, 0.5], [0.5, 1.0]]).unwrap();
    let mut state = serde_json::to_value(&model).unwrap();
    assert_eq!(state["format_version"], 1);
    state.as_object_mut().unwrap().remove("format_version");
    let error = serde_json::from_value::<VineCopula>(state.clone()).unwrap_err();
    assert!(error.to_string().contains("unversioned vine"));
    state["format_version"] = 2.into();
    assert!(serde_json::from_value::<VineCopula>(state).is_err());
}

#[test]
fn default_family_selection_reports_the_returned_vine_likelihood() {
    let model = GaussianCopula::new(array![[1.0, 0.8], [0.8, 1.0]]).unwrap();
    let data = PseudoObs::new(
        model
            .sample(80, &mut StdRng::seed_from_u64(2), &Default::default())
            .unwrap(),
    )
    .unwrap();
    for fit in [
        VineCopula::fit_c_vine(&data, &Default::default()),
        VineCopula::fit_d_vine(&data, &Default::default()),
        VineCopula::fit_r_vine(&data, &Default::default()),
    ] {
        let fit = fit.unwrap();
        let actual: f64 = fit
            .model
            .log_pdf(&data, &Default::default())
            .unwrap()
            .iter()
            .sum();
        assert!((actual - fit.diagnostics.loglik).abs() < 1e-9);
    }
}

#[test]
fn tll_bandwidth_is_the_square_root_of_the_covariance_rule() {
    let model = GaussianCopula::new(array![[1.0, 0.5], [0.5, 1.0]]).unwrap();
    let data = model
        .sample(200, &mut StdRng::seed_from_u64(2), &Default::default())
        .unwrap();
    let u = data.column(0).to_vec();
    let v = data.column(1).to_vec();
    let constant = rscopulas::tll_fit(&u, &v, rscopulas::TllOrder::Constant)
        .unwrap()
        .bandwidth();
    let linear = rscopulas::tll_fit(&u, &v, rscopulas::TllOrder::Linear)
        .unwrap()
        .bandwidth();
    let quadratic = rscopulas::tll_fit(&u, &v, rscopulas::TllOrder::Quadratic)
        .unwrap()
        .bandwidth();
    assert!((linear / constant - 1.5_f64.sqrt()).abs() < 1e-12);
    assert!((quadratic / linear - 200.0_f64.powf(1.0 / 15.0)).abs() < 1e-12);
}
