use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};
use rscopulas::{
    CopulaModel, EvalOptions, GaussianCopula, PairCopulaFamily as F, PairCopulaParams as P,
    PairCopulaSpec, PseudoObs, Rotation, VineCopula, VineEdge, VineFitOptions, VineStructureKind,
    VineTree,
};

fn pair(family: F, params: P, rotation: Rotation) -> PairCopulaSpec {
    PairCopulaSpec {
        family,
        params,
        rotation,
    }
}

fn bivariate(spec: PairCopulaSpec) -> VineCopula {
    VineCopula::from_trees(
        VineStructureKind::R,
        vec![VineTree {
            level: 1,
            edges: vec![VineEdge {
                tree: 1,
                conditioned: (0, 1),
                conditioning: vec![],
                copula: spec,
            }],
        }],
        None,
    )
    .unwrap()
}

#[test]
fn asymmetric_vines_and_axis_transposes_preserve_pair_density() {
    let khoudraji = PairCopulaSpec::khoudraji(
        PairCopulaSpec::independence(),
        pair(F::Clayton, P::One(2.0), Rotation::R0),
        0.3,
        0.8,
    )
    .unwrap();
    let specs = [
        pair(F::Clayton, P::One(2.0), Rotation::R90),
        pair(F::Tawn1, P::Two(3.0, 0.3), Rotation::R0),
        pair(F::Tawn2, P::Two(3.0, 0.3), Rotation::R270),
        khoudraji,
    ];
    let data = PseudoObs::new(array![[0.2, 0.8], [0.8, 0.2], [0.4, 0.7]]).unwrap();
    for spec in specs {
        let vine = bivariate(spec.clone());
        let actual = vine.log_pdf(&data, &EvalOptions::default()).unwrap();
        for (idx, row) in data.as_view().rows().into_iter().enumerate() {
            let expected = spec.log_pdf(row[0], row[1], 1e-12).unwrap();
            assert!((actual[idx] - expected).abs() < 1e-10, "{spec:?}");
            let transposed = spec
                .clone()
                .swap_axes()
                .log_pdf(row[1], row[0], 1e-12)
                .unwrap();
            assert!((transposed - expected).abs() < 1e-10, "{spec:?}");
        }
    }
}

#[test]
fn fitted_likelihood_matches_asymmetric_returned_models() {
    let spec = pair(F::Clayton, P::One(2.0), Rotation::R90);
    let mut rng = StdRng::seed_from_u64(2);
    let data = PseudoObs::new(
        bivariate(spec)
            .sample(200, &mut rng, &Default::default())
            .unwrap(),
    )
    .unwrap();
    let options = VineFitOptions {
        family_set: vec![F::Clayton],
        ..Default::default()
    };
    for fit in [
        VineCopula::fit_c_vine(&data, &options),
        VineCopula::fit_d_vine(&data, &options),
        VineCopula::fit_r_vine(&data, &options),
    ] {
        let fit = fit.unwrap();
        let actual: f64 = fit
            .model
            .log_pdf(&data, &Default::default())
            .unwrap()
            .iter()
            .sum();
        assert!((actual - fit.diagnostics.loglik).abs() < 1e-8);
    }
}

#[test]
fn every_vine_fitter_honors_zero_truncation() {
    let g = GaussianCopula::new(array![[1., 0.7, 0.7], [0.7, 1., 0.7], [0.7, 0.7, 1.]]).unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    let data = PseudoObs::new(g.sample(100, &mut rng, &Default::default()).unwrap()).unwrap();
    let options = VineFitOptions {
        family_set: vec![F::Gaussian],
        truncation_level: Some(0),
        ..Default::default()
    };
    for fit in [
        VineCopula::fit_c_vine(&data, &options),
        VineCopula::fit_d_vine(&data, &options),
        VineCopula::fit_r_vine(&data, &options),
    ] {
        let fit = fit.unwrap();
        assert_eq!(fit.model.truncation_level(), Some(0));
        assert_eq!(fit.diagnostics.loglik, 0.0);
        assert!(
            fit.model
                .log_pdf(&data, &Default::default())
                .unwrap()
                .iter()
                .all(|&x| x == 0.0)
        );
    }
}

#[test]
fn gaussian_conversions_preserve_strong_valid_correlations() {
    let corr = array![[1., 0.999], [0.999, 1.]];
    let data = PseudoObs::new(array![[0.5, 0.5], [0.2, 0.21]]).unwrap();
    let expected = GaussianCopula::new(corr.clone())
        .unwrap()
        .log_pdf(&data, &Default::default())
        .unwrap();
    for vine in [
        VineCopula::gaussian_c_vine(vec![0, 1], corr.clone()),
        VineCopula::gaussian_d_vine(vec![0, 1], corr),
    ] {
        let actual = vine.unwrap().log_pdf(&data, &Default::default()).unwrap();
        for (a, b) in actual.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-8);
        }
    }
}

#[test]
fn malformed_vines_are_rejected_before_runtime() {
    let valid = bivariate(PairCopulaSpec::independence());
    let mut trees = valid.trees().to_vec();
    trees[0].edges[0].conditioned = (0, 0);
    assert!(VineCopula::from_trees(VineStructureKind::R, trees, None).is_err());
    let mut json = serde_json::to_value(&valid).unwrap();
    json["dim"] = 3.into();
    assert!(serde_json::from_value::<VineCopula>(json).is_err());
    assert!(
        valid
            .rosenblatt(array![[f64::NAN, 0.5]].view(), &Default::default())
            .is_err()
    );
}

#[test]
fn invalid_options_and_pair_parameters_return_errors() {
    let g = GaussianCopula::new(array![[1., 0.5], [0.5, 1.]]).unwrap();
    let data = PseudoObs::new(array![[0.2, 0.3]]).unwrap();
    for eps in [f64::NAN, 0.6, 0., -1., 1e-30] {
        assert!(
            g.log_pdf(
                &data,
                &EvalOptions {
                    clip_eps: eps,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    assert!(
        pair(F::Gaussian, P::One(2.), Rotation::R0)
            .cond_first_given_second(0.2, 0.3, 1e-12)
            .is_err()
    );
}

#[test]
fn frank_extreme_sampling_terminates_and_preserves_uniform_margins() {
    for theta in [1e-20, 40.0, 100.0] {
        let model = rscopulas::FrankCopula::new(3, theta).unwrap();
        let mut rng = StdRng::seed_from_u64(123);
        let sample = model.sample(2000, &mut rng, &Default::default()).unwrap();
        assert!(sample.iter().all(|u| u.is_finite() && *u > 0.0 && *u < 1.0));
        for col in sample.columns() {
            assert!((col.mean().unwrap() - 0.5).abs() < 0.025);
        }
        let data = PseudoObs::new(sample).unwrap();
        assert!(
            model
                .log_pdf(&data, &Default::default())
                .unwrap()
                .iter()
                .all(|x| x.is_finite())
        );
    }
}

#[test]
fn bb_upper_tails_are_finite_and_respect_joe_reductions() {
    for (family, params, base) in [
        (
            F::Bb6,
            P::Two(1.0, 2.0),
            pair(F::Gumbel, P::One(2.0), Rotation::R0),
        ),
        (
            F::Bb7,
            P::Two(1.0, 2.0),
            pair(F::Clayton, P::One(2.0), Rotation::R0),
        ),
        (F::Bb8, P::Two(1.0, 0.6), PairCopulaSpec::independence()),
    ] {
        let spec = pair(family, params, Rotation::R0);
        for (u, v) in [(1e-10, 0.2), (0.3, 0.8), (1.0 - 1e-8, 1.0 - 1e-8)] {
            assert!(
                (spec.log_pdf(u, v, 1e-12).unwrap() - base.log_pdf(u, v, 1e-12).unwrap()).abs()
                    < 1e-8
            );
        }
    }
    let joe = pair(F::Joe, P::One(3.0), Rotation::R0);
    for u in [0.7, 1.0 - 1e-8, 1.0 - 1e-12] {
        for family in [F::Bb6, F::Bb8] {
            let spec = pair(family, P::Two(3.0, 1.0), Rotation::R0);
            assert!(
                (spec.log_pdf(u, u, 1e-12).unwrap() - joe.log_pdf(u, u, 1e-12).unwrap()).abs()
                    < 1e-8
            );
            assert!(
                (spec.cond_first_given_second(u, u, 1e-12).unwrap()
                    - joe.cond_first_given_second(u, u, 1e-12).unwrap())
                .abs()
                    < 1e-8
            );
        }
        let bb7 = pair(F::Bb7, P::Two(3.0, 1.0), Rotation::R0);
        assert!(bb7.log_pdf(u, u, 1e-12).unwrap().is_finite());
        let h = bb7.cond_first_given_second(u, u, 1e-12).unwrap();
        assert!(h > 0.1 && h < 0.9);
    }
}

#[test]
fn gaussian_factor_matches_analytic_density_even_in_tails() {
    for rho in [0.5, 0.99, -0.99] {
        let factor = rscopulas::FactorCopula::basic_1f(
            vec![pair(F::Gaussian, P::One(rho), Rotation::R0); 3],
            25,
        )
        .unwrap();
        let r = rho * rho;
        let gaussian = GaussianCopula::new(array![[1., r, r], [r, 1., r], [r, r, 1.]]).unwrap();
        let data = PseudoObs::new(array![[0.001, 0.001, 0.001], [0.5, 0.9, 0.2]]).unwrap();
        let actual = factor.log_pdf(&data, &Default::default()).unwrap();
        let expected = gaussian.log_pdf(&data, &Default::default()).unwrap();
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-10);
        }
        let policy = rscopulas::ExecPolicy::Force(rscopulas::Device::Cuda(99));
        assert!(
            factor
                .log_pdf(
                    &data,
                    &EvalOptions {
                        exec: policy,
                        ..Default::default()
                    }
                )
                .is_err()
        );
        assert!(
            factor
                .sample(
                    1,
                    &mut StdRng::seed_from_u64(1),
                    &rscopulas::SampleOptions { exec: policy }
                )
                .is_err()
        );
        let mut json = serde_json::to_value(&factor).unwrap();
        json["dim"] = 2.into();
        assert!(serde_json::from_value::<rscopulas::FactorCopula>(json).is_err());
    }
}

#[test]
fn hac_scores_and_supported_methods_are_explicit() {
    use rscopulas::{
        HacFamily, HacFitMethod as M, HacFitOptions, HacNode, HacTree as T,
        HierarchicalArchimedeanCopula as H, LikelihoodKind,
    };
    let flat = T::Node(HacNode::new(
        HacFamily::Gumbel,
        2.0,
        vec![T::Leaf(0), T::Leaf(1), T::Leaf(2)],
    ));
    let model = H::new(flat.clone()).unwrap();
    let data = PseudoObs::new(
        model
            .sample(100, &mut StdRng::seed_from_u64(31), &Default::default())
            .unwrap(),
    )
    .unwrap();
    let options = HacFitOptions {
        family_set: vec![HacFamily::Gumbel],
        ..Default::default()
    };
    let fit = H::fit_with_tree(&data, flat.clone(), &options).unwrap();
    let actual: f64 = fit
        .model
        .log_pdf(&data, &Default::default())
        .unwrap()
        .iter()
        .sum();
    assert!((actual - fit.diagnostics.loglik).abs() < 1e-10);
    assert_eq!(fit.diagnostics.likelihood_kind, LikelihoodKind::Joint);
    assert!(fit.diagnostics.n_iter < options.base.max_iter);
    assert!(!fit.model.used_smle());
    let tau_fit = H::fit_with_tree(
        &data,
        flat.clone(),
        &HacFitOptions {
            fit_method: M::TauInit,
            ..options.clone()
        },
    )
    .unwrap();
    assert_eq!(tau_fit.diagnostics.n_iter, 0);
    assert!(fit.diagnostics.n_iter > 0);
    for method in [M::FullMle, M::Smle, M::Dmle] {
        assert!(
            H::fit_with_tree(
                &data,
                flat.clone(),
                &HacFitOptions {
                    fit_method: method,
                    ..options.clone()
                }
            )
            .is_err()
        );
    }
    let nested = T::Node(HacNode::new(
        HacFamily::Gumbel,
        1.2,
        vec![
            T::Leaf(0),
            T::Node(HacNode::new(
                HacFamily::Gumbel,
                2.0,
                vec![T::Leaf(1), T::Leaf(2)],
            )),
        ],
    ));
    let model = H::new(nested).unwrap();
    assert!(model.log_pdf(&data, &Default::default()).is_err());
    assert!(
        model
            .composite_log_pdf(&data, &Default::default())
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    );
}

#[test]
fn tll_density_conditionals_margins_transpose_and_state_agree() {
    let mut rng = StdRng::seed_from_u64(91);
    let observations = bivariate(pair(F::Tawn1, P::Two(3.0, 0.3), Rotation::R0))
        .sample(600, &mut rng, &Default::default())
        .unwrap();
    let params = rscopulas::tll_fit(
        &observations.column(0).to_vec(),
        &observations.column(1).to_vec(),
        rscopulas::TllOrder::Constant,
    )
    .unwrap();
    let spec = pair(F::Tll, P::Tll(params), Rotation::R0);
    let transposed = spec.clone().swap_axes();
    for (u, v) in [(0.5, 0.5), (1e-5, 0.5), (0.2, 0.8), (1.0 - 1e-5, 0.5)] {
        let delta = 1e-7;
        let density = spec.log_pdf(u, v, 1e-12).unwrap().exp();
        let derivative = (spec.cond_first_given_second(u + delta, v, 1e-12).unwrap()
            - spec.cond_first_given_second(u - delta, v, 1e-12).unwrap())
            / (2.0 * delta);
        assert!(
            (density - derivative).abs() < 1e-4 * density.max(1.0),
            "u={u}: {density} vs {derivative}"
        );
        assert!((transposed.log_pdf(v, u, 1e-12).unwrap() - density.ln()).abs() < 1e-10);
        let h = spec.cond_first_given_second(u, v, 1e-12).unwrap();
        assert!((spec.inv_first_given_second(h, v, 1e-12).unwrap() - u).abs() < 1e-9);
    }
    // Independent midpoint integration checks both uniform margins.
    for v in [0.1, 0.5, 0.9] {
        let n = 20000;
        let mass: f64 = (0..n)
            .map(|i| {
                spec.log_pdf((i as f64 + 0.5) / n as f64, v, 1e-12)
                    .unwrap()
                    .exp()
            })
            .sum::<f64>()
            / n as f64;
        assert!((mass - 1.0).abs() < 2e-4, "margin at {v}: {mass}");
    }
    let restored: PairCopulaSpec =
        serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
    assert_eq!(
        spec.log_pdf(0.2, 0.8, 1e-12).unwrap(),
        restored.log_pdf(0.2, 0.8, 1e-12).unwrap()
    );
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_preserves_gaussian_tail_precision() {
    use rscopulas::accel::{self, GaussianPairBatchRequest};
    if !accel::is_device_available(accel::Device::Metal) {
        return;
    }
    let u = [0.5, 1.0 - 1e-8, 1.0 - 1e-12, 1e-12];
    let v = [0.5, 0.5, 1.0 - 1e-12, 0.8];
    for rho in [0.6, 0.999] {
        let result = accel::evaluate_gaussian_pair_batch(
            accel::Device::Metal,
            GaussianPairBatchRequest {
                u1: &u,
                u2: &v,
                rho,
                clip_eps: 1e-12,
            },
        )
        .unwrap();
        let spec = pair(F::Gaussian, P::One(rho), Rotation::R0);
        for i in 0..u.len() {
            let expected = spec.log_pdf(u[i], v[i], 1e-12).unwrap();
            assert!(result.log_pdf[i].is_finite());
            assert!((result.log_pdf[i] - expected).abs() < 1e-5);
            assert!(
                (result.cond_on_first[i]
                    - spec.cond_first_given_second(u[i], v[i], 1e-12).unwrap())
                .abs()
                    < 1e-5
            );
        }
    }
}
