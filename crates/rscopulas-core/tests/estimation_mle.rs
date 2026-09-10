//! Maximum-likelihood contracts for the pair-copula estimators.
//!
//! Every parametric family is simulated from known parameters through the
//! library's own inverse h-function (`u₁ ~ U(0,1)`, `u₂ = h⁻¹₂|₁(w | u₁)` with
//! `w ~ U(0,1)`), fitted with `fit_pair_copula`, and checked for
//!
//! * a fitted log-likelihood no lower than the log-likelihood at the true
//!   parameters (up to optimiser slack) — the defining property of an MLE
//!   that a grid or moment estimator does not have, and
//! * parameter recovery within a few asymptotic standard errors at n = 3000.
//!
//! The tolerances below were calibrated from the observed-information standard
//! errors printed by `report` (roughly 4× the SE, with a floor for the weakly
//! identified BB/Tawn shape parameters).

use ndarray::Array2;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rscopulas::{
    CopulaModel, FitOptions, PairCopulaFamily as F, PairCopulaParams as P, PairCopulaSpec,
    PseudoObs, Rotation, VineCopula, VineFitOptions, math, paircopula::fit_pair_copula,
};

const N: usize = 3000;
const CLIP: f64 = 1e-12;
/// Optimiser slack on the log-likelihood contract, in nats. The Nelder–Mead
/// polish stops at a relative objective tolerance of 1e-8, so on a
/// log-likelihood of a few hundred nats this is generous.
const LOGLIK_SLACK: f64 = 0.05;

fn spec(family: F, params: P) -> PairCopulaSpec {
    PairCopulaSpec {
        family,
        rotation: Rotation::R0,
        params,
    }
}

fn simulate(truth: &PairCopulaSpec, n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut u1 = Vec::with_capacity(n);
    let mut u2 = Vec::with_capacity(n);
    for _ in 0..n {
        let u = rng.random::<f64>().clamp(1e-10, 1.0 - 1e-10);
        let w = rng.random::<f64>().clamp(1e-10, 1.0 - 1e-10);
        let v = truth
            .inv_second_given_first(u, w, CLIP)
            .expect("inverse h-function should evaluate");
        u1.push(u);
        u2.push(v);
    }
    (u1, u2)
}

fn loglik(candidate: &PairCopulaSpec, u1: &[f64], u2: &[f64]) -> f64 {
    u1.iter()
        .zip(u2)
        .map(|(&u, &v)| candidate.log_pdf(u, v, CLIP).expect("log density"))
        .sum()
}

fn single_family_options(family: F) -> VineFitOptions {
    VineFitOptions {
        family_set: vec![family],
        include_rotations: false,
        ..VineFitOptions::default()
    }
}

/// Observed-information standard errors of the natural parameters at the
/// fitted point, for the diagnostic printout that calibrates the tolerances.
fn standard_errors(fitted: &PairCopulaSpec, u1: &[f64], u2: &[f64]) -> Vec<f64> {
    let values = fitted.flat_parameters();
    let hessian = math::numerical_hessian(&values, 1e-4, |x| {
        let params = match x.len() {
            1 => P::One(x[0]),
            2 => P::Two(x[0], x[1]),
            _ => unreachable!("parametric families have one or two parameters"),
        };
        let candidate = spec(fitted.family, params);
        if candidate.validate().is_err() {
            return f64::NAN;
        }
        loglik(&candidate, u1, u2)
    });
    let negated = hessian.mapv(|h| -h);
    match math::inverse(&negated) {
        Ok(covariance) => (0..values.len())
            .map(|k| covariance[(k, k)].max(0.0).sqrt())
            .collect(),
        Err(_) => vec![f64::NAN; values.len()],
    }
}

fn report(label: &str, truth: &PairCopulaSpec, fitted: &PairCopulaSpec, u1: &[f64], u2: &[f64]) {
    println!(
        "{label}: truth {:?} fitted {:?} se {:?}",
        truth.flat_parameters(),
        fitted.flat_parameters(),
        standard_errors(fitted, u1, u2)
    );
}

/// Simulates from `truth`, fits `truth.family` only, and checks the two MLE
/// contracts. `tolerances` bound `|fitted − truth|` per parameter.
fn check_recovery(truth: PairCopulaSpec, seed: u64, tolerances: &[f64]) {
    let (u1, u2) = simulate(&truth, N, seed);
    let fit = fit_pair_copula(&u1, &u2, &single_family_options(truth.family))
        .expect("pair fit should succeed");
    report(&format!("{:?}", truth.family), &truth, &fit.spec, &u1, &u2);

    assert_eq!(fit.spec.family, truth.family);
    assert_eq!(fit.spec.rotation, Rotation::R0);
    fit.spec.validate().expect("fitted spec must validate");

    let at_truth = loglik(&truth, &u1, &u2);
    let refit = loglik(&fit.spec, &u1, &u2);
    assert!(
        (refit - fit.loglik).abs() < 1e-6,
        "reported loglik {} must match the returned spec {}",
        fit.loglik,
        refit
    );
    assert!(
        fit.loglik >= at_truth - LOGLIK_SLACK,
        "{:?}: fitted loglik {} is below the truth's {}",
        truth.family,
        fit.loglik,
        at_truth
    );

    let expected = truth.flat_parameters();
    let actual = fit.spec.flat_parameters();
    assert_eq!(actual.len(), tolerances.len());
    for ((got, want), tol) in actual.iter().zip(&expected).zip(tolerances) {
        assert!(
            (got - want).abs() < *tol,
            "{:?}: fitted {got} vs truth {want} exceeds tolerance {tol}",
            truth.family
        );
    }
}

#[test]
fn gaussian_recovers_rho_by_maximum_likelihood() {
    // Observed SE(ρ̂) ≈ 0.010 at ρ = 0.6 and ≈ 0.015 at ρ = −0.35.
    check_recovery(spec(F::Gaussian, P::One(0.6)), 11, &[0.04]);
    check_recovery(spec(F::Gaussian, P::One(-0.35)), 12, &[0.06]);
}

#[test]
fn student_t_recovers_rho_and_continuous_nu() {
    // Observed SEs: (0.015, 0.53) at (0.5, 5.0) and (0.010, 0.35) at
    // (−0.7, 3.2). ν is no longer restricted to a grid, so the fitted value
    // can land between the old grid points.
    check_recovery(spec(F::StudentT, P::Two(0.5, 5.0)), 21, &[0.06, 2.1]);
    check_recovery(spec(F::StudentT, P::Two(-0.7, 3.2)), 22, &[0.04, 1.4]);
}

#[test]
fn one_parameter_archimedean_families_recover_theta() {
    // Observed SEs: Clayton 0.049, Gumbel 0.037, Joe 0.045, Frank 0.13.
    check_recovery(spec(F::Clayton, P::One(2.0)), 31, &[0.25]);
    check_recovery(spec(F::Gumbel, P::One(2.5)), 32, &[0.2]);
    check_recovery(spec(F::Joe, P::One(2.5)), 33, &[0.3]);
    check_recovery(spec(F::Frank, P::One(5.0)), 34, &[0.6]);
}

#[test]
fn frank_recovers_negative_theta() {
    // Observed SEs: 0.12 at θ = −4 and 0.11 at θ = −0.8.
    check_recovery(spec(F::Frank, P::One(-4.0)), 41, &[0.6]);
    check_recovery(spec(F::Frank, P::One(-0.8)), 42, &[0.5]);
}

#[test]
fn frank_is_selected_for_negatively_dependent_data_in_the_default_set() {
    let truth = spec(F::Frank, P::One(-4.0));
    let (u1, u2) = simulate(&truth, N, 43);
    let fit = fit_pair_copula(&u1, &u2, &VineFitOptions::default())
        .expect("pair fit should succeed with the default family set");
    assert_eq!(
        fit.spec.family,
        F::Frank,
        "default selection picked {:?}",
        fit.spec
    );
    assert_eq!(fit.spec.rotation, Rotation::R0);
    match fit.spec.params {
        P::One(theta) => assert!(theta < -3.0 && theta > -5.0, "theta = {theta}"),
        ref other => panic!("unexpected params {other:?}"),
    }
}

#[test]
fn two_parameter_families_recover_both_parameters() {
    // Observed SEs: BB1 (0.054, 0.037), BB6 (0.14, 0.083), BB7 (0.045,
    // 0.052), BB8 (0.46, 0.078), Tawn1 (0.056, 0.024), Tawn2 (0.059, 0.021).
    // BB8's θ and BB6's θ are weakly identified, hence the wide bands.
    check_recovery(spec(F::Bb1, P::Two(0.8, 1.6)), 51, &[0.3, 0.25]);
    check_recovery(spec(F::Bb6, P::Two(1.8, 1.5)), 52, &[0.6, 0.35]);
    check_recovery(spec(F::Bb7, P::Two(1.8, 1.2)), 53, &[0.3, 0.3]);
    check_recovery(spec(F::Bb8, P::Two(3.0, 0.7)), 54, &[1.9, 0.32]);
    check_recovery(spec(F::Tawn1, P::Two(2.0, 0.6)), 55, &[0.4, 0.15]);
    check_recovery(spec(F::Tawn2, P::Two(2.0, 0.6)), 56, &[0.4, 0.15]);
}

#[test]
fn gaussian_mle_beats_tau_inversion_where_they_differ() {
    // Clayton data is asymmetric, so the Gaussian τ-inversion and the
    // Gaussian MLE disagree materially; the MLE must win in likelihood.
    let truth = spec(F::Clayton, P::One(2.0));
    let (u1, u2) = simulate(&truth, N, 61);
    let fit = fit_pair_copula(&u1, &u2, &single_family_options(F::Gaussian))
        .expect("gaussian pair fit should succeed");
    let tau = rscopulas::stats::kendall_tau_bivariate(&u1, &u2).expect("tau");
    let rho_tau = (std::f64::consts::FRAC_PI_2 * tau).sin();
    let tau_inversion = spec(F::Gaussian, P::One(rho_tau));
    let tau_loglik = loglik(&tau_inversion, &u1, &u2);
    let P::One(rho_mle) = fit.spec.params else {
        panic!("gaussian fit must have one parameter");
    };
    println!(
        "gaussian mle {rho_mle} vs tau inversion {rho_tau}: {} vs {tau_loglik}",
        fit.loglik
    );
    assert!(
        (rho_mle - rho_tau).abs() > 1e-3,
        "the two estimators should differ on Clayton data"
    );
    assert!(
        fit.loglik > tau_loglik + 0.5,
        "MLE loglik {} should beat tau inversion {tau_loglik}",
        fit.loglik
    );
}

#[test]
fn gaussian_mle_never_drops_below_tau_inversion() {
    for (seed, rho) in [(71_u64, 0.95), (72, 0.1), (73, -0.6), (74, 0.999)] {
        let truth = spec(F::Gaussian, P::One(rho));
        let (u1, u2) = simulate(&truth, 800, seed);
        let fit = fit_pair_copula(&u1, &u2, &single_family_options(F::Gaussian))
            .expect("gaussian pair fit should succeed");
        let tau = rscopulas::stats::kendall_tau_bivariate(&u1, &u2).expect("tau");
        let rho_tau = (std::f64::consts::FRAC_PI_2 * tau)
            .sin()
            .clamp(-0.999_999_999, 0.999_999_999);
        let tau_loglik = loglik(&spec(F::Gaussian, P::One(rho_tau)), &u1, &u2);
        assert!(
            fit.loglik >= tau_loglik - 1e-9,
            "rho={rho}: MLE loglik {} below tau inversion {tau_loglik}",
            fit.loglik
        );
    }
}

#[test]
fn student_t_polish_improves_on_the_grid_warm_start() {
    // ν = 7.5 sits between the warm-start grid points; a joint polish must
    // reach a strictly higher likelihood than the best grid node can.
    let truth = spec(F::StudentT, P::Two(0.4, 7.5));
    let (u1, u2) = simulate(&truth, N, 81);
    let fit = fit_pair_copula(&u1, &u2, &single_family_options(F::StudentT))
        .expect("student t pair fit should succeed");
    let P::Two(rho, nu) = fit.spec.params else {
        panic!("student t fit must have two parameters");
    };
    println!("student t polish: rho {rho} nu {nu} loglik {}", fit.loglik);
    assert!(nu > 2.0 && nu < 200.0);
    // Best likelihood over the 12-point warm-start grid (ρ profiled by MLE
    // at each node) never beats the joint optimum.
    let grid_best = (0..12)
        .map(|idx| {
            let fraction = idx as f64 / 11.0;
            (2.05_f64.ln() + (200.0_f64.ln() - 2.05_f64.ln()) * fraction).exp()
        })
        .map(|grid_nu| {
            math::maximize_scalar_brent(-0.99, 0.99, Some(rho), 1e-8, 200, |r| {
                loglik(&spec(F::StudentT, P::Two(r, grid_nu)), &u1, &u2)
            })
            .value
        })
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        fit.loglik >= grid_best - 1e-6,
        "joint polish {} must not lose to the grid {grid_best}",
        fit.loglik
    );
}

#[test]
fn max_iter_is_honoured_without_breaking_the_fit() {
    let truth = spec(F::Bb1, P::Two(0.8, 1.6));
    let (u1, u2) = simulate(&truth, 500, 91);
    let mut options = single_family_options(F::Bb1);
    let full = fit_pair_copula(&u1, &u2, &options).expect("default max_iter fit");
    options.base = FitOptions {
        max_iter: 2,
        ..FitOptions::default()
    };
    let capped = fit_pair_copula(&u1, &u2, &options).expect("capped fit should still succeed");
    capped.spec.validate().expect("capped fit must validate");
    assert!(capped.loglik.is_finite());
    assert!(
        full.loglik >= capped.loglik - 1e-9,
        "more iterations must not lose likelihood: {} vs {}",
        full.loglik,
        capped.loglik
    );
}

#[test]
fn independence_test_level_controls_edge_selection() {
    let mut rng = StdRng::seed_from_u64(101);
    let n = 500;
    let u1: Vec<f64> = (0..n)
        .map(|_| rng.random::<f64>().clamp(1e-10, 1.0 - 1e-10))
        .collect();
    let u2: Vec<f64> = (0..n)
        .map(|_| rng.random::<f64>().clamp(1e-10, 1.0 - 1e-10))
        .collect();
    let tau = rscopulas::stats::kendall_tau_bivariate(&u1, &u2).expect("tau");
    assert!(
        !rscopulas::stats::kendall_tau_rejects_independence(tau, n, 0.05),
        "independent uniforms should not reject at 5% (tau = {tau})"
    );

    // Without the test and without Independence in the family set the
    // fitter must still pick a parametric family.
    let parametric = VineFitOptions {
        family_set: vec![F::Gaussian],
        include_rotations: false,
        ..VineFitOptions::default()
    };
    let untested = fit_pair_copula(&u1, &u2, &parametric).expect("fit");
    assert_eq!(untested.spec.family, F::Gaussian);

    let tested = VineFitOptions {
        independence_test_level: Some(0.05),
        ..parametric.clone()
    };
    let fit = fit_pair_copula(&u1, &u2, &tested).expect("fit");
    assert_eq!(fit.spec.family, F::Independence);
    assert_eq!(fit.loglik, 0.0);

    // Clearly dependent data rejects independence and proceeds to selection.
    let (d1, d2) = simulate(&spec(F::Gaussian, P::One(0.5)), n, 102);
    let fit = fit_pair_copula(&d1, &d2, &tested).expect("fit");
    assert_eq!(fit.spec.family, F::Gaussian);

    // Invalid levels are rejected up front.
    for level in [0.0, 1.0, -0.1, f64::NAN] {
        let invalid = VineFitOptions {
            independence_test_level: Some(level),
            ..parametric.clone()
        };
        assert!(fit_pair_copula(&u1, &u2, &invalid).is_err());
    }
}

#[test]
fn independence_test_sparsifies_a_vine_and_validates_options() {
    let mut rng = StdRng::seed_from_u64(111);
    let n = 400;
    let mut values = Array2::<f64>::zeros((n, 4));
    for row in 0..n {
        for col in 0..4 {
            values[(row, col)] = rng.random::<f64>().clamp(1e-6, 1.0 - 1e-6);
        }
    }
    let data = PseudoObs::new(values).expect("pseudo-observations");
    let options = VineFitOptions {
        family_set: vec![F::Gaussian, F::Clayton],
        independence_test_level: Some(0.05),
        ..VineFitOptions::default()
    };
    let fit = VineCopula::fit_r_vine(&data, &options).expect("vine fit");
    let independent_edges = fit
        .model
        .trees()
        .iter()
        .flat_map(|tree| tree.edges.iter())
        .filter(|edge| edge.copula.family == F::Independence)
        .count();
    assert!(
        independent_edges >= 4,
        "expected most of the 6 edges to be independence, got {independent_edges}"
    );

    let invalid = VineFitOptions {
        independence_test_level: Some(1.5),
        ..VineFitOptions::default()
    };
    assert!(invalid.validate(4).is_err());
    assert!(VineCopula::fit_r_vine(&data, &invalid).is_err());
}

#[test]
fn default_family_set_excludes_khoudraji() {
    let options = VineFitOptions::default();
    assert!(!options.family_set.contains(&F::Khoudraji));
    assert!(options.family_set.contains(&F::Gaussian));
    assert!(options.family_set.contains(&F::StudentT));
    assert_eq!(options.independence_test_level, None);
    assert_eq!(options.independence_threshold, None);
}

#[test]
fn negative_frank_spec_evaluates_consistently_through_the_public_api() {
    let negative = spec(F::Frank, P::One(-3.0));
    negative
        .validate()
        .expect("negative theta is a valid Frank copula");
    assert!(spec(F::Frank, P::One(0.0)).validate().is_err());
    let positive = spec(F::Frank, P::One(3.0));
    for (u, v) in [(0.1, 0.2), (0.3, 0.9), (0.75, 0.4), (0.5, 0.5)] {
        let a = negative.log_pdf(u, v, CLIP).unwrap();
        let b = positive.log_pdf(1.0 - u, v, CLIP).unwrap();
        assert!((a - b).abs() < 1e-12, "density reflection at ({u}, {v})");
        let h = negative.cond_second_given_first(u, v, CLIP).unwrap();
        let back = negative.inv_second_given_first(u, h, CLIP).unwrap();
        assert!(
            (back - v).abs() < 1e-8,
            "h-inverse round trip at ({u}, {v})"
        );
        let g = negative.cond_first_given_second(u, v, CLIP).unwrap();
        let back = negative.inv_first_given_second(g, v, CLIP).unwrap();
        assert!(
            (back - u).abs() < 1e-8,
            "h-inverse round trip at ({u}, {v})"
        );
    }
    // Negative dependence: the joint CDF sits below independence.
    let vine = VineCopula::from_trees(
        rscopulas::VineStructureKind::R,
        vec![rscopulas::VineTree {
            level: 1,
            edges: vec![rscopulas::VineEdge {
                tree: 1,
                conditioned: (0, 1),
                conditioning: vec![],
                copula: negative.clone(),
            }],
        }],
        None,
    )
    .expect("bivariate vine");
    let mut rng = StdRng::seed_from_u64(121);
    let sample = vine
        .sample(4000, &mut rng, &Default::default())
        .expect("sampling");
    let tau = rscopulas::stats::kendall_tau_matrix(&PseudoObs::new(sample).unwrap());
    assert!(
        tau[(0, 1)] < -0.2,
        "sampled tau {} should be negative",
        tau[(0, 1)]
    );
}
