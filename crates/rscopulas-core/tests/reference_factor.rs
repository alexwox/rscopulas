//! Parameter-recovery and self-consistency tests for the Krupskii–Joe factor
//! copula (Basic1F). Follows the same "synthetic sample → refit → assert"
//! convention as `reference_hac.rs`.
//!
//! A full external reference-fixture suite against Joe's `CopulaModel` R
//! package is future work; the bundle of tests here pins down:
//!   1. log-density numerics (finite, dimension-correct, serde round-trip),
//!   2. sampling (valid pseudo-observations, non-degenerate dependence),
//!   3. parameter recovery via `FactorCopula::fit` on data drawn from a known
//!      model with a rich link mix.

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};

use rscopulas::{
    CopulaFamily, CopulaModel, FactorCopula, FactorFitOptions, FactorLayout, PairCopulaFamily,
    PairCopulaParams, PairCopulaSpec, PseudoObs, Rotation,
};

fn frank_link(theta: f64) -> PairCopulaSpec {
    PairCopulaSpec {
        family: PairCopulaFamily::Frank,
        rotation: Rotation::R0,
        params: PairCopulaParams::One(theta),
    }
}

fn gaussian_link(rho: f64) -> PairCopulaSpec {
    PairCopulaSpec {
        family: PairCopulaFamily::Gaussian,
        rotation: Rotation::R0,
        params: PairCopulaParams::One(rho),
    }
}

fn clayton_link(theta: f64) -> PairCopulaSpec {
    PairCopulaSpec {
        family: PairCopulaFamily::Clayton,
        rotation: Rotation::R0,
        params: PairCopulaParams::One(theta),
    }
}

fn gumbel_link(theta: f64) -> PairCopulaSpec {
    PairCopulaSpec {
        family: PairCopulaFamily::Gumbel,
        rotation: Rotation::R0,
        params: PairCopulaParams::One(theta),
    }
}

/// A five-variable Basic1F model that mixes Gaussian, Clayton, and Gumbel
/// links — exercising asymmetric tail dependence alongside the symmetric
/// Gaussian baseline.
fn reference_model() -> FactorCopula {
    FactorCopula::basic_1f(
        vec![
            gaussian_link(0.7),
            gaussian_link(0.6),
            clayton_link(2.0),
            gumbel_link(1.8),
            gumbel_link(2.2),
        ],
        25,
    )
    .expect("reference factor copula should be valid")
}

#[test]
fn factor_copula_reports_factor_family_and_dim() {
    let model = reference_model();
    assert_eq!(model.family(), CopulaFamily::Factor);
    assert_eq!(model.dim(), 5);
    assert_eq!(model.num_factors(), 1);
    assert_eq!(model.layout(), FactorLayout::Basic1F);
    assert_eq!(model.links().len(), 5);
    assert_eq!(model.quadrature_nodes(), 25);
}

#[test]
fn factor_copula_rejects_too_few_variables() {
    let err = FactorCopula::basic_1f(vec![gaussian_link(0.5)], 25).expect_err("d=1 must fail");
    assert!(err.to_string().contains("at least two"));
}

#[test]
fn factor_copula_rejects_too_few_quadrature_nodes() {
    let err =
        FactorCopula::basic_1f(vec![gaussian_link(0.5), gaussian_link(0.5)], 1).expect_err(
            "too few quadrature nodes must fail",
        );
    assert!(err.to_string().contains("three nodes"));
}

#[test]
fn factor_copula_log_pdf_is_finite_and_consistent() {
    let model = reference_model();
    let mut rng = StdRng::seed_from_u64(17);
    let sample = model
        .sample(128, &mut rng, &Default::default())
        .expect("sampling should succeed");
    let data = PseudoObs::new(sample).expect("sample should yield valid pseudo-observations");

    let log_pdf = model
        .log_pdf(&data, &Default::default())
        .expect("log_pdf should evaluate");
    assert_eq!(log_pdf.len(), 128);
    for value in &log_pdf {
        assert!(value.is_finite(), "log_pdf entry {value} was non-finite");
    }

    // Evaluating again must produce bit-identical output (no rng in log_pdf).
    let log_pdf_again = model
        .log_pdf(&data, &Default::default())
        .expect("log_pdf should evaluate again");
    for (a, b) in log_pdf.iter().zip(log_pdf_again.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "log_pdf should be deterministic");
    }
}

#[test]
fn factor_copula_sample_yields_valid_pseudo_observations() {
    let model = reference_model();
    let mut rng = StdRng::seed_from_u64(23);
    let sample = model
        .sample(512, &mut rng, &Default::default())
        .expect("sampling should succeed");
    assert_eq!(sample.shape(), &[512, 5]);
    for value in sample.iter() {
        assert!(
            *value > 0.0 && *value < 1.0,
            "sampled value {value} escaped (0, 1)"
        );
        assert!(value.is_finite());
    }

    // Strong positive dependence on the first two Gaussian-linked columns:
    // τ should be comfortably positive (they both have positive-ρ Gaussian
    // links, so the induced Kendall tau across the pair is > 0 by a large
    // margin).
    let data = PseudoObs::new(sample).expect("sample should yield valid pseudo-observations");
    let tau = rscopulas::stats::kendall_tau_matrix(&data);
    assert!(
        tau[(0, 1)] > 0.25,
        "expected strong positive dependence between Gaussian-linked columns, got tau={}",
        tau[(0, 1)]
    );
    // Clayton and Gumbel columns are also positively dependent through the
    // shared factor.
    assert!(tau[(2, 3)] > 0.15, "got tau[2,3]={}", tau[(2, 3)]);
    assert!(tau[(3, 4)] > 0.3, "got tau[3,4]={}", tau[(3, 4)]);
}

#[test]
fn factor_copula_serde_round_trips() {
    let model = reference_model();

    let json = serde_json::to_string(&model).expect("serialization should succeed");
    let restored: FactorCopula =
        serde_json::from_str(&json).expect("deserialization should succeed");

    assert_eq!(restored.dim(), model.dim());
    assert_eq!(restored.num_factors(), model.num_factors());
    assert_eq!(restored.quadrature_nodes(), model.quadrature_nodes());

    // Numerically identical log-density after round-trip — the whole point of
    // serde is bit-for-bit restoration of the fitted model.
    let mut rng = StdRng::seed_from_u64(29);
    let sample = model
        .sample(64, &mut rng, &Default::default())
        .expect("sampling should succeed");
    let data = PseudoObs::new(sample).expect("sample should yield valid pseudo-observations");
    let a = model.log_pdf(&data, &Default::default()).unwrap();
    let b = restored.log_pdf(&data, &Default::default()).unwrap();
    for (left, right) in a.iter().zip(b.iter()) {
        assert_eq!(left.to_bits(), right.to_bits());
    }
}

#[test]
fn factor_copula_fit_recovers_gaussian_correlations() {
    // Mono-family Gaussian model: sampling + fit should recover link rhos to
    // within the usual Kendall-tau noise at n=2000.
    let truth = FactorCopula::basic_1f(
        vec![
            gaussian_link(0.75),
            gaussian_link(0.65),
            gaussian_link(0.55),
            gaussian_link(0.45),
        ],
        25,
    )
    .expect("reference model should be valid");

    let mut rng = StdRng::seed_from_u64(42);
    let sample = truth
        .sample(2000, &mut rng, &Default::default())
        .expect("sampling should succeed");
    let data = PseudoObs::new(sample).expect("sample should yield valid pseudo-observations");

    let options = FactorFitOptions {
        family_set: vec![PairCopulaFamily::Gaussian],
        include_rotations: false,
        ..FactorFitOptions::default()
    };
    let fit = FactorCopula::fit(&data, &options).expect("fit should succeed");

    assert_eq!(fit.model.dim(), 4);
    assert_eq!(fit.model.links().len(), 4);

    // Every link should come out Gaussian (the only family in the set).
    for link in fit.model.links() {
        assert_eq!(link.family, PairCopulaFamily::Gaussian);
    }

    // With the joint-MLE polish the sequential warm-start attenuation bias is
    // removed; recovered |ρ| should sit within standard-error-sized distance
    // of the truth. Tolerance 0.08 is comfortably above the n⁻¹ᐟ² noise at
    // n=2000 but tight enough to catch the old attenuation-biased fitter.
    let rhos: Vec<f64> = fit
        .model
        .links()
        .iter()
        .map(|link| match link.params {
            PairCopulaParams::One(rho) => rho,
            _ => panic!("expected single-parameter Gaussian link"),
        })
        .collect();
    let truth_rhos = [0.75, 0.65, 0.55, 0.45];
    for (got, expected) in rhos.iter().zip(truth_rhos.iter()) {
        assert!(
            *got > 0.0 && *got < 1.0,
            "recovered rho {got} outside (0, 1)"
        );
        assert!(
            (got - expected).abs() < 0.08,
            "rho recovery: got {got}, expected near {expected}"
        );
    }

    // Diagnostics should be finite, and the polish should report one
    // standard error per Gaussian ρ.
    assert!(fit.diagnostics.loglik.is_finite());
    assert!(fit.diagnostics.aic.is_finite());
    assert!(fit.diagnostics.bic.is_finite());
    assert!(fit.diagnostics.converged);
    assert_eq!(fit.std_errors.len(), truth_rhos.len());
    for se in &fit.std_errors {
        assert!(se.is_finite() && *se > 0.0, "std error {se} invalid");
    }
}

#[test]
fn factor_copula_fit_recovers_clayton_parameters() {
    // Clayton DGP with a Clayton-only family set: the polish stage can focus
    // purely on parameter recovery, which is where it decisively beats the
    // sequential warm-start alone. (A mixed family set with Gumbel-R180
    // available is a different test — the joint polish as implemented here
    // holds the family selection from the EM stage fixed, and R0 Clayton vs
    // R180 Gumbel is an AIC toss-up for lower-tail-dependent data. The
    // family-selection dynamic is covered indirectly by the τ-preservation
    // tail of this test.)
    let truth = FactorCopula::basic_1f(
        vec![
            clayton_link(2.5),
            clayton_link(2.5),
            clayton_link(3.0),
            clayton_link(3.0),
        ],
        25,
    )
    .expect("reference model should be valid");

    let mut rng = StdRng::seed_from_u64(101);
    let sample = truth
        .sample(2000, &mut rng, &Default::default())
        .expect("sampling should succeed");
    let data = PseudoObs::new(sample).expect("sample should yield valid pseudo-observations");
    let truth_tau = rscopulas::stats::kendall_tau_matrix(&data);

    let options = FactorFitOptions {
        family_set: vec![PairCopulaFamily::Clayton],
        include_rotations: false,
        ..FactorFitOptions::default()
    };
    let fit = FactorCopula::fit(&data, &options).expect("fit should succeed");

    for (idx, link) in fit.model.links().iter().enumerate() {
        assert_eq!(
            link.family,
            PairCopulaFamily::Clayton,
            "link {idx} selected {:?}, expected Clayton",
            link.family
        );
        assert_eq!(link.rotation, Rotation::R0);
    }

    // Parameter recovery: θ should be within 0.3 of truth at n=2000 — that's
    // roughly 2× the asymptotic SE at θ ∈ {2.5, 3.0}, comfortably above the
    // noise but tight enough to catch the old attenuation-biased fitter.
    let truth_thetas = [2.5, 2.5, 3.0, 3.0];
    for (link, expected) in fit.model.links().iter().zip(truth_thetas.iter()) {
        let theta = match link.params {
            PairCopulaParams::One(value) => value,
            _ => panic!("expected single-parameter Clayton link"),
        };
        assert!(
            (theta - expected).abs() < 0.3,
            "theta recovery: got {theta}, expected near {expected}"
        );
    }

    assert!(
        fit.diagnostics.loglik > 300.0,
        "expected strong positive log-likelihood, got {}",
        fit.diagnostics.loglik
    );
    assert_eq!(fit.std_errors.len(), truth_thetas.len());
    for se in &fit.std_errors {
        assert!(se.is_finite() && *se > 0.0, "std error {se} invalid");
    }

    // Sanity: resampling from the refitted model preserves pairwise τ.
    let mut rng2 = StdRng::seed_from_u64(202);
    let refit_sample = fit
        .model
        .sample(2000, &mut rng2, &Default::default())
        .expect("refitted model should sample");
    let refit_data = PseudoObs::new(refit_sample).expect("refit sample should be valid");
    let refit_tau = rscopulas::stats::kendall_tau_matrix(&refit_data);
    for i in 0..4 {
        for j in (i + 1)..4 {
            let gap = (refit_tau[(i, j)] - truth_tau[(i, j)]).abs();
            assert!(
                gap < 0.08,
                "pairwise tau diverged at ({i},{j}): truth={}, refit={}, gap={gap}",
                truth_tau[(i, j)],
                refit_tau[(i, j)]
            );
        }
    }
}

#[test]
fn factor_copula_fit_mixed_family_preserves_tail_behaviour() {
    // Mixed-family DGP with a rich candidate set. We don't assert exact
    // family recovery per link (Clayton R0 and Gumbel R180 are legitimate
    // AIC-equivalent choices for lower-tail dependence); instead we require
    // that (i) no link collapses to Independence, (ii) pairwise Kendall τ
    // is preserved under a refit-resample round trip within 0.1.
    let truth = FactorCopula::basic_1f(
        vec![
            clayton_link(2.5),
            clayton_link(2.5),
            clayton_link(3.0),
            clayton_link(3.0),
        ],
        25,
    )
    .expect("reference model should be valid");

    let mut rng = StdRng::seed_from_u64(101);
    let sample = truth
        .sample(2000, &mut rng, &Default::default())
        .expect("sampling should succeed");
    let data = PseudoObs::new(sample).expect("sample should yield valid pseudo-observations");
    let truth_tau = rscopulas::stats::kendall_tau_matrix(&data);

    let options = FactorFitOptions {
        family_set: vec![
            PairCopulaFamily::Gaussian,
            PairCopulaFamily::Clayton,
            PairCopulaFamily::Frank,
            PairCopulaFamily::Gumbel,
        ],
        include_rotations: true,
        ..FactorFitOptions::default()
    };
    let fit = FactorCopula::fit(&data, &options).expect("fit should succeed");

    for link in fit.model.links() {
        assert_ne!(link.family, PairCopulaFamily::Independence);
    }
    assert!(
        fit.diagnostics.loglik > 300.0,
        "expected strong positive log-likelihood, got {}",
        fit.diagnostics.loglik
    );

    let mut rng2 = StdRng::seed_from_u64(202);
    let refit_sample = fit
        .model
        .sample(2000, &mut rng2, &Default::default())
        .expect("refitted model should sample");
    let refit_data = PseudoObs::new(refit_sample).expect("refit sample should be valid");
    let refit_tau = rscopulas::stats::kendall_tau_matrix(&refit_data);
    for i in 0..4 {
        for j in (i + 1)..4 {
            let gap = (refit_tau[(i, j)] - truth_tau[(i, j)]).abs();
            assert!(
                gap < 0.1,
                "pairwise tau diverged at ({i},{j}): truth={}, refit={}, gap={gap}",
                truth_tau[(i, j)],
                refit_tau[(i, j)]
            );
        }
    }
}

#[test]
fn factor_copula_fit_recovers_frank_parameters() {
    // Frank is symmetric (no tail dependence), so the sequential warm start
    // is already close — but the polish should still not degrade the fit,
    // and parameter recovery should be within Fisher-info-sized SEs.
    let truth = FactorCopula::basic_1f(
        vec![frank_link(4.0), frank_link(6.0), frank_link(8.0)],
        25,
    )
    .expect("reference model should be valid");

    let mut rng = StdRng::seed_from_u64(303);
    let sample = truth
        .sample(2000, &mut rng, &Default::default())
        .expect("sampling should succeed");
    let data = PseudoObs::new(sample).expect("sample should yield valid pseudo-observations");

    let options = FactorFitOptions {
        family_set: vec![PairCopulaFamily::Frank],
        include_rotations: false,
        ..FactorFitOptions::default()
    };
    let fit = FactorCopula::fit(&data, &options).expect("fit should succeed");

    for link in fit.model.links() {
        assert_eq!(link.family, PairCopulaFamily::Frank);
    }
    let truth_thetas = [4.0, 6.0, 8.0];
    for (link, expected) in fit.model.links().iter().zip(truth_thetas.iter()) {
        let theta = match link.params {
            PairCopulaParams::One(value) => value,
            _ => panic!("expected single-parameter Frank link"),
        };
        assert!(
            (theta - expected).abs() < 1.0,
            "theta recovery: got {theta}, expected near {expected}"
        );
    }
    assert_eq!(fit.std_errors.len(), 3);
    for se in &fit.std_errors {
        assert!(se.is_finite() && *se > 0.0, "std error {se} invalid");
    }
}

#[test]
fn factor_copula_polish_never_degrades_loglik_vs_refinement_only() {
    // Same DGP and fit options, once with the polish enabled and once with
    // it disabled. The bail-out guard inside `fit_basic_1f` ensures the
    // polished fit's loglik is never worse than the refinement-only fit
    // (modulo floating-point noise), matching Joe's `CopulaModel` invariant.
    let truth = reference_model();
    let mut rng = StdRng::seed_from_u64(404);
    let sample = truth
        .sample(1500, &mut rng, &Default::default())
        .expect("sampling should succeed");
    let data = PseudoObs::new(sample).expect("sample should yield valid pseudo-observations");

    let refinement_only = FactorFitOptions {
        joint_polish_cycles: 0,
        ..FactorFitOptions::default()
    };
    let polished = FactorFitOptions::default();

    let fit_refinement = FactorCopula::fit(&data, &refinement_only).expect("fit without polish");
    let fit_polished = FactorCopula::fit(&data, &polished).expect("fit with polish");

    // Allow a small negative slack for floating-point noise; in practice the
    // polish either improves or is rejected by the bail-out guard.
    assert!(
        fit_polished.diagnostics.loglik + 1e-6 >= fit_refinement.diagnostics.loglik,
        "polished loglik {} < refinement-only loglik {}",
        fit_polished.diagnostics.loglik,
        fit_refinement.diagnostics.loglik,
    );

    // The polish should produce SEs; the no-polish path still computes the
    // Hessian at the refined fit, so it should also produce a vector of
    // matching length (may contain NaN entries for off-manifold MLEs).
    assert_eq!(fit_polished.std_errors.len(), fit_refinement.std_errors.len());
}

#[test]
fn factor_copula_standard_errors_scale_with_sqrt_n() {
    // Asymptotic theory: SE(θ̂) ∝ 1/√n. We fit the same mono-family Gaussian
    // DGP at n=500 and n=2000 and check that SE_2000 · √2000 agrees with
    // SE_500 · √500 within ±40% — a generous ±σ band around the expected
    // Fisher-information-sized scaling, since Gaussian-ρ SEs are small and
    // Monte-Carlo noise on one draw per size is substantial.
    let truth = FactorCopula::basic_1f(
        vec![gaussian_link(0.7), gaussian_link(0.6), gaussian_link(0.5)],
        25,
    )
    .expect("reference model should be valid");

    let options = FactorFitOptions {
        family_set: vec![PairCopulaFamily::Gaussian],
        include_rotations: false,
        ..FactorFitOptions::default()
    };

    let mut rng_small = StdRng::seed_from_u64(505);
    let sample_small = truth
        .sample(500, &mut rng_small, &Default::default())
        .expect("sampling should succeed");
    let data_small = PseudoObs::new(sample_small).expect("small sample should be valid");
    let fit_small = FactorCopula::fit(&data_small, &options).expect("small fit should succeed");

    let mut rng_big = StdRng::seed_from_u64(606);
    let sample_big = truth
        .sample(2000, &mut rng_big, &Default::default())
        .expect("sampling should succeed");
    let data_big = PseudoObs::new(sample_big).expect("big sample should be valid");
    let fit_big = FactorCopula::fit(&data_big, &options).expect("big fit should succeed");

    assert_eq!(fit_small.std_errors.len(), 3);
    assert_eq!(fit_big.std_errors.len(), 3);

    for k in 0..3 {
        let scaled_small = fit_small.std_errors[k] * (500.0_f64).sqrt();
        let scaled_big = fit_big.std_errors[k] * (2000.0_f64).sqrt();
        let ratio = scaled_big / scaled_small;
        assert!(
            (0.6..1.6).contains(&ratio),
            "SE scaling off at link {k}: scaled_small={scaled_small}, scaled_big={scaled_big}, ratio={ratio}"
        );
    }
}

#[test]
fn factor_copula_fit_vs_independence_baseline() {
    // When the data has strong factor structure, the fitted factor model
    // should comfortably beat the independence log-likelihood (0 per obs).
    let truth = reference_model();
    let mut rng = StdRng::seed_from_u64(7);
    let sample = truth
        .sample(1500, &mut rng, &Default::default())
        .expect("sampling should succeed");
    let data = PseudoObs::new(sample).expect("sample should yield valid pseudo-observations");

    let options = FactorFitOptions::default();
    let fit = FactorCopula::fit(&data, &options).expect("fit should succeed");
    assert!(
        fit.diagnostics.loglik > 50.0,
        "expected strong log-likelihood gain over independence, got {}",
        fit.diagnostics.loglik
    );
}

#[test]
fn factor_copula_log_pdf_rejects_wrong_dimension() {
    let model = reference_model(); // dim = 5
    let wrong = PseudoObs::new(Array2::from_shape_vec((3, 2), vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap())
        .expect("2D sample should be valid");
    let err = model
        .log_pdf(&wrong, &Default::default())
        .expect_err("dimension mismatch should error");
    assert!(err.to_string().contains("dimension"));
}
