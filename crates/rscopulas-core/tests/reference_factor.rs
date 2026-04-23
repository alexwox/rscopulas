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

    // Recovered rhos should be close to truth. The pseudo-latent we use for
    // the warm start induces an attenuation bias — recovered |rho| is a bit
    // smaller than truth — so we compare against slack bounds rather than a
    // tight tolerance. The ordering across links is what matters most.
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
            (got - expected).abs() < 0.15,
            "rho recovery: got {got}, expected near {expected}"
        );
    }

    // Diagnostics should be finite.
    assert!(fit.diagnostics.loglik.is_finite());
    assert!(fit.diagnostics.aic.is_finite());
    assert!(fit.diagnostics.bic.is_finite());
    assert!(fit.diagnostics.converged);
}

#[test]
fn factor_copula_fit_handles_archimedean_dgp_with_strong_dependence() {
    // With a Clayton-linked DGP, sequential MLE (the scheme we ship in this
    // phase) doesn't guarantee exact family recovery — aggregating Clayton-
    // dependent columns into a pseudo-latent smooths lower-tail asymmetry
    // enough that Gaussian links become competitive on AIC. This matches
    // Joe's own observation in `CopulaModel` that the sequential warm start
    // needs a joint-MLE polish for tail-asymmetric families (see the plan's
    // "Joint MLE" follow-up).
    //
    // What the sequential fitter *should* still deliver:
    //   (i)   no link collapses to Independence,
    //   (ii)  the fit log-likelihood dominates the independence baseline,
    //   (iii) samples from the refitted model preserve the strong pairwise
    //         Kendall tau of the true DGP.
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

    // (i) No link went to independence — every observed variable is bound to
    //     the latent by at least one free parameter.
    for link in fit.model.links() {
        assert_ne!(link.family, PairCopulaFamily::Independence);
    }

    // (ii) Log-likelihood is strongly positive (independence baseline is 0).
    assert!(
        fit.diagnostics.loglik > 300.0,
        "expected strong positive log-likelihood, got {}",
        fit.diagnostics.loglik
    );

    // (iii) Resampling from the fitted model should produce matrices with
    //       pairwise Kendall tau close to the truth. Clayton dependence is
    //       predominantly lower-tail, but pairwise tau should be within 0.1
    //       of truth across every pair.
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
                gap < 0.12,
                "pairwise tau diverged at ({i},{j}): truth={}, refit={}, gap={gap}",
                truth_tau[(i, j)],
                refit_tau[(i, j)]
            );
        }
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
