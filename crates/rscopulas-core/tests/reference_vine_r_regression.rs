use std::panic::{AssertUnwindSafe, catch_unwind};

use ndarray::Array2;
use rand::{Rng, SeedableRng, rngs::StdRng};

use rscopulas::{
    CopulaModel, GaussianCopula, PairCopulaFamily, PairCopulaSpec, PseudoObs, SampleOptions,
    VineCopula, VineFitOptions, VineStructureKind,
};

fn base_options(truncation_level: Option<usize>) -> VineFitOptions {
    VineFitOptions {
        family_set: vec![
            PairCopulaFamily::Independence,
            PairCopulaFamily::Gaussian,
            PairCopulaFamily::Clayton,
            PairCopulaFamily::Frank,
            PairCopulaFamily::Gumbel,
        ],
        truncation_level,
        ..VineFitOptions::default()
    }
}

fn gaussian_sample(correlation: Array2<f64>, n_obs: usize, seed: u64) -> PseudoObs {
    let model = GaussianCopula::new(correlation).expect("correlation should be valid");
    let mut rng = StdRng::seed_from_u64(seed);
    let sample = model
        .sample(n_obs, &mut rng, &SampleOptions::default())
        .expect("gaussian sample should succeed");
    PseudoObs::new(sample).expect("sampled pseudo-observations should be valid")
}

fn path_correlation(dim: usize, rho: f64) -> Array2<f64> {
    let mut correlation = Array2::zeros((dim, dim));
    for row in 0..dim {
        for col in 0..dim {
            correlation[(row, col)] = rho.powi((row as i32 - col as i32).abs());
        }
    }
    correlation
}

fn star_correlation(dim: usize, rho: f64) -> Array2<f64> {
    let mut correlation = Array2::from_elem((dim, dim), rho * rho);
    for idx in 0..dim {
        correlation[(idx, idx)] = 1.0;
    }
    for leaf in 1..dim {
        correlation[(0, leaf)] = rho;
        correlation[(leaf, 0)] = rho;
    }
    correlation
}

fn iid_uniform(dim: usize, n_obs: usize, seed: u64) -> PseudoObs {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut values = Array2::zeros((n_obs, dim));
    for row in 0..n_obs {
        for col in 0..dim {
            values[(row, col)] = rng.random::<f64>().clamp(1e-6, 1.0 - 1e-6);
        }
    }
    PseudoObs::new(values).expect("uniform pseudo-observations should be valid")
}

fn assert_structure_invariants(model: &VineCopula, dim: usize) {
    let order = model.order();
    assert_eq!(model.dim(), dim);
    assert_eq!(order.len(), dim);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..dim).collect::<Vec<_>>());

    let structure = model.structure_info();
    assert_eq!(structure.matrix.nrows(), dim);
    assert_eq!(structure.matrix.ncols(), dim);
    for value in structure.matrix.iter().copied() {
        assert!(value < dim);
    }
}

#[test]
fn fit_r_vine_on_d5_path_matches_dvine_loglik() {
    let data = gaussian_sample(path_correlation(5, 0.8), 512, 13);
    let options = base_options(None);

    let r_fit = VineCopula::fit_r_vine(&data, &options).expect("R-vine fit should succeed");
    let d_fit = VineCopula::fit_d_vine(&data, &options).expect("D-vine fit should succeed");

    assert_eq!(r_fit.model.structure(), VineStructureKind::R);
    assert_structure_invariants(&r_fit.model, 5);
    assert!(
        (r_fit.diagnostics.loglik - d_fit.diagnostics.loglik).abs() < 1e-6,
        "R-vine path data should match the D-vine optimum: r={}, d={}",
        r_fit.diagnostics.loglik,
        d_fit.diagnostics.loglik
    );
}

#[test]
fn fit_r_vine_on_d6_star_survives_all_trees() {
    let data = gaussian_sample(star_correlation(6, 0.7), 512, 19);

    for truncation_level in [Some(1), Some(2), Some(3), Some(4), Some(5), None] {
        let fit = VineCopula::fit_r_vine(&data, &base_options(truncation_level))
            .expect("star-correlated R-vine fit should succeed");
        assert_structure_invariants(&fit.model, 6);
    }
}

#[test]
fn fit_r_vine_never_panics_on_pathological_inputs() {
    for dim in [4usize, 5, 6, 7] {
        for seed in 0..40u64 {
            let datasets = [
                gaussian_sample(path_correlation(dim, 0.8), 192, seed + 100),
                gaussian_sample(star_correlation(dim, 0.7), 192, seed + 200),
                iid_uniform(dim, 192, seed + 300),
            ];

            for data in datasets {
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    VineCopula::fit_r_vine(&data, &base_options(None))
                }));
                assert!(
                    outcome.is_ok(),
                    "fit_r_vine panicked for dim={dim}, seed={seed}"
                );
            }
        }
    }
}

#[test]
fn fit_r_vine_on_strongly_dependent_pair_produces_finite_loglik() {
    // Regression test for a bug where fit_r_vine with the broad family set
    // (`Frank` in particular) on strongly-dependent pseudo-observations
    // silently returned `loglik = +inf` (AIC = -inf), and the subsequent
    // `sample()` produced non-uniform marginals with a pile of draws pinned
    // at the upper boundary. Root cause: catastrophic cancellation in the
    // legacy Frank log-density for large θ, which made the tainted candidate
    // "win" AIC minimization. We now compute Frank's density/h-inverse in
    // log space and reject any fit whose per-observation log-density is
    // non-finite, so the selector never picks such a candidate.
    let mut rng = StdRng::seed_from_u64(0xF3A1_BEEF);
    // A 5-d dataset whose first two variables have Kendall τ ≈ 0.93 — the
    // pathological regime for Frank in the 0.2.2 release.
    let mut values = Array2::zeros((1024, 5));
    for row in 0..1024 {
        let u: f64 = rng.random::<f64>().clamp(1e-6, 1.0 - 1e-6);
        let jitter: f64 = 1e-3 * rng.random::<f64>();
        let shifted = (u + jitter - 0.5 * 1e-3).clamp(1e-6, 1.0 - 1e-6);
        values[(row, 0)] = u;
        values[(row, 1)] = shifted;
        for col in 2..5 {
            values[(row, col)] = rng.random::<f64>().clamp(1e-6, 1.0 - 1e-6);
        }
    }
    let data = PseudoObs::new(values).expect("pseudo-obs should be valid");

    let options = VineFitOptions {
        family_set: vec![
            PairCopulaFamily::Independence,
            PairCopulaFamily::Gaussian,
            PairCopulaFamily::StudentT,
            PairCopulaFamily::Clayton,
            PairCopulaFamily::Frank,
            PairCopulaFamily::Gumbel,
        ],
        include_rotations: true,
        truncation_level: Some(2),
        ..VineFitOptions::default()
    };

    let fit = VineCopula::fit_r_vine(&data, &options).expect("R-vine fit should succeed");
    assert!(
        fit.diagnostics.loglik.is_finite(),
        "loglik must be finite after the Frank stability fix; got {}",
        fit.diagnostics.loglik
    );
    assert!(
        fit.diagnostics.aic.is_finite(),
        "aic must be finite; got {}",
        fit.diagnostics.aic
    );

    // Sample from the fitted vine and confirm every marginal stays uniform.
    let mut sample_rng = StdRng::seed_from_u64(12345);
    let samples = fit
        .model
        .sample(20_000, &mut sample_rng, &SampleOptions::default())
        .expect("sampling should succeed");
    for col in 0..data.dim() {
        let column = samples.column(col);
        let mean = column.iter().sum::<f64>() / 20_000.0;
        let above_999 = column.iter().filter(|&&v| v > 0.999).count();
        let above_99 = column.iter().filter(|&&v| v > 0.99).count();
        assert!(
            (mean - 0.5).abs() < 0.02,
            "marginal column {col} mean {mean} deviates from 0.5"
        );
        // Pinning manifests as ~2% of samples above 0.999, far more than the
        // 0.1% expected under Uniform(0, 1). Use a generous cap to avoid
        // flaky failures while still catching the pathology.
        assert!(
            (above_999 as f64) / 20_000.0 < 0.005,
            "column {col}: {above_999} / 20000 samples above 0.999 — upper-boundary pile-up"
        );
        assert!(
            (above_99 as f64 / 20_000.0 - 0.01).abs() < 0.005,
            "column {col}: upper-tail mass P(U>0.99) = {} deviates too far from 0.01",
            above_99 as f64 / 20_000.0
        );
    }
}

#[test]
fn frank_pair_fit_is_rejected_when_log_density_overflows() {
    // Direct regression at the pair-copula layer: with the stable formulas
    // the Frank density is bounded for any θ and any u, v in (0, 1), so even
    // at pathological parameters the fit returns finite diagnostics instead
    // of the old `+∞` loglik.
    let n = 1024usize;
    let mut rng = StdRng::seed_from_u64(7);
    let mut u1 = Vec::with_capacity(n);
    let mut u2 = Vec::with_capacity(n);
    for _ in 0..n {
        let u: f64 = rng.random::<f64>().clamp(1e-10, 1.0 - 1e-10);
        let jitter: f64 = 1e-4 * rng.random::<f64>();
        u1.push(u);
        u2.push((u + jitter).clamp(1e-10, 1.0 - 1e-10));
    }

    let options = VineFitOptions {
        family_set: vec![PairCopulaFamily::Frank],
        include_rotations: false,
        truncation_level: None,
        ..VineFitOptions::default()
    };
    let fit = rscopulas::paircopula::fit_pair_copula(&u1, &u2, &options)
        .expect("frank pair fit should succeed on strongly dependent data");
    let _: &PairCopulaSpec = &fit.spec;
    assert!(
        fit.loglik.is_finite(),
        "frank pair loglik must be finite for large θ; got {}",
        fit.loglik
    );
    assert!(
        fit.aic.is_finite(),
        "frank pair aic must be finite for large θ; got {}",
        fit.aic
    );
}

#[test]
fn fit_r_vine_structure_matrix_invariant() {
    let cases = [
        gaussian_sample(path_correlation(4, 0.8), 256, 41),
        gaussian_sample(path_correlation(5, 0.8), 256, 42),
        gaussian_sample(star_correlation(5, 0.7), 256, 43),
        gaussian_sample(star_correlation(6, 0.7), 256, 44),
    ];

    for (idx, data) in cases.into_iter().enumerate() {
        let fit = VineCopula::fit_r_vine(&data, &base_options(None))
            .unwrap_or_else(|err| panic!("case {idx} should succeed: {err}"));
        assert_structure_invariants(&fit.model, data.dim());
    }
}
