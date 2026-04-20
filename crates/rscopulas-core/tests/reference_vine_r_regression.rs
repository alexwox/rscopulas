use std::panic::{AssertUnwindSafe, catch_unwind};

use ndarray::Array2;
use rand::{Rng, SeedableRng, rngs::StdRng};

use rscopulas::{
    CopulaModel, GaussianCopula, PairCopulaFamily, PseudoObs, SampleOptions, VineCopula,
    VineFitOptions, VineStructureKind,
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
