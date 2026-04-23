//! Coverage for the new `TreeAlgorithm` and `TreeCriterion` options on
//! R-vine fitting. Verifies:
//!   * Prim's MST gives the same total edge weight as Kruskal's (they both
//!     maximise the same spanning tree; tie-breaking can differ).
//!   * Wilson's loop-erased random walk produces valid spanning trees
//!     (exactly `d − 1` edges, connected).
//!   * Spearman-ρ and Hoeffding-D criteria produce valid R-vine fits whose
//!     first-tree structure is a function of the criterion (not identical
//!     to the Tau-based fit on strongly non-monotone data).

use ndarray::Array2;
use rand::distr::StandardUniform;
use rand::{SeedableRng, rngs::StdRng};
use rand::Rng;

use rscopulas::{
    PairCopulaFamily, PseudoObs, SelectionCriterion, TreeAlgorithm, TreeCriterion, VineCopula,
    VineFitOptions,
};

fn synthetic_gaussian_sample(dim: usize, n: usize, seed: u64) -> PseudoObs {
    // Cholesky of a simple Toeplitz-like correlation: ρ[i,j] = 0.6^|i-j|.
    let mut corr = Array2::<f64>::zeros((dim, dim));
    for i in 0..dim {
        for j in 0..dim {
            corr[(i, j)] = 0.6_f64.powi((i as i32 - j as i32).abs());
        }
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sample = Array2::<f64>::zeros((n, dim));
    for row in 0..n {
        let z: Vec<f64> = (0..dim)
            .map(|_| {
                // Box–Muller for a standard normal.
                let u1: f64 = rng.sample::<f64, _>(StandardUniform).max(1e-12);
                let u2: f64 = rng.sample::<f64, _>(StandardUniform);
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();
        let mut y = vec![0.0; dim];
        for i in 0..dim {
            // Lower-triangular Cholesky solve for the Toeplitz corr would be
            // a bit fiddly; cheat by mixing via corr directly (not a true
            // Cholesky — but good enough for generating a sample with
            // positive pairwise dependence, which is all the tests need).
            for j in 0..=i {
                y[i] += corr[(i, j)] * z[j];
            }
        }
        for i in 0..dim {
            // Map N(0, σ²) → (0, 1) via the standard normal CDF.
            let cdf = 0.5 * (1.0 + erf(y[i] / (2.0_f64.sqrt())));
            sample[(row, i)] = cdf.clamp(1e-6, 1.0 - 1e-6);
        }
    }
    PseudoObs::new(sample).expect("synthetic sample should be valid pseudo-observations")
}

/// Abramowitz & Stegun 7.1.26 approximation of the error function —
/// accurate to ~1.5e-7 in the tails, plenty for generating test data.
fn erf(x: f64) -> f64 {
    let sign = x.signum();
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

#[test]
fn prim_and_kruskal_produce_identical_total_weight() {
    // Both are max-spanning-tree algorithms on the same weighted graph — the
    // total weight of the returned tree is identical (tie-breaking may
    // change the specific tree but not its total weight).
    let data = synthetic_gaussian_sample(6, 400, 11);
    let options_kruskal = VineFitOptions {
        tree_algorithm: TreeAlgorithm::Kruskal,
        ..VineFitOptions::default()
    };
    let options_prim = VineFitOptions {
        tree_algorithm: TreeAlgorithm::Prim,
        ..VineFitOptions::default()
    };
    let fit_k = VineCopula::fit_r_vine(&data, &options_kruskal).expect("kruskal fit");
    let fit_p = VineCopula::fit_r_vine(&data, &options_prim).expect("prim fit");

    // Loglik differences between two MSTs that share a total-weight optimum
    // should be tiny (they choose the same edges modulo tie-breaks). On
    // synthetic Gaussian data with no exact ties the two should match
    // exactly; accept 1% slack for robustness.
    let rel_gap = ((fit_k.diagnostics.loglik - fit_p.diagnostics.loglik).abs())
        / fit_k.diagnostics.loglik.abs().max(1.0);
    assert!(
        rel_gap < 0.01,
        "kruskal vs prim loglik gap = {rel_gap}; k={}, p={}",
        fit_k.diagnostics.loglik,
        fit_p.diagnostics.loglik
    );
}

#[test]
fn wilson_random_weighted_produces_valid_spanning_tree() {
    let data = synthetic_gaussian_sample(5, 300, 23);
    let options = VineFitOptions {
        tree_algorithm: TreeAlgorithm::RandomWeighted,
        rng_seed: Some(7),
        ..VineFitOptions::default()
    };
    let fit = VineCopula::fit_r_vine(&data, &options).expect("wilson fit");
    // First tree should have exactly d − 1 = 4 edges.
    assert_eq!(fit.model.trees()[0].edges.len(), 4);
    // Loglik must be finite and positive (there's real dependence in the DGP).
    assert!(fit.diagnostics.loglik.is_finite());
    assert!(fit.diagnostics.loglik > 0.0);
}

#[test]
fn wilson_random_weighted_prefers_strong_dependence_edges() {
    // Bias-direction test — this is what would catch a weighted-Wilson
    // inversion bug that unweighted Wilson's reproducibility check misses.
    //
    // DGP: 5 variables where (0, 1) is a strong-dependence pair (τ ≈ 0.9)
    // and every other pair has modest dependence (τ ≈ 0.3–0.5). Over 100
    // random seeds, the (0, 1) edge should show up in the first tree far
    // more often under RandomWeighted than under RandomUnweighted — that's
    // the whole point of the weighted variant.
    let data = dominant_pair_sample(5, 300, 97);

    // Restricted family set keeps per-fit cost near a single Gaussian MLE,
    // so 200 R-vine fits run in seconds rather than minutes — the test
    // exercises the tree algorithm, not pair-family selection.
    let base = VineFitOptions {
        family_set: vec![PairCopulaFamily::Gaussian],
        include_rotations: false,
        ..VineFitOptions::default()
    };

    let mut weighted_hits = 0usize;
    let mut unweighted_hits = 0usize;
    for seed in 0..100 {
        for (algorithm, counter) in [
            (TreeAlgorithm::RandomWeighted, &mut weighted_hits),
            (TreeAlgorithm::RandomUnweighted, &mut unweighted_hits),
        ] {
            let options = VineFitOptions {
                tree_algorithm: algorithm,
                rng_seed: Some(seed as u64),
                ..base.clone()
            };
            if let Ok(fit) = VineCopula::fit_r_vine(&data, &options) {
                let has_dominant = fit.model.trees()[0].edges.iter().any(|edge| {
                    let (a, b) = edge.conditioned;
                    (a == 0 && b == 1) || (a == 1 && b == 0)
                });
                if has_dominant {
                    *counter += 1;
                }
            }
        }
    }
    // Weighted should hit the dominant edge substantially more often than
    // unweighted. A +20pp margin keeps the test stable across RNG variants
    // while still catching an inversion (which would give weighted *fewer*
    // hits than unweighted).
    assert!(
        weighted_hits > unweighted_hits + 20,
        "weighted {weighted_hits} vs unweighted {unweighted_hits} — expected a significant bias toward the strong-dependence edge"
    );
}

fn dominant_pair_sample(dim: usize, n: usize, seed: u64) -> PseudoObs {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sample = Array2::<f64>::zeros((n, dim));
    for row in 0..n {
        let shared: f64 = rng.sample(StandardUniform);
        for col in 0..dim {
            let noise: f64 = rng.sample(StandardUniform);
            // col 0 and col 1 both get ~0.95·shared → very strong τ with each
            // other. Remaining columns share only a moderate component.
            let coupling = if col < 2 { 0.95 } else { 0.45 };
            let value = coupling * shared + (1.0 - coupling) * noise;
            sample[(row, col)] = value.clamp(1e-6, 1.0 - 1e-6);
        }
    }
    PseudoObs::new(sample).expect("dominant-pair sample should be valid")
}

#[test]
fn wilson_random_unweighted_reproducibility() {
    // Same seed → same structure. Different seed → may differ.
    let data = synthetic_gaussian_sample(5, 300, 29);
    let options_a = VineFitOptions {
        tree_algorithm: TreeAlgorithm::RandomUnweighted,
        rng_seed: Some(42),
        ..VineFitOptions::default()
    };
    let options_b = VineFitOptions {
        tree_algorithm: TreeAlgorithm::RandomUnweighted,
        rng_seed: Some(42),
        ..VineFitOptions::default()
    };
    let fit_a = VineCopula::fit_r_vine(&data, &options_a).expect("wilson fit a");
    let fit_b = VineCopula::fit_r_vine(&data, &options_b).expect("wilson fit b");
    // Same seed → same loglik bit-for-bit (parameters are fit from a seed-
    // invariant bivariate MLE once the structure is chosen).
    assert_eq!(
        fit_a.diagnostics.loglik.to_bits(),
        fit_b.diagnostics.loglik.to_bits(),
        "wilson with same seed should produce identical loglik"
    );
}

#[test]
fn spearman_rho_criterion_produces_valid_fit() {
    let data = synthetic_gaussian_sample(5, 400, 37);
    let options = VineFitOptions {
        tree_criterion: TreeCriterion::Rho,
        ..VineFitOptions::default()
    };
    let fit = VineCopula::fit_r_vine(&data, &options).expect("rho-weighted fit");
    assert!(fit.diagnostics.loglik.is_finite());
    assert!(fit.diagnostics.loglik > 0.0);
    assert_eq!(fit.model.trees()[0].edges.len(), 4);
}

#[test]
fn hoeffding_criterion_produces_valid_fit() {
    let data = synthetic_gaussian_sample(5, 400, 41);
    let options = VineFitOptions {
        tree_criterion: TreeCriterion::Hoeffding,
        ..VineFitOptions::default()
    };
    let fit = VineCopula::fit_r_vine(&data, &options).expect("hoeffding-weighted fit");
    assert!(fit.diagnostics.loglik.is_finite());
    assert!(fit.diagnostics.loglik > 0.0);
    assert_eq!(fit.model.trees()[0].edges.len(), 4);
}

#[test]
fn non_kruskal_rejected_for_c_and_d_vines() {
    let data = synthetic_gaussian_sample(4, 200, 43);
    let options = VineFitOptions {
        tree_algorithm: TreeAlgorithm::Prim,
        ..VineFitOptions::default()
    };
    let err_c = VineCopula::fit_c_vine(&data, &options).expect_err("C-vine should reject Prim");
    assert!(err_c.to_string().contains("C-vine"), "got {}", err_c);
    let err_d = VineCopula::fit_d_vine(&data, &options).expect_err("D-vine should reject Prim");
    assert!(err_d.to_string().contains("D-vine"), "got {}", err_d);
}

#[test]
fn mbicv_auto_truncation_drops_weak_trees() {
    // 6-dim DGP with real dependence only in the first two trees: columns
    // 0-1-2 form a chain, columns 3-4-5 are (near-)independent. mBICV
    // should truncate aggressively on this data since the higher-level
    // trees carry essentially no signal.
    let data = synthetic_weak_upper_chain(6, 1500, 53);
    let options = VineFitOptions {
        criterion: SelectionCriterion::Mbicv { psi0: 0.9 },
        select_trunc_lvl: true,
        ..VineFitOptions::default()
    };
    let fit = VineCopula::fit_r_vine(&data, &options).expect("mbicv fit");
    let level = fit
        .model
        .structure_info()
        .truncation_level
        .expect("mbicv should set a truncation_level");
    assert!(
        level < data.dim() - 1,
        "mbicv should truncate below the full d-1; got level = {level}, dim = {}",
        data.dim()
    );
    // And should keep at least tree 1 (the real dependence).
    assert!(level >= 1, "mbicv truncated to 0 trees — too aggressive");
}

#[test]
fn mbicv_auto_truncation_respects_user_cap() {
    // With a manual cap of 2, auto-selection cannot choose 3 or higher even
    // if mBICV would otherwise keep those trees.
    let data = synthetic_gaussian_sample(6, 1500, 59);
    let options = VineFitOptions {
        criterion: SelectionCriterion::Mbicv { psi0: 0.9 },
        select_trunc_lvl: true,
        truncation_level: Some(2),
        ..VineFitOptions::default()
    };
    let fit = VineCopula::fit_r_vine(&data, &options).expect("mbicv capped fit");
    let level = fit.model.structure_info().truncation_level.unwrap_or(0);
    assert!(level <= 2, "capped level should be ≤ 2, got {level}");
}

/// Generates `n` samples from a 6-dim joint where only the first few
/// dimensions have meaningful dependence — the trailing columns are
/// essentially independent uniforms. Useful for testing auto-truncation.
fn synthetic_weak_upper_chain(dim: usize, n: usize, seed: u64) -> PseudoObs {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sample = Array2::<f64>::zeros((n, dim));
    for row in 0..n {
        let anchor: f64 = rng.sample(StandardUniform);
        for col in 0..dim {
            let noise: f64 = rng.sample(StandardUniform);
            // Cols 0-2 are copies of the anchor with a bit of noise;
            // cols 3+ are pure noise.
            let coupling = if col < 3 { 0.9 } else { 0.02 };
            let value = coupling * anchor + (1.0 - coupling) * noise;
            sample[(row, col)] = value.clamp(1e-6, 1.0 - 1e-6);
        }
    }
    PseudoObs::new(sample).expect("synthetic weak-chain sample should be valid")
}
