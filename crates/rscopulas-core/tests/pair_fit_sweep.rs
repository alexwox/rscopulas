//! End-to-end regression sweep over the pair-copula fitters.
//!
//! `fixtures/golden/v1/pair_fit_sweep.txt` holds the fitted log-likelihood
//! and AIC that release 0.4.0 reached on 2250 fits: twelve parametric
//! families, with and without rotations, plus selection from the default
//! family set, on samples of three sizes and five dependence shapes (and
//! their negatively dependent mirror images). The optimisers behind these
//! fits were rewritten for speed; this test holds them to the old results:
//!
//! * a single-family fit must reach the stored log-likelihood (it may exceed
//!   it), and
//! * default-set selection must reach the stored AIC (it may undercut it).
//!
//! Regenerate with `RSCOPULAS_BLESS=1` only together with an entry in
//! `fixtures/golden/v1/CHANGES.md`.

#[path = "support/datagen.rs"]
mod datagen;

use std::{collections::BTreeMap, fmt::Write as _, path::PathBuf};

use datagen::Dependence;
use rscopulas::{PairCopulaFamily as F, VineFitOptions, paircopula::fit_pair_copula};

/// Fits that are allowed to end below the stored log-likelihood.
///
/// On this sample of 300 *independent* observations the 0.4.0 simplex search
/// wandered twelve units across the parameter box to `θ = 1 + e⁴` (the box
/// edge) with an asymmetry weight of 0.004, where the Tawn likelihood has an
/// isolated spike worth 1–3 log-likelihood units — the likelihood is `-inf`
/// one step beyond it and *lower* than at the start on the way there. The
/// current search returns the smooth local optimum instead. Neither is
/// selected over independence by AIC.
const TAWN_BOUNDARY_SPIKE: [&str; 4] = [
    "300/Independent/2/pos/Tawn1/true",
    "300/Independent/2/neg/Tawn1/true",
    "300/Independent/2/pos/Tawn2/true",
    "300/Independent/2/neg/Tawn2/true",
];

fn is_exempt(key: &str) -> bool {
    TAWN_BOUNDARY_SPIKE.contains(&key)
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/golden/v1/pair_fit_sweep.txt")
}

/// `key → (loglik, aic)`, or `None` for a fit that failed.
fn run_sweep() -> BTreeMap<String, Option<(f64, f64)>> {
    let families = [
        F::Gaussian,
        F::StudentT,
        F::Clayton,
        F::Frank,
        F::Gumbel,
        F::Joe,
        F::Bb1,
        F::Bb6,
        F::Bb7,
        F::Bb8,
        F::Tawn1,
        F::Tawn2,
    ];
    let mut results = BTreeMap::new();
    for n in [80usize, 300, 1500] {
        for dependence in [
            Dependence::Gaussian,
            Dependence::HeavyTail,
            Dependence::LowerTail,
            Dependence::UpperTail,
            Dependence::Independent,
        ] {
            for seed in 0..3u64 {
                let (u1, u2) = datagen::pair(n, dependence, 0x900 + seed * 17 + n as u64);
                let flipped: Vec<f64> = u1.iter().map(|u| 1.0 - u).collect();
                for (variant, first) in [("pos", &u1), ("neg", &flipped)] {
                    for family in families {
                        for include_rotations in [true, false] {
                            let options = VineFitOptions {
                                family_set: vec![family],
                                include_rotations,
                                ..VineFitOptions::default()
                            };
                            let key = format!(
                                "{n}/{dependence:?}/{seed}/{variant}/{family:?}/{include_rotations}"
                            );
                            let fit = fit_pair_copula(first, &u2, &options).ok();
                            results.insert(key, fit.map(|fit| (fit.loglik, fit.aic)));
                        }
                    }
                    let fit = fit_pair_copula(first, &u2, &VineFitOptions::default())
                        .expect("default-set fit should succeed");
                    results.insert(
                        format!("{n}/{dependence:?}/{seed}/{variant}/default"),
                        Some((fit.loglik, fit.aic)),
                    );
                }
            }
        }
    }
    results
}

#[test]
fn pair_fits_reach_the_likelihoods_of_release_0_4_0() {
    let results = run_sweep();
    let path = fixture_path();
    if std::env::var_os("RSCOPULAS_BLESS").is_some_and(|value| value != "0") {
        let mut text = String::new();
        for (key, value) in &results {
            match value {
                Some((loglik, aic)) => writeln!(text, "{key} {loglik:.10} {aic:.10}").unwrap(),
                None => writeln!(text, "{key} err err").unwrap(),
            }
        }
        std::fs::write(&path, text).expect("write sweep fixture");
        return;
    }

    let stored = std::fs::read_to_string(&path).expect("sweep fixture should exist");
    let mut failures = Vec::new();
    let mut compared = 0usize;
    let mut exempted = 0usize;
    for line in stored.lines() {
        let mut fields = line.split_whitespace();
        let (Some(key), Some(loglik), Some(aic)) = (fields.next(), fields.next(), fields.next())
        else {
            panic!("malformed fixture line: {line}");
        };
        let Some(current) = results.get(key) else {
            failures.push(format!("{key}: no longer part of the sweep"));
            continue;
        };
        compared += 1;
        let (Ok(want_loglik), Ok(want_aic)) = (loglik.parse::<f64>(), aic.parse::<f64>()) else {
            // The fit failed in 0.4.0; it may fail or succeed now.
            continue;
        };
        let Some((got_loglik, got_aic)) = *current else {
            failures.push(format!("{key}: fit used to succeed and now fails"));
            continue;
        };
        if is_exempt(key) {
            exempted += 1;
            continue;
        }
        // Same slack as `Tol::LOGLIK` in the golden snapshots, except for
        // the dozen mis-oriented fits (a positive-dependence family forced
        // onto negatively dependent data without rotations, log-likelihood
        // in the minus thousands). Their optimum is pinned against the edge
        // of the parameter box on a steep, ill-conditioned slope, where the
        // stopping point moves in the sixth digit between libm
        // implementations (glibc lands 1.7e-6 relative below macOS on one of
        // them). Such a candidate never wins selection.
        let relative = if want_loglik < -1.0 { 1e-5 } else { 1e-6 };
        let slack = 1e-5 + relative * want_loglik.abs();
        let shortfall = if key.ends_with("/default") {
            // Selected by AIC: lower is better.
            (got_aic - want_aic) / 2.0
        } else {
            want_loglik - got_loglik
        };
        if shortfall > slack {
            failures.push(format!(
                "{key}: loglik {got_loglik:.8} (aic {got_aic:.8}) vs stored {want_loglik:.8} (aic {want_aic:.8})"
            ));
        }
    }
    assert_eq!(
        compared,
        results.len(),
        "sweep and fixture cover different keys"
    );
    assert_eq!(
        exempted,
        TAWN_BOUNDARY_SPIKE.len(),
        "every exemption should match a fit"
    );
    assert!(
        failures.is_empty(),
        "{} of {compared} pair fits regressed:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
