//! Wall-clock harness for the default R-vine fit. Ignored by default because
//! timings are machine dependent; run with
//!
//! ```text
//! cargo test --release --test estimation_timing -- --ignored --nocapture
//! ```
//!
//! to compare estimator changes. The scenario is fixed (4 columns, 800 rows,
//! seeded Gaussian data with a Toeplitz correlation) so successive runs on the
//! same machine are comparable.

use std::time::{Duration, Instant};

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use rscopulas::{
    CopulaModel, GaussianCopula, PairCopulaFamily, PseudoObs, VineCopula, VineFitOptions,
};

fn toeplitz_sample(dim: usize, n_obs: usize, rho: f64, seed: u64) -> PseudoObs {
    let mut correlation = Array2::<f64>::zeros((dim, dim));
    for row in 0..dim {
        for col in 0..dim {
            correlation[(row, col)] = rho.powi((row as i32 - col as i32).abs());
        }
    }
    let model = GaussianCopula::new(correlation).expect("correlation should be valid");
    let mut rng = StdRng::seed_from_u64(seed);
    let sample = model
        .sample(n_obs, &mut rng, &Default::default())
        .expect("sampling should succeed");
    PseudoObs::new(sample).expect("sample should be valid pseudo-observations")
}

fn time_fit(label: &str, data: &PseudoObs, options: &VineFitOptions, repetitions: usize) {
    let mut best = Duration::MAX;
    let mut total = Duration::ZERO;
    let mut loglik = f64::NAN;
    for _ in 0..repetitions {
        let start = Instant::now();
        let fit = VineCopula::fit_r_vine(data, options);
        let elapsed = start.elapsed();
        best = best.min(elapsed);
        total += elapsed;
        match fit {
            Ok(fit) => loglik = fit.diagnostics.loglik,
            Err(error) => {
                println!(
                    "{label}: fit failed after {:.3}s: {error}",
                    elapsed.as_secs_f64()
                );
                return;
            }
        }
    }
    println!(
        "{label}: best {:.3}s, mean {:.3}s over {repetitions} runs (loglik {loglik:.3})",
        best.as_secs_f64(),
        total.as_secs_f64() / repetitions as f64
    );
}

#[test]
#[ignore = "wall-clock benchmark; run manually with --ignored --nocapture"]
fn default_r_vine_fit_timing_d4_n800() {
    let data = toeplitz_sample(4, 800, 0.5, 2024);
    let repetitions = 3;

    time_fit(
        "VineFitOptions::default()",
        &data,
        &VineFitOptions::default(),
        repetitions,
    );

    let explicit = VineFitOptions {
        family_set: vec![
            PairCopulaFamily::Independence,
            PairCopulaFamily::Gaussian,
            PairCopulaFamily::StudentT,
            PairCopulaFamily::Clayton,
            PairCopulaFamily::Frank,
            PairCopulaFamily::Gumbel,
            PairCopulaFamily::Joe,
            PairCopulaFamily::Bb1,
            PairCopulaFamily::Bb7,
        ],
        ..VineFitOptions::default()
    };
    time_fit(
        "explicit set without Khoudraji",
        &data,
        &explicit,
        repetitions,
    );

    for family in [
        PairCopulaFamily::Gaussian,
        PairCopulaFamily::StudentT,
        PairCopulaFamily::Clayton,
        PairCopulaFamily::Frank,
        PairCopulaFamily::Gumbel,
        PairCopulaFamily::Joe,
        PairCopulaFamily::Bb1,
        PairCopulaFamily::Bb7,
    ] {
        let single = VineFitOptions {
            family_set: vec![family],
            ..VineFitOptions::default()
        };
        time_fit(&format!("{family:?} only"), &data, &single, repetitions);
    }
}
