//! Gaussian copula: validate data, fit, log-density, sample.
use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};
use rscopulas_core::{CopulaModel, FitOptions, GaussianCopula, PseudoObs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = PseudoObs::new(array![
        [0.12_f64, 0.18],
        [0.21, 0.25],
        [0.27, 0.22],
        [0.35, 0.42],
        [0.48, 0.51],
        [0.56, 0.49],
        [0.68, 0.73],
        [0.82, 0.79],
    ])?;

    let fit = GaussianCopula::fit(&data, &FitOptions::default())?;
    println!("AIC: {}", fit.diagnostics.aic);
    println!("Correlation:\n{:?}", fit.model.correlation());

    let log_pdf = fit.model.log_pdf(&data, &Default::default())?;
    println!("First log density: {}", log_pdf[0]);

    let mut rng = StdRng::seed_from_u64(7);
    let sample = fit.model.sample(4, &mut rng, &Default::default())?;
    println!("Sample:\n{:?}", sample);
    Ok(())
}
