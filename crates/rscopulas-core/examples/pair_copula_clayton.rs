//! Clayton pair copula with rotation: density and h-function.
use rscopulas_core::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spec = PairCopulaSpec {
        family: PairCopulaFamily::Clayton,
        rotation: Rotation::R90,
        params: PairCopulaParams::One(1.4),
    };

    let log_pdf = spec.log_pdf(0.32, 0.77, 1e-12)?;
    let h = spec.cond_first_given_second(0.32, 0.77, 1e-12)?;

    println!("log_pdf = {log_pdf}");
    println!("h_1|2 = {h}");
    Ok(())
}
