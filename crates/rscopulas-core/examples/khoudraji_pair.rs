//! Khoudraji pair copula from Gaussian × Clayton base specs.
use rscopulas_core::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let first = PairCopulaSpec {
        family: PairCopulaFamily::Gaussian,
        rotation: Rotation::R0,
        params: PairCopulaParams::One(0.45),
    };
    let second = PairCopulaSpec {
        family: PairCopulaFamily::Clayton,
        rotation: Rotation::R0,
        params: PairCopulaParams::One(2.0),
    };
    let khoudraji = PairCopulaSpec::khoudraji(first, second, 0.35, 0.80)?;
    println!("log_pdf = {}", khoudraji.log_pdf(0.32, 0.77, 1e-12)?);
    Ok(())
}
