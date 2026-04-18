//! Mixed-family R-vine fit (includes Khoudraji in the candidate set).
use ndarray::array;
use rscopulas_core::{PairCopulaFamily, PseudoObs, SelectionCriterion, VineCopula, VineFitOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = PseudoObs::new(array![
        [0.12_f64, 0.18, 0.21],
        [0.21, 0.25, 0.29],
        [0.27, 0.22, 0.31],
        [0.35, 0.42, 0.39],
        [0.48, 0.51, 0.46],
        [0.56, 0.49, 0.58],
        [0.68, 0.73, 0.69],
        [0.82, 0.79, 0.76],
    ])?;

    let options = VineFitOptions {
        family_set: vec![
            PairCopulaFamily::Independence,
            PairCopulaFamily::Gaussian,
            PairCopulaFamily::Clayton,
            PairCopulaFamily::Frank,
            PairCopulaFamily::Gumbel,
            PairCopulaFamily::Khoudraji,
        ],
        include_rotations: true,
        criterion: SelectionCriterion::Aic,
        truncation_level: Some(1),
        ..VineFitOptions::default()
    };

    let fit = VineCopula::fit_r_vine(&data, &options)?;
    println!("structure = {:?}", fit.model.structure());
    println!("order = {:?}", fit.model.order());
    println!("pair parameters = {:?}", fit.model.pair_parameters());
    Ok(())
}
