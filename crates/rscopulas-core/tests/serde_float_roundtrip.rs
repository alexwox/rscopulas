//! JSON serialization must be lossless at the bit level. serde_json's default
//! float parser is best-effort and can be one ulp off, which made a fitted
//! vine compare unequal to its own JSON round trip on Windows CI; the
//! `float_roundtrip` feature makes parsing correctly rounded.

use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};
use rscopulas::{GaussianCopula, PseudoObs, VineCopula, VineFitOptions};

#[test]
fn json_text_round_trip_is_exact_for_doubles() {
    // The value that broke on Windows, plus a pseudo-random sweep.
    let literal = "0.38339426233284557";
    let parsed: f64 = serde_json::from_str(literal).unwrap();
    assert_eq!(parsed, literal.parse::<f64>().unwrap());

    let mut x = 0.123_456_789_f64;
    for i in 0..50_000 {
        x = (x * 9301.0 + 49297.0) % 233280.0 / 233280.0 + i as f64 * 1e-7;
        let text = serde_json::to_string(&x).unwrap();
        let back: f64 = serde_json::from_str(&text).unwrap();
        assert_eq!(back.to_bits(), x.to_bits(), "{text} did not round-trip");
    }
}

#[test]
fn fitted_vine_json_round_trip_is_bit_identical() {
    let model = GaussianCopula::new(array![
        [1.0, 0.45, 0.3],
        [0.45, 1.0, 0.55],
        [0.3, 0.55, 1.0]
    ])
    .unwrap();
    let data = PseudoObs::new(
        rscopulas::CopulaModel::sample(
            &model,
            400,
            &mut StdRng::seed_from_u64(3),
            &Default::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let fit = VineCopula::fit_r_vine(&data, &VineFitOptions::default()).unwrap();
    let text = serde_json::to_string(&fit.model).unwrap();
    let restored: VineCopula = serde_json::from_str(&text).unwrap();
    assert_eq!(
        serde_json::to_value(&fit.model).unwrap(),
        serde_json::to_value(&restored).unwrap()
    );
}
