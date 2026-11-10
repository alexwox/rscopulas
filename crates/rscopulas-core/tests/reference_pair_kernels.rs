//! Archimedean and BB pair kernels against 400-digit reference values.
//!
//! The fixture is produced by `scripts/reference/generate_pair_kernel_truth.py`,
//! which differentiates each family's closed-form distribution function with
//! mpmath. It shares no formula with this crate, and unlike the golden
//! snapshots it does not descend from an earlier release, so it states what
//! the kernels *should* return rather than what they used to return.
//!
//! The grid reaches `1e-6` from both edges of the unit square, where naive
//! double-precision formulas lose most of their digits.

use std::{fs, path::PathBuf};

use serde::Deserialize;

use rscopulas::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation};

const CLIP: f64 = 1e-12;

#[derive(Debug, Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    family: String,
    parameters: Vec<f64>,
    u1: Vec<f64>,
    u2: Vec<f64>,
    log_pdf: Vec<f64>,
    /// `P(U2 <= u2 | U1 = u1)`, as decimal strings with twenty digits.
    h_second_given_first: Vec<String>,
}

fn spec(case: &Case) -> PairCopulaSpec {
    let family = match case.family.as_str() {
        "clayton" => PairCopulaFamily::Clayton,
        "gumbel" => PairCopulaFamily::Gumbel,
        "frank" => PairCopulaFamily::Frank,
        "joe" => PairCopulaFamily::Joe,
        "bb1" => PairCopulaFamily::Bb1,
        "bb6" => PairCopulaFamily::Bb6,
        "bb7" => PairCopulaFamily::Bb7,
        "bb8" => PairCopulaFamily::Bb8,
        other => panic!("unexpected family {other}"),
    };
    let params = match case.parameters.as_slice() {
        [theta] => PairCopulaParams::One(*theta),
        [first, second] => PairCopulaParams::Two(*first, *second),
        other => panic!("unexpected parameter count {}", other.len()),
    };
    PairCopulaSpec {
        family,
        rotation: Rotation::R0,
        params,
    }
}

#[test]
// One index addresses four parallel arrays and its mirror point.
#[allow(clippy::needless_range_loop)]
fn archimedean_and_bb_kernels_match_high_precision_references() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/reference/mpmath/pair_kernels.json");
    let fixture: Fixture =
        serde_json::from_str(&fs::read_to_string(&path).expect("fixture should exist"))
            .expect("fixture should parse");
    assert_eq!(fixture.cases.len(), 52);

    let mut checked = 0usize;
    for case in &fixture.cases {
        let spec = spec(case);
        let label = format!("{} {:?}", case.family, case.parameters);
        let h_true: Vec<f64> = case
            .h_second_given_first
            .iter()
            .map(|text| text.parse::<f64>().expect("decimal string"))
            .collect();
        let side = (case.u1.len() as f64).sqrt() as usize;
        assert_eq!(side * side, case.u1.len(), "{label}: square grid expected");

        for idx in 0..case.u1.len() {
            let (u1, u2) = (case.u1[idx], case.u2[idx]);

            let log_pdf = spec.log_pdf(u1, u2, CLIP).expect("log pdf");
            let expected = case.log_pdf[idx];
            assert!(
                (log_pdf - expected).abs() <= 1e-10 * expected.abs().max(1.0),
                "{label} log pdf at ({u1}, {u2}): {log_pdf} vs {expected}"
            );

            // The library clips conditionals to [CLIP, 1 - CLIP].
            let expected_h = h_true[idx].clamp(CLIP, 1.0 - CLIP);
            let h21 = spec.cond_second_given_first(u1, u2, CLIP).expect("h");
            assert!(
                (h21 - expected_h).abs() <= 1e-11,
                "{label} P(U2|U1) at ({u1}, {u2}): {h21} vs {expected_h}"
            );

            // These families are exchangeable, so the other conditional is
            // the same function with its arguments swapped; the grid is
            // symmetric, so the swapped point is in the fixture too.
            let swapped = (idx % side) * side + idx / side;
            assert_eq!((case.u1[swapped], case.u2[swapped]), (u2, u1));
            let h12 = spec.cond_first_given_second(u2, u1, CLIP).expect("h");
            assert!(
                (h12 - expected_h).abs() <= 1e-11,
                "{label} P(U1|U2) at ({u2}, {u1}): {h12} vs {expected_h}"
            );

            // The inverse must return a point whose conditional probability
            // is the requested one. (Comparing against `u2` itself would test
            // the conditioning of the problem, not the solver: where the
            // density is 1e-30 the quantile is not determined to any digit.)
            if h_true[idx] > 1e-9 && h_true[idx] < 1.0 - 1e-9 {
                let solved = spec
                    .inv_second_given_first(u1, h_true[idx], CLIP)
                    .expect("inverse h");
                let back = spec
                    .cond_second_given_first(u1, solved, CLIP)
                    .expect("h at the solved point");
                assert!(
                    (back - h_true[idx]).abs() <= 1e-9,
                    "{label} inverse at ({u1}, p = {}): h(solved) = {back}",
                    h_true[idx]
                );
                let solved = spec
                    .inv_first_given_second(h_true[idx], u1, CLIP)
                    .expect("inverse h");
                let back = spec
                    .cond_first_given_second(solved, u1, CLIP)
                    .expect("h at the solved point");
                assert!(
                    (back - h_true[idx]).abs() <= 1e-9,
                    "{label} inverse (first given second) at ({u1}, p = {}): {back}",
                    h_true[idx]
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 52 * 81);
}
