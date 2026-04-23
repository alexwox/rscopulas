use std::{fs, path::PathBuf};

use serde::Deserialize;

use rscopulas::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation};

#[derive(Debug, Deserialize)]
struct Metadata {
    source_package: String,
    source_version: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct PairFixture {
    metadata: Metadata,
    family: String,
    family_code: i32,
    par: f64,
    par2: f64,
    u1: Vec<f64>,
    u2: Vec<f64>,
    p: Vec<f64>,
    expected_log_pdf: Vec<f64>,
    expected_cond_first_given_second: Vec<f64>,
    expected_cond_second_given_first: Vec<f64>,
    expected_inv_first_given_second: Vec<f64>,
    expected_inv_second_given_first: Vec<f64>,
}

#[test]
fn pair_families_match_vinecopula_fixtures() {
    for fixture_name in [
        "pair_gaussian_case01.json",
        "pair_student_t_case01.json",
        "pair_clayton_case01.json",
        "pair_frank_case01.json",
        "pair_gumbel_case01.json",
        "pair_joe_case01.json",
        "pair_bb1_case01.json",
        "pair_bb6_case01.json",
        "pair_bb7_case01.json",
        "pair_bb8_case01.json",
        "pair_tawn1_case01.json",
        "pair_tawn2_case01.json",
    ] {
        let fixture: PairFixture = load_fixture(fixture_name);
        assert_eq!(fixture.metadata.source_package, "VineCopula");
        let spec = fixture_spec(&fixture);

        for idx in 0..fixture.u1.len() {
            let u1 = fixture.u1[idx];
            let u2 = fixture.u2[idx];
            let p = fixture.p[idx];

            let actual_log_pdf = spec
                .log_pdf(u1, u2, 1e-12)
                .expect("log pdf should evaluate");
            assert!(
                (actual_log_pdf - fixture.expected_log_pdf[idx]).abs() < 1e-9,
                "{} {} log pdf mismatch at row {idx}: left={actual_log_pdf}, right={}",
                fixture.metadata.source_version,
                fixture.family,
                fixture.expected_log_pdf[idx]
            );

            let actual_h12 = spec
                .cond_first_given_second(u1, u2, 1e-12)
                .expect("h-function should evaluate");
            assert!(
                (actual_h12 - fixture.expected_cond_first_given_second[idx]).abs() < 1e-9,
                "{} {} h12 mismatch at row {idx}: left={actual_h12}, right={}",
                fixture.metadata.source_version,
                fixture.family,
                fixture.expected_cond_first_given_second[idx]
            );

            let actual_h21 = spec
                .cond_second_given_first(u1, u2, 1e-12)
                .expect("h-function should evaluate");
            assert!(
                (actual_h21 - fixture.expected_cond_second_given_first[idx]).abs() < 1e-9,
                "{} {} h21 mismatch at row {idx}: left={actual_h21}, right={}",
                fixture.metadata.source_version,
                fixture.family,
                fixture.expected_cond_second_given_first[idx]
            );

            let actual_inv_h12 = spec
                .inv_first_given_second(p, u2, 1e-12)
                .expect("inverse h-function should evaluate");
            assert!(
                (actual_inv_h12 - fixture.expected_inv_first_given_second[idx]).abs() < 1e-9,
                "{} {} hinv12 mismatch at row {idx}: left={actual_inv_h12}, right={}",
                fixture.metadata.source_version,
                fixture.family,
                fixture.expected_inv_first_given_second[idx]
            );

            let actual_inv_h21 = spec
                .inv_second_given_first(u1, p, 1e-12)
                .expect("inverse h-function should evaluate");
            assert!(
                (actual_inv_h21 - fixture.expected_inv_second_given_first[idx]).abs() < 1e-9,
                "{} {} hinv21 mismatch at row {idx}: left={actual_inv_h21}, right={}",
                fixture.metadata.source_version,
                fixture.family,
                fixture.expected_inv_second_given_first[idx]
            );
        }
    }
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/vinecopula/v2")
}

fn load_fixture<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = fixture_dir().join(name);
    let bytes = fs::read(path).expect("fixture should exist");
    serde_json::from_slice(&bytes).expect("fixture should deserialize")
}

fn fixture_spec(fixture: &PairFixture) -> PairCopulaSpec {
    let family = match fixture.family.as_str() {
        "gaussian" => PairCopulaFamily::Gaussian,
        "student_t" => PairCopulaFamily::StudentT,
        "clayton" => PairCopulaFamily::Clayton,
        "frank" => PairCopulaFamily::Frank,
        "gumbel" => PairCopulaFamily::Gumbel,
        "joe" => PairCopulaFamily::Joe,
        "bb1" => PairCopulaFamily::Bb1,
        "bb6" => PairCopulaFamily::Bb6,
        "bb7" => PairCopulaFamily::Bb7,
        "bb8" => PairCopulaFamily::Bb8,
        "tawn1" => PairCopulaFamily::Tawn1,
        "tawn2" => PairCopulaFamily::Tawn2,
        other => panic!("unexpected unrotated family fixture {other}"),
    };

    let params = match family {
        PairCopulaFamily::StudentT
        | PairCopulaFamily::Bb1
        | PairCopulaFamily::Bb6
        | PairCopulaFamily::Bb7
        | PairCopulaFamily::Bb8
        | PairCopulaFamily::Tawn1
        | PairCopulaFamily::Tawn2 => PairCopulaParams::Two(fixture.par, fixture.par2),
        _ => PairCopulaParams::One(fixture.par),
    };

    PairCopulaSpec {
        family,
        rotation: Rotation::R0,
        params,
    }
}
