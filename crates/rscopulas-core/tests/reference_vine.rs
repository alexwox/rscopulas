use std::{fs, path::PathBuf};

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;

use rscopulas_core::{
    CopulaModel, EvalOptions, PseudoObs, SampleOptions, VineCopula, VineStructureKind,
};

#[derive(Debug, Deserialize)]
struct Metadata {
    source_package: String,
    source_version: String,
}

#[derive(Debug, Deserialize)]
struct VineLogPdfFixture {
    metadata: Metadata,
    structure: String,
    order: Vec<usize>,
    correlation: Vec<Vec<f64>>,
    pair_parameters: Vec<f64>,
    inputs: Vec<Vec<f64>>,
    expected_log_pdf: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct VineSampleSummaryFixture {
    metadata: Metadata,
    structure: String,
    order: Vec<usize>,
    correlation: Vec<Vec<f64>>,
    pair_parameters: Vec<f64>,
    seed: u64,
    sample_size: usize,
    expected_mean: Vec<f64>,
    expected_kendall_tau: Vec<Vec<f64>>,
}

#[test]
fn gaussian_c_vine_matches_vinecopula_fixture() {
    let fixture: VineLogPdfFixture = load_fixture("gaussian_c_vine_log_pdf_d4_case01.json");
    assert_eq!(fixture.metadata.source_package, "VineCopula");
    assert_eq!(fixture.structure, "C");
    let model = VineCopula::gaussian_c_vine(fixture.order.clone(), array2(&fixture.correlation))
        .expect("fixture parameters should be valid");
    assert_eq!(model.structure(), VineStructureKind::C);
    assert_eq!(model.order(), fixture.order.as_slice());

    for (actual, expected) in model
        .pair_parameters()
        .iter()
        .zip(fixture.pair_parameters.iter())
    {
        assert!((actual - expected).abs() < 1e-10);
    }

    let input = PseudoObs::new(array2(&fixture.inputs)).expect("fixture inputs should be valid");
    let actual = model
        .log_pdf(&input, &EvalOptions::default())
        .expect("log pdf should evaluate");

    for (idx, (left, right)) in actual
        .iter()
        .zip(fixture.expected_log_pdf.iter())
        .enumerate()
    {
        assert!(
            (left - right).abs() < 2e-10,
            "fixture {} mismatch at row {idx}: left={left}, right={right}",
            fixture.metadata.source_version
        );
    }
}

#[test]
fn gaussian_d_vine_matches_vinecopula_fixture() {
    let fixture: VineLogPdfFixture = load_fixture("gaussian_d_vine_log_pdf_d4_case01.json");
    assert_eq!(fixture.metadata.source_package, "VineCopula");
    assert_eq!(fixture.structure, "D");
    let model = VineCopula::gaussian_d_vine(fixture.order.clone(), array2(&fixture.correlation))
        .expect("fixture parameters should be valid");
    assert_eq!(model.structure(), VineStructureKind::D);
    assert_eq!(model.order(), fixture.order.as_slice());

    for (actual, expected) in model
        .pair_parameters()
        .iter()
        .zip(fixture.pair_parameters.iter())
    {
        assert!((actual - expected).abs() < 1e-10);
    }

    let input = PseudoObs::new(array2(&fixture.inputs)).expect("fixture inputs should be valid");
    let actual = model
        .log_pdf(&input, &EvalOptions::default())
        .expect("log pdf should evaluate");

    for (idx, (left, right)) in actual
        .iter()
        .zip(fixture.expected_log_pdf.iter())
        .enumerate()
    {
        assert!(
            (left - right).abs() < 1e-10,
            "fixture {} mismatch at row {idx}: left={left}, right={right}",
            fixture.metadata.source_version
        );
    }
}

#[test]
fn gaussian_c_vine_sample_statistics_match_fixture() {
    let fixture: VineSampleSummaryFixture =
        load_fixture("gaussian_c_vine_sample_summary_d4_case01.json");
    assert_eq!(fixture.metadata.source_package, "VineCopula");
    assert_eq!(fixture.structure, "C");
    let model = VineCopula::gaussian_c_vine(fixture.order.clone(), array2(&fixture.correlation))
        .expect("fixture parameters should be valid");
    let mut rng = StdRng::seed_from_u64(fixture.seed);

    let samples = model
        .sample(fixture.sample_size, &mut rng, &SampleOptions)
        .expect("sampling should succeed");
    let sample_obs = PseudoObs::new(samples).expect("sample should be valid");
    let means = column_means(&sample_obs);
    let tau = rscopulas_core::stats::kendall_tau_matrix(&sample_obs);

    for (actual, expected) in model
        .pair_parameters()
        .iter()
        .zip(fixture.pair_parameters.iter())
    {
        assert!((actual - expected).abs() < 1e-10);
    }

    for (idx, (actual, expected)) in means.iter().zip(fixture.expected_mean.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1.2e-2,
            "mean mismatch at column {idx}: left={actual}, right={expected}"
        );
    }

    for row in 0..tau.nrows() {
        for col in 0..tau.ncols() {
            let expected = fixture.expected_kendall_tau[row][col];
            assert!((tau[(row, col)] - expected).abs() < 2.5e-2);
        }
    }
}

#[test]
fn gaussian_d_vine_sample_statistics_match_fixture() {
    let fixture: VineSampleSummaryFixture =
        load_fixture("gaussian_d_vine_sample_summary_d4_case01.json");
    assert_eq!(fixture.metadata.source_package, "VineCopula");
    assert_eq!(fixture.structure, "D");
    let model = VineCopula::gaussian_d_vine(fixture.order.clone(), array2(&fixture.correlation))
        .expect("fixture parameters should be valid");
    let mut rng = StdRng::seed_from_u64(fixture.seed);

    let samples = model
        .sample(fixture.sample_size, &mut rng, &SampleOptions)
        .expect("sampling should succeed");
    let sample_obs = PseudoObs::new(samples).expect("sample should be valid");
    let means = column_means(&sample_obs);
    let tau = rscopulas_core::stats::kendall_tau_matrix(&sample_obs);

    for (actual, expected) in model
        .pair_parameters()
        .iter()
        .zip(fixture.pair_parameters.iter())
    {
        assert!((actual - expected).abs() < 1e-10);
    }

    for (idx, (actual, expected)) in means.iter().zip(fixture.expected_mean.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-2,
            "mean mismatch at column {idx}: left={actual}, right={expected}"
        );
    }

    for row in 0..tau.nrows() {
        for col in 0..tau.ncols() {
            let expected = fixture.expected_kendall_tau[row][col];
            assert!((tau[(row, col)] - expected).abs() < 2e-2);
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

fn array2(rows: &[Vec<f64>]) -> Array2<f64> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let data = rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    Array2::from_shape_vec((nrows, ncols), data).expect("rows should form a matrix")
}

fn column_means(data: &PseudoObs) -> Vec<f64> {
    (0..data.dim())
        .map(|col| data.as_view().column(col).iter().sum::<f64>() / data.n_obs() as f64)
        .collect()
}
