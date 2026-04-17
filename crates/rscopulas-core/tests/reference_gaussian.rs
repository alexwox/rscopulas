use std::{fs, path::PathBuf};

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;

use rscopulas_core::{
    CopulaModel, EvalOptions, FitOptions, GaussianCopula, PseudoObs, SampleOptions,
};

#[derive(Debug, Deserialize)]
struct Metadata {
    source_package: String,
    source_version: String,
}

#[derive(Debug, Deserialize)]
struct GaussianLogPdfFixture {
    metadata: Metadata,
    correlation: Vec<Vec<f64>>,
    inputs: Vec<Vec<f64>>,
    expected_log_pdf: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct GaussianFitFixture {
    metadata: Metadata,
    input_pobs: Vec<Vec<f64>>,
    expected_correlation: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct GaussianSampleSummaryFixture {
    metadata: Metadata,
    correlation: Vec<Vec<f64>>,
    seed: u64,
    sample_size: usize,
    expected_mean: Vec<f64>,
    expected_kendall_tau: Vec<Vec<f64>>,
}

#[test]
fn gaussian_log_pdf_matches_r_fixture() {
    let fixture: GaussianLogPdfFixture = load_fixture("gaussian_log_pdf_d2_case01.json");
    assert_eq!(fixture.metadata.source_package, "copula");
    let model = GaussianCopula::new(array2(&fixture.correlation))
        .expect("fixture correlation should be valid");
    let input = PseudoObs::new(array2(&fixture.inputs)).expect("fixture inputs should be valid");

    let actual = model
        .log_pdf(&input, &EvalOptions::default())
        .expect("log pdf should evaluate");

    assert_eq!(actual.len(), fixture.expected_log_pdf.len());
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
fn gaussian_fit_matches_r_itau_fixture() {
    let fixture: GaussianFitFixture = load_fixture("gaussian_fit_d2_case01.json");
    assert_eq!(fixture.metadata.source_package, "copula");
    let input =
        PseudoObs::new(array2(&fixture.input_pobs)).expect("fixture inputs should be valid");

    let fit = GaussianCopula::fit(&input, &FitOptions::default()).expect("fit should succeed");
    let expected = array2(&fixture.expected_correlation);

    for ((row, col), expected_value) in expected.indexed_iter() {
        let actual = fit.model.correlation()[(row, col)];
        assert!(
            (actual - expected_value).abs() < 1e-10,
            "correlation mismatch at ({row}, {col}): left={actual}, right={expected_value}"
        );
    }

    assert!(fit.diagnostics.converged);
}

#[test]
fn gaussian_sample_statistics_match_r_fixture() {
    let fixture: GaussianSampleSummaryFixture =
        load_fixture("gaussian_sample_summary_d2_case01.json");
    assert_eq!(fixture.metadata.source_package, "copula");
    let model = GaussianCopula::new(array2(&fixture.correlation))
        .expect("fixture correlation should be valid");
    let mut rng = StdRng::seed_from_u64(fixture.seed);

    let samples = model
        .sample(fixture.sample_size, &mut rng, &SampleOptions::default())
        .expect("sampling should succeed");
    let sample_obs = PseudoObs::new(samples).expect("generated sample should stay inside (0,1)");
    let means = column_means(&sample_obs);
    let tau = rscopulas_core::stats::kendall_tau_matrix(&sample_obs);

    for (idx, (actual, expected)) in means.iter().zip(fixture.expected_mean.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 5e-3,
            "mean mismatch at column {idx}: left={actual}, right={expected}"
        );
    }

    for row in 0..tau.nrows() {
        for col in 0..tau.ncols() {
            let expected = fixture.expected_kendall_tau[row][col];
            assert!(
                (tau[(row, col)] - expected).abs() < 2e-2,
                "kendall tau mismatch at ({row}, {col}): left={}, right={expected}",
                tau[(row, col)]
            );
        }
    }
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/r-copula/v1_1_3")
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
