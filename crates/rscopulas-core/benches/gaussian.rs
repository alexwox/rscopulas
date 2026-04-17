use std::{fs, hint::black_box, path::PathBuf};

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;

use rscopulas_core::{
    CopulaModel, EvalOptions, FitOptions, GaussianCopula, PseudoObs, SampleOptions,
};

#[derive(Debug, Deserialize)]
struct GaussianLogPdfFixture {
    correlation: Vec<Vec<f64>>,
    inputs: Vec<Vec<f64>>,
    expected_log_pdf: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct GaussianFitFixture {
    input_pobs: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct GaussianSampleSummaryFixture {
    correlation: Vec<Vec<f64>>,
    seed: u64,
    sample_size: usize,
}

fn gaussian_log_pdf_benchmark(criterion: &mut Criterion) {
    let fixture: GaussianLogPdfFixture = load_fixture("gaussian_log_pdf_d2_case01.json");
    let model = GaussianCopula::new(array2(&fixture.correlation))
        .expect("fixture correlation should be valid");
    let input = PseudoObs::new(array2(&fixture.inputs)).expect("fixture inputs should be valid");
    let options = EvalOptions::default();
    let baseline = model
        .log_pdf(&input, &options)
        .expect("log pdf should evaluate");
    assert_eq!(baseline.len(), fixture.expected_log_pdf.len());

    criterion.bench_function("gaussian_log_pdf_fixture_case01", |bench| {
        bench.iter(|| {
            model
                .log_pdf(black_box(&input), black_box(&options))
                .expect("log pdf should evaluate")
        });
    });
}

fn gaussian_fit_benchmark(criterion: &mut Criterion) {
    let fixture: GaussianFitFixture = load_fixture("gaussian_fit_d2_case01.json");
    let input =
        PseudoObs::new(array2(&fixture.input_pobs)).expect("fixture inputs should be valid");
    let options = FitOptions::default();

    criterion.bench_function("gaussian_fit_fixture_case01", |bench| {
        bench.iter(|| {
            GaussianCopula::fit(black_box(&input), black_box(&options))
                .expect("fit should evaluate")
        });
    });
}

fn gaussian_sample_benchmark(criterion: &mut Criterion) {
    let fixture: GaussianSampleSummaryFixture =
        load_fixture("gaussian_sample_summary_d2_case01.json");
    let model = GaussianCopula::new(array2(&fixture.correlation))
        .expect("fixture correlation should be valid");
    let options = SampleOptions::default();

    criterion.bench_function("gaussian_sample_fixture_case01", |bench| {
        bench.iter(|| {
            let mut rng = StdRng::seed_from_u64(fixture.seed);
            model
                .sample(
                    black_box(fixture.sample_size),
                    black_box(&mut rng),
                    black_box(&options),
                )
                .expect("sampling should evaluate")
        });
    });
}

criterion_group!(
    benches,
    gaussian_log_pdf_benchmark,
    gaussian_fit_benchmark,
    gaussian_sample_benchmark
);
criterion_main!(benches);

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
