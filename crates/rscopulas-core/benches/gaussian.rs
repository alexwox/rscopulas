use std::{fs, hint::black_box, path::PathBuf};

use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;

use rscopulas::accel::is_device_available as accel_device_available;
use rscopulas::{
    ClaytonCopula, CopulaModel, Device, EvalOptions, ExecPolicy, FitOptions, FrankCopula,
    GaussianCopula, GumbelHougaardCopula, PairCopulaFamily, PseudoObs, SampleOptions,
    StudentTCopula, VineCopula, VineFitOptions, paircopula::fit_pair_copula,
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

#[derive(Debug, Deserialize)]
struct StudentTLogPdfFixture {
    correlation: Vec<Vec<f64>>,
    degrees_of_freedom: f64,
    inputs: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct ClaytonLogPdfFixture {
    theta: f64,
    inputs: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct FrankLogPdfFixture {
    theta: f64,
    inputs: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct GumbelLogPdfFixture {
    theta: f64,
    inputs: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct VineLogPdfFixture {
    order: Vec<usize>,
    correlation: Vec<Vec<f64>>,
    inputs: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct PairFixture {
    u1: Vec<f64>,
    u2: Vec<f64>,
}

fn gaussian_log_pdf_benchmark(criterion: &mut Criterion) {
    let fixture: GaussianLogPdfFixture = load_fixture("gaussian_log_pdf_d2_case01.json");
    let model = GaussianCopula::new(array2(&fixture.correlation))
        .expect("fixture correlation should be valid");
    let input = PseudoObs::new(array2(&fixture.inputs)).expect("fixture inputs should be valid");
    let serial = serial_eval_options();
    let auto = EvalOptions::default();
    let baseline = model
        .log_pdf(&input, &serial)
        .expect("log pdf should evaluate");
    assert_eq!(baseline.len(), fixture.expected_log_pdf.len());

    criterion.bench_function("gaussian_log_pdf_serial_fixture_case01", |bench| {
        bench.iter(|| {
            model
                .log_pdf(black_box(&input), black_box(&serial))
                .expect("log pdf should evaluate")
        });
    });
    criterion.bench_function("gaussian_log_pdf_auto_fixture_case01", |bench| {
        bench.iter(|| {
            model
                .log_pdf(black_box(&input), black_box(&auto))
                .expect("log pdf should evaluate")
        });
    });
}

fn gaussian_fit_benchmark(criterion: &mut Criterion) {
    let fixture: GaussianFitFixture = load_fixture("gaussian_fit_d2_case01.json");
    let input =
        PseudoObs::new(array2(&fixture.input_pobs)).expect("fixture inputs should be valid");
    let serial = serial_fit_options();
    let auto = FitOptions::default();

    criterion.bench_function("gaussian_fit_serial_fixture_case01", |bench| {
        bench.iter(|| {
            GaussianCopula::fit(black_box(&input), black_box(&serial)).expect("fit should evaluate")
        });
    });
    criterion.bench_function("gaussian_fit_auto_fixture_case01", |bench| {
        bench.iter(|| {
            GaussianCopula::fit(black_box(&input), black_box(&auto)).expect("fit should evaluate")
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

fn student_t_log_pdf_benchmark(criterion: &mut Criterion) {
    let fixture: StudentTLogPdfFixture = load_fixture("student_t_log_pdf_d2_case01.json");
    let model = StudentTCopula::new(array2(&fixture.correlation), fixture.degrees_of_freedom)
        .expect("fixture parameters should be valid");
    let input = PseudoObs::new(array2(&fixture.inputs)).expect("fixture inputs should be valid");
    let options = EvalOptions::default();

    criterion.bench_function("student_t_log_pdf_fixture_case01", |bench| {
        bench.iter(|| {
            model
                .log_pdf(black_box(&input), black_box(&options))
                .expect("log pdf should evaluate")
        });
    });
}

fn clayton_log_pdf_benchmark(criterion: &mut Criterion) {
    let fixture: ClaytonLogPdfFixture = load_fixture("clayton_log_pdf_d2_case01.json");
    let model = ClaytonCopula::new(2, fixture.theta).expect("fixture theta should be valid");
    let input = PseudoObs::new(array2(&fixture.inputs)).expect("fixture inputs should be valid");
    let options = EvalOptions::default();

    criterion.bench_function("clayton_log_pdf_fixture_case01", |bench| {
        bench.iter(|| {
            model
                .log_pdf(black_box(&input), black_box(&options))
                .expect("log pdf should evaluate")
        });
    });
}

fn frank_log_pdf_benchmark(criterion: &mut Criterion) {
    let fixture: FrankLogPdfFixture = load_fixture("frank_log_pdf_d2_case01.json");
    let model = FrankCopula::new(2, fixture.theta).expect("fixture theta should be valid");
    let input = PseudoObs::new(array2(&fixture.inputs)).expect("fixture inputs should be valid");
    let options = EvalOptions::default();

    criterion.bench_function("frank_log_pdf_fixture_case01", |bench| {
        bench.iter(|| {
            model
                .log_pdf(black_box(&input), black_box(&options))
                .expect("log pdf should evaluate")
        });
    });
}

fn gumbel_log_pdf_benchmark(criterion: &mut Criterion) {
    let fixture: GumbelLogPdfFixture = load_fixture("gumbel_log_pdf_d2_case01.json");
    let model = GumbelHougaardCopula::new(2, fixture.theta).expect("fixture theta should be valid");
    let input = PseudoObs::new(array2(&fixture.inputs)).expect("fixture inputs should be valid");
    let options = EvalOptions::default();

    criterion.bench_function("gumbel_log_pdf_fixture_case01", |bench| {
        bench.iter(|| {
            model
                .log_pdf(black_box(&input), black_box(&options))
                .expect("log pdf should evaluate")
        });
    });
}

fn gaussian_c_vine_log_pdf_benchmark(criterion: &mut Criterion) {
    let fixture: VineLogPdfFixture = load_vine_fixture("gaussian_c_vine_log_pdf_d4_case01.json");
    let model = VineCopula::gaussian_c_vine(fixture.order, array2(&fixture.correlation))
        .expect("fixture parameters should be valid");
    let input = PseudoObs::new(array2(&fixture.inputs)).expect("fixture inputs should be valid");
    let serial = serial_eval_options();
    let auto = EvalOptions::default();

    criterion.bench_function("gaussian_c_vine_log_pdf_serial_fixture_case01", |bench| {
        bench.iter(|| {
            model
                .log_pdf(black_box(&input), black_box(&serial))
                .expect("log pdf should evaluate")
        });
    });
    criterion.bench_function("gaussian_c_vine_log_pdf_auto_fixture_case01", |bench| {
        bench.iter(|| {
            model
                .log_pdf(black_box(&input), black_box(&auto))
                .expect("log pdf should evaluate")
        });
    });
    if accel_device_available(rscopulas::accel::Device::Metal) {
        let metal = metal_eval_options();
        criterion.bench_function("gaussian_c_vine_log_pdf_metal_fixture_case01", |bench| {
            bench.iter(|| {
                model
                    .log_pdf(black_box(&input), black_box(&metal))
                    .expect("log pdf should evaluate")
            });
        });
    }
    if accel_device_available(rscopulas::accel::Device::Cuda(0)) {
        let cuda = cuda_eval_options();
        criterion.bench_function("gaussian_c_vine_log_pdf_cuda_fixture_case01", |bench| {
            bench.iter(|| {
                model
                    .log_pdf(black_box(&input), black_box(&cuda))
                    .expect("log pdf should evaluate")
            });
        });
    }
}

fn gaussian_d_vine_log_pdf_benchmark(criterion: &mut Criterion) {
    let fixture: VineLogPdfFixture = load_vine_fixture("gaussian_d_vine_log_pdf_d4_case01.json");
    let model = VineCopula::gaussian_d_vine(fixture.order, array2(&fixture.correlation))
        .expect("fixture parameters should be valid");
    let input = PseudoObs::new(array2(&fixture.inputs)).expect("fixture inputs should be valid");
    let serial = serial_eval_options();
    let auto = EvalOptions::default();

    criterion.bench_function("gaussian_d_vine_log_pdf_serial_fixture_case01", |bench| {
        bench.iter(|| {
            model
                .log_pdf(black_box(&input), black_box(&serial))
                .expect("log pdf should evaluate")
        });
    });
    criterion.bench_function("gaussian_d_vine_log_pdf_auto_fixture_case01", |bench| {
        bench.iter(|| {
            model
                .log_pdf(black_box(&input), black_box(&auto))
                .expect("log pdf should evaluate")
        });
    });
    if accel_device_available(rscopulas::accel::Device::Metal) {
        let metal = metal_eval_options();
        criterion.bench_function("gaussian_d_vine_log_pdf_metal_fixture_case01", |bench| {
            bench.iter(|| {
                model
                    .log_pdf(black_box(&input), black_box(&metal))
                    .expect("log pdf should evaluate")
            });
        });
    }
    if accel_device_available(rscopulas::accel::Device::Cuda(0)) {
        let cuda = cuda_eval_options();
        criterion.bench_function("gaussian_d_vine_log_pdf_cuda_fixture_case01", |bench| {
            bench.iter(|| {
                model
                    .log_pdf(black_box(&input), black_box(&cuda))
                    .expect("log pdf should evaluate")
            });
        });
    }
}

fn gaussian_pair_fit_backend_benchmark(criterion: &mut Criterion) {
    let fixture: PairFixture = load_vine_fixture("pair_gaussian_case01.json");
    let serial = gaussian_pair_fit_options(ExecPolicy::Force(Device::Cpu));

    criterion.bench_function("gaussian_pair_fit_serial_fixture_case01", |bench| {
        bench.iter(|| {
            fit_pair_copula(
                black_box(&fixture.u1),
                black_box(&fixture.u2),
                black_box(&serial),
            )
            .expect("pair fit should evaluate")
        });
    });
    if accel_device_available(rscopulas::accel::Device::Metal) {
        let metal = gaussian_pair_fit_options(ExecPolicy::Force(Device::Metal));
        criterion.bench_function("gaussian_pair_fit_metal_fixture_case01", |bench| {
            bench.iter(|| {
                fit_pair_copula(
                    black_box(&fixture.u1),
                    black_box(&fixture.u2),
                    black_box(&metal),
                )
                .expect("pair fit should evaluate")
            });
        });
    }
    if accel_device_available(rscopulas::accel::Device::Cuda(0)) {
        let cuda = gaussian_pair_fit_options(ExecPolicy::Force(Device::Cuda(0)));
        criterion.bench_function("gaussian_pair_fit_cuda_fixture_case01", |bench| {
            bench.iter(|| {
                fit_pair_copula(
                    black_box(&fixture.u1),
                    black_box(&fixture.u2),
                    black_box(&cuda),
                )
                .expect("pair fit should evaluate")
            });
        });
    }
}

criterion_group!(
    benches,
    gaussian_log_pdf_benchmark,
    gaussian_fit_benchmark,
    gaussian_sample_benchmark,
    student_t_log_pdf_benchmark,
    clayton_log_pdf_benchmark,
    frank_log_pdf_benchmark,
    gumbel_log_pdf_benchmark,
    gaussian_pair_fit_backend_benchmark,
    gaussian_c_vine_log_pdf_benchmark,
    gaussian_d_vine_log_pdf_benchmark
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

fn vine_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/vinecopula/v2")
}

fn load_vine_fixture<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = vine_fixture_dir().join(name);
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

fn serial_eval_options() -> EvalOptions {
    EvalOptions {
        exec: ExecPolicy::Force(Device::Cpu),
        ..EvalOptions::default()
    }
}

fn serial_fit_options() -> FitOptions {
    FitOptions {
        exec: ExecPolicy::Force(Device::Cpu),
        ..FitOptions::default()
    }
}

fn metal_eval_options() -> EvalOptions {
    EvalOptions {
        exec: ExecPolicy::Force(Device::Metal),
        ..EvalOptions::default()
    }
}

fn cuda_eval_options() -> EvalOptions {
    EvalOptions {
        exec: ExecPolicy::Force(Device::Cuda(0)),
        ..EvalOptions::default()
    }
}

fn gaussian_pair_fit_options(exec: ExecPolicy) -> VineFitOptions {
    VineFitOptions {
        base: FitOptions {
            exec,
            ..FitOptions::default()
        },
        family_set: vec![PairCopulaFamily::Gaussian],
        include_rotations: false,
        ..VineFitOptions::default()
    }
}
