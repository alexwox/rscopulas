#[path = "../examples/benchmark_support.rs"]
mod benchmark_support;

use std::hint::black_box;

use benchmark_support::{
    PairFixture, SingleFamilyFixture, VineFixture, default_vine_fit_options, fit_single_family,
    implementation_requested, load_json_fixture, load_manifest, pair_spec_from_fixture,
    repeat_pair_kernels, sample_seed, sample_size, single_family_fit_input, single_family_input,
    single_family_model, vine_fit_input, vine_input, vine_model_from_fixture, vine_sample_seed,
    vine_sample_size, workspace_root,
};
use criterion::{Criterion, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};
use rscopulas_core::{CopulaModel, Device, EvalOptions, ExecPolicy, FitOptions, SampleOptions, VineCopula};

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

fn serial_sample_options() -> SampleOptions {
    SampleOptions {
        exec: ExecPolicy::Force(Device::Cpu),
    }
}

fn benchmark_manifest() -> benchmark_support::BenchmarkManifest {
    let path = workspace_root().join("benchmarks/cases.json");
    load_manifest(path.to_str().expect("manifest path should be utf-8"))
}

fn single_family_log_pdf_benchmarks(criterion: &mut Criterion) {
    let options = serial_eval_options();
    for case in benchmark_manifest()
        .cases
        .into_iter()
        .filter(|case| {
            implementation_requested(case, "rust")
                && case.category == "single_family"
                && case.operation == "log_pdf"
        })
    {
        let family = case
            .family
            .clone()
            .expect("single-family cases should declare a family");
        let fixture: SingleFamilyFixture = load_json_fixture(&case.fixture);
        let model = single_family_model(&family, &fixture);
        let input = single_family_input(&fixture);
        let name = format!("cross_language_{}", case.id);
        criterion.bench_function(&name, |bench| {
            bench.iter(|| {
                model
                    .log_pdf(black_box(&input), black_box(&options))
                    .expect("single-family log_pdf should succeed")
            });
        });
    }
}

fn single_family_fit_benchmarks(criterion: &mut Criterion) {
    let options = serial_fit_options();
    for case in benchmark_manifest()
        .cases
        .into_iter()
        .filter(|case| {
            implementation_requested(case, "rust")
                && case.category == "single_family"
                && case.operation == "fit"
        })
    {
        let family = case
            .family
            .clone()
            .expect("single-family cases should declare a family");
        let fixture: SingleFamilyFixture = load_json_fixture(&case.fixture);
        let input = single_family_fit_input(&fixture);
        let name = format!("cross_language_{}", case.id);
        criterion.bench_function(&name, |bench| {
            bench.iter(|| {
                fit_single_family(black_box(&family), black_box(&input), black_box(&options))
                    .expect("single-family fit should succeed")
            });
        });
    }
}

fn single_family_sample_benchmarks(criterion: &mut Criterion) {
    let options = serial_sample_options();
    for case in benchmark_manifest()
        .cases
        .into_iter()
        .filter(|case| {
            implementation_requested(case, "rust")
                && case.category == "single_family"
                && case.operation == "sample"
        })
    {
        let family = case
            .family
            .clone()
            .expect("single-family cases should declare a family");
        let fixture: SingleFamilyFixture = load_json_fixture(&case.fixture);
        let model = single_family_model(&family, &fixture);
        let sample_n = sample_size(&fixture);
        let seed = sample_seed(&fixture);
        let name = format!("cross_language_{}", case.id);
        criterion.bench_function(&name, |bench| {
            bench.iter(|| {
                let mut rng = StdRng::seed_from_u64(seed);
                model
                    .sample(black_box(sample_n), black_box(&mut rng), black_box(&options))
                    .expect("single-family sample should succeed")
            });
        });
    }
}

fn pair_kernel_benchmarks(criterion: &mut Criterion) {
    for case in benchmark_manifest()
        .cases
        .into_iter()
        .filter(|case| implementation_requested(case, "rust") && case.category == "pair_copula")
    {
        let fixture: PairFixture = load_json_fixture(&case.fixture);
        let spec = pair_spec_from_fixture(&fixture);
        let name = format!("cross_language_{}", case.id);
        criterion.bench_function(&name, |bench| {
            bench.iter(|| {
                repeat_pair_kernels(black_box(&spec), black_box(&fixture), black_box(1e-12))
                    .expect("pair kernels should succeed")
            });
        });
    }
}

fn vine_log_pdf_benchmarks(criterion: &mut Criterion) {
    let options = serial_eval_options();
    for case in benchmark_manifest()
        .cases
        .into_iter()
        .filter(|case| {
            implementation_requested(case, "rust")
                && case.category == "vine"
                && case.operation == "log_pdf"
        })
    {
        let fixture: VineFixture = load_json_fixture(&case.fixture);
        let model = vine_model_from_fixture(&fixture).expect("vine fixture should build");
        let input = vine_input(&fixture);
        let name = format!("cross_language_{}", case.id);
        criterion.bench_function(&name, |bench| {
            bench.iter(|| {
                model
                    .log_pdf(black_box(&input), black_box(&options))
                    .expect("vine log_pdf should succeed")
            });
        });
    }
}

fn vine_sample_benchmarks(criterion: &mut Criterion) {
    let options = serial_sample_options();
    for case in benchmark_manifest()
        .cases
        .into_iter()
        .filter(|case| {
            implementation_requested(case, "rust")
                && case.category == "vine"
                && case.operation == "sample"
        })
    {
        let fixture: VineFixture = load_json_fixture(&case.fixture);
        let model = vine_model_from_fixture(&fixture).expect("vine fixture should build");
        let sample_n = vine_sample_size(&fixture);
        let seed = vine_sample_seed(&fixture);
        let name = format!("cross_language_{}", case.id);
        criterion.bench_function(&name, |bench| {
            bench.iter(|| {
                let mut rng = StdRng::seed_from_u64(seed);
                model
                    .sample(black_box(sample_n), black_box(&mut rng), black_box(&options))
                    .expect("vine sampling should succeed")
            });
        });
    }
}

fn vine_fit_benchmarks(criterion: &mut Criterion) {
    let options = default_vine_fit_options();
    for case in benchmark_manifest()
        .cases
        .into_iter()
        .filter(|case| {
            implementation_requested(case, "rust")
                && case.category == "vine"
                && case.operation == "fit"
        })
    {
        let fixture: VineFixture = load_json_fixture(&case.fixture);
        let input = vine_fit_input(&fixture);
        let name = format!("cross_language_{}", case.id);
        criterion.bench_function(&name, |bench| {
            bench.iter(|| {
                VineCopula::fit_r_vine(black_box(&input), black_box(&options))
                    .expect("vine fit should succeed")
            });
        });
    }
}

criterion_group!(
    benches,
    single_family_log_pdf_benchmarks,
    single_family_fit_benchmarks,
    single_family_sample_benchmarks,
    pair_kernel_benchmarks,
    vine_log_pdf_benchmarks,
    vine_sample_benchmarks,
    vine_fit_benchmarks
);
criterion_main!(benches);
