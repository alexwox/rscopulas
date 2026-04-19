mod benchmark_support;

use std::{env, fs, path::PathBuf, time::Instant};

use benchmark_support::{
    BenchmarkCase, PairFixture, SingleFamilyFixture, VineFixture, default_vine_fit_options,
    fit_single_family, implementation_requested, load_json_fixture, load_manifest,
    pair_spec_from_fixture, repeat_pair_kernels, sample_seed, sample_size, single_family_fit_input,
    single_family_input, single_family_model, vine_fit_input, vine_input, vine_model_from_fixture,
    vine_sample_seed, vine_sample_size,
};
use rand::{SeedableRng, rngs::StdRng};
use rscopulas::{
    CopulaModel, Device, EvalOptions, ExecPolicy, FitOptions, SampleOptions, VineCopula,
};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct RunnerPayload {
    implementation: &'static str,
    runner: &'static str,
    environment: RunnerEnvironment,
    results: Vec<ResultRow>,
}

#[derive(Debug, Serialize)]
struct RunnerEnvironment {
    crate_name: &'static str,
    crate_version: &'static str,
    profile: &'static str,
}

#[derive(Debug, Serialize)]
struct ResultRow {
    case_id: String,
    category: String,
    family: Option<String>,
    operation: String,
    structure: Option<String>,
    fixture: String,
    iterations: usize,
    total_ns: u128,
    mean_ns: u128,
    observations: Option<usize>,
    sample_size: Option<usize>,
    dim: Option<usize>,
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

fn serial_sample_options() -> SampleOptions {
    SampleOptions {
        exec: ExecPolicy::Force(Device::Cpu),
    }
}

fn measure_iterations<F>(iterations: usize, mut callback: F) -> (u128, u128)
where
    F: FnMut(),
{
    callback();
    let start = Instant::now();
    for _ in 0..iterations {
        callback();
    }
    let total_ns = start.elapsed().as_nanos();
    (total_ns, total_ns / iterations as u128)
}

fn run_single_family(case: &BenchmarkCase) -> ResultRow {
    let family = case
        .family
        .as_deref()
        .expect("single-family cases should declare a family");
    let fixture: SingleFamilyFixture = load_json_fixture(&case.fixture);
    match case.operation.as_str() {
        "log_pdf" => {
            let model = single_family_model(family, &fixture);
            let input = single_family_input(&fixture);
            let observations = input.n_obs();
            let dim = model.dim();
            let options = serial_eval_options();
            let (total_ns, mean_ns) = measure_iterations(case.iterations, || {
                model
                    .log_pdf(&input, &options)
                    .expect("log-pdf benchmark should succeed");
            });
            ResultRow {
                case_id: case.id.clone(),
                category: case.category.clone(),
                family: case.family.clone(),
                operation: case.operation.clone(),
                structure: case.structure.clone(),
                fixture: case.fixture.clone(),
                iterations: case.iterations,
                total_ns,
                mean_ns,
                observations: Some(observations),
                sample_size: None,
                dim: Some(dim),
            }
        }
        "fit" => {
            let input = single_family_fit_input(&fixture);
            let observations = input.n_obs();
            let dim = input.dim();
            let (total_ns, mean_ns) = measure_iterations(case.iterations, || {
                fit_single_family(family, &input, &serial_fit_options())
                    .expect("fit should succeed");
            });
            ResultRow {
                case_id: case.id.clone(),
                category: case.category.clone(),
                family: case.family.clone(),
                operation: case.operation.clone(),
                structure: case.structure.clone(),
                fixture: case.fixture.clone(),
                iterations: case.iterations,
                total_ns,
                mean_ns,
                observations: Some(observations),
                sample_size: None,
                dim: Some(dim),
            }
        }
        "sample" => {
            let model = single_family_model(family, &fixture);
            let sample_n = sample_size(&fixture);
            let seed = sample_seed(&fixture);
            let dim = model.dim();
            let options = serial_sample_options();
            let (total_ns, mean_ns) = measure_iterations(case.iterations, || {
                let mut rng = StdRng::seed_from_u64(seed);
                model
                    .sample(sample_n, &mut rng, &options)
                    .expect("sampling should succeed");
            });
            ResultRow {
                case_id: case.id.clone(),
                category: case.category.clone(),
                family: case.family.clone(),
                operation: case.operation.clone(),
                structure: case.structure.clone(),
                fixture: case.fixture.clone(),
                iterations: case.iterations,
                total_ns,
                mean_ns,
                observations: None,
                sample_size: Some(sample_n),
                dim: Some(dim),
            }
        }
        other => panic!("unsupported single-family operation: {other}"),
    }
}

fn run_pair(case: &BenchmarkCase) -> ResultRow {
    let fixture: PairFixture = load_json_fixture(&case.fixture);
    let spec = pair_spec_from_fixture(&fixture);
    let observations = fixture.u1.len();
    let (total_ns, mean_ns) = measure_iterations(case.iterations, || {
        repeat_pair_kernels(&spec, &fixture, 1e-12).expect("pair kernels should evaluate");
    });
    ResultRow {
        case_id: case.id.clone(),
        category: case.category.clone(),
        family: case.family.clone(),
        operation: case.operation.clone(),
        structure: case.structure.clone(),
        fixture: case.fixture.clone(),
        iterations: case.iterations,
        total_ns,
        mean_ns,
        observations: Some(observations),
        sample_size: None,
        dim: Some(2),
    }
}

fn run_vine(case: &BenchmarkCase) -> ResultRow {
    let fixture: VineFixture = load_json_fixture(&case.fixture);
    match case.operation.as_str() {
        "log_pdf" => {
            let model = vine_model_from_fixture(&fixture).expect("vine fixture should build");
            let input = vine_input(&fixture);
            let observations = input.n_obs();
            let dim = input.dim();
            let options = serial_eval_options();
            let (total_ns, mean_ns) = measure_iterations(case.iterations, || {
                model
                    .log_pdf(&input, &options)
                    .expect("vine log-pdf benchmark should succeed");
            });
            ResultRow {
                case_id: case.id.clone(),
                category: case.category.clone(),
                family: case.family.clone(),
                operation: case.operation.clone(),
                structure: case.structure.clone(),
                fixture: case.fixture.clone(),
                iterations: case.iterations,
                total_ns,
                mean_ns,
                observations: Some(observations),
                sample_size: None,
                dim: Some(dim),
            }
        }
        "sample" => {
            let model = vine_model_from_fixture(&fixture).expect("vine fixture should build");
            let sample_n = vine_sample_size(&fixture);
            let seed = vine_sample_seed(&fixture);
            let dim = model.dim();
            let options = serial_sample_options();
            let (total_ns, mean_ns) = measure_iterations(case.iterations, || {
                let mut rng = StdRng::seed_from_u64(seed);
                model
                    .sample(sample_n, &mut rng, &options)
                    .expect("vine sampling should succeed");
            });
            ResultRow {
                case_id: case.id.clone(),
                category: case.category.clone(),
                family: case.family.clone(),
                operation: case.operation.clone(),
                structure: case.structure.clone(),
                fixture: case.fixture.clone(),
                iterations: case.iterations,
                total_ns,
                mean_ns,
                observations: None,
                sample_size: Some(sample_n),
                dim: Some(dim),
            }
        }
        "fit" => {
            let input = vine_fit_input(&fixture);
            let observations = input.n_obs();
            let dim = input.dim();
            let options = default_vine_fit_options();
            let (total_ns, mean_ns) = measure_iterations(case.iterations, || {
                VineCopula::fit_r_vine(&input, &options).expect("vine fitting should succeed");
            });
            ResultRow {
                case_id: case.id.clone(),
                category: case.category.clone(),
                family: case.family.clone(),
                operation: case.operation.clone(),
                structure: case.structure.clone(),
                fixture: case.fixture.clone(),
                iterations: case.iterations,
                total_ns,
                mean_ns,
                observations: Some(observations),
                sample_size: None,
                dim: Some(dim),
            }
        }
        other => panic!("unsupported vine operation: {other}"),
    }
}

fn run_case(case: &BenchmarkCase) -> ResultRow {
    match case.category.as_str() {
        "single_family" => run_single_family(case),
        "pair_copula" => run_pair(case),
        "vine" => run_vine(case),
        other => panic!("unsupported benchmark category: {other}"),
    }
}

fn parse_args() -> (String, String) {
    let mut args = env::args().skip(1);
    let mut cases_path = None;
    let mut output_path = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--cases" => cases_path = args.next(),
            "--output" => output_path = args.next(),
            other => panic!("unsupported argument: {other}"),
        }
    }
    (
        cases_path.expect("--cases is required"),
        output_path.expect("--output is required"),
    )
}

fn main() {
    let (cases_path, output_path) = parse_args();
    let manifest = load_manifest(&cases_path);
    assert_eq!(
        manifest.schema_version, 1,
        "unsupported benchmark schema version"
    );
    let results = manifest
        .cases
        .iter()
        .filter(|case| implementation_requested(case, "rust"))
        .map(run_case)
        .collect::<Vec<_>>();
    let payload = RunnerPayload {
        implementation: "rust",
        runner: "crates/rscopulas-core/examples/benchmark_runner.rs",
        environment: RunnerEnvironment {
            crate_name: env!("CARGO_PKG_NAME"),
            crate_version: env!("CARGO_PKG_VERSION"),
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
        },
        results,
    };
    let output_path = PathBuf::from(output_path);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).expect("runner output parent should exist");
    }
    fs::write(
        output_path,
        serde_json::to_vec_pretty(&payload).expect("runner payload should serialize"),
    )
    .expect("runner output should be written");
}
