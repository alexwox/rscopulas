use std::{fs, path::PathBuf};

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;

use rscopulas_core::{
    CopulaError, CopulaModel, Device, EvalOptions, ExecPolicy, FitOptions, GaussianCopula,
    PairCopulaFamily, PairCopulaParams, PairCopulaSpec, PseudoObs, Rotation, SampleOptions,
    StudentTCopula, VineCopula, VineEdge, VineFitOptions, VineStructureKind, VineTree,
    paircopula::fit_pair_copula,
    stats::try_kendall_tau_matrix,
};

#[derive(Debug, Deserialize)]
struct PairFixture {
    u1: Vec<f64>,
    u2: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct GaussianLogPdfFixture {
    correlation: Vec<Vec<f64>>,
    inputs: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct StudentTLogPdfFixture {
    correlation: Vec<Vec<f64>>,
    degrees_of_freedom: f64,
    inputs: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct GaussianFitFixture {
    input_pobs: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct VineLogPdfFixture {
    order: Vec<usize>,
    correlation: Vec<Vec<f64>>,
    inputs: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct EdgeFixture {
    conditioned: [usize; 2],
    conditioning: Vec<usize>,
    family: String,
    rotation: String,
    params: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct TreeFixture {
    level: usize,
    edges: Vec<EdgeFixture>,
}

#[derive(Debug, Deserialize)]
struct RVineLogPdfFixture {
    truncation_level: Option<usize>,
    trees: Vec<TreeFixture>,
    inputs: Vec<Vec<f64>>,
}

#[test]
fn gaussian_log_pdf_auto_matches_forced_cpu() {
    let fixture: GaussianLogPdfFixture = load_copula_fixture("gaussian_log_pdf_d2_case01.json");
    let model = GaussianCopula::new(array2(&fixture.correlation)).expect("correlation should be valid");
    let data = PseudoObs::new(array2(&fixture.inputs)).expect("inputs should be valid");

    let auto = model
        .log_pdf(&data, &EvalOptions::default())
        .expect("auto log pdf should evaluate");
    let serial = model
        .log_pdf(&data, &serial_eval_options())
        .expect("serial log pdf should evaluate");

    assert_eq!(auto, serial);
}

#[test]
fn student_t_log_pdf_auto_matches_forced_cpu() {
    let fixture: StudentTLogPdfFixture = load_copula_fixture("student_t_log_pdf_d2_case01.json");
    let model = StudentTCopula::new(array2(&fixture.correlation), fixture.degrees_of_freedom)
        .expect("parameters should be valid");
    let data = PseudoObs::new(array2(&fixture.inputs)).expect("inputs should be valid");

    let auto = model
        .log_pdf(&data, &EvalOptions::default())
        .expect("auto log pdf should evaluate");
    let serial = model
        .log_pdf(&data, &serial_eval_options())
        .expect("serial log pdf should evaluate");

    assert_eq!(auto, serial);
}

#[test]
fn gaussian_fit_auto_matches_forced_cpu() {
    let fixture: GaussianFitFixture = load_copula_fixture("gaussian_fit_d2_case01.json");
    let data = PseudoObs::new(array2(&fixture.input_pobs)).expect("input should be valid");

    let auto = GaussianCopula::fit(&data, &FitOptions::default()).expect("auto fit should succeed");
    let serial =
        GaussianCopula::fit(&data, &serial_fit_options()).expect("serial fit should succeed");

    assert_eq!(auto.model.correlation(), serial.model.correlation());
    assert_eq!(auto.diagnostics.loglik, serial.diagnostics.loglik);
}

#[test]
fn kendall_tau_auto_matches_forced_cpu() {
    let fixture: GaussianFitFixture = load_copula_fixture("gaussian_fit_d2_case01.json");
    let data = PseudoObs::new(array2(&fixture.input_pobs)).expect("input should be valid");

    let auto =
        try_kendall_tau_matrix(&data, ExecPolicy::Auto).expect("auto tau matrix should succeed");
    let serial = try_kendall_tau_matrix(&data, ExecPolicy::Force(Device::Cpu))
        .expect("serial tau matrix should succeed");

    assert_eq!(auto, serial);
}

#[test]
fn pair_fit_auto_matches_forced_cpu_for_base_and_rotation() {
    for fixture_name in ["pair_clayton_case01.json", "pair_clayton_rot90_case01.json"] {
        let fixture: PairFixture = load_vine_fixture(fixture_name);
        let auto = fit_pair_copula(&fixture.u1, &fixture.u2, &pair_fit_options(ExecPolicy::Auto))
            .expect("auto pair fit should succeed");
        let serial = fit_pair_copula(
            &fixture.u1,
            &fixture.u2,
            &pair_fit_options(ExecPolicy::Force(Device::Cpu)),
        )
        .expect("serial pair fit should succeed");

        assert_eq!(auto.spec, serial.spec);
        assert_eq!(auto.cond_on_first, serial.cond_on_first);
        assert_eq!(auto.cond_on_second, serial.cond_on_second);
        assert!((auto.loglik - serial.loglik).abs() < 1e-12);
    }
}

#[test]
fn gaussian_c_and_d_vine_auto_match_forced_cpu() {
    for fixture_name in [
        ("gaussian_c_vine_log_pdf_d4_case01.json", VineStructureKind::C),
        ("gaussian_d_vine_log_pdf_d4_case01.json", VineStructureKind::D),
    ] {
        let fixture: VineLogPdfFixture = load_vine_fixture(fixture_name.0);
        let model = match fixture_name.1 {
            VineStructureKind::C => {
                VineCopula::gaussian_c_vine(fixture.order.clone(), array2(&fixture.correlation))
            }
            VineStructureKind::D => {
                VineCopula::gaussian_d_vine(fixture.order.clone(), array2(&fixture.correlation))
            }
            VineStructureKind::R => unreachable!("only C and D vines are exercised here"),
        }
        .expect("vine model should build");
        let data = PseudoObs::new(array2(&fixture.inputs)).expect("inputs should be valid");

        let auto = model
            .log_pdf(&data, &EvalOptions::default())
            .expect("auto vine log pdf should evaluate");
        let serial = model
            .log_pdf(&data, &serial_eval_options())
            .expect("serial vine log pdf should evaluate");

        assert_eq!(auto, serial);
    }
}

#[test]
fn mixed_r_vine_auto_matches_forced_cpu() {
    let fixture: RVineLogPdfFixture = load_vine_fixture("mixed_r_vine_log_pdf_d5_case01.json");
    let model = build_r_vine_model(&fixture.trees, fixture.truncation_level);
    let data = PseudoObs::new(array2(&fixture.inputs)).expect("inputs should be valid");

    let auto = model
        .log_pdf(&data, &EvalOptions::default())
        .expect("auto log pdf should evaluate");
    let serial = model
        .log_pdf(&data, &serial_eval_options())
        .expect("serial log pdf should evaluate");

    assert_eq!(auto, serial);
}

#[test]
fn unsupported_accelerator_requests_surface_backend_errors() {
    let fixture: GaussianLogPdfFixture = load_copula_fixture("gaussian_log_pdf_d2_case01.json");
    let model = GaussianCopula::new(array2(&fixture.correlation)).expect("correlation should be valid");
    let data = PseudoObs::new(array2(&fixture.inputs)).expect("inputs should be valid");

    for exec in [
        ExecPolicy::Force(Device::Cuda(0)),
        ExecPolicy::Force(Device::Metal),
    ] {
        let error = model
            .log_pdf(
                &data,
                &EvalOptions {
                    exec,
                    ..EvalOptions::default()
                },
            )
            .expect_err("unsupported accelerator request should fail");
        assert!(matches!(error, CopulaError::Backend(_)));
    }
}

#[test]
fn sample_force_cpu_matches_auto_for_gaussian() {
    let fixture: GaussianLogPdfFixture = load_copula_fixture("gaussian_log_pdf_d2_case01.json");
    let model = GaussianCopula::new(array2(&fixture.correlation)).expect("correlation should be valid");
    let mut auto_rng = StdRng::seed_from_u64(1234);
    let mut serial_rng = StdRng::seed_from_u64(1234);

    let auto = model
        .sample(128, &mut auto_rng, &SampleOptions::default())
        .expect("auto sampling should succeed");
    let serial = model
        .sample(
            128,
            &mut serial_rng,
            &SampleOptions {
                exec: ExecPolicy::Force(Device::Cpu),
            },
        )
        .expect("serial sampling should succeed");

    assert_eq!(auto, serial);
}

fn pair_fit_options(exec: ExecPolicy) -> VineFitOptions {
    VineFitOptions {
        base: FitOptions {
            exec,
            ..FitOptions::default()
        },
        family_set: vec![
            PairCopulaFamily::Independence,
            PairCopulaFamily::Gaussian,
            PairCopulaFamily::StudentT,
            PairCopulaFamily::Clayton,
            PairCopulaFamily::Frank,
            PairCopulaFamily::Gumbel,
        ],
        include_rotations: true,
        ..VineFitOptions::default()
    }
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

fn build_r_vine_model(trees: &[TreeFixture], truncation_level: Option<usize>) -> VineCopula {
    let trees = trees
        .iter()
        .map(|tree| VineTree {
            level: tree.level,
            edges: tree
                .edges
                .iter()
                .map(|edge| VineEdge {
                    tree: tree.level,
                    conditioned: (edge.conditioned[0], edge.conditioned[1]),
                    conditioning: edge.conditioning.clone(),
                    copula: PairCopulaSpec {
                        family: parse_family(&edge.family),
                        rotation: parse_rotation(&edge.rotation),
                        params: parse_params(&edge.params),
                    },
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    VineCopula::from_trees(VineStructureKind::R, trees, truncation_level)
        .expect("fixture trees should form a valid vine")
}

fn parse_family(value: &str) -> PairCopulaFamily {
    match value {
        "Independence" | "independence" => PairCopulaFamily::Independence,
        "Gaussian" | "gaussian" => PairCopulaFamily::Gaussian,
        "StudentT" | "student_t" => PairCopulaFamily::StudentT,
        "Clayton" | "clayton" => PairCopulaFamily::Clayton,
        "Frank" | "frank" => PairCopulaFamily::Frank,
        "Gumbel" | "gumbel" => PairCopulaFamily::Gumbel,
        other => panic!("unexpected family {other}"),
    }
}

fn parse_rotation(value: &str) -> Rotation {
    match value {
        "R0" => Rotation::R0,
        "R90" => Rotation::R90,
        "R180" => Rotation::R180,
        "R270" => Rotation::R270,
        other => panic!("unexpected rotation {other}"),
    }
}

fn parse_params(values: &[f64]) -> PairCopulaParams {
    match values {
        [] => PairCopulaParams::None,
        [value] => PairCopulaParams::One(*value),
        [first, second] => PairCopulaParams::Two(*first, *second),
        _ => panic!("unexpected parameter arity"),
    }
}

fn copula_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/r-copula/v1_1_3")
}

fn vine_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/vinecopula/v2")
}

fn load_copula_fixture<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = copula_fixture_dir().join(name);
    let bytes = fs::read(path).expect("fixture should exist");
    serde_json::from_slice(&bytes).expect("fixture should deserialize")
}

fn load_vine_fixture<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = vine_fixture_dir().join(name);
    let bytes = fs::read(path).expect("fixture should exist");
    serde_json::from_slice(&bytes).expect("fixture should deserialize")
}

fn array2(rows: &[Vec<f64>]) -> Array2<f64> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let data = rows.iter().flat_map(|row| row.iter().copied()).collect::<Vec<_>>();
    Array2::from_shape_vec((nrows, ncols), data).expect("rows should form a matrix")
}
