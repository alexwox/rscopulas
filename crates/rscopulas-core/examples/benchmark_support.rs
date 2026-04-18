#![allow(dead_code)]

use std::{fs, path::PathBuf};

use ndarray::Array2;
use rand::rngs::StdRng;
use serde::{Deserialize, Deserializer};
use serde_json::Value;

use rscopulas_core::{
    ClaytonCopula, CopulaError, CopulaModel, EvalOptions, FitOptions, FrankCopula, GaussianCopula,
    GumbelHougaardCopula, KhoudrajiParams, PairCopulaFamily, PairCopulaParams, PairCopulaSpec,
    PseudoObs, Rotation, SampleOptions, SelectionCriterion, StudentTCopula, VineCopula, VineEdge,
    VineFitOptions, VineStructureKind, VineTree,
};

#[derive(Debug, Deserialize)]
pub struct BenchmarkManifest {
    pub schema_version: u32,
    pub cases: Vec<BenchmarkCase>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BenchmarkCase {
    pub id: String,
    pub category: String,
    pub family: Option<String>,
    pub operation: String,
    pub structure: Option<String>,
    pub fixture: String,
    pub iterations: usize,
    pub implementations: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct SingleFamilyFixture {
    pub correlation: Option<Vec<Vec<f64>>>,
    pub degrees_of_freedom: Option<f64>,
    pub theta: Option<f64>,
    pub inputs: Option<Vec<Vec<f64>>>,
    pub input_pobs: Option<Vec<Vec<f64>>>,
    pub seed: Option<u64>,
    pub sample_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct PairFixture {
    pub family: Option<String>,
    pub family_code: Option<usize>,
    pub rotation: Option<String>,
    pub par: Option<f64>,
    pub par2: Option<f64>,
    #[serde(default)]
    pub base_copula_1: Option<PairSpecFixture>,
    #[serde(default)]
    pub base_copula_2: Option<PairSpecFixture>,
    #[serde(default)]
    pub shape_1: Option<f64>,
    #[serde(default)]
    pub shape_2: Option<f64>,
    pub u1: Vec<f64>,
    pub u2: Vec<f64>,
    pub p: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PairSpecFixture {
    pub family: String,
    #[serde(default)]
    pub rotation: Option<String>,
    #[serde(default, deserialize_with = "deserialize_parameter_vec")]
    pub parameters: Vec<f64>,
    #[serde(default)]
    pub base_copula_1: Option<Box<PairSpecFixture>>,
    #[serde(default)]
    pub base_copula_2: Option<Box<PairSpecFixture>>,
    #[serde(default)]
    pub shape_1: Option<f64>,
    #[serde(default)]
    pub shape_2: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VineFixture {
    pub structure: String,
    pub truncation_level: Option<usize>,
    pub trees: Vec<VineTreeFixture>,
    pub inputs: Option<Vec<Vec<f64>>>,
    pub data: Option<Vec<Vec<f64>>>,
    pub seed: Option<u64>,
    pub sample_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VineTreeFixture {
    pub level: usize,
    pub edges: Vec<VineEdgeFixture>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VineEdgeFixture {
    pub conditioned: Vec<usize>,
    pub conditioning: Vec<usize>,
    pub family: String,
    pub rotation: String,
    pub params: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum SingleFamilyModel {
    Gaussian(GaussianCopula),
    StudentT(StudentTCopula),
    Clayton(ClaytonCopula),
    Frank(FrankCopula),
    Gumbel(GumbelHougaardCopula),
}

impl SingleFamilyModel {
    pub fn dim(&self) -> usize {
        match self {
            Self::Gaussian(model) => model.dim(),
            Self::StudentT(model) => model.dim(),
            Self::Clayton(model) => model.dim(),
            Self::Frank(model) => model.dim(),
            Self::Gumbel(model) => model.dim(),
        }
    }

    pub fn log_pdf(
        &self,
        input: &PseudoObs,
        options: &EvalOptions,
    ) -> Result<Vec<f64>, CopulaError> {
        match self {
            Self::Gaussian(model) => model.log_pdf(input, options),
            Self::StudentT(model) => model.log_pdf(input, options),
            Self::Clayton(model) => model.log_pdf(input, options),
            Self::Frank(model) => model.log_pdf(input, options),
            Self::Gumbel(model) => model.log_pdf(input, options),
        }
    }

    pub fn sample(
        &self,
        n: usize,
        rng: &mut StdRng,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        match self {
            Self::Gaussian(model) => model.sample(n, rng, options),
            Self::StudentT(model) => model.sample(n, rng, options),
            Self::Clayton(model) => model.sample(n, rng, options),
            Self::Frank(model) => model.sample(n, rng, options),
            Self::Gumbel(model) => model.sample(n, rng, options),
        }
    }
}

pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn load_json_fixture<T: for<'de> Deserialize<'de>>(relative_path: &str) -> T {
    let path = workspace_root().join(relative_path);
    let bytes = fs::read(path).expect("benchmark fixture should exist");
    serde_json::from_slice(&bytes).expect("benchmark fixture should deserialize")
}

pub fn load_manifest(path: &str) -> BenchmarkManifest {
    let bytes = fs::read(path).expect("benchmark manifest should exist");
    serde_json::from_slice(&bytes).expect("benchmark manifest should deserialize")
}

pub fn array2(rows: &[Vec<f64>]) -> Array2<f64> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let data = rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    Array2::from_shape_vec((nrows, ncols), data).expect("rows should form a matrix")
}

pub fn pseudo_obs(rows: &[Vec<f64>]) -> PseudoObs {
    PseudoObs::new(array2(rows)).expect("fixture pseudo-observations should be valid")
}

pub fn single_family_model(family: &str, fixture: &SingleFamilyFixture) -> SingleFamilyModel {
    match family {
        "gaussian" => SingleFamilyModel::Gaussian(
            GaussianCopula::new(array2(
                fixture
                    .correlation
                    .as_ref()
                    .expect("gaussian fixture should contain a correlation matrix"),
            ))
            .expect("gaussian parameters should be valid"),
        ),
        "student_t" => SingleFamilyModel::StudentT(
            StudentTCopula::new(
                array2(
                    fixture
                        .correlation
                        .as_ref()
                        .expect("student t fixture should contain a correlation matrix"),
                ),
                fixture
                    .degrees_of_freedom
                    .expect("student t fixture should contain degrees of freedom"),
            )
            .expect("student t parameters should be valid"),
        ),
        "clayton" => SingleFamilyModel::Clayton(
            ClaytonCopula::new(
                fixture_dim_from_data(fixture),
                fixture.theta.expect("clayton fixture should contain theta"),
            )
            .expect("clayton parameters should be valid"),
        ),
        "frank" => SingleFamilyModel::Frank(
            FrankCopula::new(
                fixture_dim_from_data(fixture),
                fixture.theta.expect("frank fixture should contain theta"),
            )
            .expect("frank parameters should be valid"),
        ),
        "gumbel" => SingleFamilyModel::Gumbel(
            GumbelHougaardCopula::new(
                fixture_dim_from_data(fixture),
                fixture.theta.expect("gumbel fixture should contain theta"),
            )
            .expect("gumbel parameters should be valid"),
        ),
        other => panic!("unsupported single-family benchmark family: {other}"),
    }
}

pub fn fixture_dim_from_data(fixture: &SingleFamilyFixture) -> usize {
    fixture
        .inputs
        .as_ref()
        .or(fixture.input_pobs.as_ref())
        .map(|rows| rows.first().map_or(0, Vec::len))
        .or_else(|| fixture.correlation.as_ref().map(Vec::len))
        .or_else(|| fixture.theta.map(|_| 2))
        .expect("fixture should provide enough information to infer its dimension")
}

pub fn fit_single_family(
    family: &str,
    data: &PseudoObs,
    options: &FitOptions,
) -> Result<(), CopulaError> {
    match family {
        "gaussian" => {
            GaussianCopula::fit(data, options)?;
        }
        "student_t" => {
            StudentTCopula::fit(data, options)?;
        }
        "clayton" => {
            ClaytonCopula::fit(data, options)?;
        }
        "frank" => {
            FrankCopula::fit(data, options)?;
        }
        "gumbel" => {
            GumbelHougaardCopula::fit(data, options)?;
        }
        other => panic!("unsupported single-family benchmark family: {other}"),
    }
    Ok(())
}

pub fn pair_spec_from_fixture(fixture: &PairFixture) -> PairCopulaSpec {
    if fixture.family.as_deref() == Some("khoudraji") {
        return PairCopulaSpec {
            family: PairCopulaFamily::Khoudraji,
            rotation: parse_rotation(
                fixture
                    .rotation
                    .as_deref()
                    .expect("khoudraji benchmark fixture should contain a rotation"),
            ),
            params: PairCopulaParams::Khoudraji(
                KhoudrajiParams::new(
                    pair_spec_from_component(
                        fixture
                            .base_copula_1
                            .as_ref()
                            .expect("khoudraji benchmark fixture should contain base_copula_1"),
                    ),
                    pair_spec_from_component(
                        fixture
                            .base_copula_2
                            .as_ref()
                            .expect("khoudraji benchmark fixture should contain base_copula_2"),
                    ),
                    fixture
                        .shape_1
                        .expect("khoudraji benchmark fixture should contain shape_1"),
                    fixture
                        .shape_2
                        .expect("khoudraji benchmark fixture should contain shape_2"),
                )
                .expect("khoudraji benchmark fixture should be valid"),
            ),
        };
    }

    let (family, rotation) = match fixture.family_code {
        Some(0) => (PairCopulaFamily::Independence, Rotation::R0),
        Some(1) => (PairCopulaFamily::Gaussian, Rotation::R0),
        Some(2) => (PairCopulaFamily::StudentT, Rotation::R0),
        Some(3) => (PairCopulaFamily::Clayton, Rotation::R0),
        Some(13) => (PairCopulaFamily::Clayton, Rotation::R180),
        Some(23) => (PairCopulaFamily::Clayton, Rotation::R90),
        Some(33) => (PairCopulaFamily::Clayton, Rotation::R270),
        Some(4) => (PairCopulaFamily::Gumbel, Rotation::R0),
        Some(14) => (PairCopulaFamily::Gumbel, Rotation::R180),
        Some(24) => (PairCopulaFamily::Gumbel, Rotation::R90),
        Some(34) => (PairCopulaFamily::Gumbel, Rotation::R270),
        Some(5) => (PairCopulaFamily::Frank, Rotation::R0),
        other => panic!("unsupported pair fixture family code: {other:?}"),
    };
    let params = match family {
        PairCopulaFamily::Independence => PairCopulaParams::None,
        PairCopulaFamily::StudentT => PairCopulaParams::Two(
            fixture
                .par
                .expect("student t pair fixture should contain par"),
            fixture
                .par2
                .expect("student t pair fixture should contain par2"),
        ),
        _ => PairCopulaParams::One(fixture.par.expect("pair fixture should contain par")),
    };
    PairCopulaSpec {
        family,
        rotation,
        params,
    }
}

pub fn vine_model_from_fixture(fixture: &VineFixture) -> Result<VineCopula, CopulaError> {
    let kind = match fixture.structure.to_ascii_lowercase().as_str() {
        "c" => VineStructureKind::C,
        "d" => VineStructureKind::D,
        "r" => VineStructureKind::R,
        other => panic!("unsupported vine structure kind: {other}"),
    };
    let trees = fixture
        .trees
        .iter()
        .map(|tree| VineTree {
            level: tree.level,
            edges: tree
                .edges
                .iter()
                .map(|edge| VineEdge {
                    tree: tree.level,
                    conditioned: (
                        *edge
                            .conditioned
                            .first()
                            .expect("conditioned edges should contain two variables"),
                        *edge
                            .conditioned
                            .get(1)
                            .expect("conditioned edges should contain two variables"),
                    ),
                    conditioning: edge.conditioning.clone(),
                    copula: PairCopulaSpec {
                        family: match edge.family.as_str() {
                            "Independence" => PairCopulaFamily::Independence,
                            "Gaussian" => PairCopulaFamily::Gaussian,
                            "StudentT" => PairCopulaFamily::StudentT,
                            "Clayton" => PairCopulaFamily::Clayton,
                            "Frank" => PairCopulaFamily::Frank,
                            "Gumbel" => PairCopulaFamily::Gumbel,
                            other => panic!("unsupported vine edge family: {other}"),
                        },
                        rotation: match edge.rotation.as_str() {
                            "R0" => Rotation::R0,
                            "R90" => Rotation::R90,
                            "R180" => Rotation::R180,
                            "R270" => Rotation::R270,
                            other => panic!("unsupported vine edge rotation: {other}"),
                        },
                        params: match edge.params.as_slice() {
                            [] => PairCopulaParams::None,
                            [value] => PairCopulaParams::One(*value),
                            [first, second] => PairCopulaParams::Two(*first, *second),
                            values => {
                                panic!("unsupported vine edge parameter count: {}", values.len())
                            }
                        },
                    },
                })
                .collect(),
        })
        .collect();
    VineCopula::from_trees(kind, trees, fixture.truncation_level)
}

pub fn default_vine_fit_options() -> VineFitOptions {
    VineFitOptions {
        family_set: vec![
            PairCopulaFamily::Independence,
            PairCopulaFamily::Gaussian,
            PairCopulaFamily::Clayton,
            PairCopulaFamily::Frank,
            PairCopulaFamily::Gumbel,
        ],
        include_rotations: true,
        criterion: SelectionCriterion::Aic,
        truncation_level: Some(2),
        ..VineFitOptions::default()
    }
}

pub fn implementation_requested(case: &BenchmarkCase, implementation: &str) -> bool {
    case.implementations
        .iter()
        .any(|value| value == implementation)
}

pub fn repeat_pair_kernels(
    spec: &PairCopulaSpec,
    fixture: &PairFixture,
    clip_eps: f64,
) -> Result<(), CopulaError> {
    for ((u1, u2), p) in fixture
        .u1
        .iter()
        .zip(&fixture.u2)
        .zip(&fixture.p)
        .map(|((left, right), p)| ((*left, *right), *p))
    {
        spec.log_pdf(u1, u2, clip_eps)?;
        spec.cond_first_given_second(u1, u2, clip_eps)?;
        spec.cond_second_given_first(u1, u2, clip_eps)?;
        spec.inv_first_given_second(p, u2, clip_eps)?;
        spec.inv_second_given_first(u1, p, clip_eps)?;
    }
    Ok(())
}

pub fn single_family_input(fixture: &SingleFamilyFixture) -> PseudoObs {
    let rows = fixture
        .inputs
        .as_ref()
        .expect("log-pdf benchmark fixture should contain inputs");
    pseudo_obs(rows)
}

pub fn single_family_fit_input(fixture: &SingleFamilyFixture) -> PseudoObs {
    let rows = fixture
        .input_pobs
        .as_ref()
        .expect("fit benchmark fixture should contain input_pobs");
    pseudo_obs(rows)
}

pub fn vine_input(fixture: &VineFixture) -> PseudoObs {
    let rows = fixture
        .inputs
        .as_ref()
        .expect("vine fixture should contain inputs");
    pseudo_obs(rows)
}

pub fn vine_fit_input(fixture: &VineFixture) -> PseudoObs {
    let rows = fixture
        .data
        .as_ref()
        .expect("vine fit fixture should contain data");
    pseudo_obs(rows)
}

pub fn sample_size(fixture: &SingleFamilyFixture) -> usize {
    fixture
        .sample_size
        .expect("sample benchmark fixture should contain sample_size")
}

pub fn sample_seed(fixture: &SingleFamilyFixture) -> u64 {
    fixture
        .seed
        .expect("sample benchmark fixture should contain seed")
}

pub fn vine_sample_size(fixture: &VineFixture) -> usize {
    fixture
        .sample_size
        .expect("vine sample fixture should contain sample_size")
}

pub fn vine_sample_seed(fixture: &VineFixture) -> u64 {
    fixture
        .seed
        .expect("vine sample fixture should contain seed")
}

fn main() {}

fn pair_spec_from_component(fixture: &PairSpecFixture) -> PairCopulaSpec {
    if fixture.family == "khoudraji" {
        return PairCopulaSpec {
            family: PairCopulaFamily::Khoudraji,
            rotation: parse_rotation(
                fixture
                    .rotation
                    .as_deref()
                    .expect("nested khoudraji benchmark fixture should contain a rotation"),
            ),
            params: PairCopulaParams::Khoudraji(
                KhoudrajiParams::new(
                    pair_spec_from_component(
                        fixture.base_copula_1.as_deref().expect(
                            "nested khoudraji benchmark fixture should contain base_copula_1",
                        ),
                    ),
                    pair_spec_from_component(
                        fixture.base_copula_2.as_deref().expect(
                            "nested khoudraji benchmark fixture should contain base_copula_2",
                        ),
                    ),
                    fixture
                        .shape_1
                        .expect("nested khoudraji benchmark fixture should contain shape_1"),
                    fixture
                        .shape_2
                        .expect("nested khoudraji benchmark fixture should contain shape_2"),
                )
                .expect("nested khoudraji benchmark fixture should be valid"),
            ),
        };
    }

    PairCopulaSpec {
        family: match fixture.family.as_str() {
            "independence" => PairCopulaFamily::Independence,
            "gaussian" => PairCopulaFamily::Gaussian,
            "student_t" => PairCopulaFamily::StudentT,
            "clayton" => PairCopulaFamily::Clayton,
            "frank" => PairCopulaFamily::Frank,
            "gumbel" => PairCopulaFamily::Gumbel,
            other => panic!("unsupported pair component family: {other}"),
        },
        rotation: parse_rotation(fixture.rotation.as_deref().unwrap_or("R0")),
        params: match fixture.parameters.as_slice() {
            [] => PairCopulaParams::None,
            [value] => PairCopulaParams::One(*value),
            [first, second] => PairCopulaParams::Two(*first, *second),
            values => panic!(
                "unsupported pair component parameter count: {}",
                values.len()
            ),
        },
    }
}

fn parse_rotation(value: &str) -> Rotation {
    match value {
        "R0" => Rotation::R0,
        "R90" => Rotation::R90,
        "R180" => Rotation::R180,
        "R270" => Rotation::R270,
        other => panic!("unsupported pair rotation: {other}"),
    }
}

fn deserialize_parameter_vec<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Number(number)) => {
            Ok(vec![number.as_f64().ok_or_else(|| {
                serde::de::Error::custom("parameter scalar must be numeric")
            })?])
        }
        Some(Value::Array(values)) => values
            .into_iter()
            .map(|value| {
                value
                    .as_f64()
                    .ok_or_else(|| serde::de::Error::custom("parameter entries must be numeric"))
            })
            .collect(),
        Some(other) => Err(serde::de::Error::custom(format!(
            "unexpected parameter payload {other}"
        ))),
    }
}
