use std::{fs, path::PathBuf};

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

use rscopulas::paircopula::fit_pair_copula;
use rscopulas::{
    CopulaError, CopulaModel, FitOptions, KhoudrajiParams, PairCopulaFamily, PairCopulaParams,
    PairCopulaSpec, PseudoObs, Rotation, SampleOptions, VineCopula, VineEdge, VineFitOptions,
    VineStructureKind, VineTree,
};

#[derive(Debug, Deserialize)]
struct Metadata {
    source_package: String,
    source_version: String,
}

#[derive(Debug, Deserialize)]
struct PairSpecFixture {
    family: String,
    rotation: String,
    #[serde(default, deserialize_with = "deserialize_parameter_vec")]
    parameters: Vec<f64>,
    #[serde(default)]
    base_copula_1: Option<Box<PairSpecFixture>>,
    #[serde(default)]
    base_copula_2: Option<Box<PairSpecFixture>>,
    #[serde(default)]
    shape_1: Option<f64>,
    #[serde(default)]
    shape_2: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct KhoudrajiPairFixture {
    metadata: Metadata,
    family: String,
    rotation: String,
    base_copula_1: PairSpecFixture,
    base_copula_2: PairSpecFixture,
    shape_1: f64,
    shape_2: f64,
    u1: Vec<f64>,
    u2: Vec<f64>,
    p: Vec<f64>,
    expected_log_pdf: Vec<f64>,
    expected_cond_first_given_second: Vec<f64>,
    expected_cond_second_given_first: Vec<f64>,
    expected_inv_first_given_second: Vec<f64>,
    expected_inv_second_given_first: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct KhoudrajiSampleSummaryFixture {
    metadata: Metadata,
    family: String,
    rotation: String,
    base_copula_1: PairSpecFixture,
    base_copula_2: PairSpecFixture,
    shape_1: f64,
    shape_2: f64,
    seed: u64,
    sample_size: usize,
    expected_mean: Vec<f64>,
    expected_kendall_tau: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct KhoudrajiFitFixture {
    metadata: Metadata,
    family: String,
    rotation: String,
    base_copula_1: PairSpecFixture,
    base_copula_2: PairSpecFixture,
    input_pobs: Vec<Vec<f64>>,
    expected_theta: f64,
    expected_shape_1: f64,
    expected_shape_2: f64,
}

#[test]
fn khoudraji_pair_matches_r_fixture() {
    let fixture: KhoudrajiPairFixture = load_fixture("khoudraji_pair_case01.json");
    assert_eq!(fixture.metadata.source_package, "copula");
    assert_eq!(fixture.family, "khoudraji");
    assert_eq!(fixture.rotation, "R0");

    let spec = build_khoudraji_spec(
        &fixture.base_copula_1,
        &fixture.base_copula_2,
        fixture.shape_1,
        fixture.shape_2,
        &fixture.rotation,
    )
    .expect("fixture should describe a valid khoudraji pair");

    for idx in 0..fixture.u1.len() {
        let u1 = fixture.u1[idx];
        let u2 = fixture.u2[idx];
        let p = fixture.p[idx];

        let actual_log_pdf = spec
            .log_pdf(u1, u2, 1e-12)
            .expect("log pdf should evaluate");
        assert!(
            (actual_log_pdf - fixture.expected_log_pdf[idx]).abs() < 2e-3,
            "{} khoudraji log pdf mismatch at row {idx}: left={actual_log_pdf}, right={}",
            fixture.metadata.source_version,
            fixture.expected_log_pdf[idx]
        );

        let actual_h12 = spec
            .cond_first_given_second(u1, u2, 1e-12)
            .expect("h-function should evaluate");
        assert!(
            (actual_h12 - fixture.expected_cond_first_given_second[idx]).abs() < 2e-3,
            "{} khoudraji h12 mismatch at row {idx}: left={actual_h12}, right={}",
            fixture.metadata.source_version,
            fixture.expected_cond_first_given_second[idx]
        );

        let actual_h21 = spec
            .cond_second_given_first(u1, u2, 1e-12)
            .expect("h-function should evaluate");
        assert!(
            (actual_h21 - fixture.expected_cond_second_given_first[idx]).abs() < 2e-3,
            "{} khoudraji h21 mismatch at row {idx}: left={actual_h21}, right={}",
            fixture.metadata.source_version,
            fixture.expected_cond_second_given_first[idx]
        );

        let actual_inv_h12 = spec
            .inv_first_given_second(p, u2, 1e-12)
            .expect("inverse h-function should evaluate");
        assert!(
            (actual_inv_h12 - fixture.expected_inv_first_given_second[idx]).abs() < 2e-3,
            "{} khoudraji hinv12 mismatch at row {idx}: left={actual_inv_h12}, right={}",
            fixture.metadata.source_version,
            fixture.expected_inv_first_given_second[idx]
        );

        let actual_inv_h21 = spec
            .inv_second_given_first(u1, p, 1e-12)
            .expect("inverse h-function should evaluate");
        assert!(
            (actual_inv_h21 - fixture.expected_inv_second_given_first[idx]).abs() < 2e-3,
            "{} khoudraji hinv21 mismatch at row {idx}: left={actual_inv_h21}, right={}",
            fixture.metadata.source_version,
            fixture.expected_inv_second_given_first[idx]
        );
    }
}

#[test]
fn khoudraji_sampling_matches_r_fixture_statistics() {
    let fixture: KhoudrajiSampleSummaryFixture =
        load_fixture("khoudraji_sample_summary_case01.json");
    assert_eq!(fixture.metadata.source_package, "copula");
    assert_eq!(fixture.family, "khoudraji");

    let spec = build_khoudraji_spec(
        &fixture.base_copula_1,
        &fixture.base_copula_2,
        fixture.shape_1,
        fixture.shape_2,
        &fixture.rotation,
    )
    .expect("fixture should describe a valid khoudraji pair");
    let model = vine_from_pair(spec);

    let mut rng = StdRng::seed_from_u64(fixture.seed);
    let samples = model
        .sample(fixture.sample_size, &mut rng, &SampleOptions::default())
        .expect("sampling should succeed");
    let sample_obs = PseudoObs::new(samples).expect("sample should stay inside (0,1)");
    let means = column_means(&sample_obs);
    let tau = rscopulas::stats::kendall_tau_matrix(&sample_obs);

    for (idx, (actual, expected)) in means.iter().zip(fixture.expected_mean.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1.5e-2,
            "mean mismatch at column {idx}: left={actual}, right={expected}"
        );
    }

    for row in 0..tau.nrows() {
        for col in 0..tau.ncols() {
            let expected = fixture.expected_kendall_tau[row][col];
            assert!(
                (tau[(row, col)] - expected).abs() < 2.5e-2,
                "kendall tau mismatch at ({row}, {col}): left={}, right={expected}",
                tau[(row, col)]
            );
        }
    }
}

#[test]
fn khoudraji_fit_tracks_r_fixture_for_indep_clayton_case() {
    let fixture: KhoudrajiFitFixture = load_fixture("khoudraji_fit_case01.json");
    assert_eq!(fixture.metadata.source_package, "copula");
    assert_eq!(fixture.family, "khoudraji");
    assert_eq!(fixture.rotation, "R0");
    assert_eq!(fixture.base_copula_1.family, "independence");
    assert_eq!(fixture.base_copula_2.family, "clayton");

    let input =
        PseudoObs::new(array2(&fixture.input_pobs)).expect("fixture inputs should be valid");
    let options = VineFitOptions {
        family_set: vec![PairCopulaFamily::Khoudraji],
        include_rotations: false,
        base: FitOptions {
            max_iter: 8,
            ..FitOptions::default()
        },
        ..VineFitOptions::default()
    };
    let fit = fit_pair_copula(
        &input
            .as_view()
            .column(0)
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        &input
            .as_view()
            .column(1)
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        &options,
    )
    .expect("khoudraji fit should succeed");

    let PairCopulaParams::Khoudraji(ref params) = fit.spec.params else {
        panic!("expected khoudraji parameters");
    };
    assert_eq!(fit.spec.family, PairCopulaFamily::Khoudraji);
    assert!(fit.loglik.is_finite());
    assert!(fit.loglik > 0.0);
    assert!((0.0..=1.0).contains(&params.shape_first));
    assert!((0.0..=1.0).contains(&params.shape_second));
    let representative = match &params.second.params {
        PairCopulaParams::None => 0.0,
        PairCopulaParams::One(value) => *value,
        PairCopulaParams::Two(first, _) => *first,
        PairCopulaParams::Khoudraji(_) => unreachable!("nested khoudraji should not be fitted"),
        PairCopulaParams::Tll(_) => unreachable!("tll inner khoudraji base is not fitted"),
    };
    assert!(representative.is_finite());
    assert!((representative - fixture.expected_theta).abs() < 5.0);
    assert!((params.shape_first - fixture.expected_shape_1).abs() < 0.7);
    assert!((params.shape_second - fixture.expected_shape_2).abs() < 0.7);
}

#[test]
fn khoudraji_vine_edges_can_be_selected_and_serialized() {
    let spec = PairCopulaSpec::khoudraji(
        PairCopulaSpec::independence(),
        PairCopulaSpec {
            family: PairCopulaFamily::Clayton,
            rotation: Rotation::R0,
            params: PairCopulaParams::One(2.2),
        },
        0.35,
        0.75,
    )
    .expect("khoudraji spec should be valid");
    let source = vine_from_pair(spec.clone());
    let mut rng = StdRng::seed_from_u64(404);
    let samples = source
        .sample(96, &mut rng, &SampleOptions::default())
        .expect("sampling should succeed");
    let data = PseudoObs::new(samples).expect("sample should stay inside (0,1)");
    let options = VineFitOptions {
        family_set: vec![PairCopulaFamily::Independence, PairCopulaFamily::Khoudraji],
        include_rotations: true,
        base: FitOptions {
            max_iter: 6,
            ..FitOptions::default()
        },
        ..VineFitOptions::default()
    };
    let fit = VineCopula::fit_r_vine(&data, &options).expect("vine fit should succeed");
    let selected = &fit.model.trees()[0].edges[0].copula;
    assert_eq!(selected.family, PairCopulaFamily::Khoudraji);

    let encoded = serde_json::to_vec(&fit.model).expect("model should serialize");
    let restored: VineCopula = serde_json::from_slice(&encoded).expect("model should deserialize");
    assert_eq!(
        restored.trees()[0].edges[0].copula.family,
        PairCopulaFamily::Khoudraji
    );
}

fn build_khoudraji_spec(
    first: &PairSpecFixture,
    second: &PairSpecFixture,
    shape_1: f64,
    shape_2: f64,
    rotation: &str,
) -> Result<PairCopulaSpec, CopulaError> {
    Ok(PairCopulaSpec {
        family: PairCopulaFamily::Khoudraji,
        rotation: parse_rotation(rotation),
        params: PairCopulaParams::Khoudraji(KhoudrajiParams::new(
            build_pair_spec(first)?,
            build_pair_spec(second)?,
            shape_1,
            shape_2,
        )?),
    })
}

fn build_pair_spec(fixture: &PairSpecFixture) -> Result<PairCopulaSpec, CopulaError> {
    if fixture.family == "khoudraji" {
        return build_khoudraji_spec(
            fixture
                .base_copula_1
                .as_deref()
                .expect("khoudraji fixtures require base_copula_1"),
            fixture
                .base_copula_2
                .as_deref()
                .expect("khoudraji fixtures require base_copula_2"),
            fixture.shape_1.expect("khoudraji fixtures require shape_1"),
            fixture.shape_2.expect("khoudraji fixtures require shape_2"),
            &fixture.rotation,
        );
    }
    Ok(PairCopulaSpec {
        family: parse_family(&fixture.family),
        rotation: parse_rotation(&fixture.rotation),
        params: match fixture.parameters.as_slice() {
            [] => PairCopulaParams::None,
            [value] => PairCopulaParams::One(*value),
            [first, second] => PairCopulaParams::Two(*first, *second),
            values => {
                panic!("unexpected parameter vector {values:?}");
            }
        },
    })
}

fn vine_from_pair(spec: PairCopulaSpec) -> VineCopula {
    VineCopula::from_trees(
        VineStructureKind::R,
        vec![VineTree {
            level: 1,
            edges: vec![VineEdge {
                tree: 1,
                conditioned: (0, 1),
                conditioning: Vec::new(),
                copula: spec,
            }],
        }],
        None,
    )
    .expect("pair vine should be valid")
}

fn parse_family(value: &str) -> PairCopulaFamily {
    match value {
        "independence" | "Independence" => PairCopulaFamily::Independence,
        "gaussian" | "Gaussian" => PairCopulaFamily::Gaussian,
        "student_t" | "StudentT" => PairCopulaFamily::StudentT,
        "clayton" | "Clayton" => PairCopulaFamily::Clayton,
        "frank" | "Frank" => PairCopulaFamily::Frank,
        "gumbel" | "Gumbel" => PairCopulaFamily::Gumbel,
        "khoudraji" | "Khoudraji" => PairCopulaFamily::Khoudraji,
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
