use std::{fs, path::PathBuf};

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;

use rscopulas_core::{
    CopulaModel, EvalOptions, PairCopulaFamily, PairCopulaParams, PairCopulaSpec, PseudoObs,
    Rotation, SampleOptions, VineCopula, VineEdge, VineFitOptions, VineStructureKind, VineTree,
};

#[derive(Debug, Deserialize)]
struct Metadata {
    source_package: String,
    source_version: String,
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
struct VineLogPdfFixture {
    metadata: Metadata,
    structure: String,
    truncation_level: Option<usize>,
    matrix: Vec<Vec<usize>>,
    trees: Vec<TreeFixture>,
    inputs: Vec<Vec<f64>>,
    expected_log_pdf: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct VineSampleSummaryFixture {
    metadata: Metadata,
    structure: String,
    truncation_level: Option<usize>,
    matrix: Vec<Vec<usize>>,
    trees: Vec<TreeFixture>,
    seed: u64,
    sample_size: usize,
    expected_mean: Vec<f64>,
    expected_kendall_tau: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct VineFitDataFixture {
    metadata: Metadata,
    structure: String,
    trees: Vec<TreeFixture>,
    data: Vec<Vec<f64>>,
}

#[test]
fn mixed_r_vine_matches_vinecopula_log_pdf_fixture() {
    let fixture: VineLogPdfFixture = load_fixture("mixed_r_vine_log_pdf_d5_case01.json");
    assert_eq!(fixture.metadata.source_package, "VineCopula");
    assert_eq!(fixture.structure, "R");

    let model = build_model(&fixture.trees, fixture.truncation_level);
    assert_eq!(model.structure(), VineStructureKind::R);
    assert_eq!(model.structure_info().matrix, array2_usize(&fixture.matrix));

    let input =
        PseudoObs::new(array2_f64(&fixture.inputs)).expect("fixture inputs should be valid");
    let actual = model
        .log_pdf(&input, &EvalOptions::default())
        .expect("log pdf should evaluate");

    for (idx, (left, right)) in actual
        .iter()
        .zip(fixture.expected_log_pdf.iter())
        .enumerate()
    {
        assert!(
            (left - right).abs() < 1e-8,
            "fixture {} mismatch at row {idx}: left={left}, right={right}",
            fixture.metadata.source_version
        );
    }
}

#[test]
fn mixed_r_vine_sample_statistics_match_vinecopula_fixture() {
    let fixture: VineSampleSummaryFixture =
        load_fixture("mixed_r_vine_sample_summary_d5_case01.json");
    assert_eq!(fixture.metadata.source_package, "VineCopula");
    assert_eq!(fixture.structure, "R");

    let model = build_model(&fixture.trees, fixture.truncation_level);
    assert_eq!(model.structure_info().matrix, array2_usize(&fixture.matrix));

    let mut rng = StdRng::seed_from_u64(fixture.seed);
    let samples = model
        .sample(fixture.sample_size, &mut rng, &SampleOptions::default())
        .expect("sampling should succeed");
    let sample_obs = PseudoObs::new(samples).expect("sample should be valid");
    let means = column_means(&sample_obs);
    let tau = rscopulas_core::stats::kendall_tau_matrix(&sample_obs);

    for (idx, (actual, expected)) in means.iter().zip(fixture.expected_mean.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1.5e-2,
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
fn mixed_r_vine_serde_round_trip_preserves_log_pdf_and_sampling() {
    let log_pdf_fixture: VineLogPdfFixture = load_fixture("mixed_r_vine_log_pdf_d5_case01.json");
    let sample_fixture: VineSampleSummaryFixture =
        load_fixture("mixed_r_vine_sample_summary_d5_case01.json");
    let model = build_model(&log_pdf_fixture.trees, log_pdf_fixture.truncation_level);
    let input = PseudoObs::new(array2_f64(&log_pdf_fixture.inputs))
        .expect("fixture inputs should be valid");

    let encoded = serde_json::to_vec(&model).expect("vine model should serialize");
    let restored: VineCopula =
        serde_json::from_slice(&encoded).expect("vine model should deserialize");

    assert_eq!(
        restored.structure_info().matrix,
        model.structure_info().matrix
    );

    let original_log_pdf = model
        .log_pdf(&input, &EvalOptions::default())
        .expect("original log pdf should evaluate");
    let restored_log_pdf = restored
        .log_pdf(&input, &EvalOptions::default())
        .expect("restored log pdf should evaluate");
    assert_eq!(original_log_pdf, restored_log_pdf);

    let mut original_rng = StdRng::seed_from_u64(sample_fixture.seed);
    let mut restored_rng = StdRng::seed_from_u64(sample_fixture.seed);
    let original_samples = model
        .sample(256, &mut original_rng, &SampleOptions::default())
        .expect("original sampling should succeed");
    let restored_samples = restored
        .sample(256, &mut restored_rng, &SampleOptions::default())
        .expect("restored sampling should succeed");
    assert_eq!(original_samples, restored_samples);
}

#[test]
fn fit_r_vine_is_not_a_relabel_of_canonical_vines_on_mixed_reference_data() {
    let fixture: VineFitDataFixture = load_fixture("mixed_r_vine_fit_data_d5_case01.json");
    assert_eq!(fixture.metadata.source_package, "VineCopula");
    assert_eq!(fixture.structure, "R");
    assert_eq!(fixture.trees.len(), 4);

    let data = PseudoObs::new(array2_f64(&fixture.data)).expect("fixture data should be valid");
    let options = VineFitOptions {
        family_set: vec![
            PairCopulaFamily::Independence,
            PairCopulaFamily::Gaussian,
            PairCopulaFamily::Clayton,
            PairCopulaFamily::Frank,
            PairCopulaFamily::Gumbel,
        ],
        truncation_level: Some(2),
        ..VineFitOptions::default()
    };

    let reference_trees = fixture
        .trees
        .iter()
        .take(
            options
                .truncation_level
                .expect("truncation level should be set"),
        )
        .map(tree_signature)
        .collect::<Vec<_>>();
    let r_fit = VineCopula::fit_r_vine(&data, &options).expect("R-vine fit should succeed");
    let c_fit = VineCopula::fit_c_vine(&data, &options).expect("C-vine fit should succeed");
    let d_fit = VineCopula::fit_d_vine(&data, &options).expect("D-vine fit should succeed");

    assert_eq!(r_fit.model.structure(), VineStructureKind::R);
    let fitted_trees = r_fit
        .model
        .trees()
        .iter()
        .map(tree_signature)
        .collect::<Vec<_>>();
    let shared_first_tree = fitted_trees[0]
        .iter()
        .filter(|edge| reference_trees[0].contains(*edge))
        .count();
    assert!(
        shared_first_tree >= 2,
        "fitted R-vine should recover at least two fixture first-tree edges; fitted={:?}, reference={:?}",
        fitted_trees[0],
        reference_trees[0]
    );
    assert!(
        fitted_trees[0].contains(&((2, 4), Vec::new())),
        "fitted R-vine should recover the fixture's strongest cross-branch edge"
    );
    assert!(
        r_fit
            .model
            .trees()
            .iter()
            .flat_map(|tree| tree.edges.iter())
            .any(|edge| !edge.conditioning.is_empty()),
        "fitted R-vine should contain higher-tree conditioned edges"
    );

    let canonical_best = c_fit.diagnostics.loglik.max(d_fit.diagnostics.loglik);
    assert!(
        (r_fit.diagnostics.loglik - canonical_best).abs() > 1e-6,
        "R-vine fit should not be a relabeled best-of-C/D fit: r={}, c={}, d={}",
        r_fit.diagnostics.loglik,
        c_fit.diagnostics.loglik,
        d_fit.diagnostics.loglik
    );
}

fn build_model(trees: &[TreeFixture], truncation_level: Option<usize>) -> VineCopula {
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
        "Independence" => PairCopulaFamily::Independence,
        "Gaussian" => PairCopulaFamily::Gaussian,
        "StudentT" => PairCopulaFamily::StudentT,
        "Clayton" => PairCopulaFamily::Clayton,
        "Frank" => PairCopulaFamily::Frank,
        "Gumbel" => PairCopulaFamily::Gumbel,
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

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/vinecopula/v2")
}

fn load_fixture<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = fixture_dir().join(name);
    let bytes = fs::read(path).expect("fixture should exist");
    serde_json::from_slice(&bytes).expect("fixture should deserialize")
}

fn array2_f64(rows: &[Vec<f64>]) -> Array2<f64> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let data = rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    Array2::from_shape_vec((nrows, ncols), data).expect("rows should form a matrix")
}

fn array2_usize(rows: &[Vec<usize>]) -> Array2<usize> {
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

fn tree_signature<T>(tree: &T) -> Vec<((usize, usize), Vec<usize>)>
where
    T: TreeLike,
{
    let mut edges = tree
        .edges()
        .iter()
        .map(|edge| {
            let conditioned = if edge.conditioned().0 < edge.conditioned().1 {
                edge.conditioned()
            } else {
                (edge.conditioned().1, edge.conditioned().0)
            };
            (conditioned, edge.conditioning().to_vec())
        })
        .collect::<Vec<_>>();
    edges.sort();
    edges
}

trait TreeLike {
    type Edge: EdgeLike;

    fn edges(&self) -> &[Self::Edge];
}

trait EdgeLike {
    fn conditioned(&self) -> (usize, usize);
    fn conditioning(&self) -> &[usize];
}

impl TreeLike for TreeFixture {
    type Edge = EdgeFixture;

    fn edges(&self) -> &[Self::Edge] {
        &self.edges
    }
}

impl TreeLike for VineTree {
    type Edge = VineEdge;

    fn edges(&self) -> &[Self::Edge] {
        &self.edges
    }
}

impl EdgeLike for EdgeFixture {
    fn conditioned(&self) -> (usize, usize) {
        (self.conditioned[0], self.conditioned[1])
    }

    fn conditioning(&self) -> &[usize] {
        &self.conditioning
    }
}

impl EdgeLike for VineEdge {
    fn conditioned(&self) -> (usize, usize) {
        self.conditioned
    }

    fn conditioning(&self) -> &[usize] {
        &self.conditioning
    }
}
