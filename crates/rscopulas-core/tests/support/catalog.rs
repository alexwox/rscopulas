//! Fixed models shared by the golden-snapshot tests and the perf suite, so
//! "what is covered by the regression net" and "what is timed" are the same
//! list.
#![allow(dead_code)]

use ndarray::Array2;
use rscopulas::{
    ClaytonCopula, FactorCopula, FrankCopula, GaussianCopula, GumbelHougaardCopula, HacFamily,
    HacNode, HacTree, HierarchicalArchimedeanCopula, PairCopulaFamily, PairCopulaParams,
    PairCopulaSpec, Rotation, StudentTCopula, TllOrder, VineCopula, VineEdge, VineStructureKind,
    VineTree, tll_fit,
};

use super::datagen::{self, Dependence};

pub fn spec(
    family: PairCopulaFamily,
    rotation: Rotation,
    params: PairCopulaParams,
) -> PairCopulaSpec {
    PairCopulaSpec {
        family,
        rotation,
        params,
    }
}

pub fn one(family: PairCopulaFamily, rotation: Rotation, value: f64) -> PairCopulaSpec {
    spec(family, rotation, PairCopulaParams::One(value))
}

pub fn two(
    family: PairCopulaFamily,
    rotation: Rotation,
    first: f64,
    second: f64,
) -> PairCopulaSpec {
    spec(family, rotation, PairCopulaParams::Two(first, second))
}

const ALL_ROTATIONS: [Rotation; 4] = [Rotation::R0, Rotation::R90, Rotation::R180, Rotation::R270];

fn rotation_tag(rotation: Rotation) -> &'static str {
    match rotation {
        Rotation::R0 => "r0",
        Rotation::R90 => "r90",
        Rotation::R180 => "r180",
        Rotation::R270 => "r270",
    }
}

/// One representative spec per family (moderate dependence, no rotation):
/// the list the perf suite times kernel by kernel.
pub fn representative_pair_specs() -> Vec<(String, PairCopulaSpec)> {
    use PairCopulaFamily as F;
    use Rotation::R0;
    let mut out = vec![
        ("independence".to_string(), PairCopulaSpec::independence()),
        ("gaussian".to_string(), one(F::Gaussian, R0, 0.6)),
        ("student_t".to_string(), two(F::StudentT, R0, 0.5, 4.0)),
        ("clayton".to_string(), one(F::Clayton, R0, 2.0)),
        ("frank".to_string(), one(F::Frank, R0, 5.0)),
        ("gumbel".to_string(), one(F::Gumbel, R0, 1.8)),
        ("joe".to_string(), one(F::Joe, R0, 3.0)),
        ("bb1".to_string(), two(F::Bb1, R0, 0.5, 1.5)),
        ("bb6".to_string(), two(F::Bb6, R0, 1.5, 1.5)),
        ("bb7".to_string(), two(F::Bb7, R0, 1.5, 0.8)),
        ("bb8".to_string(), two(F::Bb8, R0, 2.0, 0.7)),
        ("tawn1".to_string(), two(F::Tawn1, R0, 2.0, 0.6)),
        ("tawn2".to_string(), two(F::Tawn2, R0, 3.0, 0.3)),
    ];
    out.push(("khoudraji".to_string(), khoudraji_gumbel_clayton()));
    out.push(("tll".to_string(), tll_spec(TllOrder::Constant)));
    out
}

pub fn khoudraji_gumbel_clayton() -> PairCopulaSpec {
    PairCopulaSpec::khoudraji(
        one(PairCopulaFamily::Gumbel, Rotation::R0, 2.5),
        one(PairCopulaFamily::Clayton, Rotation::R0, 3.0),
        0.4,
        0.8,
    )
    .expect("khoudraji spec should validate")
}

pub fn tll_spec(order: TllOrder) -> PairCopulaSpec {
    let (u1, u2) = datagen::pair(400, Dependence::Gaussian, 0x711);
    let params = tll_fit(&u1, &u2, order).expect("tll fit should succeed");
    spec(
        PairCopulaFamily::Tll,
        Rotation::R0,
        PairCopulaParams::Tll(params),
    )
}

/// Every (family, parameter regime, rotation) combination pinned by the
/// golden snapshots. Parametric specs only — TLL is fitted, so it is
/// snapshotted through its own fit test.
pub fn pair_specs() -> Vec<(String, PairCopulaSpec)> {
    use PairCopulaFamily as F;
    use Rotation::R0;
    let mut out: Vec<(String, PairCopulaSpec)> = Vec::new();
    out.push(("independence".into(), PairCopulaSpec::independence()));
    for rho in [-0.7, 0.3, 0.95] {
        out.push((format!("gaussian[{rho}]"), one(F::Gaussian, R0, rho)));
    }
    for (rho, nu) in [(0.5, 4.0), (-0.6, 2.5), (0.8, 30.0)] {
        out.push((
            format!("student_t[{rho},{nu}]"),
            two(F::StudentT, R0, rho, nu),
        ));
    }
    for theta in [-6.0, 0.5, 12.0] {
        out.push((format!("frank[{theta}]"), one(F::Frank, R0, theta)));
    }
    // One-parameter Archimedean families: three regimes unrotated, plus all
    // rotations at the moderate one.
    for (family, name, thetas) in [
        (F::Clayton, "clayton", [0.3, 2.0, 8.0]),
        (F::Gumbel, "gumbel", [1.0, 1.4, 4.0]),
        (F::Joe, "joe", [1.2, 3.0, 8.0]),
    ] {
        for theta in thetas {
            out.push((format!("{name}[{theta}]"), one(family, R0, theta)));
        }
        for rotation in &ALL_ROTATIONS[1..] {
            out.push((
                format!("{name}[{}]/{}", thetas[1], rotation_tag(*rotation)),
                one(family, *rotation, thetas[1]),
            ));
        }
    }
    for (family, name, params) in [
        (F::Bb1, "bb1", [(0.5, 1.5), (2.0, 3.0)]),
        (F::Bb6, "bb6", [(1.5, 1.5), (3.0, 2.0)]),
        (F::Bb7, "bb7", [(1.5, 0.8), (3.0, 3.0)]),
        (F::Bb8, "bb8", [(2.0, 0.7), (5.0, 0.95)]),
        (F::Tawn1, "tawn1", [(2.0, 0.6), (4.0, 0.9)]),
        (F::Tawn2, "tawn2", [(3.0, 0.3), (1.5, 0.8)]),
    ] {
        for (first, second) in params {
            out.push((
                format!("{name}[{first},{second}]"),
                two(family, R0, first, second),
            ));
        }
        for rotation in &ALL_ROTATIONS[1..] {
            out.push((
                format!(
                    "{name}[{},{}]/{}",
                    params[0].0,
                    params[0].1,
                    rotation_tag(*rotation)
                ),
                two(family, *rotation, params[0].0, params[0].1),
            ));
        }
    }
    out.push((
        "khoudraji[gumbel,clayton]".into(),
        khoudraji_gumbel_clayton(),
    ));
    out.push((
        "khoudraji[independence,clayton]".into(),
        PairCopulaSpec::khoudraji(
            PairCopulaSpec::independence(),
            one(F::Clayton, R0, 4.9),
            0.76,
            0.41,
        )
        .expect("khoudraji spec should validate"),
    ));
    let mut rotated =
        PairCopulaSpec::khoudraji(one(F::Gaussian, R0, 0.5), one(F::Frank, R0, 6.0), 0.3, 0.6)
            .expect("khoudraji spec should validate");
    rotated.rotation = Rotation::R90;
    out.push(("khoudraji[gaussian,frank]/r90".into(), rotated));
    out
}

/// Evaluation points: a tail-heavy grid plus pseudo-random interior points.
pub fn pair_points() -> Vec<(f64, f64)> {
    const GRID: [f64; 7] = [1e-6, 1e-3, 0.05, 0.5, 0.95, 0.999, 1.0 - 1e-6];
    let mut points = Vec::new();
    for &u in &GRID {
        for &v in &GRID {
            points.push((u, v));
        }
    }
    let extra = datagen::uniforms(31, 2, 0xA11CE);
    for row in extra.rows() {
        points.push((row[0], row[1]));
    }
    points
}

pub fn gaussian_copula(dim: usize) -> GaussianCopula {
    GaussianCopula::new(datagen::correlation(dim)).expect("correlation should be SPD")
}

pub fn student_t_copula(dim: usize, nu: f64) -> StudentTCopula {
    StudentTCopula::new(datagen::correlation(dim), nu).expect("correlation should be SPD")
}

pub fn clayton_copula(dim: usize) -> ClaytonCopula {
    ClaytonCopula::new(dim, 1.7).expect("valid clayton")
}

pub fn frank_copula(dim: usize) -> FrankCopula {
    FrankCopula::new(dim, 4.5).expect("valid frank")
}

pub fn gumbel_copula(dim: usize) -> GumbelHougaardCopula {
    GumbelHougaardCopula::new(dim, 1.9).expect("valid gumbel")
}

pub fn hac_tree(family: HacFamily, outer: f64, inner: f64) -> HacTree {
    HacTree::Node(HacNode::new(
        family,
        outer,
        vec![
            HacTree::Leaf(0),
            HacTree::Leaf(1),
            HacTree::Node(HacNode::new(
                family,
                inner,
                vec![HacTree::Leaf(2), HacTree::Leaf(3)],
            )),
        ],
    ))
}

fn flat_hac(family: HacFamily, theta: f64) -> HierarchicalArchimedeanCopula {
    HierarchicalArchimedeanCopula::new(HacTree::Node(HacNode::new(
        family,
        theta,
        (0..4).map(HacTree::Leaf).collect(),
    )))
    .expect("valid flat hac")
}

/// Nested trees (sampling and composite scores) and flat trees (the only
/// shape with an exact joint density).
pub fn hac_models() -> Vec<(&'static str, HierarchicalArchimedeanCopula)> {
    vec![
        ("flat_gumbel", flat_hac(HacFamily::Gumbel, 1.6)),
        ("flat_clayton", flat_hac(HacFamily::Clayton, 1.2)),
        ("flat_frank", flat_hac(HacFamily::Frank, 3.5)),
        (
            "gumbel",
            HierarchicalArchimedeanCopula::new(hac_tree(HacFamily::Gumbel, 1.3, 2.2))
                .expect("valid hac"),
        ),
        (
            "clayton",
            HierarchicalArchimedeanCopula::new(hac_tree(HacFamily::Clayton, 0.8, 2.5))
                .expect("valid hac"),
        ),
        (
            "frank",
            HierarchicalArchimedeanCopula::new(hac_tree(HacFamily::Frank, 2.0, 6.0))
                .expect("valid hac"),
        ),
    ]
}

pub fn factor_model() -> FactorCopula {
    use PairCopulaFamily as F;
    use Rotation::R0;
    FactorCopula::basic_1f(
        vec![
            one(F::Gaussian, R0, 0.7),
            one(F::Clayton, R0, 1.5),
            one(F::Gumbel, R0, 1.8),
            one(F::Frank, R0, 4.0),
            one(F::Gumbel, Rotation::R180, 1.4),
        ],
        25,
    )
    .expect("valid factor model")
}

/// D-vine trees along `order` with one spec per edge (tree by tree).
fn d_vine_trees(order: &[usize], specs: &[PairCopulaSpec]) -> Vec<VineTree> {
    let dim = order.len();
    let mut trees = Vec::new();
    let mut offset = 0;
    for gap in 1..dim {
        let mut edges = Vec::new();
        for start in (0..(dim - gap)).rev() {
            edges.push(VineEdge {
                tree: gap,
                conditioned: (order[start], order[start + gap]),
                conditioning: order[(start + 1)..(start + gap)].to_vec(),
                copula: specs[offset].clone(),
            });
            offset += 1;
        }
        trees.push(VineTree { level: gap, edges });
    }
    trees
}

/// C-vine trees rooted along `order` with one spec per edge (tree by tree).
fn c_vine_trees(order: &[usize], specs: &[PairCopulaSpec]) -> Vec<VineTree> {
    let dim = order.len();
    let mut trees = Vec::new();
    let mut offset = 0;
    for level in 1..dim {
        let root = order[level - 1];
        let mut edges = Vec::new();
        for target in (level..dim).rev() {
            edges.push(VineEdge {
                tree: level,
                conditioned: (root, order[target]),
                conditioning: order[..(level - 1)].to_vec(),
                copula: specs[offset].clone(),
            });
            offset += 1;
        }
        trees.push(VineTree { level, edges });
    }
    trees
}

/// Ten edge copulas for a 5-dimensional vine, mixing closed-form inverses,
/// bisection-based inverses, rotations and two-parameter families.
fn mixed_edge_specs() -> Vec<PairCopulaSpec> {
    use PairCopulaFamily as F;
    use Rotation::{R0, R90, R180, R270};
    vec![
        one(F::Gaussian, R0, 0.6),
        two(F::StudentT, R0, -0.4, 5.0),
        one(F::Clayton, R180, 1.8),
        one(F::Gumbel, R0, 1.6),
        one(F::Frank, R0, -3.0),
        one(F::Joe, R90, 1.7),
        two(F::Bb1, R0, 0.4, 1.3),
        two(F::Bb7, R270, 1.4, 0.6),
        one(F::Gaussian, R0, -0.2),
        PairCopulaSpec::independence(),
    ]
}

pub fn mixed_d_vine() -> VineCopula {
    let trees = d_vine_trees(&[0, 1, 2, 3, 4], &mixed_edge_specs());
    VineCopula::from_trees(VineStructureKind::D, trees, None).expect("valid d-vine")
}

pub fn mixed_c_vine() -> VineCopula {
    let trees = c_vine_trees(&[2, 0, 4, 1, 3], &mixed_edge_specs());
    VineCopula::from_trees(VineStructureKind::C, trees, None).expect("valid c-vine")
}

/// A Gaussian vine in `dim` dimensions (closed-form kernels only).
pub fn gaussian_d_vine(dim: usize) -> VineCopula {
    VineCopula::gaussian_d_vine((0..dim).collect(), datagen::correlation(dim))
        .expect("valid gaussian d-vine")
}

pub fn to_rows(matrix: &Array2<f64>) -> Vec<f64> {
    matrix.iter().copied().collect()
}
