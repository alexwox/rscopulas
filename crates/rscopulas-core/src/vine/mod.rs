mod eval;
mod fit;
mod rosenblatt;
mod sample;
mod structure;

use std::ops::Deref;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::paircopula::{PairCopulaFamily, PairCopulaSpec};

pub use fit::{SelectionCriterion, TreeAlgorithm, TreeCriterion, VineFitOptions};

/// Supported vine structure families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VineStructureKind {
    C,
    D,
    R,
}

/// One edge in a vine tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VineEdge {
    pub tree: usize,
    pub conditioned: (usize, usize),
    pub conditioning: Vec<usize>,
    pub copula: PairCopulaSpec,
}

/// A single tree in a vine decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VineTree {
    pub level: usize,
    pub edges: Vec<VineEdge>,
}

/// Structural metadata for a vine copula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VineStructure {
    pub kind: VineStructureKind,
    pub matrix: Array2<usize>,
    pub truncation_level: Option<usize>,
}

/// Fitted vine copula together with its structural matrices.
#[derive(Debug, Clone, Serialize)]
pub struct VineCopula {
    pub(crate) format_version: u32,
    pub(crate) dim: usize,
    pub(crate) structure: VineStructure,
    pub(crate) trees: Vec<VineTree>,
    pub(crate) normalized_matrix: Array2<usize>,
    pub(crate) variable_order: Vec<usize>,
    pub(crate) pair_matrix: Array2<Option<PairCopulaSpec>>,
    pub(crate) max_matrix: Array2<usize>,
    pub(crate) cond_direct: Array2<bool>,
    pub(crate) cond_indirect: Array2<bool>,
    #[serde(skip, default)]
    pub(crate) runtime: CompiledVineRuntime,
}

impl<'de> Deserialize<'de> for VineCopula {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct State {
            format_version: Option<u32>,
            dim: usize,
            structure: VineStructure,
            trees: Vec<VineTree>,
        }
        let state = State::deserialize(deserializer)?;
        if state.format_version != Some(1) {
            return Err(serde::de::Error::custom(
                "unsupported or unversioned vine state; rscopulas 0.2 used a different pair orientation; refit or explicitly rebuild trees under the 0.3 contract",
            ));
        }
        let model = Self::from_trees(
            state.structure.kind,
            state.trees,
            state.structure.truncation_level,
        )
        .map_err(serde::de::Error::custom)?;
        if state.dim != model.dim || state.structure.matrix != model.structure.matrix {
            return Err(serde::de::Error::custom(
                "serialized vine dimension or matrix disagrees with its trees",
            ));
        }
        Ok(model)
    }
}

impl VineCopula {
    /// Builds a vine copula from explicit trees and an optional truncation level.
    pub fn from_trees(
        kind: VineStructureKind,
        trees: Vec<VineTree>,
        truncation_level: Option<usize>,
    ) -> Result<Self, crate::errors::CopulaError> {
        structure::build_model_from_trees(kind, trees, truncation_level)
    }

    /// Returns the vine structure family.
    pub fn structure(&self) -> VineStructureKind {
        self.structure.kind
    }

    /// Returns structure metadata including the R-vine matrix and truncation level.
    pub fn structure_info(&self) -> &VineStructure {
        &self.structure
    }

    /// Returns the vine trees in evaluation order.
    pub fn trees(&self) -> &[VineTree] {
        &self.trees
    }

    /// Returns the top-level variable order implied by the structure.
    ///
    /// For C-vines the first element is the tree-1 root; for D-vines the
    /// result is the tree-1 path. Both are reconstructed from the edge set,
    /// so hand-built trees with arbitrary edge order or orientation give the
    /// same answer as canonically built ones. R-vines return the reversed
    /// diagonal of the structure matrix.
    pub fn order(&self) -> Vec<usize> {
        let Some(first_tree) = self.trees.first() else {
            return Vec::new();
        };
        match self.structure.kind {
            VineStructureKind::C => {
                let edges = &first_tree.edges;
                let Some(first_edge) = edges.first() else {
                    return Vec::new();
                };
                let is_root = |v: usize| {
                    edges
                        .iter()
                        .all(|edge| edge.conditioned.0 == v || edge.conditioned.1 == v)
                };
                let root = if is_root(first_edge.conditioned.0) {
                    first_edge.conditioned.0
                } else {
                    first_edge.conditioned.1
                };
                let mut order = vec![root];
                order.extend(edges.iter().rev().map(|edge| {
                    if edge.conditioned.0 == root {
                        edge.conditioned.1
                    } else {
                        edge.conditioned.0
                    }
                }));
                order
            }
            VineStructureKind::D => {
                let edges = &first_tree.edges;
                let Some(last_edge) = edges.last() else {
                    return Vec::new();
                };
                let mut neighbours: std::collections::BTreeMap<usize, Vec<usize>> =
                    std::collections::BTreeMap::new();
                for edge in edges {
                    let (a, b) = edge.conditioned;
                    neighbours.entry(a).or_default().push(b);
                    neighbours.entry(b).or_default().push(a);
                }
                let degree_one = |v: usize| neighbours.get(&v).is_some_and(|n| n.len() == 1);
                // Prefer the endpoint a canonical build would start from so
                // existing callers see unchanged output.
                let start = if degree_one(last_edge.conditioned.0) {
                    last_edge.conditioned.0
                } else if degree_one(last_edge.conditioned.1) {
                    last_edge.conditioned.1
                } else {
                    neighbours
                        .iter()
                        .find(|(_, n)| n.len() == 1)
                        .map(|(v, _)| *v)
                        .unwrap_or(last_edge.conditioned.0)
                };
                let mut order = vec![start];
                let mut previous = None;
                let mut current = start;
                while order.len() < neighbours.len() {
                    let Some(next) = neighbours
                        .get(&current)
                        .and_then(|n| n.iter().copied().find(|&v| Some(v) != previous))
                    else {
                        break;
                    };
                    order.push(next);
                    previous = Some(current);
                    current = next;
                }
                order
            }
            VineStructureKind::R => self.structure.matrix.diag().iter().rev().copied().collect(),
        }
    }

    /// Returns the primary parameter from each pair-copula edge.
    pub fn pair_parameters(&self) -> Vec<f64> {
        self.trees
            .iter()
            .flat_map(|tree| tree.edges.iter())
            .map(|edge| {
                edge.copula
                    .flat_parameters()
                    .into_iter()
                    .next()
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// Returns the configured truncation level, if any.
    pub fn truncation_level(&self) -> Option<usize> {
        self.structure.truncation_level
    }

    pub(crate) fn compiled_runtime(&self) -> Result<RuntimeView<'_>, crate::errors::CopulaError> {
        if self.runtime.is_empty() {
            Ok(RuntimeView::Owned(structure::compile_runtime(
                &self.normalized_matrix,
                &self.max_matrix,
                &self.cond_indirect,
                &self.pair_matrix,
                &self.variable_order,
            )?))
        } else {
            Ok(RuntimeView::Borrowed(&self.runtime))
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CompiledVineRuntime {
    pub(crate) dim: usize,
    pub(crate) variable_order: Vec<usize>,
    pub(crate) sample_steps: Vec<CompiledSampleStep>,
    pub(crate) eval_steps: Vec<CompiledEvalStep>,
    pub(crate) all_gaussian: bool,
}

impl CompiledVineRuntime {
    pub(crate) fn is_empty(&self) -> bool {
        self.sample_steps.is_empty() || self.eval_steps.is_empty()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CompiledSampleStep {
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) label: usize,
    pub(crate) source_from_direct: bool,
    pub(crate) write_indirect: bool,
    pub(crate) spec: PairCopulaSpec,
}

#[derive(Debug, Clone)]
pub(crate) struct CompiledEvalStep {
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) label: usize,
    pub(crate) source_from_direct: bool,
    pub(crate) write_indirect: bool,
    pub(crate) spec: PairCopulaSpec,
}

pub(crate) enum RuntimeView<'a> {
    Borrowed(&'a CompiledVineRuntime),
    Owned(CompiledVineRuntime),
}

impl Deref for RuntimeView<'_> {
    type Target = CompiledVineRuntime;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(runtime) => runtime,
            Self::Owned(runtime) => runtime,
        }
    }
}

/// Families considered by [`VineFitOptions::default`].
///
/// Khoudraji is deliberately absent since 0.4: its fit enumerates 25 base
/// pairs with a nested shape search, which dominated the default fit time
/// while rarely winning selection. Opt in by listing
/// `PairCopulaFamily::Khoudraji` in `family_set`.
fn default_family_set() -> Vec<PairCopulaFamily> {
    vec![
        PairCopulaFamily::Independence,
        PairCopulaFamily::Gaussian,
        PairCopulaFamily::StudentT,
        PairCopulaFamily::Clayton,
        PairCopulaFamily::Frank,
        PairCopulaFamily::Gumbel,
        PairCopulaFamily::Joe,
        PairCopulaFamily::Bb1,
        PairCopulaFamily::Bb7,
    ]
}
