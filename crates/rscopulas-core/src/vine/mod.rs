mod eval;
mod fit;
mod sample;
mod structure;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::paircopula::{PairCopulaFamily, PairCopulaSpec};

pub use fit::{SelectionCriterion, VineFitOptions};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VineCopula {
    pub(crate) dim: usize,
    pub(crate) structure: VineStructure,
    pub(crate) trees: Vec<VineTree>,
    pub(crate) normalized_matrix: Array2<usize>,
    pub(crate) variable_order: Vec<usize>,
    pub(crate) pair_matrix: Array2<Option<PairCopulaSpec>>,
    pub(crate) max_matrix: Array2<usize>,
    pub(crate) cond_direct: Array2<bool>,
    pub(crate) cond_indirect: Array2<bool>,
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
    pub fn order(&self) -> Vec<usize> {
        match self.structure.kind {
            VineStructureKind::C => {
                let mut order = Vec::new();
                if let Some(first_tree) = self.trees.first()
                    && let Some(first_edge) = first_tree.edges.first()
                {
                    order.push(first_edge.conditioned.0);
                    order.extend(first_tree.edges.iter().rev().map(|edge| edge.conditioned.1));
                }
                order
            }
            VineStructureKind::D => {
                let mut order = Vec::new();
                if let Some(first_tree) = self.trees.first() {
                    for (idx, edge) in first_tree.edges.iter().rev().enumerate() {
                        if idx == 0 {
                            order.push(edge.conditioned.0);
                        }
                        order.push(edge.conditioned.1);
                    }
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
}

fn default_family_set() -> Vec<PairCopulaFamily> {
    vec![
        PairCopulaFamily::Independence,
        PairCopulaFamily::Gaussian,
        PairCopulaFamily::StudentT,
        PairCopulaFamily::Clayton,
        PairCopulaFamily::Frank,
        PairCopulaFamily::Gumbel,
        PairCopulaFamily::Khoudraji,
    ]
}
