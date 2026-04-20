use ndarray::Array2;

use crate::{
    errors::{CopulaError, FitError},
    paircopula::PairCopulaSpec,
};

use super::{
    CompiledEvalStep, CompiledSampleStep, CompiledVineRuntime, VineCopula, VineEdge, VineStructure,
    VineStructureKind, VineTree,
};

pub(crate) fn validate_order(order: &[usize], dim: usize) -> Result<(), CopulaError> {
    if order.len() != dim {
        return Err(FitError::Failed {
            reason: "vine order length must match the correlation dimension",
        }
        .into());
    }

    let mut seen = vec![false; dim];
    for &value in order {
        if value >= dim || seen[value] {
            return Err(FitError::Failed {
                reason: "vine order must be a permutation of 0..d-1",
            }
            .into());
        }
        seen[value] = true;
    }

    Ok(())
}

pub(crate) fn build_model_from_trees(
    kind: VineStructureKind,
    trees: Vec<VineTree>,
    truncation_level: Option<usize>,
) -> Result<VineCopula, CopulaError> {
    let dim = trees.len() + 1;
    validate_tree_variables(&trees, dim)?;
    let (matrix, pair_matrix) = to_structure_matrices(&trees)?;
    let variable_order = matrix.diag().iter().rev().copied().collect::<Vec<_>>();
    let normalized_matrix = normalize_matrix(&matrix, &variable_order)?;
    let max_matrix = create_max_matrix(&normalized_matrix);
    let (cond_direct, cond_indirect) =
        needed_conditional_distributions(&normalized_matrix, &max_matrix);
    let runtime = compile_runtime(
        &normalized_matrix,
        &max_matrix,
        &cond_indirect,
        &pair_matrix,
        &variable_order,
    )?;
    Ok(VineCopula {
        dim,
        structure: VineStructure {
            kind,
            matrix,
            truncation_level,
        },
        trees,
        normalized_matrix,
        variable_order,
        pair_matrix,
        max_matrix,
        cond_direct,
        cond_indirect,
        runtime,
    })
}

pub(crate) fn compile_runtime(
    normalized_matrix: &Array2<usize>,
    max_matrix: &Array2<usize>,
    cond_indirect: &Array2<bool>,
    pair_matrix: &Array2<Option<PairCopulaSpec>>,
    variable_order: &[usize],
) -> Result<CompiledVineRuntime, CopulaError> {
    let mat = revert_matrix(normalized_matrix);
    let maxmat = revert_matrix(max_matrix);
    let cindirect = revert_matrix(cond_indirect);
    let specs = revert_pair_matrix(pair_matrix);
    let d = normalized_matrix.nrows();
    let mut sample_steps = Vec::with_capacity(d.saturating_mul(d.saturating_sub(1)) / 2);
    let mut eval_steps = Vec::with_capacity(sample_steps.capacity());
    let mut all_gaussian = true;

    for i in 1..d {
        for k in (0..i).rev() {
            let spec = specs[(k, i)].clone().ok_or(FitError::Failed {
                reason: "vine structure is missing pair-copula specs for one or more edges",
            })?;
            all_gaussian &= matches!(
                (spec.family, spec.rotation, &spec.params),
                (
                    crate::paircopula::PairCopulaFamily::Gaussian,
                    crate::paircopula::Rotation::R0,
                    crate::paircopula::PairCopulaParams::One(_)
                )
            );
            sample_steps.push(CompiledSampleStep {
                row: k,
                col: i,
                label: maxmat[(k, i)],
                source_from_direct: mat[(k, i)] == maxmat[(k, i)],
                write_indirect: i + 1 < d && cindirect[(k + 1, i)],
                spec,
            });
        }
    }

    for i in 1..d {
        for k in 0..i {
            let spec = specs[(k, i)].clone().ok_or(FitError::Failed {
                reason: "vine structure is missing pair-copula specs for one or more edges",
            })?;
            eval_steps.push(CompiledEvalStep {
                row: k,
                col: i,
                label: maxmat[(k, i)],
                source_from_direct: mat[(k, i)] == maxmat[(k, i)],
                write_indirect: i + 1 < d && cindirect[(k + 1, i)],
                spec,
            });
        }
    }

    Ok(CompiledVineRuntime {
        dim: d,
        variable_order: variable_order.to_vec(),
        sample_steps,
        eval_steps,
        all_gaussian,
    })
}

fn to_structure_matrices(
    trees: &[VineTree],
) -> Result<(Array2<usize>, Array2<Option<PairCopulaSpec>>), CopulaError> {
    let n = trees.len() + 1;
    let mut matrix = Array2::zeros((n, n));
    let mut pair_matrix = Array2::from_elem((n, n), None);
    let mut remaining = trees
        .iter()
        .map(|tree| tree.edges.clone())
        .collect::<Vec<_>>();

    for k in 0..(n - 1) {
        let source_tree = n - k - 2;
        let first = remaining[source_tree]
            .first()
            .ok_or(FitError::Failed {
                reason: "vine tree is missing required edges for structure conversion",
            })?
            .clone();
        let w = first.conditioned.0;
        matrix[(k, k)] = w;
        matrix[(k + 1, k)] = first.conditioned.1;
        pair_matrix[(k + 1, k)] = Some(first.copula.clone());

        if k == n - 2 {
            matrix[(k + 1, k + 1)] = first.conditioned.1;
            continue;
        }

        for i in (k + 2)..n {
            let tree_idx = n - i - 1;
            let mut found = None;
            for (idx, edge) in remaining[tree_idx].iter().enumerate() {
                if edge.conditioned.0 == w {
                    found = Some((idx, edge.conditioned.1, edge.copula.clone()));
                    break;
                }
                if edge.conditioned.1 == w {
                    found = Some((idx, edge.conditioned.0, edge.copula.clone().swap_axes()));
                    break;
                }
            }

            let (idx, value, spec) = found.ok_or(FitError::Failed {
                reason: "failed to convert selected vine trees into a structure matrix",
            })?;
            matrix[(i, k)] = value;
            pair_matrix[(i, k)] = Some(spec);
            remaining[tree_idx].remove(idx);
        }
    }

    Ok((matrix, pair_matrix))
}

fn validate_tree_variables(trees: &[VineTree], dim: usize) -> Result<(), CopulaError> {
    for tree in trees {
        for edge in &tree.edges {
            if edge.conditioned.0 >= dim
                || edge.conditioned.1 >= dim
                || edge.conditioning.iter().any(|&value| value >= dim)
            {
                return Err(FitError::Failed {
                    reason: "vine tree references variables outside the declared dimension",
                }
                .into());
            }
        }
    }
    Ok(())
}

fn create_max_matrix(matrix: &Array2<usize>) -> Array2<usize> {
    let mut max_matrix = matrix.clone();
    let n = max_matrix.nrows();
    for col in 0..(n - 1) {
        for row in (col..=(n - 2)).rev() {
            max_matrix[(row, col)] = max_matrix[(row, col)].max(max_matrix[(row + 1, col)]);
        }
    }
    max_matrix
}

fn normalize_matrix(
    matrix: &Array2<usize>,
    variable_order: &[usize],
) -> Result<Array2<usize>, CopulaError> {
    let mut mapping = vec![0usize; variable_order.len()];
    for (idx, &value) in variable_order.iter().enumerate() {
        let entry = mapping.get_mut(value).ok_or(FitError::Failed {
            reason: "vine structure matrix contains variables outside the declared order",
        })?;
        *entry = idx;
    }

    let mut normalized = Array2::zeros(matrix.raw_dim());
    for ((row, col), value) in matrix.indexed_iter() {
        normalized[(row, col)] = *mapping.get(*value).ok_or(FitError::Failed {
            reason: "vine structure matrix contains variables outside the declared order",
        })?;
    }
    Ok(normalized)
}

fn needed_conditional_distributions(
    matrix: &Array2<usize>,
    max_matrix: &Array2<usize>,
) -> (Array2<bool>, Array2<bool>) {
    let d = matrix.nrows();
    let mut direct = Array2::from_elem((d, d), false);
    let mut indirect = Array2::from_elem((d, d), false);
    for row in 1..d {
        direct[(row, 0)] = true;
    }

    for i in 1..(d - 1) {
        let v = d - i - 1;
        for row in i..d {
            let mut any_bw_and_not_direct = false;
            let mut any_first = false;
            for col in 0..i {
                let bw = max_matrix[(row, col)] == v;
                let is_direct = matrix[(row, col)] == v;
                any_bw_and_not_direct |= bw && !is_direct;
                if row == i {
                    any_first |= bw && is_direct;
                }
            }
            indirect[(row, i)] = any_bw_and_not_direct;
            direct[(row, i)] = true;
            if row == i {
                direct[(i, i)] = any_first;
            }
        }
    }

    (direct, indirect)
}

pub(crate) fn revert_matrix<T: Clone + Default>(matrix: &Array2<T>) -> Array2<T> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let mut reverted = Array2::default((nrows, ncols));
    for row in 0..nrows {
        for col in 0..ncols {
            reverted[(row, col)] = matrix[(nrows - row - 1, ncols - col - 1)].clone();
        }
    }
    reverted
}

pub(crate) fn revert_pair_matrix(
    matrix: &Array2<Option<PairCopulaSpec>>,
) -> Array2<Option<PairCopulaSpec>> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let mut reverted = Array2::from_elem((nrows, ncols), None);
    for row in 0..nrows {
        for col in 0..ncols {
            reverted[(row, col)] = matrix[(nrows - row - 1, ncols - col - 1)].clone();
        }
    }
    reverted
}

pub(crate) fn canonical_c_vine_trees(order: &[usize], specs: &[PairCopulaSpec]) -> Vec<VineTree> {
    let dim = order.len();
    let mut trees = Vec::with_capacity(dim - 1);
    let mut offset = 0usize;
    for level in 0..(dim - 1) {
        let root = order[level];
        let conditioning = order[..level].to_vec();
        let mut edges = Vec::with_capacity(dim - level - 1);
        for idx in (level + 1..dim).rev() {
            edges.push(VineEdge {
                tree: level + 1,
                conditioned: (root, order[idx]),
                conditioning: conditioning.clone(),
                copula: specs[offset].clone(),
            });
            offset += 1;
        }
        trees.push(VineTree {
            level: level + 1,
            edges,
        });
    }
    trees
}

pub(crate) fn canonical_d_vine_trees(order: &[usize], specs: &[PairCopulaSpec]) -> Vec<VineTree> {
    let dim = order.len();
    let mut trees = Vec::with_capacity(dim - 1);
    let mut offset = 0usize;
    for gap in 1..dim {
        let mut edges = Vec::with_capacity(dim - gap);
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
