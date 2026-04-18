use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    backend::{ExecutionStrategy, Operation, parallel_try_map_range_collect, resolve_strategy},
    data::PseudoObs,
    domain::{Device, ExecPolicy},
    errors::{CopulaError, FitError},
    fit::FitResult,
    math::{inverse, validate_correlation_matrix},
    paircopula::{
        PairCopulaFamily, PairCopulaParams, PairCopulaSpec, PairFitResult, Rotation,
        fit_pair_copula,
    },
    stats::try_kendall_tau_matrix,
};

use super::{
    VineCopula, VineEdge, VineStructureKind, VineTree, default_family_set,
    structure::{
        build_model_from_trees, canonical_c_vine_trees, canonical_d_vine_trees, validate_order,
    },
};

/// Information criterion used to compare candidate pair-copula fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionCriterion {
    Aic,
    Bic,
}

/// Options controlling vine structure and pair-copula selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VineFitOptions {
    /// Base options shared with the single-family fitters.
    pub base: crate::domain::FitOptions,
    /// Candidate pair-copula families considered for each edge.
    pub family_set: Vec<PairCopulaFamily>,
    /// Whether rotated Archimedean families may be selected.
    pub include_rotations: bool,
    /// Criterion used when comparing candidate pair-copula fits.
    pub criterion: SelectionCriterion,
    /// Optional maximum tree level to fit and retain.
    pub truncation_level: Option<usize>,
    /// Optional absolute Kendall tau threshold for selecting independence.
    pub independence_threshold: Option<f64>,
}

impl Default for VineFitOptions {
    fn default() -> Self {
        Self {
            base: crate::domain::FitOptions::default(),
            family_set: default_family_set(),
            include_rotations: true,
            criterion: SelectionCriterion::Aic,
            truncation_level: None,
            independence_threshold: None,
        }
    }
}

#[derive(Clone)]
struct InternalEdgeFit {
    conditioned: (usize, usize),
    conditioning: Vec<usize>,
    cond_on_first: SharedSeries,
    cond_on_second: SharedSeries,
}

#[derive(Clone)]
struct InternalTree {
    edges: Vec<InternalEdgeFit>,
}

#[derive(Clone)]
struct GraphNode {
    conditioned_set: Vec<usize>,
    conditioning_set: Vec<usize>,
}

#[derive(Clone)]
struct GraphEdge {
    endpoints: (usize, usize),
    weight: f64,
    conditioned_set: (usize, usize),
    conditioning_set: Vec<usize>,
    left_data: SharedSeries,
    right_data: SharedSeries,
}

#[derive(Clone)]
struct Graph {
    vertices: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
}

impl VineCopula {
    /// Constructs a Gaussian C-vine from an explicit variable order and correlation matrix.
    pub fn gaussian_c_vine(
        order: Vec<usize>,
        correlation: Array2<f64>,
    ) -> Result<Self, CopulaError> {
        validate_correlation_matrix(&correlation)?;
        validate_order(&order, correlation.nrows())?;
        let specs = c_vine_gaussian_specs(&order, &correlation)?;
        let trees = canonical_c_vine_trees(&order, &specs);
        build_model_from_trees(VineStructureKind::C, trees, None)
    }

    /// Constructs a Gaussian D-vine from an explicit variable order and correlation matrix.
    pub fn gaussian_d_vine(
        order: Vec<usize>,
        correlation: Array2<f64>,
    ) -> Result<Self, CopulaError> {
        validate_correlation_matrix(&correlation)?;
        validate_order(&order, correlation.nrows())?;
        let specs = d_vine_gaussian_specs(&order, &correlation)?;
        let trees = canonical_d_vine_trees(&order, &specs);
        build_model_from_trees(VineStructureKind::D, trees, None)
    }

    /// Fits a simplified C-vine with pair-copula selection on each edge.
    pub fn fit_c_vine(
        data: &PseudoObs,
        options: &VineFitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        let order = c_vine_order(data, options)?;
        let (trees, diagnostics) = fit_canonical_vine(data, &order, VineStructureKind::C, options)?;
        let model = build_model_from_trees(VineStructureKind::C, trees, options.truncation_level)?;
        Ok(FitResult { model, diagnostics })
    }

    /// Fits a simplified D-vine with pair-copula selection on each edge.
    pub fn fit_d_vine(
        data: &PseudoObs,
        options: &VineFitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        let order = d_vine_order(data, options)?;
        let (trees, diagnostics) = fit_canonical_vine(data, &order, VineStructureKind::D, options)?;
        let model = build_model_from_trees(VineStructureKind::D, trees, options.truncation_level)?;
        Ok(FitResult { model, diagnostics })
    }

    /// Fits a simplified R-vine using a Dissmann-style maximum spanning tree procedure.
    pub fn fit_r_vine(
        data: &PseudoObs,
        options: &VineFitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        let truncation_level = options.truncation_level.unwrap_or(data.dim() - 1);
        let columns = collect_column_data(data);
        let mut graph = initialize_first_graph(data, &columns, options)?;
        let mut internal_trees = Vec::with_capacity(data.dim() - 1);
        let mut public_trees = Vec::with_capacity(data.dim() - 1);
        let mut total_loglik = 0.0;
        let mut n_iter = 0usize;

        for level in 1..data.dim() {
            let truncated = level > truncation_level;
            let mst = match find_max_tree(&graph, VineStructureKind::R, truncated) {
                Ok(tree) => tree,
                Err(_err) if truncated => {
                    graph = build_next_graph(
                        &Graph {
                            vertices: graph.vertices.clone(),
                            edges: Vec::new(),
                        },
                        internal_trees.last(),
                        true,
                    )?;
                    continue;
                }
                Err(err) => return Err(err),
            };

            let fitted = if level == 1 {
                fit_first_tree(&mst, options, truncated)?
            } else {
                fit_tree_from_graph(&mst, options, truncated)?
            };
            total_loglik += fitted.iter().map(|edge| edge.loglik()).sum::<f64>();
            n_iter += fitted.len();
            public_trees.push(VineTree {
                level,
                edges: fitted
                    .iter()
                    .map(|edge| VineEdge {
                        tree: level,
                        conditioned: edge.conditioned,
                        conditioning: edge.conditioning.clone(),
                        copula: edge.spec.clone(),
                    })
                    .collect(),
            });
            internal_trees.push(InternalTree {
                edges: fitted
                    .iter()
                    .map(|edge| InternalEdgeFit {
                        conditioned: edge.conditioned,
                        conditioning: edge.conditioning.clone(),
                        cond_on_first: edge.cond_on_first.clone(),
                        cond_on_second: edge.cond_on_second.clone(),
                    })
                    .collect(),
            });

            if level < data.dim() - 1 {
                graph = build_next_graph(&mst, internal_trees.last(), truncated)?;
            }
        }

        let model =
            build_model_from_trees(VineStructureKind::R, public_trees, options.truncation_level)?;
        let n_obs = data.n_obs() as f64;
        let parameter_count = model
            .trees
            .iter()
            .flat_map(|tree| tree.edges.iter())
            .map(|edge| edge.copula.parameter_count() as f64)
            .sum::<f64>();
        let diagnostics = crate::domain::FitDiagnostics {
            loglik: total_loglik,
            aic: 2.0 * parameter_count - 2.0 * total_loglik,
            bic: parameter_count * n_obs.ln() - 2.0 * total_loglik,
            converged: true,
            n_iter,
        };
        Ok(FitResult { model, diagnostics })
    }
}

#[derive(Clone)]
struct FittedGraphEdge {
    conditioned: (usize, usize),
    conditioning: Vec<usize>,
    spec: PairCopulaSpec,
    cond_on_first: SharedSeries,
    cond_on_second: SharedSeries,
    loglik: f64,
}

impl FittedGraphEdge {
    fn loglik(&self) -> f64 {
        self.loglik
    }
}

type SharedSeries = Arc<[f64]>;

fn fit_canonical_vine(
    data: &PseudoObs,
    order: &[usize],
    kind: VineStructureKind,
    options: &VineFitOptions,
) -> Result<(Vec<VineTree>, crate::domain::FitDiagnostics), CopulaError> {
    validate_order(order, data.dim())?;
    let mut internal_trees = Vec::with_capacity(data.dim() - 1);
    let mut public_trees = Vec::with_capacity(data.dim() - 1);
    let mut total_loglik = 0.0;
    let mut iterations = 0usize;

    let edge_defs = match kind {
        VineStructureKind::C => canonical_c_vine_trees(
            order,
            &vec![PairCopulaSpec::independence(); data.dim() * (data.dim() - 1) / 2],
        ),
        VineStructureKind::D => canonical_d_vine_trees(
            order,
            &vec![PairCopulaSpec::independence(); data.dim() * (data.dim() - 1) / 2],
        ),
        VineStructureKind::R => unreachable!(),
    };

    for tree in edge_defs {
        let mut fitted_edges = Vec::with_capacity(tree.edges.len());
        for edge in tree.edges {
            let left_data = if edge.conditioning.is_empty() {
                shared_column(data, edge.conditioned.0)
            } else {
                resolve_internal_conditional(
                    internal_trees.last(),
                    edge.conditioned.0,
                    &edge.conditioning,
                )?
            };
            let right_data = if edge.conditioning.is_empty() {
                shared_column(data, edge.conditioned.1)
            } else {
                resolve_internal_conditional(
                    internal_trees.last(),
                    edge.conditioned.1,
                    &edge.conditioning,
                )?
            };

            let fit = fit_pair_copula(left_data.as_ref(), right_data.as_ref(), options)?;
            total_loglik += fit.loglik;
            iterations += 1;
            fitted_edges.push(FittedGraphEdge {
                conditioned: edge.conditioned,
                conditioning: edge.conditioning.clone(),
                spec: fit.spec,
                cond_on_first: Arc::from(fit.cond_on_first),
                cond_on_second: Arc::from(fit.cond_on_second),
                loglik: fit.loglik,
            });
        }

        public_trees.push(VineTree {
            level: tree.level,
            edges: fitted_edges
                .iter()
                .map(|edge| VineEdge {
                    tree: tree.level,
                    conditioned: edge.conditioned,
                    conditioning: edge.conditioning.clone(),
                    copula: edge.spec.clone(),
                })
                .collect(),
        });
        internal_trees.push(InternalTree {
            edges: fitted_edges
                .iter()
                .map(|edge| InternalEdgeFit {
                    conditioned: edge.conditioned,
                    conditioning: edge.conditioning.clone(),
                    cond_on_first: edge.cond_on_first.clone(),
                    cond_on_second: edge.cond_on_second.clone(),
                })
                .collect(),
        });
    }

    let n_obs = data.n_obs() as f64;
    let parameter_count = public_trees
        .iter()
        .flat_map(|tree| tree.edges.iter())
        .map(|edge| edge.copula.parameter_count() as f64)
        .sum::<f64>();
    let diagnostics = crate::domain::FitDiagnostics {
        loglik: total_loglik,
        aic: 2.0 * parameter_count - 2.0 * total_loglik,
        bic: parameter_count * n_obs.ln() - 2.0 * total_loglik,
        converged: true,
        n_iter: iterations,
    };
    Ok((public_trees, diagnostics))
}

fn resolve_internal_conditional(
    previous: Option<&InternalTree>,
    target: usize,
    conditioning: &[usize],
) -> Result<SharedSeries, CopulaError> {
    let tree = previous.ok_or(FitError::Failed {
        reason: "missing previous tree while resolving canonical vine conditionals",
    })?;

    for &shared in conditioning {
        let base = conditioning
            .iter()
            .copied()
            .filter(|value| *value != shared)
            .collect::<Vec<_>>();
        for edge in &tree.edges {
            if edge.conditioning != base {
                continue;
            }
            if edge.conditioned == (target, shared) {
                return Ok(edge.cond_on_first.clone());
            }
            if edge.conditioned == (shared, target) {
                return Ok(edge.cond_on_second.clone());
            }
        }
    }

    Err(FitError::Failed {
        reason: "failed to resolve conditional data in canonical vine fitting",
    }
    .into())
}

fn initialize_first_graph(
    data: &PseudoObs,
    columns: &[SharedSeries],
    options: &VineFitOptions,
) -> Result<Graph, CopulaError> {
    let tau = try_kendall_tau_matrix(data, options.base.exec)?;
    let mut edges = Vec::new();
    for left in 0..data.dim() {
        for right in (left + 1)..data.dim() {
            edges.push(GraphEdge {
                endpoints: (left, right),
                weight: tau[(left, right)].abs(),
                conditioned_set: (left, right),
                conditioning_set: Vec::new(),
                left_data: columns[left].clone(),
                right_data: columns[right].clone(),
            });
        }
    }

    Ok(Graph {
        vertices: (0..data.dim())
            .map(|idx| GraphNode {
                conditioned_set: vec![idx],
                conditioning_set: Vec::new(),
            })
            .collect(),
        edges,
    })
}

fn find_max_tree(
    graph: &Graph,
    kind: VineStructureKind,
    truncated: bool,
) -> Result<Graph, CopulaError> {
    let vertices = graph.vertices.clone();
    let edges = if truncated {
        spanning_tree_by_dfs(graph)?
    } else {
        match kind {
            VineStructureKind::C => star_tree(graph)?,
            VineStructureKind::D => path_tree(graph)?,
            VineStructureKind::R => maximum_spanning_tree(graph)?,
        }
    };
    Ok(Graph { vertices, edges })
}

fn maximum_spanning_tree(graph: &Graph) -> Result<Vec<GraphEdge>, CopulaError> {
    let mut edges = graph.edges.clone();
    edges.sort_by(|left, right| {
        right
            .weight
            .total_cmp(&left.weight)
            .then_with(|| left.endpoints.cmp(&right.endpoints))
    });
    let mut dsu = DisjointSet::new(graph.vertices.len());
    let mut selected = Vec::with_capacity(graph.vertices.len().saturating_sub(1));
    for edge in edges {
        if dsu.union(edge.endpoints.0, edge.endpoints.1) {
            selected.push(edge);
        }
        if selected.len() + 1 == graph.vertices.len() {
            break;
        }
    }
    if selected.len() + 1 != graph.vertices.len() {
        return Err(FitError::Failed {
            reason: "graph is disconnected and cannot form a vine tree",
        }
        .into());
    }
    selected.sort_by_key(|left| left.endpoints);
    Ok(selected)
}

fn star_tree(graph: &Graph) -> Result<Vec<GraphEdge>, CopulaError> {
    let mut sums = vec![0.0; graph.vertices.len()];
    for edge in &graph.edges {
        sums[edge.endpoints.0] += edge.weight;
        sums[edge.endpoints.1] += edge.weight;
    }
    let root = sums
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(idx, _)| idx)
        .ok_or(FitError::Failed {
            reason: "failed to select a C-vine root",
        })?;
    let mut selected = graph
        .edges
        .iter()
        .filter(|edge| edge.endpoints.0 == root || edge.endpoints.1 == root)
        .cloned()
        .collect::<Vec<_>>();
    selected.sort_by_key(|left| left.endpoints);
    Ok(selected)
}

fn path_tree(graph: &Graph) -> Result<Vec<GraphEdge>, CopulaError> {
    let d = graph.vertices.len();
    let mut weights = Array2::zeros((d, d));
    for edge in &graph.edges {
        weights[(edge.endpoints.0, edge.endpoints.1)] = edge.weight;
        weights[(edge.endpoints.1, edge.endpoints.0)] = edge.weight;
    }
    let order = d_vine_order_from_weights(&weights);
    let mut selected = Vec::with_capacity(d.saturating_sub(1));
    for pair in order.windows(2) {
        let mut edge = graph
            .edges
            .iter()
            .find(|edge| {
                (edge.endpoints.0, edge.endpoints.1) == (pair[0], pair[1])
                    || (edge.endpoints.0, edge.endpoints.1) == (pair[1], pair[0])
            })
            .cloned()
            .ok_or(FitError::Failed {
                reason: "failed to construct a D-vine path from the candidate graph",
            })?;
        edge.endpoints = (pair[0], pair[1]);
        selected.push(edge);
    }
    Ok(selected)
}

fn spanning_tree_by_dfs(graph: &Graph) -> Result<Vec<GraphEdge>, CopulaError> {
    if graph.vertices.len() <= 1 {
        return Ok(Vec::new());
    }
    let mut adjacency = vec![Vec::new(); graph.vertices.len()];
    for edge in &graph.edges {
        adjacency[edge.endpoints.0].push(edge.endpoints.1);
        adjacency[edge.endpoints.1].push(edge.endpoints.0);
    }
    let mut visited = vec![false; graph.vertices.len()];
    let mut stack = vec![0usize];
    visited[0] = true;
    let mut selected = Vec::new();
    while let Some(node) = stack.pop() {
        for &neighbor in &adjacency[node] {
            if !visited[neighbor] {
                visited[neighbor] = true;
                stack.push(neighbor);
                let edge = graph
                    .edges
                    .iter()
                    .find(|edge| {
                        (edge.endpoints.0, edge.endpoints.1) == (node, neighbor)
                            || (edge.endpoints.0, edge.endpoints.1) == (neighbor, node)
                    })
                    .cloned()
                    .ok_or(FitError::Failed {
                        reason: "failed to recover an edge during truncated vine construction",
                    })?;
                selected.push(edge);
            }
        }
    }
    if selected.len() + 1 != graph.vertices.len() {
        return Err(FitError::Failed {
            reason: "graph is disconnected and cannot form a spanning tree",
        }
        .into());
    }
    Ok(selected)
}

fn fit_first_tree(
    tree: &Graph,
    options: &VineFitOptions,
    truncated: bool,
) -> Result<Vec<FittedGraphEdge>, CopulaError> {
    let strategy = resolve_strategy(options.base.exec, Operation::PairFitScoring, tree.edges.len())?;
    let inner_options = edge_fit_options(options, strategy);
    parallel_try_map_range_collect(tree.edges.len(), strategy, |idx| {
        let edge = &tree.edges[idx];
        let fit = if truncated {
            truncated_pair_fit(&edge.left_data, &edge.right_data)
        } else {
            fit_pair_copula(edge.left_data.as_ref(), edge.right_data.as_ref(), &inner_options)?
        };
        Ok(FittedGraphEdge {
            conditioned: edge.conditioned_set,
            conditioning: edge.conditioning_set.clone(),
            spec: fit.spec,
            cond_on_first: Arc::from(fit.cond_on_first),
            cond_on_second: Arc::from(fit.cond_on_second),
            loglik: fit.loglik,
        })
    })
}

fn fit_tree_from_graph(
    tree: &Graph,
    options: &VineFitOptions,
    truncated: bool,
) -> Result<Vec<FittedGraphEdge>, CopulaError> {
    let strategy = resolve_strategy(options.base.exec, Operation::PairFitScoring, tree.edges.len())?;
    let inner_options = edge_fit_options(options, strategy);
    parallel_try_map_range_collect(tree.edges.len(), strategy, |idx| {
        let edge = &tree.edges[idx];
        let fit = if truncated {
            truncated_pair_fit(&edge.left_data, &edge.right_data)
        } else {
            fit_pair_copula(edge.left_data.as_ref(), edge.right_data.as_ref(), &inner_options)?
        };
        Ok(FittedGraphEdge {
            conditioned: edge.conditioned_set,
            conditioning: edge.conditioning_set.clone(),
            spec: fit.spec,
            cond_on_first: Arc::from(fit.cond_on_first),
            cond_on_second: Arc::from(fit.cond_on_second),
            loglik: fit.loglik,
        })
    })
}

fn edge_fit_options(options: &VineFitOptions, strategy: ExecutionStrategy) -> VineFitOptions {
    let mut adjusted = options.clone();
    if matches!(strategy, ExecutionStrategy::CpuParallel) {
        adjusted.base.exec = ExecPolicy::Force(Device::Cpu);
    }
    adjusted
}

fn collect_column_data(data: &PseudoObs) -> Vec<SharedSeries> {
    (0..data.dim()).map(|column| shared_column(data, column)).collect()
}

fn shared_column(data: &PseudoObs, column: usize) -> SharedSeries {
    Arc::from(
        data.as_view()
            .column(column)
            .iter()
            .copied()
            .collect::<Vec<_>>(),
    )
}

fn truncated_pair_fit(left: &SharedSeries, right: &SharedSeries) -> PairFitResult {
    PairFitResult {
        spec: PairCopulaSpec::independence(),
        loglik: 0.0,
        aic: 0.0,
        bic: 0.0,
        cond_on_first: left.as_ref().to_vec(),
        cond_on_second: right.as_ref().to_vec(),
    }
}

fn build_next_graph(
    _tree: &Graph,
    previous_fitted: Option<&InternalTree>,
    truncated: bool,
) -> Result<Graph, CopulaError> {
    let previous = previous_fitted.ok_or(FitError::Failed {
        reason: "missing previous fitted tree while building the next vine graph",
    })?;

    let vertices = previous
        .edges
        .iter()
        .map(|edge| GraphNode {
            conditioned_set: vec![edge.conditioned.0, edge.conditioned.1],
            conditioning_set: edge.conditioning.clone(),
        })
        .collect::<Vec<_>>();

    let mut edges = Vec::new();
    for left in 0..vertices.len() {
        for right in (left + 1)..vertices.len() {
            if let Some(edge) = edge_info(left, right, &vertices, previous, truncated)? {
                edges.push(edge);
            }
        }
    }

    Ok(Graph { vertices, edges })
}

fn edge_info(
    left_idx: usize,
    right_idx: usize,
    vertices: &[GraphNode],
    previous: &InternalTree,
    truncated: bool,
) -> Result<Option<GraphEdge>, CopulaError> {
    let left = &vertices[left_idx];
    let right = &vertices[right_idx];
    let l1 = left
        .conditioned_set
        .iter()
        .chain(left.conditioning_set.iter())
        .copied()
        .collect::<Vec<_>>();
    let l2 = right
        .conditioned_set
        .iter()
        .chain(right.conditioning_set.iter())
        .copied()
        .collect::<Vec<_>>();
    let conditioned = (
        *set_difference(&l1, &l2).first().ok_or(FitError::Failed {
            reason: "proximity condition failed while building the next vine graph",
        })?,
        *set_difference(&l2, &l1).first().ok_or(FitError::Failed {
            reason: "proximity condition failed while building the next vine graph",
        })?,
    );
    let conditioning = set_intersection(&l1, &l2);
    if conditioned.0 == conditioned.1 {
        return Ok(None);
    }

    let same = if left.conditioned_set[0] == right.conditioned_set[0]
        || left.conditioned_set[1] == right.conditioned_set[0]
    {
        right.conditioned_set[0]
    } else if left.conditioned_set[0] == right.conditioned_set[1]
        || left.conditioned_set[1] == right.conditioned_set[1]
    {
        right.conditioned_set[1]
    } else {
        return Ok(None);
    };

    let left_edge = &previous.edges[left_idx];
    let right_edge = &previous.edges[right_idx];
    let left_data = if left_edge.conditioned.0 == same {
        left_edge.cond_on_second.clone()
    } else {
        left_edge.cond_on_first.clone()
    };
    let right_data = if right_edge.conditioned.0 == same {
        right_edge.cond_on_second.clone()
    } else {
        right_edge.cond_on_first.clone()
    };
    let weight = if truncated {
        1.0
    } else {
        crate::stats::kendall_tau_bivariate(left_data.as_ref(), right_data.as_ref())?.abs()
    };

    Ok(Some(GraphEdge {
        endpoints: (left_idx, right_idx),
        weight,
        conditioned_set: conditioned,
        conditioning_set: conditioning,
        left_data,
        right_data,
    }))
}


fn set_difference(left: &[usize], right: &[usize]) -> Vec<usize> {
    left.iter()
        .copied()
        .filter(|value| !right.contains(value))
        .collect()
}

fn set_intersection(left: &[usize], right: &[usize]) -> Vec<usize> {
    left.iter()
        .copied()
        .filter(|value| right.contains(value))
        .collect()
}

fn c_vine_order(data: &PseudoObs, options: &VineFitOptions) -> Result<Vec<usize>, CopulaError> {
    let tau = try_kendall_tau_matrix(data, options.base.exec)?;
    let mut indices = (0..data.dim()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        let left_score = (0..data.dim())
            .filter(|idx| *idx != *left)
            .map(|idx| tau[(*left, idx)].abs())
            .sum::<f64>();
        let right_score = (0..data.dim())
            .filter(|idx| *idx != *right)
            .map(|idx| tau[(*right, idx)].abs())
            .sum::<f64>();
        right_score
            .total_cmp(&left_score)
            .then_with(|| left.cmp(right))
    });
    Ok(indices)
}

fn d_vine_order(data: &PseudoObs, options: &VineFitOptions) -> Result<Vec<usize>, CopulaError> {
    let tau = try_kendall_tau_matrix(data, options.base.exec)?;
    Ok(d_vine_order_from_weights(&tau.mapv(f64::abs)))
}

fn d_vine_order_from_weights(weights: &Array2<f64>) -> Vec<usize> {
    let dim = weights.nrows();
    let mut best_pair = (0, 1);
    let mut best_score = f64::NEG_INFINITY;
    for left in 0..dim {
        for right in (left + 1)..dim {
            let score = weights[(left, right)];
            if score > best_score {
                best_score = score;
                best_pair = (left, right);
            }
        }
    }

    let mut order = vec![best_pair.0, best_pair.1];
    let mut remaining = (0..dim)
        .filter(|idx| *idx != best_pair.0 && *idx != best_pair.1)
        .collect::<Vec<_>>();

    while !remaining.is_empty() {
        let left_end = order[0];
        let right_end = *order.last().expect("order is non-empty");
        let mut best_idx = 0usize;
        let mut append_left = false;
        let mut score = f64::NEG_INFINITY;

        for (idx, candidate) in remaining.iter().enumerate() {
            let left_score = weights[(*candidate, left_end)];
            if left_score > score {
                score = left_score;
                best_idx = idx;
                append_left = true;
            }

            let right_score = weights[(*candidate, right_end)];
            if right_score > score {
                score = right_score;
                best_idx = idx;
                append_left = false;
            }
        }

        let candidate = remaining.remove(best_idx);
        if append_left {
            order.insert(0, candidate);
        } else {
            order.push(candidate);
        }
    }

    order
}

fn c_vine_gaussian_specs(
    order: &[usize],
    correlation: &Array2<f64>,
) -> Result<Vec<PairCopulaSpec>, CopulaError> {
    let dim = order.len();
    let mut specs = Vec::with_capacity(dim * (dim - 1) / 2);
    for level in 0..(dim - 1) {
        let root = order[level];
        let conditioning = &order[..level];
        for idx in (level + 1..dim).rev() {
            let variable = order[idx];
            let rho = partial_correlation(correlation, root, variable, conditioning)?;
            specs.push(PairCopulaSpec {
                family: PairCopulaFamily::Gaussian,
                rotation: Rotation::R0,
                params: PairCopulaParams::One(rho.clamp(-0.98, 0.98)),
            });
        }
    }
    Ok(specs)
}

fn d_vine_gaussian_specs(
    order: &[usize],
    correlation: &Array2<f64>,
) -> Result<Vec<PairCopulaSpec>, CopulaError> {
    let dim = order.len();
    let mut specs = Vec::with_capacity(dim * (dim - 1) / 2);
    for gap in 1..dim {
        for start in (0..(dim - gap)).rev() {
            let left = order[start];
            let right = order[start + gap];
            let conditioning = &order[(start + 1)..(start + gap)];
            let rho = partial_correlation(correlation, left, right, conditioning)?;
            specs.push(PairCopulaSpec {
                family: PairCopulaFamily::Gaussian,
                rotation: Rotation::R0,
                params: PairCopulaParams::One(rho.clamp(-0.98, 0.98)),
            });
        }
    }
    Ok(specs)
}

fn partial_correlation(
    correlation: &Array2<f64>,
    left: usize,
    right: usize,
    conditioning: &[usize],
) -> Result<f64, CopulaError> {
    if conditioning.is_empty() {
        return Ok(correlation[(left, right)]);
    }

    let mut indices = Vec::with_capacity(conditioning.len() + 2);
    indices.push(left);
    indices.push(right);
    indices.extend_from_slice(conditioning);

    let mut sub = Array2::zeros((indices.len(), indices.len()));
    for (row, source_row) in indices.iter().enumerate() {
        for (col, source_col) in indices.iter().enumerate() {
            sub[(row, col)] = correlation[(*source_row, *source_col)];
        }
    }

    let precision = inverse(&sub)?;
    Ok(-precision[(0, 1)] / (precision[(0, 0)] * precision[(1, 1)]).sqrt())
}

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            let parent = self.parent[value];
            self.parent[value] = self.find(parent);
        }
        self.parent[value]
    }

    fn union(&mut self, left: usize, right: usize) -> bool {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return false;
        }

        if self.rank[left_root] < self.rank[right_root] {
            self.parent[left_root] = right_root;
        } else if self.rank[left_root] > self.rank[right_root] {
            self.parent[right_root] = left_root;
        } else {
            self.parent[right_root] = left_root;
            self.rank[left_root] += 1;
        }
        true
    }
}
