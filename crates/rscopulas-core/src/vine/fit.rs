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
    stats::{
        hoeffding_d_bivariate, kendall_tau_bivariate, spearman_rho_bivariate,
        try_hoeffding_d_matrix, try_kendall_tau_matrix, try_spearman_rho_matrix,
    },
};

use super::{
    VineCopula, VineEdge, VineStructureKind, VineTree, default_family_set,
    structure::{
        build_model_from_trees, canonical_c_vine_trees, canonical_d_vine_trees, validate_order,
    },
};

/// Information criterion used to compare candidate pair-copula fits, and to
/// (optionally) decide vine truncation depth.
///
/// `Aic` and `Bic` apply at the per-edge level as before. `Mbicv` is the
/// modified vine BIC of Nagler, Bumann & Czado 2019 — per-edge family
/// selection still uses BIC (mBICV is not defined at the per-edge level),
/// but the **tree-level** mBICV penalty drives automatic truncation when
/// [`VineFitOptions::select_trunc_lvl`] is enabled. `psi0 ∈ (0, 1)` is a
/// prior on "edge is non-independent"; vinecopulib's default is `0.9`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SelectionCriterion {
    Aic,
    Bic,
    Mbicv { psi0: f64 },
}

/// Measure used to weight candidate edges when building each vine tree.
///
/// All three criteria produce a scalar in `[-1, 1]` and edge weights take the
/// absolute value, matching vinecopulib's `tools_select.cpp`. Choose `Rho`
/// when rank-based but monotone-dependence is enough (O(n log n), cheaper
/// than Kendall for some inputs); choose `Hoeffding` when pairs may exhibit
/// U-shaped or otherwise non-monotone dependence that rank correlations miss.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TreeCriterion {
    Tau,
    Rho,
    Hoeffding,
}

/// Algorithm used to build each vine tree from the candidate-edge graph.
///
/// `Kruskal` is the default maximum-spanning-tree search (same behaviour as
/// pre-this-change). `Prim` is an alternative MST that can differ only in
/// tie-breaking. `RandomWeighted` and `RandomUnweighted` sample spanning
/// trees via Wilson's loop-erased random walk, useful for structure-
/// uncertainty diagnostics; they require [`VineFitOptions::rng_seed`] for
/// reproducibility and are supported only for R-vines (C-vine's star and
/// D-vine's path are structural, not tree-search problems).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TreeAlgorithm {
    Kruskal,
    Prim,
    RandomWeighted,
    RandomUnweighted,
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
    /// Optional maximum tree level to fit and retain. When combined with
    /// `select_trunc_lvl`, this acts as an **upper cap** on the auto-
    /// selected level (matching vinecopulib's behaviour).
    pub truncation_level: Option<usize>,
    /// Optional absolute-dependence threshold (in the current tree
    /// criterion's units) for selecting Independence at an edge.
    pub independence_threshold: Option<f64>,
    /// Measure used to weight edges in the candidate-edge graph.
    pub tree_criterion: TreeCriterion,
    /// Algorithm used to build each tree from the weighted candidate graph.
    pub tree_algorithm: TreeAlgorithm,
    /// When `true` and `criterion = Mbicv { .. }`, truncation depth is
    /// selected automatically by running trees until cumulative mBICV stops
    /// improving; see [`SelectionCriterion::Mbicv`]. Ignored when the
    /// criterion is `Aic` or `Bic`.
    pub select_trunc_lvl: bool,
    /// RNG seed used by the stochastic tree algorithms. Ignored by the
    /// deterministic ones. `None` draws from the OS RNG.
    pub rng_seed: Option<u64>,
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
            tree_criterion: TreeCriterion::Tau,
            tree_algorithm: TreeAlgorithm::Kruskal,
            select_trunc_lvl: false,
            rng_seed: None,
        }
    }
}

#[derive(Clone)]
struct InternalEdgeFit {
    conditioned: (usize, usize),
    conditioning: Vec<usize>,
    parent_endpoints: (usize, usize),
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
    parent_endpoints: (usize, usize),
}

#[derive(Clone)]
struct GraphEdge {
    endpoints: (usize, usize),
    weight: f64,
    conditioned_set: (usize, usize),
    conditioning_set: Vec<usize>,
    parent_endpoints: (usize, usize),
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
        Self::fit_c_vine_with_order(data, &order, options)
    }

    /// Fits a C-vine with an explicit variable order.
    ///
    /// The first element of `order` becomes the C-vine root, and the resulting
    /// structure has that variable at the Rosenblatt anchor position. Use this
    /// to set up exact conditional sampling via [`Self::inverse_rosenblatt`]:
    /// variables intended to be conditioned on should occupy the leading
    /// positions of `order`.
    pub fn fit_c_vine_with_order(
        data: &PseudoObs,
        order: &[usize],
        options: &VineFitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        reject_non_default_tree_algorithm(options, "C")?;
        let (trees, diagnostics) = fit_canonical_vine(data, order, VineStructureKind::C, options)?;
        let model = build_model_from_trees(VineStructureKind::C, trees, options.truncation_level)?;
        Ok(FitResult { model, diagnostics })
    }

    /// Fits a simplified D-vine with pair-copula selection on each edge.
    pub fn fit_d_vine(
        data: &PseudoObs,
        options: &VineFitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        let order = d_vine_order(data, options)?;
        Self::fit_d_vine_with_order(data, &order, options)
    }

    /// Fits a D-vine with an explicit variable order.
    ///
    /// `order` defines the D-vine path in order, so `order[0]` is again the
    /// Rosenblatt anchor. Use this to pin conditioning variables to the
    /// leading diagonal positions for exact conditional sampling.
    pub fn fit_d_vine_with_order(
        data: &PseudoObs,
        order: &[usize],
        options: &VineFitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        reject_non_default_tree_algorithm(options, "D")?;
        let (trees, diagnostics) = fit_canonical_vine(data, order, VineStructureKind::D, options)?;
        let model = build_model_from_trees(VineStructureKind::D, trees, options.truncation_level)?;
        Ok(FitResult { model, diagnostics })
    }

    /// Fits a simplified R-vine using a Dissmann-style maximum spanning tree procedure.
    pub fn fit_r_vine(
        data: &PseudoObs,
        options: &VineFitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        let user_cap = options.truncation_level.unwrap_or(data.dim() - 1);
        let columns = collect_column_data(data);
        let mut graph = initialize_first_graph(data, &columns, options)?;
        let mut internal_trees = Vec::with_capacity(data.dim() - 1);
        let mut public_trees = Vec::with_capacity(data.dim() - 1);
        let mut per_tree_stats: Vec<TreeStats> = Vec::with_capacity(data.dim() - 1);
        let mut total_loglik = 0.0;
        let mut n_iter = 0usize;
        let mut rng = rng_from_seed(options.rng_seed);

        for level in 1..data.dim() {
            let truncated = level > user_cap;
            let mst = find_max_tree(
                &graph,
                VineStructureKind::R,
                truncated,
                options.tree_algorithm,
                &mut rng,
            )?;

            let fitted = if level == 1 {
                fit_first_tree(&mst, options, truncated)?
            } else {
                fit_tree_from_graph(&mst, options, truncated)?
            };
            let tree_loglik: f64 = fitted.iter().map(|edge| edge.loglik()).sum();
            let tree_df: f64 = fitted
                .iter()
                .map(|edge| edge.spec.parameter_count() as f64)
                .sum();
            let non_indep = fitted
                .iter()
                .filter(|edge| edge.spec.family != PairCopulaFamily::Independence)
                .count();
            total_loglik += tree_loglik;
            n_iter += fitted.len();
            per_tree_stats.push(TreeStats {
                loglik: tree_loglik,
                df: tree_df,
                non_indep,
                total_pairs: fitted.len(),
            });
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
                        parent_endpoints: edge.parent_endpoints,
                        cond_on_first: edge.cond_on_first.clone(),
                        cond_on_second: edge.cond_on_second.clone(),
                    })
                    .collect(),
            });

            if level < data.dim() - 1 {
                graph = build_next_graph(
                    &mst,
                    internal_trees.last(),
                    truncated,
                    options.tree_criterion,
                )?;
            }
        }

        if public_trees.len() != data.dim() - 1 {
            return Err(FitError::Failed {
                reason: "R-vine fit produced fewer trees than dim - 1",
            }
            .into());
        }

        // Auto-truncation via mBICV. Walks backward from the full fit,
        // dropping trees whose mBICV contribution is positive (i.e. adding
        // them hurts the total criterion more than it helps). The user cap
        // `options.truncation_level` is a strict upper bound.
        let n_obs = data.n_obs() as f64;
        let auto_trunc_level = auto_truncate_mbicv(
            &per_tree_stats,
            &options.criterion,
            options.select_trunc_lvl,
            user_cap,
            n_obs,
        );

        // Apply truncation: replace every edge at level > auto_trunc_level
        // with Independence, and drop its loglik contribution. This reuses
        // the existing "fill with Independence" representation so that the
        // downstream model builder doesn't need to know about mBICV at all.
        if auto_trunc_level < public_trees.len() {
            for level_idx in auto_trunc_level..public_trees.len() {
                total_loglik -= per_tree_stats[level_idx].loglik;
                for edge in &mut public_trees[level_idx].edges {
                    edge.copula = PairCopulaSpec::independence();
                }
            }
        }

        // The model's `truncation_level` records the *effective* truncation
        // (min of user cap and auto-selected cap). `None` means "no
        // truncation applied" — only set this when trees run to full depth.
        let effective_trunc = if auto_trunc_level == public_trees.len() {
            options.truncation_level
        } else {
            Some(auto_trunc_level)
        };

        let model = build_model_from_trees(VineStructureKind::R, public_trees, effective_trunc)?;
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

/// Per-tree aggregates needed for the mBICV auto-truncation pass.
#[derive(Clone, Copy)]
struct TreeStats {
    loglik: f64,
    df: f64,
    non_indep: usize,
    total_pairs: usize,
}

/// Computes per-tree mBICV contributions and returns the largest truncation
/// level `L` (number of trees to retain) such that including trees `1..=L`
/// minimises cumulative mBICV, clamped at `user_cap`.
///
/// When `select` is false or the criterion is not `Mbicv`, this returns the
/// user cap unchanged — i.e. the pre-this-change behaviour.
fn auto_truncate_mbicv(
    stats: &[TreeStats],
    criterion: &SelectionCriterion,
    select: bool,
    user_cap: usize,
    n_obs: f64,
) -> usize {
    let psi0 = match criterion {
        SelectionCriterion::Mbicv { psi0 } if select => *psi0,
        _ => return user_cap.min(stats.len()),
    };
    if !(0.0..1.0).contains(&psi0) || stats.is_empty() {
        return user_cap.min(stats.len());
    }

    // Per-tree mBICV: `-2·ll_t + log(n)·df_t − 2·log_prior_t`, with
    // `log_prior_t = q_t·log(ψ₀^(t+1)) + (M_t − q_t)·log(1 − ψ₀^(t+1))`.
    // Tree index `t` in the paper's notation is 0-based; we use 1-based
    // levels here, so the depth exponent is `level` rather than `level + 1`.
    let contributions: Vec<f64> = stats
        .iter()
        .enumerate()
        .map(|(idx, tree)| {
            let level = idx + 1;
            let psi_level = psi0.powi(level as i32);
            let q = tree.non_indep as f64;
            let mt = tree.total_pairs as f64;
            let log_prior = q * psi_level.ln() + (mt - q) * (1.0 - psi_level).ln();
            -2.0 * tree.loglik + n_obs.ln() * tree.df - 2.0 * log_prior
        })
        .collect();

    // Walk backwards: drop any tail tree whose contribution is positive.
    // Bounded above by the user cap.
    let mut selected = contributions.len().min(user_cap);
    while selected > 0 && contributions[selected - 1] > 0.0 {
        selected -= 1;
    }
    selected
}

#[derive(Clone)]
struct FittedGraphEdge {
    conditioned: (usize, usize),
    conditioning: Vec<usize>,
    parent_endpoints: (usize, usize),
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
                parent_endpoints: edge.conditioned,
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
                    parent_endpoints: edge.parent_endpoints,
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
    let criterion = tree_criterion_matrix(data, options.base.exec, options.tree_criterion)?;
    let mut edges = Vec::new();
    for left in 0..data.dim() {
        for right in (left + 1)..data.dim() {
            edges.push(GraphEdge {
                endpoints: (left, right),
                weight: criterion[(left, right)].abs(),
                conditioned_set: (left, right),
                conditioning_set: Vec::new(),
                parent_endpoints: (left, right),
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
                parent_endpoints: (idx, idx),
            })
            .collect(),
        edges,
    })
}

fn find_max_tree(
    graph: &Graph,
    kind: VineStructureKind,
    truncated: bool,
    algorithm: TreeAlgorithm,
    rng: &mut rand::rngs::StdRng,
) -> Result<Graph, CopulaError> {
    let vertices = graph.vertices.clone();
    let edges = if truncated {
        spanning_tree_by_dfs(graph)?
    } else {
        match kind {
            VineStructureKind::C => star_tree(graph)?,
            VineStructureKind::D => path_tree(graph)?,
            VineStructureKind::R => match algorithm {
                TreeAlgorithm::Kruskal => kruskal_max_spanning_tree(graph)?,
                TreeAlgorithm::Prim => prim_max_spanning_tree(graph)?,
                TreeAlgorithm::RandomWeighted => wilson_spanning_tree(graph, rng, true)?,
                TreeAlgorithm::RandomUnweighted => wilson_spanning_tree(graph, rng, false)?,
            },
        }
    };
    Ok(Graph { vertices, edges })
}

/// Rejects non-default tree algorithms for C-vine and D-vine fits. Those
/// structures are built by fixed routines (star / path) rather than a
/// tree-search, so the `TreeAlgorithm` choice does not apply.
fn reject_non_default_tree_algorithm(
    options: &VineFitOptions,
    kind_label: &'static str,
) -> Result<(), CopulaError> {
    match options.tree_algorithm {
        TreeAlgorithm::Kruskal => Ok(()),
        _ => Err(FitError::Failed {
            reason: match kind_label {
                "C" => "tree_algorithm override is not supported for C-vines (use Kruskal)",
                _ => "tree_algorithm override is not supported for D-vines (use Kruskal)",
            },
        }
        .into()),
    }
}

fn rng_from_seed(seed: Option<u64>) -> rand::rngs::StdRng {
    use rand::SeedableRng;
    match seed {
        Some(value) => rand::rngs::StdRng::seed_from_u64(value),
        None => rand::rngs::StdRng::from_os_rng(),
    }
}

fn kruskal_max_spanning_tree(graph: &Graph) -> Result<Vec<GraphEdge>, CopulaError> {
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

/// Prim's maximum-spanning-tree algorithm. Grows a tree from vertex 0 using
/// a max-heap on the frontier edges. On a complete weighted graph the total
/// weight of the returned tree equals Kruskal's up to tie-breaking; this
/// function is primarily an alternative MST the user can opt into for
/// structure-uncertainty diagnostics.
fn prim_max_spanning_tree(graph: &Graph) -> Result<Vec<GraphEdge>, CopulaError> {
    let n = graph.vertices.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let adjacency = build_adjacency(graph);

    // Max-heap over (weight, tie-break endpoints, edge index).
    use std::collections::BinaryHeap;
    let mut in_tree = vec![false; n];
    in_tree[0] = true;
    let mut heap: BinaryHeap<(WeightKey, (usize, usize), usize)> = BinaryHeap::new();
    for &(other, edge_idx) in &adjacency[0] {
        let edge = &graph.edges[edge_idx];
        heap.push((
            WeightKey(edge.weight),
            (edge.endpoints.0, edge.endpoints.1),
            edge_idx,
        ));
        let _ = other; // silence unused warning; `other` is implicit in endpoints
    }
    let mut selected = Vec::with_capacity(n.saturating_sub(1));
    while let Some((_, _, edge_idx)) = heap.pop() {
        let edge = &graph.edges[edge_idx];
        let a = edge.endpoints.0;
        let b = edge.endpoints.1;
        let (keep, added) = match (in_tree[a], in_tree[b]) {
            (true, false) => (true, b),
            (false, true) => (true, a),
            _ => (false, 0),
        };
        if !keep {
            continue;
        }
        selected.push(edge.clone());
        in_tree[added] = true;
        for &(_, new_idx) in &adjacency[added] {
            let new_edge = &graph.edges[new_idx];
            let (a, b) = new_edge.endpoints;
            if in_tree[a] ^ in_tree[b] {
                heap.push((
                    WeightKey(new_edge.weight),
                    (a, b),
                    new_idx,
                ));
            }
        }
        if selected.len() + 1 == n {
            break;
        }
    }

    if selected.len() + 1 != n {
        return Err(FitError::Failed {
            reason: "graph is disconnected and cannot form a vine tree",
        }
        .into());
    }
    selected.sort_by_key(|edge| edge.endpoints);
    Ok(selected)
}

/// Samples a spanning tree by Wilson's algorithm — loop-erased random walks
/// from each uncovered vertex back to the growing tree. With `weighted =
/// true`, transition probability from `u` to a neighbour `v` is proportional
/// to `1 / (edge_weight(u, v) + 1e-10)` — matching vinecopulib's weighting,
/// which favours edges with strong measured dependence. With `weighted =
/// false`, transitions are uniform over neighbours; the resulting
/// distribution over spanning trees is uniform (among trees satisfying the
/// R-vine proximity condition imposed by the candidate graph).
fn wilson_spanning_tree(
    graph: &Graph,
    rng: &mut rand::rngs::StdRng,
    weighted: bool,
) -> Result<Vec<GraphEdge>, CopulaError> {
    use rand::Rng;
    let n = graph.vertices.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let adjacency = build_adjacency(graph);
    // Check connectivity up front; a disconnected candidate graph cannot
    // yield any spanning tree regardless of sampling scheme.
    if !is_connected(&adjacency, n) {
        return Err(FitError::Failed {
            reason: "graph is disconnected and cannot form a vine tree",
        }
        .into());
    }

    let root = rng.random_range(0..n);
    let mut in_tree = vec![false; n];
    in_tree[root] = true;
    // For each non-root vertex we store the edge index that enters it in the
    // sampled tree (parent in the LERW rooted at `root`).
    let mut entry_edge: Vec<Option<usize>> = vec![None; n];

    for start in 0..n {
        if in_tree[start] {
            continue;
        }
        // next[v] = (neighbour u it points to, edge idx used). Loop erasure
        // is achieved by re-assigning `next[v]` each time we visit `v`; the
        // final walk follows the last-assigned pointer from `start` until
        // hitting the tree.
        let mut next: Vec<Option<(usize, usize)>> = vec![None; n];
        let mut current = start;
        while !in_tree[current] {
            let neighbours = &adjacency[current];
            if neighbours.is_empty() {
                return Err(FitError::Failed {
                    reason: "graph is disconnected and cannot form a vine tree",
                }
                .into());
            }
            let (choice_vertex, choice_edge) = if weighted {
                choose_weighted_neighbour(neighbours, &graph.edges, rng)
            } else {
                let pick = rng.random_range(0..neighbours.len());
                neighbours[pick]
            };
            next[current] = Some((choice_vertex, choice_edge));
            current = choice_vertex;
        }
        // Walk forward from `start`, stamping entry edges.
        let mut walker = start;
        while !in_tree[walker] {
            let (next_vertex, edge_idx) = next[walker].expect("walker in unvisited vertex");
            entry_edge[walker] = Some(edge_idx);
            in_tree[walker] = true;
            walker = next_vertex;
        }
    }

    let mut selected = Vec::with_capacity(n.saturating_sub(1));
    for v in 0..n {
        if let Some(edge_idx) = entry_edge[v] {
            selected.push(graph.edges[edge_idx].clone());
        }
    }
    if selected.len() + 1 != n {
        return Err(FitError::Failed {
            reason: "graph is disconnected and cannot form a vine tree",
        }
        .into());
    }
    selected.sort_by_key(|edge| edge.endpoints);
    Ok(selected)
}

fn build_adjacency(graph: &Graph) -> Vec<Vec<(usize, usize)>> {
    let mut adjacency: Vec<Vec<(usize, usize)>> = vec![Vec::new(); graph.vertices.len()];
    for (idx, edge) in graph.edges.iter().enumerate() {
        let (a, b) = edge.endpoints;
        adjacency[a].push((b, idx));
        adjacency[b].push((a, idx));
    }
    adjacency
}

fn is_connected(adjacency: &[Vec<(usize, usize)>], n: usize) -> bool {
    if n == 0 {
        return true;
    }
    let mut visited = vec![false; n];
    let mut stack = vec![0usize];
    visited[0] = true;
    let mut count = 1;
    while let Some(node) = stack.pop() {
        for &(neighbour, _) in &adjacency[node] {
            if !visited[neighbour] {
                visited[neighbour] = true;
                count += 1;
                stack.push(neighbour);
            }
        }
    }
    count == n
}

fn choose_weighted_neighbour(
    neighbours: &[(usize, usize)],
    edges: &[GraphEdge],
    rng: &mut rand::rngs::StdRng,
) -> (usize, usize) {
    use rand::Rng;
    // Transition weight ∝ 1 / (1 − |crit| + ε). Equivalent to vinecopulib's
    // `inv_weights[e] = 1.0 / (original_weights[e] + 1e-10)` with
    // `original_weights[e] = 1 − crit` — see `tools_select.ipp:950-965`.
    //
    // Our stored `edge.weight` IS `|crit|` (we maximise weight), so the
    // conversion is `1 − edge.weight` before inverting. `boost::random_
    // spanning_tree` samples a tree with probability proportional to the
    // product of its weight_map entries, i.e. Wilson's loop-erased walk
    // takes transitions with probability ∝ `weight_map[e]` per edge. High
    // `|crit|` → small `(1 − |crit|)` → large `weight_map` → preferred.
    let inv_weight = |idx: usize| -> f64 { 1.0 / ((1.0 - edges[idx].weight) + 1e-10) };
    let total: f64 = neighbours.iter().map(|(_, idx)| inv_weight(*idx)).sum();
    let mut draw = rng.random::<f64>() * total;
    for &(vertex, edge_idx) in neighbours {
        let w = inv_weight(edge_idx);
        if draw <= w {
            return (vertex, edge_idx);
        }
        draw -= w;
    }
    *neighbours.last().expect("neighbours are non-empty")
}

/// Newtype-wrapped `f64` that implements a total order via `total_cmp`, so we
/// can store `(weight, …)` tuples directly in a `BinaryHeap` — vanilla `f64`
/// is only `PartialOrd`.
#[derive(Clone, Copy, PartialEq)]
struct WeightKey(f64);

impl Eq for WeightKey {}
impl Ord for WeightKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
impl PartialOrd for WeightKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
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
    let strategy = resolve_strategy(
        options.base.exec,
        Operation::PairFitScoring,
        tree.edges.len(),
    )?;
    let inner_options = edge_fit_options(options, strategy);
    parallel_try_map_range_collect(tree.edges.len(), strategy, |idx| {
        let edge = &tree.edges[idx];
        let fit = if truncated {
            truncated_pair_fit(&edge.left_data, &edge.right_data)
        } else {
            fit_pair_copula(
                edge.left_data.as_ref(),
                edge.right_data.as_ref(),
                &inner_options,
            )?
        };
        Ok(FittedGraphEdge {
            conditioned: edge.conditioned_set,
            conditioning: edge.conditioning_set.clone(),
            parent_endpoints: edge.parent_endpoints,
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
    let strategy = resolve_strategy(
        options.base.exec,
        Operation::PairFitScoring,
        tree.edges.len(),
    )?;
    let inner_options = edge_fit_options(options, strategy);
    parallel_try_map_range_collect(tree.edges.len(), strategy, |idx| {
        let edge = &tree.edges[idx];
        let fit = if truncated {
            truncated_pair_fit(&edge.left_data, &edge.right_data)
        } else {
            fit_pair_copula(
                edge.left_data.as_ref(),
                edge.right_data.as_ref(),
                &inner_options,
            )?
        };
        Ok(FittedGraphEdge {
            conditioned: edge.conditioned_set,
            conditioning: edge.conditioning_set.clone(),
            parent_endpoints: edge.parent_endpoints,
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
    (0..data.dim())
        .map(|column| shared_column(data, column))
        .collect()
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
    tree_criterion: TreeCriterion,
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
            parent_endpoints: edge.parent_endpoints,
        })
        .collect::<Vec<_>>();

    let mut edges = Vec::new();
    for left in 0..vertices.len() {
        for right in (left + 1)..vertices.len() {
            if let Some(edge) = edge_info(
                left,
                right,
                &vertices,
                previous,
                truncated,
                tree_criterion,
            )? {
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
    options_tree_criterion: TreeCriterion,
) -> Result<Option<GraphEdge>, CopulaError> {
    let left = &vertices[left_idx];
    let right = &vertices[right_idx];
    let Some(shared) = shared_parent_endpoint(left.parent_endpoints, right.parent_endpoints) else {
        return Ok(None);
    };
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
    let left_only = set_difference(&l1, &l2);
    let right_only = set_difference(&l2, &l1);
    if left_only.len() != 1 || right_only.len() != 1 {
        return Ok(None);
    }
    let conditioned = (left_only[0], right_only[0]);
    let conditioning = set_intersection(&l1, &l2);
    if conditioned.0 == conditioned.1 {
        return Ok(None);
    }

    let left_edge = &previous.edges[left_idx];
    let right_edge = &previous.edges[right_idx];
    let left_data = if left.parent_endpoints.0 == shared {
        left_edge.cond_on_second.clone()
    } else if left.parent_endpoints.1 == shared {
        left_edge.cond_on_first.clone()
    } else {
        return Ok(None);
    };
    let right_data = if right.parent_endpoints.0 == shared {
        right_edge.cond_on_second.clone()
    } else if right.parent_endpoints.1 == shared {
        right_edge.cond_on_first.clone()
    } else {
        return Ok(None);
    };
    let weight = if truncated {
        1.0
    } else {
        tree_criterion_bivariate(
            left_data.as_ref(),
            right_data.as_ref(),
            options_tree_criterion,
        )?
        .abs()
    };

    Ok(Some(GraphEdge {
        endpoints: (left_idx, right_idx),
        weight,
        conditioned_set: conditioned,
        conditioning_set: conditioning,
        parent_endpoints: (left_idx, right_idx),
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

fn shared_parent_endpoint(left: (usize, usize), right: (usize, usize)) -> Option<usize> {
    if left.0 == right.0 || left.0 == right.1 {
        Some(left.0)
    } else if left.1 == right.0 || left.1 == right.1 {
        Some(left.1)
    } else {
        None
    }
}

fn c_vine_order(data: &PseudoObs, options: &VineFitOptions) -> Result<Vec<usize>, CopulaError> {
    let criterion = tree_criterion_matrix(data, options.base.exec, options.tree_criterion)?;
    let mut indices = (0..data.dim()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        let left_score = (0..data.dim())
            .filter(|idx| *idx != *left)
            .map(|idx| criterion[(*left, idx)].abs())
            .sum::<f64>();
        let right_score = (0..data.dim())
            .filter(|idx| *idx != *right)
            .map(|idx| criterion[(*right, idx)].abs())
            .sum::<f64>();
        right_score
            .total_cmp(&left_score)
            .then_with(|| left.cmp(right))
    });
    Ok(indices)
}

fn d_vine_order(data: &PseudoObs, options: &VineFitOptions) -> Result<Vec<usize>, CopulaError> {
    let criterion = tree_criterion_matrix(data, options.base.exec, options.tree_criterion)?;
    Ok(d_vine_order_from_weights(&criterion.mapv(f64::abs)))
}

/// Dispatches the pairwise tree-criterion matrix computation to the backend
/// helper that matches the caller's [`TreeCriterion`] choice. All three
/// helpers return a symmetric `d × d` matrix with unit diagonal and values
/// in `[−1, 1]`.
fn tree_criterion_matrix(
    data: &PseudoObs,
    exec: ExecPolicy,
    criterion: TreeCriterion,
) -> Result<Array2<f64>, CopulaError> {
    match criterion {
        TreeCriterion::Tau => try_kendall_tau_matrix(data, exec),
        TreeCriterion::Rho => try_spearman_rho_matrix(data, exec),
        TreeCriterion::Hoeffding => try_hoeffding_d_matrix(data, exec),
    }
}

/// Bivariate-input version of [`tree_criterion_matrix`], used by the tree-
/// level edge-info pass on each vine level beyond the first. Returns the
/// raw (signed) measure — callers take `.abs()` to get the edge weight.
fn tree_criterion_bivariate(
    u: &[f64],
    v: &[f64],
    criterion: TreeCriterion,
) -> Result<f64, CopulaError> {
    match criterion {
        TreeCriterion::Tau => kendall_tau_bivariate(u, v),
        TreeCriterion::Rho => spearman_rho_bivariate(u, v),
        TreeCriterion::Hoeffding => hoeffding_d_bivariate(u, v),
    }
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
