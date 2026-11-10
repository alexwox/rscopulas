//! Hierarchical Archimedean copulas: tree validation, `log_pdf`, sampling, and fitting.
//!
//! **User guide** (density paths, which `sample()` scenarios are reliable, mixed-family
//! caveats): see `docs/hac.md` in the repository root.

use ndarray::Array2;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Exp1, Gamma};
use statrs::function::gamma::ln_gamma;

use crate::{
    archimedean_math::frank,
    data::PseudoObs,
    domain::{
        ClaytonCopula, CopulaModel, EvalOptions, FitDiagnostics, FrankCopula, GumbelHougaardCopula,
        HacFamily, HacFitMethod, HacFitOptions, HacNode, HacStructureMethod, HacTree,
        HierarchicalArchimedeanCopula, SampleOptions,
    },
    errors::{CopulaError, FitError},
    fit::FitResult,
    paircopula::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation},
    stats::try_kendall_tau_matrix,
};

#[derive(Debug, Clone, Copy)]
pub struct ValidationSummary {
    pub dim: usize,
    pub exact: bool,
    pub exact_loglik: bool,
}

#[derive(Debug, Clone)]
struct Cluster {
    tree: HacTree,
    leaves: Vec<usize>,
}

#[derive(Debug, Clone)]
struct NodeWork {
    family: HacFamily,
    theta: f64,
    direct_pairs: Vec<(usize, usize)>,
}

pub fn validate_tree(root: &HacTree) -> Result<ValidationSummary, CopulaError> {
    let mut leaves = Vec::new();
    validate_tree_inner(root, None, &mut leaves)?;
    leaves.sort_unstable();
    leaves.dedup();
    if leaves.len() < 2 {
        return Err(FitError::UnsupportedDimension {
            family: "HierarchicalArchimedean",
            dim: leaves.len(),
        }
        .into());
    }
    for (expected, actual) in leaves.iter().enumerate() {
        if *actual != expected {
            return Err(FitError::Failed {
                reason: "HAC trees must cover each column index exactly once starting at 0",
            }
            .into());
        }
    }
    Ok(ValidationSummary {
        dim: leaves.len(),
        exact: tree_is_exact(root),
        exact_loglik: tree_has_exact_loglik(root),
    })
}

pub fn leaf_order(root: &HacTree) -> Vec<usize> {
    let mut out = Vec::new();
    collect_leaf_order(root, &mut out);
    out
}

pub fn families_preorder(root: &HacTree) -> Vec<HacFamily> {
    let mut out = Vec::new();
    collect_families_preorder(root, &mut out);
    out
}

pub fn parameters_preorder(root: &HacTree) -> Vec<f64> {
    let mut out = Vec::new();
    collect_parameters_preorder(root, &mut out);
    out
}

pub fn fit_hac(
    data: &PseudoObs,
    root: Option<HacTree>,
    options: &HacFitOptions,
) -> Result<FitResult<HierarchicalArchimedeanCopula>, CopulaError> {
    options.base.validate()?;
    if matches!(
        options.fit_method,
        HacFitMethod::FullMle | HacFitMethod::Smle | HacFitMethod::Dmle
    ) {
        return Err(FitError::Failed {
            reason: "requested HAC optimizer is not implemented; use tau_init or composite_mle",
        }
        .into());
    }
    if options.family_set.is_empty()
        || !options.collapse_eps.is_finite()
        || options.collapse_eps < 0.0
    {
        return Err(FitError::Failed {
            reason: "invalid HAC fitting options",
        }
        .into());
    }
    if options.mc_samples != 0 {
        return Err(FitError::Failed {
            reason: "HAC simulated likelihood is not implemented; mc_samples must be zero",
        }
        .into());
    }
    let mut progress = HacFitProgress {
        iterations: 0,
        converged: true,
    };
    let tau = try_kendall_tau_matrix(data, options.base.exec)?;
    let structure_method = if root.is_some() {
        HacStructureMethod::GivenTree
    } else {
        options.structure_method
    };
    let mut tree = match root {
        Some(root) => {
            let summary = validate_tree(&root)?;
            if summary.dim != data.dim() {
                return Err(FitError::Failed {
                    reason: "input dimension does not match HAC tree dimension",
                }
                .into());
            }
            root
        }
        None if options.structure_method == HacStructureMethod::GivenTree => {
            return Err(FitError::Failed {
                reason: "given_tree requires an explicit HAC tree",
            }
            .into());
        }
        None => build_agglomerative_tree(&tau)?,
    };

    tree = fit_tree_node(tree, data, &tau, options, &mut progress)?;
    if matches!(
        structure_method,
        HacStructureMethod::AgglomerativeTauThenCollapse
    ) {
        tree = collapse_tree(tree, options.collapse_eps);
        tree = fit_tree_node(tree, data, &tau, options, &mut progress)?;
    }
    project_same_family_nesting(&mut tree);

    let validation = validate_tree(&tree)?;
    if !options.allow_experimental && !validation.exact {
        return Err(FitError::Failed {
            reason: "experimental mixed-family HAC fitting is disabled",
        }
        .into());
    }
    let model = HierarchicalArchimedeanCopula::from_parts(
        tree,
        structure_method,
        options.fit_method,
        validation.exact,
        validation.exact_loglik,
        false,
        0,
    )?;

    let loglik = if validation.exact_loglik {
        model
            .log_pdf(
                data,
                &EvalOptions {
                    exec: options.base.exec,
                    clip_eps: options.base.clip_eps,
                },
            )?
            .iter()
            .sum()
    } else {
        composite_loglik(&model, data, options.base.clip_eps)?
    };
    let n_params = internal_node_count(model.tree()) as f64;
    let n_obs = data.n_obs() as f64;
    let diagnostics = FitDiagnostics {
        likelihood_kind: if validation.exact_loglik {
            crate::domain::LikelihoodKind::Joint
        } else {
            crate::domain::LikelihoodKind::Composite
        },
        loglik,
        aic: if validation.exact_loglik {
            2.0 * n_params - 2.0 * loglik
        } else {
            f64::NAN
        },
        bic: if validation.exact_loglik {
            n_params * n_obs.ln() - 2.0 * loglik
        } else {
            f64::NAN
        },
        converged: progress.converged,
        n_iter: progress.iterations,
    };
    Ok(FitResult { model, diagnostics })
}

pub fn log_pdf(
    model: &HierarchicalArchimedeanCopula,
    data: &PseudoObs,
    options: &EvalOptions,
) -> Result<Vec<f64>, CopulaError> {
    options.validate()?;
    if data.dim() != model.dim() {
        return Err(FitError::Failed {
            reason: "input dimension does not match model dimension",
        }
        .into());
    }
    if let Some(values) = exact_exchangeable_log_pdf(model, data, options)? {
        return Ok(values);
    }
    Err(FitError::Failed { reason: "joint density is unavailable for nested HAC; use composite_log_pdf for an explicitly composite score" }.into())
}

pub fn sample<R: Rng + ?Sized>(
    model: &HierarchicalArchimedeanCopula,
    n: usize,
    rng: &mut R,
    options: &SampleOptions,
) -> Result<Array2<f64>, CopulaError> {
    crate::backend::resolve_strategy(options.exec, crate::backend::Operation::Sample, n)?;
    if let Some(values) = exact_exchangeable_sample(model, n, rng, options)? {
        return Ok(values);
    }
    // Phase 1 (sequential): consume the RNG exactly as a row-by-row sampler
    // would, recording the raw draws. Phase 2 (parallel): turn each row's
    // draws into frailties and observations. Only phase 2 is expensive — a
    // nested Clayton or Frank frailty is a root search on a numerically
    // inverted Laplace transform — and it touches no random state.
    let root = node_of(model.tree())?;
    let draws_per_row = 1 + count_subtree_draws(model.tree());
    let mut draws = Vec::with_capacity(n * draws_per_row);
    for _ in 0..n {
        draws.push(sample_root_frailty(root.family, root.theta, rng)?);
        draw_subtree(model.tree(), rng, &mut draws);
    }
    debug_assert_eq!(draws.len(), n * draws_per_row);

    let dim = model.dim();
    let weights = stehfest_weights(12);
    let mut values = vec![0.0; n * dim];
    let fill = |(row, row_draws): (&mut [f64], &[f64])| -> Result<(), CopulaError> {
        let mut cursor = 1;
        apply_subtree(
            model.tree(),
            Frailty::root(root.family, row_draws[0]),
            row_draws,
            &mut cursor,
            &weights,
            row,
        )
    };
    if n >= 16 {
        use rayon::prelude::*;
        values
            .par_chunks_mut(dim)
            .zip(draws.par_chunks(draws_per_row))
            .with_min_len(8)
            .try_for_each(fill)?;
    } else {
        values
            .chunks_mut(dim)
            .zip(draws.chunks(draws_per_row))
            .try_for_each(fill)?;
    }
    Ok(Array2::from_shape_vec((n, dim), values)
        .expect("sample buffer should match the requested shape"))
}

fn validate_tree_inner(
    tree: &HacTree,
    parent: Option<(HacFamily, f64)>,
    leaves: &mut Vec<usize>,
) -> Result<(), CopulaError> {
    match tree {
        HacTree::Leaf(index) => {
            if leaves.contains(index) {
                return Err(FitError::Failed {
                    reason: "HAC tree contains duplicate leaf indices",
                }
                .into());
            }
            leaves.push(*index);
        }
        HacTree::Node(node) => {
            validate_node_parameter(node.family, node.theta)?;
            if node.children.len() < 2 {
                return Err(FitError::Failed {
                    reason: "HAC internal nodes must have at least two children",
                }
                .into());
            }
            if let Some((parent_family, parent_theta)) = parent
                && parent_family == node.family
                && parent_theta > node.theta + 1e-12
            {
                return Err(FitError::Failed {
                    reason: "same-family HAC nodes must satisfy theta_parent <= theta_child",
                }
                .into());
            }
            for child in &node.children {
                validate_tree_inner(child, Some((node.family, node.theta)), leaves)?;
            }
        }
    }
    Ok(())
}

fn collect_leaf_order(tree: &HacTree, out: &mut Vec<usize>) {
    match tree {
        HacTree::Leaf(index) => out.push(*index),
        HacTree::Node(node) => {
            for child in &node.children {
                collect_leaf_order(child, out);
            }
        }
    }
}

fn collect_families_preorder(tree: &HacTree, out: &mut Vec<HacFamily>) {
    if let HacTree::Node(node) = tree {
        out.push(node.family);
        for child in &node.children {
            collect_families_preorder(child, out);
        }
    }
}

fn collect_parameters_preorder(tree: &HacTree, out: &mut Vec<f64>) {
    if let HacTree::Node(node) = tree {
        out.push(node.theta);
        for child in &node.children {
            collect_parameters_preorder(child, out);
        }
    }
}

fn build_agglomerative_tree(tau: &Array2<f64>) -> Result<HacTree, CopulaError> {
    let dim = tau.nrows();
    let mut clusters = (0..dim)
        .map(|index| Cluster {
            tree: HacTree::Leaf(index),
            leaves: vec![index],
        })
        .collect::<Vec<_>>();
    while clusters.len() > 1 {
        let mut best_pair = None;
        let mut best_score = f64::NEG_INFINITY;
        for left in 0..clusters.len() {
            for right in (left + 1)..clusters.len() {
                let score = average_cross_tau(&clusters[left].leaves, &clusters[right].leaves, tau);
                if score > best_score {
                    best_score = score;
                    best_pair = Some((left, right));
                }
            }
        }
        let (left_idx, right_idx) = best_pair.ok_or(FitError::Failed {
            reason: "failed to build HAC clustering tree",
        })?;
        let right = clusters.remove(right_idx);
        let left = clusters.remove(left_idx);
        let mut leaves = left.leaves.clone();
        leaves.extend(right.leaves.iter().copied());
        leaves.sort_unstable();
        clusters.push(Cluster {
            tree: HacTree::Node(HacNode::new(
                HacFamily::Gumbel,
                1.01,
                vec![left.tree, right.tree],
            )),
            leaves,
        });
    }
    Ok(clusters.remove(0).tree)
}

fn average_cross_tau(left: &[usize], right: &[usize], tau: &Array2<f64>) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for &i in left {
        for &j in right {
            total += tau[(i, j)];
            count += 1;
        }
    }
    total / count as f64
}

/// Absolute tolerance on a node parameter in the composite-likelihood search.
const HAC_THETA_TOL: f64 = 1e-9;

struct HacFitProgress {
    iterations: usize,
    converged: bool,
}

fn fit_tree_node(
    tree: HacTree,
    data: &PseudoObs,
    tau: &Array2<f64>,
    options: &HacFitOptions,
    progress: &mut HacFitProgress,
) -> Result<HacTree, CopulaError> {
    match tree {
        HacTree::Leaf(_) => Ok(tree),
        HacTree::Node(node) => {
            let children = node
                .children
                .into_iter()
                .map(|child| fit_tree_node(child, data, tau, options, progress))
                .collect::<Result<Vec<_>, _>>()?;
            let child_groups = children.iter().map(leaves_of).collect::<Vec<_>>();
            let direct_pairs = direct_pairs_from_groups(&child_groups);
            let (family, theta) =
                fit_best_family_for_pairs(&direct_pairs, data, tau, options, progress)?;
            Ok(HacTree::Node(HacNode::new(family, theta, children)))
        }
    }
}

fn fit_best_family_for_pairs(
    direct_pairs: &[(usize, usize)],
    data: &PseudoObs,
    tau: &Array2<f64>,
    options: &HacFitOptions,
    progress: &mut HacFitProgress,
) -> Result<(HacFamily, f64), CopulaError> {
    let mut best = None;
    let view = data.as_view();
    let mean_tau = if direct_pairs.is_empty() {
        0.2
    } else {
        direct_pairs.iter().map(|&(i, j)| tau[(i, j)]).sum::<f64>() / direct_pairs.len() as f64
    }
    .clamp(1e-4, 1.0 - 1e-4);

    // Candidate families are independent one-dimensional searches.
    let fit_family = |idx: usize| -> Result<(HacFamily, f64, f64, bool, usize), CopulaError> {
        let family = &options.family_set[idx];
        let init = theta_from_tau(*family, mean_tau)?;
        let (low, upper_seed) = family_parameter_bounds(*family);
        let upper = (init * 4.0 + 2.0).max(upper_seed);
        let (theta, iterations, converged) = if options.fit_method == HacFitMethod::TauInit {
            (init, 0, true)
        } else {
            // Composite log-likelihood of the node: one independent sum per
            // direct pair, added up in pair order so the result does not
            // depend on thread scheduling.
            let objective = |theta: f64| {
                let spec = pair_spec(*family, theta);
                let pair_sum = |idx: usize| {
                    let (left, right) = direct_pairs[idx];
                    (0..data.n_obs())
                        .map(|row| {
                            spec.log_pdf(
                                view[(row, left)],
                                view[(row, right)],
                                options.base.clip_eps,
                            )
                            .unwrap_or(-1e12)
                        })
                        .sum::<f64>()
                };
                let sums: Vec<f64> = if direct_pairs.len() * data.n_obs() >= 2048 {
                    crate::accel::parallel_map_range_collect(direct_pairs.len(), pair_sum)
                } else {
                    (0..direct_pairs.len()).map(pair_sum).collect()
                };
                sums.iter().sum::<f64>()
            };
            let search = crate::math::maximize_scalar_brent(
                low,
                upper,
                Some(init),
                HAC_THETA_TOL,
                options.base.max_iter,
                objective,
            );
            (search.x, search.evaluations, search.converged)
        };
        let score = pair_loglik_sum(*family, theta, direct_pairs, data, options.base.clip_eps)?;
        Ok((*family, theta, score, converged, iterations))
    };
    let strategy = if options.family_set.len() > 1 && direct_pairs.len() * data.n_obs() >= 512 {
        crate::backend::ExecutionStrategy::CpuParallel
    } else {
        crate::backend::ExecutionStrategy::CpuSerial
    };
    let fits = crate::backend::parallel_try_map_range_collect(
        options.family_set.len(),
        strategy,
        fit_family,
    )?;
    for (family, theta, score, converged, iterations) in fits {
        progress.iterations += iterations;
        match best {
            Some((_, _, best_score, _)) if score <= best_score => {}
            _ => best = Some((family, theta, score, converged)),
        }
    }

    best.map(|(family, theta, _, converged)| {
        progress.converged &= converged;
        (family, theta)
    })
    .ok_or_else(|| {
        FitError::Failed {
            reason: "HAC family selection requires at least one candidate family",
        }
        .into()
    })
}

fn pair_loglik_sum(
    family: HacFamily,
    theta: f64,
    direct_pairs: &[(usize, usize)],
    data: &PseudoObs,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let spec = pair_spec(family, theta);
    let view = data.as_view();
    let mut total = 0.0;
    for &(left, right) in direct_pairs {
        for row in 0..data.n_obs() {
            total += spec.log_pdf(view[(row, left)], view[(row, right)], clip_eps)?;
        }
    }
    Ok(total)
}

fn collapse_tree(tree: HacTree, eps: f64) -> HacTree {
    match tree {
        HacTree::Leaf(_) => tree,
        HacTree::Node(mut node) => {
            node.children = node
                .children
                .into_iter()
                .map(|child| collapse_tree(child, eps))
                .collect();
            let mut merged_thetas = vec![node.theta];
            let mut merged_children = Vec::new();
            for child in node.children {
                match child {
                    HacTree::Node(child_node)
                        if child_node.family == node.family
                            && (child_node.theta - node.theta).abs() <= eps =>
                    {
                        merged_thetas.push(child_node.theta);
                        merged_children.extend(child_node.children);
                    }
                    other => merged_children.push(other),
                }
            }
            node.theta = merged_thetas.iter().sum::<f64>() / merged_thetas.len() as f64;
            node.children = merged_children;
            HacTree::Node(node)
        }
    }
}

fn project_same_family_nesting(tree: &mut HacTree) -> f64 {
    match tree {
        HacTree::Leaf(_) => f64::INFINITY,
        HacTree::Node(node) => {
            let child_limits = node
                .children
                .iter_mut()
                .map(project_same_family_nesting)
                .collect::<Vec<_>>();
            for (child, limit) in node.children.iter().zip(child_limits.iter()) {
                if let HacTree::Node(child_node) = child
                    && child_node.family == node.family
                    && node.theta > *limit
                {
                    node.theta = *limit;
                }
            }
            node.theta
        }
    }
}

pub(crate) fn composite_log_pdf_rows(
    model: &HierarchicalArchimedeanCopula,
    data: &PseudoObs,
    clip_eps: f64,
) -> Result<Vec<f64>, CopulaError> {
    let work = collect_node_work(model.tree());
    let view = data.as_view();
    let mut rows = vec![0.0; data.n_obs()];
    for node in work {
        let spec = pair_spec(node.family, node.theta);
        for &(left, right) in &node.direct_pairs {
            for row in 0..data.n_obs() {
                rows[row] += spec.log_pdf(view[(row, left)], view[(row, right)], clip_eps)?;
            }
        }
    }
    Ok(rows)
}

fn composite_loglik(
    model: &HierarchicalArchimedeanCopula,
    data: &PseudoObs,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    Ok(composite_log_pdf_rows(model, data, clip_eps)?
        .into_iter()
        .sum())
}

fn collect_node_work(tree: &HacTree) -> Vec<NodeWork> {
    let mut out = Vec::new();
    collect_node_work_inner(tree, &mut out);
    out
}

fn collect_node_work_inner(tree: &HacTree, out: &mut Vec<NodeWork>) -> Vec<usize> {
    match tree {
        HacTree::Leaf(index) => vec![*index],
        HacTree::Node(node) => {
            let groups = node
                .children
                .iter()
                .map(|child| collect_node_work_inner(child, out))
                .collect::<Vec<_>>();
            out.push(NodeWork {
                family: node.family,
                theta: node.theta,
                direct_pairs: direct_pairs_from_groups(&groups),
            });
            groups.into_iter().flatten().collect()
        }
    }
}

fn direct_pairs_from_groups(groups: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for left in 0..groups.len() {
        for right in (left + 1)..groups.len() {
            for &i in &groups[left] {
                for &j in &groups[right] {
                    pairs.push((i, j));
                }
            }
        }
    }
    pairs
}

fn exact_exchangeable_log_pdf(
    model: &HierarchicalArchimedeanCopula,
    data: &PseudoObs,
    options: &EvalOptions,
) -> Result<Option<Vec<f64>>, CopulaError> {
    let node = match model.tree() {
        HacTree::Node(node)
            if node
                .children
                .iter()
                .all(|child| matches!(child, HacTree::Leaf(_))) =>
        {
            node
        }
        _ => return Ok(None),
    };
    let values = match node.family {
        HacFamily::Clayton => {
            ClaytonCopula::new(model.dim(), node.theta)?.log_pdf(data, options)?
        }
        HacFamily::Frank => FrankCopula::new(model.dim(), node.theta)?.log_pdf(data, options)?,
        HacFamily::Gumbel => {
            GumbelHougaardCopula::new(model.dim(), node.theta)?.log_pdf(data, options)?
        }
    };
    Ok(Some(values))
}

fn exact_exchangeable_sample<R: Rng + ?Sized>(
    model: &HierarchicalArchimedeanCopula,
    n: usize,
    rng: &mut R,
    options: &SampleOptions,
) -> Result<Option<Array2<f64>>, CopulaError> {
    let node = match model.tree() {
        HacTree::Node(node)
            if node
                .children
                .iter()
                .all(|child| matches!(child, HacTree::Leaf(_))) =>
        {
            node
        }
        _ => return Ok(None),
    };
    let values = match node.family {
        HacFamily::Clayton => {
            ClaytonCopula::new(model.dim(), node.theta)?.sample(n, rng, options)?
        }
        HacFamily::Frank => FrankCopula::new(model.dim(), node.theta)?.sample(n, rng, options)?,
        HacFamily::Gumbel => {
            GumbelHougaardCopula::new(model.dim(), node.theta)?.sample(n, rng, options)?
        }
    };
    Ok(Some(values))
}

/// How a child node's frailty is obtained from its parent's.
enum ChildFrailty {
    /// Same generator: the frailty is shared, nothing is drawn.
    Shared,
    /// Gumbel within Gumbel: multiply by a positive stable variate with this
    /// index (drawn directly).
    GumbelStable(f64),
    /// Clayton within Clayton: an exponentially tilted positive stable
    /// variate with this index, drawn exactly by rejection.
    ClaytonTilted(f64),
    /// Frank within Frank: a sum of tilted Sibuya variates.
    FrankSum { parent_theta: f64, child_theta: f64 },
    /// Mixed families (experimental): invert a numerically inverted
    /// conditional frailty CDF at a uniform draw.
    InverseCdf,
}

/// A frailty carried both as a value and as its logarithm. Frank frailties
/// reach `e^θ/θ` and nested Clayton ones `V^{θ₁/θ₀}`, either of which leaves
/// the range of `f64`, so the Clayton and Frank paths work on the logarithm.
#[derive(Clone, Copy)]
struct Frailty {
    value: f64,
    log: f64,
}

impl Frailty {
    fn from_value(value: f64) -> Self {
        Self {
            value,
            log: value.ln(),
        }
    }

    fn from_log(log: f64) -> Self {
        Self {
            value: log.exp(),
            log,
        }
    }

    /// The root frailty from the draw recorded by [`sample_root_frailty`].
    fn root(family: HacFamily, draw: f64) -> Self {
        match family {
            HacFamily::Frank => Self::from_log(draw),
            HacFamily::Clayton | HacFamily::Gumbel => Self::from_value(draw),
        }
    }
}

fn child_frailty_kind(parent: &HacNode, child: &HacNode) -> ChildFrailty {
    if parent.family == child.family && parent.theta == child.theta {
        ChildFrailty::Shared
    } else if parent.family == HacFamily::Gumbel && child.family == HacFamily::Gumbel {
        ChildFrailty::GumbelStable(parent.theta / child.theta)
    } else if parent.family == HacFamily::Clayton
        && child.family == HacFamily::Clayton
        && parent.theta < child.theta
    {
        ChildFrailty::ClaytonTilted(parent.theta / child.theta)
    } else if parent.family == HacFamily::Frank
        && child.family == HacFamily::Frank
        && parent.theta < child.theta
    {
        ChildFrailty::FrankSum {
            parent_theta: parent.theta,
            child_theta: child.theta,
        }
    } else {
        ChildFrailty::InverseCdf
    }
}

/// Number of values [`draw_subtree`] records for one row.
fn count_subtree_draws(tree: &HacTree) -> usize {
    match tree {
        HacTree::Leaf(_) => 0,
        HacTree::Node(node) => node
            .children
            .iter()
            .map(|child| match child {
                HacTree::Leaf(_) => 1,
                HacTree::Node(child_node) => {
                    let own = match child_frailty_kind(node, child_node) {
                        ChildFrailty::Shared => 0,
                        ChildFrailty::GumbelStable(_)
                        | ChildFrailty::ClaytonTilted(_)
                        | ChildFrailty::FrankSum { .. }
                        | ChildFrailty::InverseCdf => 1,
                    };
                    own + count_subtree_draws(child)
                }
            })
            .sum(),
    }
}

/// Records the random draws of one row in traversal order: an exponential
/// per leaf, and per nested node whatever its frailty needs. A frailty drawn
/// by rejection needs a variable number of variates, so it records a seed
/// (an integer below 2⁵³, exact in an `f64`) for a generator of its own.
fn draw_subtree<R: Rng + ?Sized>(tree: &HacTree, rng: &mut R, draws: &mut Vec<f64>) {
    let HacTree::Node(node) = tree else {
        return;
    };
    for child in &node.children {
        match child {
            HacTree::Leaf(_) => draws.push(Exp1.sample(rng)),
            HacTree::Node(child_node) => {
                match child_frailty_kind(node, child_node) {
                    ChildFrailty::Shared => {}
                    ChildFrailty::GumbelStable(alpha) => {
                        draws.push(sample_positive_stable(rng, alpha));
                    }
                    ChildFrailty::ClaytonTilted(_) | ChildFrailty::FrankSum { .. } => {
                        draws.push((rng.random::<u64>() >> 11) as f64);
                    }
                    ChildFrailty::InverseCdf => draws.push(rng.random::<f64>()),
                }
                draw_subtree(child, rng, draws);
            }
        }
    }
}

/// Turns the recorded draws of one row into observations.
fn apply_subtree(
    tree: &HacTree,
    frailty: Frailty,
    draws: &[f64],
    cursor: &mut usize,
    weights: &[f64],
    row: &mut [f64],
) -> Result<(), CopulaError> {
    let HacTree::Node(node) = tree else {
        return Err(FitError::Failed {
            reason: "leaf nodes cannot be sampled without a parent generator",
        }
        .into());
    };
    for child in &node.children {
        match child {
            HacTree::Leaf(index) => {
                let exponential = draws[*cursor];
                *cursor += 1;
                row[*index] = leaf_from_exponential(node.family, node.theta, frailty, exponential);
            }
            HacTree::Node(child_node) => {
                let child_frailty = match child_frailty_kind(node, child_node) {
                    ChildFrailty::Shared => frailty,
                    ChildFrailty::GumbelStable(alpha) => {
                        let stable = draws[*cursor];
                        *cursor += 1;
                        Frailty::from_value(frailty.value.powf(1.0 / alpha) * stable)
                    }
                    ChildFrailty::ClaytonTilted(alpha) => {
                        let mut rng = StdRng::seed_from_u64(draws[*cursor] as u64);
                        *cursor += 1;
                        Frailty::from_log(ln_tilted_stable(&mut rng, alpha, frailty.log, 0.0))
                    }
                    ChildFrailty::FrankSum {
                        parent_theta,
                        child_theta,
                    } => {
                        let mut rng = StdRng::seed_from_u64(draws[*cursor] as u64);
                        *cursor += 1;
                        Frailty::from_log(ln_nested_frank_frailty(
                            &mut rng,
                            parent_theta,
                            child_theta,
                            frailty.log,
                        ))
                    }
                    ChildFrailty::InverseCdf => {
                        let u = draws[*cursor];
                        *cursor += 1;
                        if !frailty.value.is_finite() {
                            return Err(crate::errors::NumericalError::Failed {
                                reason: "mixed-family HAC frailty exceeds f64 range",
                            }
                            .into());
                        }
                        Frailty::from_value(invert_child_frailty_cdf(
                            node.family,
                            node.theta,
                            child_node.family,
                            child_node.theta,
                            frailty.value,
                            u,
                            weights,
                        ))
                    }
                };
                apply_subtree(child, child_frailty, draws, cursor, weights, row)?;
            }
        }
    }
    Ok(())
}

/// Draws the root frailty: its value for Clayton and Gumbel, its logarithm
/// for Frank (see [`Frailty::root`]).
fn sample_root_frailty<R: Rng + ?Sized>(
    family: HacFamily,
    theta: f64,
    rng: &mut R,
) -> Result<f64, CopulaError> {
    match family {
        HacFamily::Clayton => {
            let dist = Gamma::new(1.0 / theta, 1.0).map_err(|_| FitError::Failed {
                reason: "failed to construct Clayton root frailty sampler",
            })?;
            Ok(dist.sample(rng))
        }
        // The logarithm: a Frank frailty is of order `e^θ/θ`.
        HacFamily::Frank => Ok(frank::sample_log_frailty(rng, theta)),
        HacFamily::Gumbel => Ok(sample_positive_stable(rng, 1.0 / theta)),
    }
}

/// Child frailty at conditional probability `u` given the parent frailty,
/// by root search on the conditional CDF.
fn invert_child_frailty_cdf(
    parent_family: HacFamily,
    parent_theta: f64,
    child_family: HacFamily,
    child_theta: f64,
    parent_frailty: f64,
    u: f64,
    weights: &[f64],
) -> f64 {
    let mut low = 0.0_f64;
    let mut high = (parent_frailty + 1.0).max(1.0);
    let cdf = |x: f64| {
        conditional_frailty_cdf(
            parent_family,
            parent_theta,
            child_family,
            child_theta,
            parent_frailty,
            x,
            weights,
        )
    };
    let mut residual_high = cdf(high) - u;
    while residual_high < 0.0 && high < 1e6 {
        low = high;
        high *= 2.0;
        residual_high = cdf(high) - u;
    }
    if residual_high < 0.0 {
        // The quantile lies beyond the search cap.
        return high;
    }
    // The numerically inverted CDF carries ~1e-9 of noise, so the root is
    // resolved to that level and no further.
    let residual_low = if low == 0.0 { -u } else { cdf(low) - u };
    let root = crate::math::find_root_bracketed(
        (low, high),
        (residual_low, residual_high),
        0.0,
        FRAILTY_ROOT_REL_TOL,
        80,
        |x| cdf(x) - u,
    );
    root.max(1e-9)
}

/// Relative tolerance on a sampled child frailty.
const FRAILTY_ROOT_REL_TOL: f64 = 1e-10;

fn conditional_frailty_cdf(
    parent_family: HacFamily,
    parent_theta: f64,
    child_family: HacFamily,
    child_theta: f64,
    parent_frailty: f64,
    x: f64,
    weights: &[f64],
) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let transform = |s: f64| -> f64 {
        let child_gen = generator(child_family, child_theta, s);
        let parent_inv = generator_inverse(parent_family, parent_theta, child_gen);
        (-parent_frailty * parent_inv).exp() / s
    };
    inverse_laplace_stehfest(transform, x, weights).clamp(0.0, 1.0)
}

fn inverse_laplace_stehfest<F>(transform: F, x: f64, weights: &[f64]) -> f64
where
    F: Fn(f64) -> f64,
{
    let ln2 = std::f64::consts::LN_2;
    let mut total = 0.0;
    for (idx, weight) in weights.iter().enumerate() {
        let k = (idx + 1) as f64;
        total += weight * transform(k * ln2 / x);
    }
    (ln2 / x) * total
}

fn stehfest_weights(n: usize) -> Vec<f64> {
    assert!(n.is_multiple_of(2));
    let m = n / 2;
    (1..=n)
        .map(|k| {
            let sign = if (k + m).is_multiple_of(2) { 1.0 } else { -1.0 };
            let mut total = 0.0;
            let lower = k.div_ceil(2);
            let upper = k.min(m);
            for j in lower..=upper {
                total += (j as f64).powi(m as i32) * factorial(2 * j)
                    / (factorial(m - j)
                        * factorial(j)
                        * factorial(j - 1)
                        * factorial(k - j)
                        * factorial(2 * j - k));
            }
            sign * total
        })
        .collect()
}

fn factorial(n: usize) -> f64 {
    (1..=n.max(1)).fold(1.0, |acc, value| acc * value as f64)
}

fn leaf_from_exponential(family: HacFamily, theta: f64, frailty: Frailty, e: f64) -> f64 {
    match family {
        HacFamily::Clayton => (-(e.ln() - frailty.log).exp().ln_1p() / theta).exp(),
        HacFamily::Frank => frank::generator_from_log_argument(e.ln() - frailty.log, theta),
        HacFamily::Gumbel => (-(e / frailty.value).powf(1.0 / theta)).exp(),
    }
}

/// Logarithm of a positive stable variate with Laplace transform
/// `exp(−t^α)`, `0 < α < 1` (Kanter's representation, evaluated in logs so
/// that a small `α` does not overflow).
fn ln_positive_stable<R: Rng + ?Sized>(rng: &mut R, alpha: f64) -> f64 {
    // Open interval: the endpoints make a sine vanish.
    let uniform = loop {
        let u = rng.random::<f64>() * std::f64::consts::PI;
        if u > 0.0 {
            break u;
        }
    };
    let exponential: f64 = Exp1.sample(rng);
    (alpha * uniform).sin().ln() - uniform.sin().ln() / alpha
        + ((1.0 - alpha) / alpha) * (((1.0 - alpha) * uniform).sin().ln() - exponential.ln())
}

/// Logarithm of a variate with Laplace transform
/// `exp(−V₀ [(h + t)^α − h^α])`, an exponentially tilted positive stable
/// law, given `ln V₀` and `ln h`.
///
/// Hofert's fast rejection (2011, Algorithm 3.5): the variate is a sum of
/// `m ≈ V₀ h^α` independent pieces, each a stable variate of scale
/// `(V₀/m)^{1/α}` accepted with probability `exp(−h S)`; every piece is
/// accepted with probability about `1/e` or better.
fn ln_tilted_stable<R: Rng + ?Sized>(rng: &mut R, alpha: f64, ln_v0: f64, ln_tilt: f64) -> f64 {
    let pieces = (ln_v0 + alpha * ln_tilt).exp().round().clamp(1.0, 1e9);
    let ln_scale = (ln_v0 - pieces.ln()) / alpha;
    let mut ln_total = f64::NEG_INFINITY;
    for _ in 0..pieces as u64 {
        let ln_piece = loop {
            let ln_s = ln_scale + ln_positive_stable(rng, alpha);
            let exponential: f64 = Exp1.sample(rng);
            if exponential >= (ln_tilt + ln_s).exp() {
                break ln_s;
            }
        };
        ln_total = crate::math::logaddexp(ln_total, ln_piece);
    }
    ln_total
}

/// Parent frailties above this are handled by the stable limit in
/// [`ln_nested_frank_frailty`]; the same threshold as R's `copula` package.
const FRANK_SUM_LIMIT: f64 = 1e4;

/// Logarithm of the frailty of a Frank node with parameter `θ₁` nested in a
/// Frank node with parameter `θ₀ < θ₁`, given the logarithm of the parent's
/// frailty `N`.
///
/// The child frailty is a sum of `N` independent variates with mass function
/// `p_k = a_k q₁^k / q₀`, where `a_k` is the Sibuya(`α = θ₀/θ₁`) mass
/// function and `q_i = 1 − e^{−θ_i}` (Hofert 2011, §4.2). For `N` up to
/// [`FRANK_SUM_LIMIT`] the sum is drawn term by term, which is exact. Beyond
/// it the sum is replaced by its limit, the stable law tilted by
/// `h = −ln q₁`, whose Laplace transform differs from the exact one by a
/// relative `O(1/N)`.
fn ln_nested_frank_frailty<R: Rng + ?Sized>(
    rng: &mut R,
    parent_theta: f64,
    child_theta: f64,
    ln_parent: f64,
) -> f64 {
    let alpha = parent_theta / child_theta;
    // ln q₁ = ln(1 − e^{−θ₁}), so the tilt is h = −ln q₁.
    let ln_q1 = crate::math::log1mexp(-child_theta);
    if ln_parent > FRANK_SUM_LIMIT.ln() {
        return ln_tilted_stable(rng, alpha, ln_parent, (-ln_q1).ln());
    }
    let terms = ln_parent.exp().round().max(1.0) as u64;
    let sampler = TiltedSibuya::new(parent_theta, child_theta);
    let mut total = 0.0;
    for _ in 0..terms {
        total += sampler.sample(rng);
    }
    total.ln()
}

/// Sampler for the mass function `p_k = a_k q₁^k / q₀` on `k = 1, 2, …`,
/// with `a_k` the Sibuya(`α`) mass function.
///
/// Rejection from whichever proposal is accepted more often: a
/// logarithmic-series variate with parameter `q₁` (accepted with probability
/// `q₀/θ₀`), or a Sibuya variate (accepted with probability `q₀/q₁`). The
/// better of the two is accepted at least 63 % of the time.
struct TiltedSibuya {
    alpha: f64,
    child_theta: f64,
    ln_q1: f64,
    ln_gamma_1m_alpha: f64,
    gamma_1m_alpha: f64,
    log_series_proposal: bool,
}

impl TiltedSibuya {
    fn new(parent_theta: f64, child_theta: f64) -> Self {
        let alpha = parent_theta / child_theta;
        let ln_q1 = crate::math::log1mexp(-child_theta);
        let ln_gamma_1m_alpha = ln_gamma(1.0 - alpha);
        Self {
            alpha,
            child_theta,
            ln_q1,
            ln_gamma_1m_alpha,
            gamma_1m_alpha: ln_gamma_1m_alpha.exp(),
            log_series_proposal: parent_theta < ln_q1.exp(),
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        if self.log_series_proposal {
            loop {
                let ln_k = frank::sample_log_frailty(rng, self.child_theta);
                if ln_k == 0.0 {
                    return 1.0;
                }
                // The logarithm of an integer; beyond 2⁵³ rounding is moot.
                let k = ln_k.exp().round();
                // p_k / (envelope) = Γ(k − α) / (Γ(1 − α) Γ(k)).
                let ln_accept = ln_gamma(k - self.alpha) - self.ln_gamma_1m_alpha - ln_gamma(k);
                if rng.random::<f64>().ln() <= ln_accept {
                    return k;
                }
            }
        }
        loop {
            let k = self.sibuya(rng);
            // Accept with probability q₁^{k − 1}.
            if k == 1.0 || rng.random::<f64>().ln() <= (k - 1.0) * self.ln_q1 {
                return k;
            }
        }
    }

    /// Sibuya(`α`) by inversion: the survival function is
    /// `P(X > k) = 1 / (k B(k, 1 − α))`, and the quantile is the floor or
    /// the ceiling of `((1 − U) Γ(1 − α))^{−1/α}` (Hofert 2011, Prop. 3.2).
    fn sibuya<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.random();
        if u <= self.alpha {
            return 1.0;
        }
        let guess = ((1.0 - u) * self.gamma_1m_alpha).powf(-1.0 / self.alpha);
        let floor = guess.floor();
        if guess > 1.0 / f64::EPSILON {
            return floor;
        }
        let ln_survival = -(floor.ln() + ln_gamma(floor) + self.ln_gamma_1m_alpha
            - ln_gamma(floor + 1.0 - self.alpha));
        if (1.0 - u).ln() < ln_survival {
            guess.ceil()
        } else {
            floor
        }
    }
}

fn sample_positive_stable<R: Rng + ?Sized>(rng: &mut R, alpha: f64) -> f64 {
    if (alpha - 1.0).abs() < 1e-12 {
        return 1.0;
    }
    let uniform: f64 = rng.random::<f64>() * std::f64::consts::PI;
    let exponential: f64 = Exp1.sample(rng);
    let left = (alpha * uniform).sin() / uniform.sin().powf(1.0 / alpha);
    let right = (((1.0 - alpha) * uniform).sin() / exponential).powf((1.0 - alpha) / alpha);
    left * right
}

fn pair_spec(family: HacFamily, theta: f64) -> PairCopulaSpec {
    PairCopulaSpec {
        family: match family {
            HacFamily::Clayton => PairCopulaFamily::Clayton,
            HacFamily::Frank => PairCopulaFamily::Frank,
            HacFamily::Gumbel => PairCopulaFamily::Gumbel,
        },
        rotation: Rotation::R0,
        params: PairCopulaParams::One(theta),
    }
}

fn theta_from_tau(family: HacFamily, tau: f64) -> Result<f64, CopulaError> {
    match family {
        HacFamily::Clayton => {
            if tau <= 0.0 || tau >= 1.0 {
                return Err(FitError::Failed {
                    reason: "Clayton HAC fitting requires tau in (0, 1)",
                }
                .into());
            }
            Ok(2.0 * tau / (1.0 - tau))
        }
        HacFamily::Frank => frank::invert_tau(
            tau,
            "Frank HAC fitting failed to bracket the tau inversion root",
        ),
        HacFamily::Gumbel => {
            if tau <= 0.0 || tau >= 1.0 {
                return Err(FitError::Failed {
                    reason: "Gumbel HAC fitting requires tau in (0, 1)",
                }
                .into());
            }
            Ok(1.0 / (1.0 - tau))
        }
    }
}

fn family_parameter_bounds(family: HacFamily) -> (f64, f64) {
    match family {
        HacFamily::Clayton | HacFamily::Frank => (1e-6, 20.0),
        HacFamily::Gumbel => (1.0 + 1e-6, 20.0),
    }
}

fn validate_node_parameter(family: HacFamily, theta: f64) -> Result<(), CopulaError> {
    let valid = match family {
        HacFamily::Clayton | HacFamily::Frank => theta.is_finite() && theta > 0.0,
        HacFamily::Gumbel => theta.is_finite() && theta >= 1.0,
    };
    if valid {
        Ok(())
    } else {
        Err(FitError::Failed {
            reason: "invalid HAC node parameter",
        }
        .into())
    }
}

fn generator(family: HacFamily, theta: f64, t: f64) -> f64 {
    match family {
        HacFamily::Clayton => (1.0 + t).powf(-1.0 / theta),
        HacFamily::Frank => frank::generator(t, theta),
        HacFamily::Gumbel => (-t.powf(1.0 / theta)).exp(),
    }
}

fn generator_inverse(family: HacFamily, theta: f64, u: f64) -> f64 {
    let clipped = u.clamp(1e-15, 1.0 - 1e-15);
    match family {
        HacFamily::Clayton => clipped.powf(-theta) - 1.0,
        HacFamily::Frank => {
            let numerator = (-theta * clipped).exp_m1();
            let denominator = (-theta).exp_m1();
            -((numerator / denominator).abs()).ln()
        }
        HacFamily::Gumbel => (-clipped.ln()).powf(theta),
    }
}

fn tree_is_exact(tree: &HacTree) -> bool {
    match tree {
        HacTree::Leaf(_) => true,
        HacTree::Node(node) => node.children.iter().all(|child| match child {
            HacTree::Leaf(_) => true,
            HacTree::Node(child_node) => {
                child_node.family == node.family
                    && node.theta <= child_node.theta + 1e-12
                    && tree_is_exact(child)
            }
        }),
    }
}

fn tree_has_exact_loglik(tree: &HacTree) -> bool {
    matches!(tree, HacTree::Node(node) if node.children.iter().all(|child| matches!(child, HacTree::Leaf(_))))
}

fn leaves_of(tree: &HacTree) -> Vec<usize> {
    match tree {
        HacTree::Leaf(index) => vec![*index],
        HacTree::Node(node) => node.children.iter().flat_map(leaves_of).collect(),
    }
}

fn node_of(tree: &HacTree) -> Result<&HacNode, CopulaError> {
    match tree {
        HacTree::Node(node) => Ok(node),
        HacTree::Leaf(_) => Err(FitError::Failed {
            reason: "HAC root cannot be a leaf",
        }
        .into()),
    }
}

fn internal_node_count(tree: &HacTree) -> usize {
    match tree {
        HacTree::Leaf(_) => 0,
        HacTree::Node(node) => 1 + node.children.iter().map(internal_node_count).sum::<usize>(),
    }
}

#[cfg(test)]
mod frailty_tests {
    use rand::{SeedableRng, rngs::StdRng};
    use statrs::function::gamma::ln_gamma;

    /// `ln a_k` of the Sibuya(α) mass function `α Γ(k − α) / (Γ(1 − α) k!)`.
    fn ln_sibuya_pmf(alpha: f64, k: f64) -> f64 {
        alpha.ln() + ln_gamma(k - alpha) - ln_gamma(1.0 - alpha) - ln_gamma(k + 1.0)
    }

    #[test]
    fn tilted_sibuya_matches_its_mass_function() {
        // One case per proposal: θ₀ < q₁ uses the logarithmic series,
        // θ₀ ≥ q₁ the Sibuya proposal.
        for (parent, child) in [(0.4_f64, 3.0_f64), (2.5, 4.0), (0.9, 0.95), (6.0, 20.0)] {
            let sampler = super::TiltedSibuya::new(parent, child);
            let alpha = parent / child;
            let (q0, q1) = (-(-parent).exp_m1(), -(-child).exp_m1());
            let mut rng = StdRng::seed_from_u64(99);
            let n = 400_000;
            let mut counts = [0usize; 6];
            for _ in 0..n {
                let k = sampler.sample(&mut rng);
                assert!(k >= 1.0 && k.fract() == 0.0, "{k}");
                counts[(k as usize - 1).min(5)] += 1;
            }
            let mut tail = 1.0;
            for (idx, count) in counts.iter().enumerate() {
                let k = (idx + 1) as f64;
                let expected = if idx < 5 {
                    let p = (ln_sibuya_pmf(alpha, k) + k * q1.ln() - q0.ln()).exp();
                    tail -= p;
                    p
                } else {
                    tail
                };
                let observed = *count as f64 / n as f64;
                let standard_error = (expected * (1.0 - expected) / n as f64).sqrt();
                assert!(
                    (observed - expected).abs() < 4.5 * standard_error + 1e-12,
                    "θ₀ = {parent}, θ₁ = {child}, k = {k}: {observed} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn tilted_stable_matches_its_laplace_transform() {
        // E[exp(−t V)] = exp(−V₀ [(h + t)^α − h^α]).
        for (alpha, v0, tilt) in [
            (0.5_f64, 0.7_f64, 1.0_f64),
            (0.2, 6.3, 1.0),
            (0.9, 40.0, 0.01),
        ] {
            let mut rng = StdRng::seed_from_u64(5);
            let n = 200_000;
            let draws: Vec<f64> = (0..n)
                .map(|_| super::ln_tilted_stable(&mut rng, alpha, v0.ln(), tilt.ln()).exp())
                .collect();
            for t in [0.05, 0.5, 3.0] {
                let exact = (-v0 * ((tilt + t).powf(alpha) - tilt.powf(alpha))).exp();
                let values: Vec<f64> = draws.iter().map(|v| (-t * v).exp()).collect();
                let mean = values.iter().sum::<f64>() / n as f64;
                let variance =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                let standard_error = (variance / n as f64).sqrt();
                assert!(
                    (mean - exact).abs() < 4.5 * standard_error,
                    "α = {alpha}, V₀ = {v0}, h = {tilt}, t = {t}: {mean} vs {exact}"
                );
            }
        }
    }
}

#[cfg(test)]
mod inversion_tests {
    #[test]
    fn stehfest_recovers_known_laplace_transforms() {
        assert_eq!(super::stehfest_weights(2), vec![2.0, -2.0]);
        let weights = super::stehfest_weights(12);
        for x in [0.1, 1.0, 3.0] {
            let value = super::inverse_laplace_stehfest(|s| 1.0 / (s + 1.0), x, &weights);
            assert!((value - (-x).exp()).abs() < 1e-3);
        }
    }
}
