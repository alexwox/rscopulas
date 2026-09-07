//! Hierarchical Archimedean copulas: tree validation, `log_pdf`, sampling, and fitting.
//!
//! **User guide** (density paths, which `sample()` scenarios are reliable, mixed-family
//! caveats): see `docs/hac.md` in the repository root.

use ndarray::Array2;
use rand::Rng;
use rand_distr::{Distribution, Exp1, Gamma};

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
    math::maximize_scalar_with_diagnostics,
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
    let mut samples = Array2::zeros((n, model.dim()));
    let weights = stehfest_weights(12);
    for row in 0..n {
        let root_frailty = sample_root_frailty(
            node_of(model.tree())?.family,
            node_of(model.tree())?.theta,
            rng,
        )?;
        sample_subtree(
            model.tree(),
            root_frailty,
            rng,
            &weights,
            samples
                .row_mut(row)
                .as_slice_mut()
                .ok_or(FitError::Failed {
                    reason: "row storage should be contiguous",
                })?,
        )?;
    }
    Ok(samples)
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

    for family in &options.family_set {
        let init = theta_from_tau(*family, mean_tau)?;
        let (low, upper_seed) = family_parameter_bounds(*family);
        let upper = (init * 4.0 + 2.0).max(upper_seed);
        let (theta, iterations, converged) = if options.fit_method == HacFitMethod::TauInit {
            (init, 0, true)
        } else {
            maximize_scalar_with_diagnostics(low, upper, options.base.max_iter, |theta| {
                direct_pairs
                    .iter()
                    .map(|&(left, right)| {
                        let spec = pair_spec(*family, theta);
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
                    })
                    .sum::<f64>()
            })
        };
        progress.iterations += iterations;
        let score = pair_loglik_sum(*family, theta, direct_pairs, data, options.base.clip_eps)?;
        match best {
            Some((_, _, best_score, _)) if score <= best_score => {}
            _ => best = Some((*family, theta, score, converged)),
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

fn sample_subtree<R: Rng + ?Sized>(
    tree: &HacTree,
    frailty: f64,
    rng: &mut R,
    weights: &[f64],
    row: &mut [f64],
) -> Result<(), CopulaError> {
    match tree {
        HacTree::Leaf(_) => Err(FitError::Failed {
            reason: "leaf nodes cannot be sampled without a parent generator",
        }
        .into()),
        HacTree::Node(node) => {
            for child in &node.children {
                match child {
                    HacTree::Leaf(index) => {
                        row[*index] = sample_leaf(node.family, node.theta, frailty, rng);
                    }
                    HacTree::Node(child_node) => {
                        let child_frailty = sample_child_frailty(
                            node.family,
                            node.theta,
                            child_node.family,
                            child_node.theta,
                            frailty,
                            rng,
                            weights,
                        )?;
                        sample_subtree(child, child_frailty, rng, weights, row)?;
                    }
                }
            }
            Ok(())
        }
    }
}

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
        HacFamily::Frank => {
            let frailty = frank::sample_log_frailty(rng, theta).exp();
            if !frailty.is_finite() {
                return Err(crate::errors::NumericalError::Failed {
                    reason: "nested Frank frailty exceeds f64 range",
                }
                .into());
            }
            Ok(frailty)
        }
        HacFamily::Gumbel => Ok(sample_positive_stable(rng, 1.0 / theta)),
    }
}

fn sample_child_frailty<R: Rng + ?Sized>(
    parent_family: HacFamily,
    parent_theta: f64,
    child_family: HacFamily,
    child_theta: f64,
    parent_frailty: f64,
    rng: &mut R,
    weights: &[f64],
) -> Result<f64, CopulaError> {
    if parent_family == child_family && parent_theta == child_theta {
        return Ok(parent_frailty);
    }
    if parent_family == HacFamily::Gumbel && child_family == HacFamily::Gumbel {
        let alpha = parent_theta / child_theta;
        let stable = sample_positive_stable(rng, alpha);
        return Ok(parent_frailty.powf(1.0 / alpha) * stable);
    }

    let u: f64 = rng.random();
    let mut low = 0.0;
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
    while cdf(high) < u && high < 1e6 {
        high *= 2.0;
    }
    for _ in 0..60 {
        let mid = 0.5 * (low + high);
        if cdf(mid) < u {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(0.5 * (low + high)).map(|value| value.max(1e-9))
}

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

fn sample_leaf<R: Rng + ?Sized>(family: HacFamily, theta: f64, frailty: f64, rng: &mut R) -> f64 {
    let e: f64 = Exp1.sample(rng);
    match family {
        HacFamily::Clayton => (1.0 + e / frailty).powf(-1.0 / theta),
        HacFamily::Frank => frank::generator(e / frailty, theta),
        HacFamily::Gumbel => (-(e / frailty).powf(1.0 / theta)).exp(),
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
