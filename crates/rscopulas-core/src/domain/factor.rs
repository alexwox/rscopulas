//! Krupskii–Joe factor copulas (Joe 2014, ch. 3–4; Krupskii & Joe 2013, 2015).
//!
//! This module exposes the **Basic1F** layout only: `d` observed variables
//! linked to a single latent factor `V ~ U[0, 1]` through arbitrary
//! bivariate pair-copulas. The joint density is
//!
//! ```text
//!     c(u_1, …, u_d) = ∫₀¹  ∏_{j=1..d}  c_{U_j,V}(u_j, v)  dv,
//! ```
//!
//! and is evaluated via an `n`-point Gauss–Legendre rule on `[0, 1]`
//! (default `n = 25`, matching Joe's `CopulaModel` R package).
//!
//! Extensions (Nested2F, Structured, BiFactor) are intentionally deferred —
//! adding them only requires extending the `FactorLayout` enum and
//! `log_pdf_single` / `sample` to walk a richer link tree; the rest of the
//! infrastructure (quadrature, fit driver, serialization, Python wrapper)
//! is laid out so those additions are local.

use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::{
    data::PseudoObs,
    errors::{CopulaError, FitError},
    fit::FitResult,
    math::gauss_legendre_01,
    paircopula::{PairCopulaFamily, PairCopulaSpec, fit_pair_copula},
    vine::{SelectionCriterion, VineFitOptions},
};

use super::{CopulaFamily, CopulaModel, EvalOptions, FitDiagnostics, FitOptions, SampleOptions};

/// Topology of latent factors and link copulas.
///
/// Only `Basic1F` is supported today; the variant-less enum still shapes the
/// public API so that richer layouts (two-factor nested, structured,
/// bi-factor) can be added without breaking existing call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactorLayout {
    /// One common latent factor shared by every observed variable.
    Basic1F,
}

/// Configuration for fitting a [`FactorCopula`] to pseudo-observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorFitOptions {
    /// Base options (clipping, max_iter, execution policy) shared with other
    /// fitters in the crate.
    pub base: FitOptions,
    /// Target layout. Defaults to `Basic1F`.
    pub layout: FactorLayout,
    /// Candidate pair-copula families considered for every link.
    pub family_set: Vec<PairCopulaFamily>,
    /// Whether rotated Archimedean links may be selected.
    pub include_rotations: bool,
    /// Criterion used when comparing candidate link fits.
    pub criterion: SelectionCriterion,
    /// Number of Gauss–Legendre nodes used for the latent integral.
    pub quadrature_nodes: usize,
    /// Number of EM-style refinement passes after the initial sequential
    /// MLE. Each pass recomputes `E[V | U_i]` under the current fit (the
    /// posterior mean of the latent), rank-normalises it, and refits every
    /// link. One pass fixes most of the warm-start attenuation bias; more
    /// give diminishing returns.
    pub refine_iterations: usize,
}

impl Default for FactorFitOptions {
    fn default() -> Self {
        Self {
            base: FitOptions::default(),
            layout: FactorLayout::Basic1F,
            // Conservative default family set — mirrors the single-family
            // copulas already first-class in the crate. Archimedean 2-param
            // families (BB1/6/7/8) and Tawn can be opted into via
            // `family_set` explicitly, same as vine fitting.
            family_set: vec![
                PairCopulaFamily::Independence,
                PairCopulaFamily::Gaussian,
                PairCopulaFamily::Clayton,
                PairCopulaFamily::Frank,
                PairCopulaFamily::Gumbel,
            ],
            include_rotations: true,
            criterion: SelectionCriterion::Aic,
            // 25 nodes is exact for polynomials up to degree 49 on [0, 1] and
            // matches the default used by Joe's `CopulaModel` R package.
            quadrature_nodes: 25,
            refine_iterations: 2,
        }
    }
}

/// Fitted factor copula.
///
/// The model stores one `PairCopulaSpec` per observed variable, giving the
/// joint density via quadrature over the single latent factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorCopula {
    dim: usize,
    layout: FactorLayout,
    /// One link per observed variable. For `Basic1F`, `links[j]` is the
    /// bivariate copula between variable `j` and the common latent factor.
    links: Vec<PairCopulaSpec>,
    quadrature_nodes: usize,
}

impl FactorCopula {
    /// Builds a `Basic1F` factor copula directly from fully-specified links.
    pub fn basic_1f(
        links: Vec<PairCopulaSpec>,
        quadrature_nodes: usize,
    ) -> Result<Self, CopulaError> {
        if links.len() < 2 {
            return Err(FitError::Failed {
                reason: "factor copula requires at least two observed variables",
            }
            .into());
        }
        if quadrature_nodes < 3 {
            return Err(FitError::Failed {
                reason: "factor copula quadrature requires at least three nodes",
            }
            .into());
        }
        Ok(Self {
            dim: links.len(),
            layout: FactorLayout::Basic1F,
            links,
            quadrature_nodes,
        })
    }

    /// Fits a factor copula to pseudo-observations using a two-stage sequential
    /// MLE: (i) build a pseudo-latent via normal-score PCA-like projection,
    /// (ii) fit each link by bivariate MLE against the pseudo-latent.
    ///
    /// The final log-likelihood reported in `FitDiagnostics` is the true
    /// factor-copula log-likelihood (evaluated with the fitted links and the
    /// same Gauss–Legendre rule used at inference), not the per-link sum —
    /// so AIC/BIC comparisons against other models (vines, HAC, elliptical)
    /// are apples-to-apples.
    pub fn fit(
        data: &PseudoObs,
        options: &FactorFitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        match options.layout {
            FactorLayout::Basic1F => fit_basic_1f(data, options),
        }
    }

    /// Returns the factor layout.
    pub fn layout(&self) -> FactorLayout {
        self.layout
    }

    /// Number of latent factors in this model. Always 1 for `Basic1F`.
    pub fn num_factors(&self) -> usize {
        match self.layout {
            FactorLayout::Basic1F => 1,
        }
    }

    /// Returns the fitted link specifications in variable-index order.
    pub fn links(&self) -> &[PairCopulaSpec] {
        &self.links
    }

    /// Number of Gauss–Legendre nodes used by `log_pdf`.
    pub fn quadrature_nodes(&self) -> usize {
        self.quadrature_nodes
    }

    /// Evaluates the factor-copula log-density at a single observation using
    /// the model's stored quadrature size and a numerically stable log-sum-exp
    /// accumulation.
    fn log_pdf_single(
        &self,
        obs: &[f64],
        nodes: &[f64],
        weights: &[f64],
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        // Integrand at v: ∏_j c_j(u_j, v) = exp(Σ_j log c_j(u_j, v)).
        // So log ∫ = log Σ_q w_q exp(Σ_j log c_j(u_j, v_q)) — classical
        // log-sum-exp with `log w_q` added to the per-node accumulation.
        let mut log_terms = Vec::with_capacity(nodes.len());
        for (v, w) in nodes.iter().zip(weights.iter()) {
            let mut s = w.ln();
            for (j, link) in self.links.iter().enumerate() {
                let contrib = link.log_pdf(obs[j], *v, clip_eps)?;
                s += contrib;
                if !s.is_finite() {
                    // Short-circuit: any one link evaluating to -∞ kills the
                    // whole quadrature contribution at this node.
                    break;
                }
            }
            log_terms.push(s);
        }
        Ok(log_sum_exp(&log_terms))
    }
}

impl CopulaModel for FactorCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::Factor
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn log_pdf(&self, data: &PseudoObs, options: &EvalOptions) -> Result<Vec<f64>, CopulaError> {
        let view = data.as_view();
        if view.ncols() != self.dim {
            return Err(FitError::Failed {
                reason: "factor copula input dimension does not match model dimension",
            }
            .into());
        }
        // Compute quadrature once per call and reuse across observations — the
        // rule is observation-independent.
        let (nodes, weights) = gauss_legendre_01(self.quadrature_nodes);
        let mut out = Vec::with_capacity(view.nrows());
        let mut obs = vec![0.0_f64; self.dim];
        for row in view.rows() {
            for (dst, src) in obs.iter_mut().zip(row.iter()) {
                *dst = *src;
            }
            out.push(self.log_pdf_single(&obs, &nodes, &weights, options.clip_eps)?);
        }
        Ok(out)
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        _options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        // Latent-first sampling: draw V = v, then for each observed j draw a
        // fresh uniform p and set U_j = h_{1|2}^{-1}(p | v; link_j).
        // This is exactly the conditional-inverse construction used by vines
        // on an anchored tree, but specialised to a star graph with V at the
        // centre.
        let clip_eps = 1e-12;
        let mut out = Array2::<f64>::zeros((n, self.dim));
        for row in 0..n {
            let v = rng.random::<f64>().clamp(clip_eps, 1.0 - clip_eps);
            for (j, link) in self.links.iter().enumerate() {
                let p = rng.random::<f64>().clamp(clip_eps, 1.0 - clip_eps);
                let u = link.inv_first_given_second(p, v, clip_eps)?;
                out[(row, j)] = u.clamp(clip_eps, 1.0 - clip_eps);
            }
        }
        Ok(out)
    }
}

fn fit_basic_1f(
    data: &PseudoObs,
    options: &FactorFitOptions,
) -> Result<FitResult<FactorCopula>, CopulaError> {
    let view = data.as_view();
    let n = view.nrows();
    let dim = view.ncols();
    if dim < 2 {
        return Err(FitError::Failed {
            reason: "factor copula requires at least two variables",
        }
        .into());
    }
    if options.quadrature_nodes < 3 {
        return Err(FitError::Failed {
            reason: "factor copula quadrature requires at least three nodes",
        }
        .into());
    }

    let clip = options.base.clip_eps;
    let vine_options = VineFitOptions {
        base: options.base.clone(),
        family_set: options.family_set.clone(),
        include_rotations: options.include_rotations,
        criterion: options.criterion,
        truncation_level: None,
        independence_threshold: None,
    };

    // Stage 1: pseudo-latent via normal-score projection.
    //
    // Intuition: if the factor-model assumption holds, z_ij = Φ⁻¹(u_ij) has
    // approximately a one-factor Gaussian structure, i.e. z_ij ≈ a_j ζ_i + ε_ij.
    // The row-wise sum Σ_j z_ij is then proportional to ζ_i with noise; rank-
    // normalising that sum onto (0, 1) gives a uniform-margin pseudo-latent
    // whose rank agrees with the true factor rank up to noise.
    //
    // We *rank-normalise* instead of using Φ of the standardised sum so the
    // marginal distribution matches V ~ U(0, 1) exactly (avoids spurious
    // tail-mass when the sum's variance is mis-specified), and so rotations
    // like R180 can still pick up lower-tail asymmetry in the raw data.
    let normal = Normal::new(0.0, 1.0).expect("standard normal parameters should be valid");
    let mut score_sum = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..dim {
            let u = view[(i, j)].clamp(clip, 1.0 - clip);
            s += normal.inverse_cdf(u);
        }
        score_sum[i] = s;
    }
    let mut v_pseudo = rank_normalise_01(&score_sum, clip);

    // Stage 2: fit each link against the pseudo-latent.
    //
    // We reuse the existing `fit_pair_copula` machinery so every family
    // (Gaussian, Clayton, Gumbel, Joe, BB*, Tawn*, TLL, Khoudraji) supported
    // by the pair layer is automatically available here. The convention
    // inside PairCopulaSpec — "first" → u_j, "second" → v — matches the
    // sampling path (`inv_first_given_second(p, v)`), so rotation selection
    // from the fitter lines up with how the link is used at inference time.
    let columns: Vec<Vec<f64>> = (0..dim)
        .map(|j| view.column(j).iter().copied().collect())
        .collect();

    let mut links = fit_links_against_latent(&columns, &v_pseudo, &vine_options)?;
    let mut model = FactorCopula::basic_1f(links.clone(), options.quadrature_nodes)?;

    // Stage 3: EM-style refinement. Each pass:
    //   1. Compute E[V | U_i] under the current fit (quadrature-integrated
    //      posterior mean of the latent factor given the observation).
    //   2. Rank-normalise those posterior means into a fresh pseudo-latent.
    //   3. Refit every link against the refined pseudo-latent.
    //
    // One pass fixes most of the attenuation bias in the naive pseudo-latent;
    // two give diminishing returns. We track log-likelihood and bail early
    // if the refined fit fails to improve (safety against pathological data
    // where the refinement would otherwise degrade the model).
    let mut best_loglik = factor_log_likelihood(&model, data, options.base.clip_eps)?;
    for _ in 0..options.refine_iterations {
        let posterior = posterior_latent_mean(&model, data, clip)?;
        v_pseudo = rank_normalise_01(&posterior, clip);
        let refined_links = fit_links_against_latent(&columns, &v_pseudo, &vine_options)?;
        let refined = FactorCopula::basic_1f(refined_links.clone(), options.quadrature_nodes)?;
        let refined_loglik = factor_log_likelihood(&refined, data, options.base.clip_eps)?;
        if refined_loglik > best_loglik {
            best_loglik = refined_loglik;
            links = refined_links;
            model = refined;
        } else {
            break;
        }
    }

    let total_parameters: usize = links.iter().map(|link| link.parameter_count()).sum();

    // Final diagnostics use the true factor-copula log-likelihood (not the
    // per-link pseudo-likelihood sum), so AIC/BIC are comparable with other
    // domain-level models in the crate.
    let loglik = best_loglik;
    let k = total_parameters as f64;
    let n_f = n as f64;
    let (aic, bic) = if loglik.is_finite() {
        (2.0 * k - 2.0 * loglik, k * n_f.ln() - 2.0 * loglik)
    } else {
        (f64::INFINITY, f64::INFINITY)
    };

    Ok(FitResult {
        model,
        diagnostics: FitDiagnostics {
            loglik,
            aic,
            bic,
            converged: true,
            n_iter: 1 + options.refine_iterations,
        },
    })
}

/// Fits every observed-variable-to-latent link against a supplied latent
/// vector using the existing pair-copula fitter.
fn fit_links_against_latent(
    columns: &[Vec<f64>],
    v_pseudo: &[f64],
    options: &VineFitOptions,
) -> Result<Vec<PairCopulaSpec>, CopulaError> {
    let mut links = Vec::with_capacity(columns.len());
    for column in columns {
        let result = fit_pair_copula(column, v_pseudo, options)?;
        links.push(result.spec);
    }
    Ok(links)
}

/// Sum of log-densities over the pseudo-observations (the factor copula
/// log-likelihood).
fn factor_log_likelihood(
    model: &FactorCopula,
    data: &PseudoObs,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let eval = EvalOptions {
        exec: super::ExecPolicy::Auto,
        clip_eps,
    };
    let log_pdfs = model.log_pdf(data, &eval)?;
    if log_pdfs.iter().any(|value| !value.is_finite()) {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(log_pdfs.iter().sum())
}

/// Posterior mean `E[V | U_i]` under the current factor-copula fit, evaluated
/// via the same Gauss–Legendre rule used at inference. Returns one value per
/// observation, in `[0, 1]`.
fn posterior_latent_mean(
    model: &FactorCopula,
    data: &PseudoObs,
    clip: f64,
) -> Result<Vec<f64>, CopulaError> {
    let (nodes, weights) = gauss_legendre_01(model.quadrature_nodes);
    let view = data.as_view();
    let mut out = Vec::with_capacity(view.nrows());
    let mut obs = vec![0.0_f64; model.dim];
    let mut log_terms = vec![0.0_f64; nodes.len()];
    for row in view.rows() {
        for (dst, src) in obs.iter_mut().zip(row.iter()) {
            *dst = *src;
        }
        // Compute log of the unnormalised posterior at each quadrature node.
        for (idx, (v, w)) in nodes.iter().zip(weights.iter()).enumerate() {
            let mut s = w.ln();
            for (j, link) in model.links.iter().enumerate() {
                s += link.log_pdf(obs[j], *v, clip)?;
                if !s.is_finite() {
                    break;
                }
            }
            log_terms[idx] = s;
        }
        // Normalise via the usual log-sum-exp shift and compute the weighted
        // mean of `v` under the posterior.
        let m = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !m.is_finite() {
            out.push(0.5);
            continue;
        }
        let mut num = 0.0;
        let mut den = 0.0;
        for (idx, &log_term) in log_terms.iter().enumerate() {
            let w = (log_term - m).exp();
            num += nodes[idx] * w;
            den += w;
        }
        let posterior = if den > 0.0 { num / den } else { 0.5 };
        out.push(posterior.clamp(clip, 1.0 - clip));
    }
    Ok(out)
}

/// Maps a vector of arbitrary real-valued scores onto uniform pseudo-
/// observations in `(clip, 1 - clip)` by mid-rank transform. Ties break
/// deterministically via index ordering (same convention as NumPy's default
/// rank). The result's empirical CDF is the standard `(rank - 0.5) / n`.
fn rank_normalise_01(values: &[f64], clip: f64) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0_f64; n];
    for (rank, (orig_idx, _)) in indexed.iter().enumerate() {
        let uniform = (rank as f64 + 0.5) / n as f64;
        out[*orig_idx] = uniform.clamp(clip, 1.0 - clip);
    }
    out
}

/// Numerically stable log-sum-exp over a slice of log-weights. Returns
/// `f64::NEG_INFINITY` if every entry is `-∞`.
fn log_sum_exp(xs: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &x in xs {
        if x > m {
            m = x;
        }
    }
    if !m.is_finite() {
        return m;
    }
    let mut s = 0.0;
    for &x in xs {
        s += (x - m).exp();
    }
    m + s.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_sum_exp_matches_naive_for_benign_inputs() {
        let xs = [-1.0_f64, -0.5, 0.25, 1.5];
        let naive = xs.iter().map(|x| x.exp()).sum::<f64>().ln();
        let stable = log_sum_exp(&xs);
        assert!((naive - stable).abs() < 1e-14);
    }

    #[test]
    fn log_sum_exp_is_stable_for_extreme_inputs() {
        // Naive exp would overflow; stable path should just return the max.
        let xs = [900.0_f64, 901.0, 902.0];
        let stable = log_sum_exp(&xs);
        let shifted: f64 = (-2.0_f64).exp() + (-1.0_f64).exp() + 0.0_f64.exp();
        assert!((stable - (902.0 + shifted.ln())).abs() < 1e-12);
    }
}
