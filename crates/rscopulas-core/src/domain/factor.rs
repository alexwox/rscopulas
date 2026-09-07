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
//! Gaussian links use an exact Gaussian density. Other links use normal-scale
//! Gauss-Legendre quadrature on [-8,8], refining intervals by their estimated
//! error. The default relative tolerance is 1e-7 with a budget of 4096 integrand
//! evaluations per row. Fixed-node evaluation is an explicit approximate mode.
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
    math::{coord_ascent_with_diagnostics, gauss_legendre_01, inverse, numerical_hessian},
    paircopula::{
        PairCopulaFamily, PairCopulaSpec, decode_params, encode_brackets, encode_jacobian,
        encode_params, fit_pair_copula,
    },
    vine::{SelectionCriterion, VineFitOptions},
};

use super::{CopulaFamily, CopulaModel, EvalOptions, FitDiagnostics, FitOptions, SampleOptions};

/// Result of fitting a [`FactorCopula`], including the delta-method standard
/// errors for every polished link parameter.
///
/// `std_errors` has one entry per free parameter returned by the polish stage,
/// in the flat layout produced by concatenating `encode_params` across the
/// fitted links. Entries for skipped families (Independence, TLL, Khoudraji,
/// and Student-t's ν block) are absent — those parameters do not participate
/// in the joint-MLE polish. If the numerical Hessian at the MLE was not
/// positive-definite (typical when the polish stage bailed out) every entry
/// is `f64::NAN` so callers can distinguish "uninformative" from a zero SE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorFitResult {
    pub model: FactorCopula,
    pub diagnostics: FitDiagnostics,
    pub std_errors: Vec<f64>,
}

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

/// Work and accuracy controls for the latent integral. Stored with the model
/// so fitting diagnostics and subsequent evaluation use the same rule.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FactorQuadrature {
    /// Refine by estimated integration error. False uses exactly the model's
    /// `quadrature_nodes` and makes no convergence guarantee.
    pub adaptive: bool,
    /// Maximum integrand evaluations per row in adaptive mode, including
    /// coarse rules and discarded parent intervals.
    pub max_nodes: usize,
    /// Relative error target for adaptive integration.
    pub rel_tol: f64,
}

impl Default for FactorQuadrature {
    fn default() -> Self {
        Self {
            adaptive: true,
            max_nodes: 4096,
            rel_tol: 1e-7,
        }
    }
}

impl FactorQuadrature {
    fn validate(self, min_nodes: usize) -> Result<(), CopulaError> {
        if !self.rel_tol.is_finite()
            || self.rel_tol <= 0.0
            || self.rel_tol >= 1.0
            || !(24..=65_536).contains(&self.max_nodes)
            || (self.adaptive && self.max_nodes < min_nodes)
        {
            return Err(FitError::Failed {
                reason: "invalid factor quadrature tolerance or node budget",
            }
            .into());
        }
        Ok(())
    }
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
    /// Minimum work in adaptive mode, or exact node count in fixed mode.
    pub quadrature_nodes: usize,
    #[serde(default)]
    pub quadrature: FactorQuadrature,
    /// Number of EM-style refinement passes after the initial sequential
    /// MLE. Each pass recomputes `E[V | U_i]` under the current fit (the
    /// posterior mean of the latent), rank-normalises it, and refits every
    /// link. One pass fixes most of the warm-start attenuation bias; more
    /// give diminishing returns.
    pub refine_iterations: usize,
    /// Number of full coordinate-ascent sweeps over all link parameters in
    /// the final joint-MLE polish. Set to `0` to disable the polish (e.g.
    /// when reproducing the pre-polish fit for benchmarking). Each sweep
    /// cycles every link's unconstrained parameter through a golden-section
    /// step against the true quadrature-integrated factor log-likelihood.
    pub joint_polish_cycles: usize,
    /// Relative-tolerance stop criterion for the joint-MLE polish: a sweep
    /// that improves the log-likelihood by less than `rel_tol * |loglik|`
    /// (absolute when the loglik is near zero) is treated as convergence
    /// and ends the polish early. Matches the resolution of the golden-
    /// section inner steps.
    pub joint_polish_rel_tol: f64,
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
            // Small initial work budget; adaptive accuracy and maximum work
            // are controlled separately by `quadrature`.
            quadrature_nodes: 25,
            quadrature: FactorQuadrature::default(),
            refine_iterations: 2,
            // Bound the number of sweeps; diagnostics report whether the
            // stopping criterion was reached before this cap.
            joint_polish_cycles: 5,
            joint_polish_rel_tol: 1e-6,
        }
    }
}

/// Fitted factor copula.
///
/// The model stores one `PairCopulaSpec` per observed variable, giving the
/// joint density via quadrature over the single latent factor.
#[derive(Debug, Clone, Serialize)]
pub struct FactorCopula {
    dim: usize,
    layout: FactorLayout,
    /// One link per observed variable. For `Basic1F`, `links[j]` is the
    /// bivariate copula between variable `j` and the common latent factor.
    links: Vec<PairCopulaSpec>,
    quadrature_nodes: usize,
    quadrature: FactorQuadrature,
}

impl<'de> Deserialize<'de> for FactorCopula {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct State {
            dim: usize,
            layout: FactorLayout,
            links: Vec<PairCopulaSpec>,
            quadrature_nodes: usize,
            #[serde(default)]
            quadrature: FactorQuadrature,
        }
        let state = State::deserialize(deserializer)?;
        let model = Self::basic_1f(state.links, state.quadrature_nodes)
            .and_then(|model| model.with_quadrature(state.quadrature))
            .map_err(serde::de::Error::custom)?;
        if model.dim != state.dim || model.layout != state.layout {
            return Err(serde::de::Error::custom(
                "factor dimension or layout disagrees with its links",
            ));
        }
        Ok(model)
    }
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
        if !(3..=4096).contains(&quadrature_nodes) {
            return Err(FitError::Failed {
                reason: "factor copula quadrature requires at least three nodes and at most 4096",
            }
            .into());
        }
        for link in &links {
            link.validate()?;
        }
        Ok(Self {
            dim: links.len(),
            layout: FactorLayout::Basic1F,
            links,
            quadrature_nodes,
            quadrature: FactorQuadrature::default(),
        })
    }

    /// Sets the work/accuracy policy used for every subsequent density call.
    pub fn with_quadrature(mut self, quadrature: FactorQuadrature) -> Result<Self, CopulaError> {
        quadrature.validate(self.quadrature_nodes)?;
        self.quadrature = quadrature;
        Ok(self)
    }

    pub fn quadrature(&self) -> FactorQuadrature {
        self.quadrature
    }

    /// Fits a factor copula to pseudo-observations using a two-stage sequential
    /// MLE: (i) build a pseudo-latent via normal-score PCA-like projection,
    /// (ii) fit each link by bivariate MLE against the pseudo-latent.
    ///
    /// `FitDiagnostics` evaluates the joint likelihood with the fitted links
    /// and the same integration policy used at inference. Fixed quadrature
    /// explicitly forgoes the adaptive accuracy checks.
    /// Ordinary AIC/BIC uses this joint likelihood. Nested HAC composite
    /// scores cannot be compared using ordinary likelihood criteria.
    pub fn fit(
        data: &PseudoObs,
        options: &FactorFitOptions,
    ) -> Result<FactorFitResult, CopulaError> {
        options.base.validate()?;
        options.quadrature.validate(options.quadrature_nodes)?;
        if !options.joint_polish_rel_tol.is_finite() || options.joint_polish_rel_tol <= 0.0 {
            return Err(FitError::Failed {
                reason: "polish tolerance must be finite and positive",
            }
            .into());
        }
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

    /// Minimum Gauss-Legendre node budget used by `log_pdf`.
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
        normal_nodes: &[f64],
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        // Integrand at v: ∏_j c_j(u_j, v) = exp(Σ_j log c_j(u_j, v)).
        // So log ∫ = log Σ_q w_q exp(Σ_j log c_j(u_j, v_q)) — classical
        // log-sum-exp with `log w_q` added to the per-node accumulation.
        let mut log_terms: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
        for (j, link) in self.links.iter().enumerate() {
            // Quantiles of each observation and node are invariant across the
            // other axis. Reuse them in mixed-family models as well.
            if let (PairCopulaFamily::Gaussian, crate::paircopula::PairCopulaParams::One(rho)) =
                (link.family, &link.params)
            {
                let rho = if matches!(
                    link.rotation,
                    crate::paircopula::Rotation::R90 | crate::paircopula::Rotation::R270
                ) {
                    -*rho
                } else {
                    *rho
                };
                let z = Normal::new(0.0, 1.0)
                    .unwrap()
                    .inverse_cdf(obs[j].clamp(clip_eps, 1.0 - clip_eps));
                let variance = 1.0 - rho * rho;
                for (term, &v) in log_terms.iter_mut().zip(normal_nodes) {
                    *term += -0.5 * variance.ln()
                        - (rho * rho * (z * z + v * v) - 2.0 * rho * z * v) / (2.0 * variance);
                }
            } else if link.family != PairCopulaFamily::Independence {
                for (term, &v) in log_terms.iter_mut().zip(nodes) {
                    *term += link.log_pdf(obs[j], v, clip_eps)?;
                }
            }
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
        options.validate()?;
        crate::backend::resolve_strategy(
            options.exec,
            crate::backend::Operation::DensityEval,
            data.n_obs(),
        )?;
        let view = data.as_view();
        if view.ncols() != self.dim {
            return Err(FitError::Failed {
                reason: "factor copula input dimension does not match model dimension",
            }
            .into());
        }
        if self
            .links
            .iter()
            .filter(|link| link.family != PairCopulaFamily::Independence)
            .count()
            <= 1
        {
            return Ok(vec![0.0; data.n_obs()]);
        }
        // Gaussian links have an exact Gaussian integral; use it rather than
        // attempting to resolve a very narrow latent posterior on a fixed grid.
        let loadings: Option<Vec<f64>> = self
            .links
            .iter()
            .map(|link| {
                use crate::paircopula::{PairCopulaParams as P, Rotation};
                match (&link.family, &link.params) {
                    (PairCopulaFamily::Independence, P::None) => Some(0.0),
                    (PairCopulaFamily::Gaussian, P::One(rho)) => {
                        Some(if matches!(link.rotation, Rotation::R90 | Rotation::R270) {
                            -*rho
                        } else {
                            *rho
                        })
                    }
                    _ => None,
                }
            })
            .collect();
        if let Some(loadings) = loadings {
            let mut correlation = Array2::eye(self.dim);
            for i in 0..self.dim {
                for j in 0..i {
                    correlation[(i, j)] = loadings[i] * loadings[j];
                    correlation[(j, i)] = correlation[(i, j)];
                }
            }
            return super::GaussianCopula::new(correlation)?.log_pdf(data, options);
        }
        let normal = Normal::new(0.0, 1.0).unwrap();
        if !self.quadrature.adaptive {
            let (nodes, weights) = factor_quadrature(self.quadrature_nodes);
            let normal_nodes: Vec<_> = nodes.iter().map(|&v| normal.inverse_cdf(v)).collect();
            return view
                .rows()
                .into_iter()
                .map(|row| {
                    let obs: Vec<_> = row
                        .iter()
                        .map(|u| u.clamp(options.clip_eps, 1.0 - options.clip_eps))
                        .collect();
                    self.log_pdf_single(&obs, &nodes, &weights, &normal_nodes, f64::EPSILON / 2.0)
                })
                .collect();
        }
        // Include TLL knots, where derivatives change, in every row's mesh.
        let mut knots = Vec::new();
        for link in &self.links {
            if let crate::paircopula::PairCopulaParams::Tll(params) = &link.params {
                for &knot in params.second_margin_knots() {
                    let v = if matches!(
                        link.rotation,
                        crate::paircopula::Rotation::R180 | crate::paircopula::Rotation::R270
                    ) {
                        1.0 - knot
                    } else {
                        knot
                    };
                    if v > 0.0 && v < 1.0 {
                        let z = normal.inverse_cdf(v);
                        if z > -8.0 && z < 8.0 {
                            knots.push(z);
                        }
                    }
                }
            }
        }
        view.rows()
            .into_iter()
            .map(|row| {
                let obs: Vec<_> = row
                    .iter()
                    .map(|u| u.clamp(options.clip_eps, 1.0 - options.clip_eps))
                    .collect();
                let normal_obs: Vec<_> = obs.iter().map(|&u| normal.inverse_cdf(u)).collect();
                let mut breaks = vec![-8.0, 0.0, 8.0];
                breaks.extend(&knots);
                for (&z, link) in normal_obs.iter().zip(&self.links) {
                    let z = if matches!(
                        link.rotation,
                        crate::paircopula::Rotation::R90 | crate::paircopula::Rotation::R270
                    ) {
                        -z
                    } else {
                        z
                    };
                    if z > -8.0 && z < 8.0 {
                        breaks.push(z);
                    }
                }
                breaks.sort_by(f64::total_cmp);
                breaks.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
                let integrand = |z: f64| -> Result<f64, CopulaError> {
                    let v = normal.cdf(z);
                    let tail = normal.cdf(-z.abs());
                    let (log_v, log_sv) = if z > 0.0 {
                        ((-tail).ln_1p(), tail.ln())
                    } else {
                        (tail.ln(), (-tail).ln_1p())
                    };
                    let mut value = -0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln();
                    for (j, link) in self.links.iter().enumerate() {
                        if let (
                            PairCopulaFamily::Gaussian,
                            crate::paircopula::PairCopulaParams::One(rho),
                        ) = (link.family, &link.params)
                        {
                            let rho = if matches!(
                                link.rotation,
                                crate::paircopula::Rotation::R90
                                    | crate::paircopula::Rotation::R270
                            ) {
                                -*rho
                            } else {
                                *rho
                            };
                            let x = normal_obs[j];
                            let variance = 1.0 - rho * rho;
                            value += -0.5 * variance.ln()
                                - (rho * rho * (x * x + z * z) - 2.0 * rho * x * z)
                                    / (2.0 * variance);
                        } else if link.family != PairCopulaFamily::Independence {
                            // The public clipping policy applies to observations.
                            // Clipping latent integration nodes at 1e-12 creates an
                            // artificial tail plateau and prevents convergence.
                            value += match link.log_pdf_from_log_probabilities(
                                [obs[j].ln(), log_v],
                                [(-obs[j]).ln_1p(), log_sv],
                            ) {
                                Some(value) => value,
                                None => link.log_pdf(obs[j], v, f64::EPSILON / 2.0)?,
                            };
                        }
                    }
                    if value.is_nan() || value == f64::INFINITY {
                        return Err(crate::errors::NumericalError::Failed {
                            reason: "factor integrand is not finite",
                        }
                        .into());
                    }
                    Ok(value)
                };
                adaptive_factor_integral(
                    &integrand,
                    &breaks,
                    self.quadrature_nodes,
                    self.quadrature,
                )
            })
            .collect()
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        crate::backend::resolve_strategy(options.exec, crate::backend::Operation::Sample, n)?;
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

#[derive(Clone, Copy)]
struct IntegralInterval {
    lower: f64,
    upper: f64,
    value: f64,
    error: f64,
}

fn factor_interval<F: Fn(f64) -> Result<f64, CopulaError>>(
    f: &F,
    lower: f64,
    upper: f64,
) -> Result<IntegralInterval, CopulaError> {
    use std::sync::OnceLock;
    type Rule = (Vec<f64>, Vec<f64>);
    static RULES: OnceLock<(Rule, Rule)> = OnceLock::new();
    let (small, large) = RULES.get_or_init(|| (gauss_legendre_01(8), gauss_legendre_01(16)));
    let integrate = |rule: &Rule| -> Result<f64, CopulaError> {
        let mut terms = Vec::with_capacity(rule.0.len());
        for (&x, &w) in rule.0.iter().zip(&rule.1) {
            terms.push(w.ln() + f(lower + (upper - lower) * x)?);
        }
        Ok((upper - lower).ln() + log_sum_exp(&terms))
    };
    let coarse = integrate(small)?;
    let value = integrate(large)?;
    let error = if coarse == value {
        f64::NEG_INFINITY
    } else {
        coarse.max(value) + crate::math::log1mexp(-(coarse - value).abs())
    };
    Ok(IntegralInterval {
        lower,
        upper,
        value,
        error,
    })
}

fn adaptive_factor_integral<F: Fn(f64) -> Result<f64, CopulaError>>(
    f: &F,
    breaks: &[f64],
    min_nodes: usize,
    policy: FactorQuadrature,
) -> Result<f64, CopulaError> {
    let budget_error = || {
        CopulaError::from(crate::errors::NumericalError::Failed {
            reason: "factor quadrature exhausted its node budget; increase quadrature_max_nodes, relax quadrature_rel_tol, or explicitly use fixed quadrature",
        })
    };
    let mut spent = 24 * (breaks.len() - 1);
    if spent > policy.max_nodes {
        return Err(budget_error());
    }
    let mut intervals = breaks
        .windows(2)
        .map(|w| factor_interval(f, w[0], w[1]))
        .collect::<Result<Vec<_>, _>>()?;
    loop {
        let value = log_sum_exp(&intervals.iter().map(|x| x.value).collect::<Vec<_>>());
        let error = log_sum_exp(&intervals.iter().map(|x| x.error).collect::<Vec<_>>());
        if value.is_finite() && spent >= min_nodes && error <= value + policy.rel_tol.ln() {
            return Ok(value);
        }
        if spent + 48 > policy.max_nodes {
            return Err(budget_error());
        }
        let worst = intervals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.error.total_cmp(&b.1.error))
            .unwrap()
            .0;
        let old = intervals[worst];
        let mid = (old.lower + old.upper) * 0.5;
        intervals[worst] = factor_interval(f, old.lower, mid)?;
        intervals.push(factor_interval(f, mid, old.upper)?);
        spent += 48;
    }
}

/// Normal-scale quadrature resolves the latent tails without an endpoint cutoff
/// at the first probability-scale Gauss node. Cache observation-independent rules.
fn factor_quadrature(n: usize) -> (Vec<f64>, Vec<f64>) {
    use std::{
        collections::BTreeMap,
        sync::{Mutex, OnceLock},
    };
    type Rules = BTreeMap<usize, (Vec<f64>, Vec<f64>)>;
    static RULES: OnceLock<Mutex<Rules>> = OnceLock::new();
    let cache = RULES.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache
        .entry(n)
        .or_insert_with(|| {
            let normal = Normal::new(0.0, 1.0).unwrap();
            let (mut nodes, mut weights) = gauss_legendre_01(n);
            for (node, weight) in nodes.iter_mut().zip(weights.iter_mut()) {
                let z = 16.0 * *node - 8.0;
                *node = normal.cdf(z);
                *weight *= 16.0 * (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
            }
            (nodes, weights)
        })
        .clone()
}

fn fit_basic_1f(
    data: &PseudoObs,
    options: &FactorFitOptions,
) -> Result<FactorFitResult, CopulaError> {
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
        ..VineFitOptions::default()
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
    let mut model = FactorCopula::basic_1f(links.clone(), options.quadrature_nodes)?
        .with_quadrature(options.quadrature)?;

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
    let mut n_iter = 1;
    for _ in 0..options.refine_iterations {
        n_iter += 1;
        let posterior = posterior_latent_mean(&model, data, clip)?;
        v_pseudo = rank_normalise_01(&posterior, clip);
        let refined_links = fit_links_against_latent(&columns, &v_pseudo, &vine_options)?;
        let refined = FactorCopula::basic_1f(refined_links.clone(), options.quadrature_nodes)?
            .with_quadrature(options.quadrature)?;
        let refined_loglik = match factor_log_likelihood(&refined, data, options.base.clip_eps) {
            Ok(value) => value,
            Err(CopulaError::Numerical(_)) => break,
            Err(error) => return Err(error),
        };
        if refined_loglik > best_loglik {
            best_loglik = refined_loglik;
            links = refined_links;
            model = refined;
        } else {
            break;
        }
    }

    // Stage 4: joint-MLE polish.
    //
    // The sequential+EM stages above never optimise the full quadrature-
    // integrated factor log-likelihood — each link is fit against a rank-
    // normalised pseudo-latent. For tail-asymmetric families (Clayton, Joe,
    // BB*) this leaves real money on the table: Joe's own `CopulaModel`
    // reference package always runs a joint-MLE polish via `nlm` after the
    // two-stage warm start. We do the same here with coordinate ascent over
    // unconstrained-space reparametrisations of each link's free params.
    //
    // The bail-out guard mirrors the EM pattern above: accept the polished
    // fit only if it strictly improves the log-likelihood, never worse.
    let mut polished_loglik = best_loglik;
    let mut converged = false;
    if options.joint_polish_cycles > 0 {
        let (candidate_model, candidate_loglik, cycles, stopped) = polish_factor_model(
            &model,
            data,
            options.base.clip_eps,
            options.joint_polish_rel_tol,
            options.joint_polish_cycles,
        )?;
        n_iter += cycles;
        converged = stopped;
        if candidate_loglik > polished_loglik {
            polished_loglik = candidate_loglik;
            model = candidate_model;
            links = model.links.clone();
        }
    }

    let total_parameters: usize = links.iter().map(|link| link.parameter_count()).sum();

    // Final diagnostics use the true factor-copula log-likelihood (not the
    // per-link pseudo-likelihood sum), so AIC/BIC are comparable with other
    // domain-level models in the crate.
    let loglik = polished_loglik;
    let k = total_parameters as f64;
    let n_f = n as f64;
    let (aic, bic) = if loglik.is_finite() {
        (2.0 * k - 2.0 * loglik, k * n_f.ln() - 2.0 * loglik)
    } else {
        (f64::INFINITY, f64::INFINITY)
    };

    // Delta-method standard errors from the numerical Hessian of the factor
    // log-likelihood at the polished MLE. When the polish was disabled or
    // bailed out, the Hessian is computed at the pre-polish parameters, and
    // callers should interpret the SEs as only approximately valid — the
    // point is still a stationary point along each coordinate from the EM
    // refinement's perspective, though not necessarily jointly.
    let std_errors = factor_standard_errors(&model, data, options.base.clip_eps);

    Ok(FactorFitResult {
        model,
        diagnostics: FitDiagnostics {
            likelihood_kind: crate::domain::LikelihoodKind::Joint,
            loglik,
            aic,
            bic,
            converged,
            n_iter,
        },
        std_errors,
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
/// via a fixed probability-scale rule for initialization/refinement only.
/// Candidate acceptance and joint polishing use the checked density. Returns one value per
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

/// Joint-MLE polish: coordinate ascent over the unconstrained-space
/// reparametrisations of every link's free parameters against the factor-
/// copula log-likelihood. Returns the polished model plus its final loglik.
///
/// The polish holds skipped families (Independence, TLL, Khoudraji, plus the
/// ν block of Student-t) fixed at the sequential-MLE values; only parameters
/// that `encode_params` reports are updated. When no link has polishable
/// parameters the input model is returned unchanged.
fn polish_factor_model(
    model: &FactorCopula,
    data: &PseudoObs,
    clip_eps: f64,
    rel_tol: f64,
    max_cycles: usize,
) -> Result<(FactorCopula, f64, usize, bool), CopulaError> {
    let per_link_counts: Vec<usize> = model
        .links
        .iter()
        .map(|link| encode_params(link).len())
        .collect();
    let total_polishable: usize = per_link_counts.iter().sum();
    if total_polishable == 0 {
        let loglik = factor_log_likelihood(model, data, clip_eps)?;
        return Ok((model.clone(), loglik, 0, true));
    }

    let mut x0 = Vec::with_capacity(total_polishable);
    let mut brackets = Vec::with_capacity(total_polishable);
    for link in &model.links {
        x0.extend(encode_params(link));
        brackets.extend(encode_brackets(link.family));
    }

    let template_links = model.links.clone();
    let quadrature_nodes = model.quadrature_nodes;
    let counts = per_link_counts.clone();

    let evaluate = |x: &[f64]| -> f64 {
        match build_model_from_flat(
            &template_links,
            &counts,
            quadrature_nodes,
            model.quadrature,
            x,
        ) {
            Ok(candidate) => {
                factor_log_likelihood(&candidate, data, clip_eps).unwrap_or(f64::NEG_INFINITY)
            }
            Err(_) => f64::NEG_INFINITY,
        }
    };

    let (x_polished, polished_loglik, cycles, converged) =
        coord_ascent_with_diagnostics(&x0, &brackets, rel_tol, max_cycles, 60, evaluate);

    let polished_model = build_model_from_flat(
        &template_links,
        &per_link_counts,
        quadrature_nodes,
        model.quadrature,
        &x_polished,
    )?;
    Ok((polished_model, polished_loglik, cycles, converged))
}

/// Delta-method standard errors for every polished link parameter.
///
/// Computes the numerical Hessian `H` of the factor log-likelihood at the
/// fitted MLE in unconstrained space, inverts `-H` (observed Fisher
/// information) for the covariance, and maps back to natural-parameter
/// standard errors through `|dθ/dx|` supplied by `encode_jacobian`:
///
/// ```text
///     SE(θ_k) = |dθ_k / dx_k| · sqrt([(-H)⁻¹]_{kk}).
/// ```
///
/// Returns one entry per *polished* parameter, matching the concatenation of
/// `encode_params` across links. When the inverse of `-H` is not positive-
/// definite — typical when the polish bailed out or when the MLE sits at a
/// bound — every entry is `f64::NAN` so callers can distinguish a well-
/// identified parameter from an uninformative one.
fn factor_standard_errors(model: &FactorCopula, data: &PseudoObs, clip_eps: f64) -> Vec<f64> {
    let mut x_star = Vec::new();
    let mut jacobians = Vec::new();
    let mut per_link_counts = Vec::new();
    for link in &model.links {
        let encoded = encode_params(link);
        per_link_counts.push(encoded.len());
        jacobians.extend(encode_jacobian(link));
        x_star.extend(encoded);
    }
    let p = x_star.len();
    if p == 0 {
        return Vec::new();
    }

    let template_links = model.links.clone();
    let quadrature_nodes = model.quadrature_nodes;
    let counts = per_link_counts.clone();
    let evaluate = |x: &[f64]| -> f64 {
        match build_model_from_flat(
            &template_links,
            &counts,
            quadrature_nodes,
            model.quadrature,
            x,
        ) {
            Ok(candidate) => {
                factor_log_likelihood(&candidate, data, clip_eps).unwrap_or(f64::NEG_INFINITY)
            }
            Err(_) => f64::NEG_INFINITY,
        }
    };

    // Step size 1e-4 is a robust default for 2nd-order central differences on
    // a loglik objective scaled by n — the relative truncation error is
    // O(h²) ~ 1e-8 and the relative roundoff is ~ε·|f|/h² ~ ε·n·1e8 which is
    // below 1e-5 for typical n. Users with unusual scales can adjust via the
    // options struct in a future patch.
    let hess = numerical_hessian(&x_star, 1e-4, evaluate);
    let coarser = numerical_hessian(&x_star, 2e-4, evaluate);
    if hess.iter().zip(coarser.iter()).any(|(a, b)| {
        !a.is_finite() || !b.is_finite() || (a - b).abs() > 0.01 * (1.0 + a.abs().max(b.abs()))
    }) {
        return vec![f64::NAN; p];
    }

    // Observed Fisher information is the Hessian of the *negative* log-
    // likelihood; since `hess` is the Hessian of the (unnegated) log-
    // likelihood, the Fisher information matrix is `-hess`.
    let mut neg_hess = ndarray::Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            neg_hess[(i, j)] = -hess[(i, j)];
        }
    }
    if crate::math::cholesky(&neg_hess).is_err() {
        return vec![f64::NAN; p];
    }
    match inverse(&neg_hess) {
        Ok(cov) => (0..p)
            .map(|k| {
                let var = cov[(k, k)];
                if var > 0.0 && var.is_finite() {
                    jacobians[k].abs() * var.sqrt()
                } else {
                    f64::NAN
                }
            })
            .collect(),
        Err(_) => vec![f64::NAN; p],
    }
}

/// Rebuilds a [`FactorCopula`] from a flat vector of unconstrained parameters
/// by slicing per link and decoding each slice against the link template. A
/// shared helper between the polish's inner loop and the Hessian evaluator
/// so both are guaranteed to agree on the encoding convention.
fn build_model_from_flat(
    template_links: &[PairCopulaSpec],
    per_link_counts: &[usize],
    quadrature_nodes: usize,
    quadrature: FactorQuadrature,
    x: &[f64],
) -> Result<FactorCopula, CopulaError> {
    let mut links = Vec::with_capacity(template_links.len());
    let mut cursor = 0;
    for (template, count) in template_links.iter().zip(per_link_counts.iter()) {
        let slice = &x[cursor..cursor + *count];
        cursor += *count;
        links.push(if *count == 0 {
            template.clone()
        } else {
            decode_params(template, slice)
        });
    }
    FactorCopula::basic_1f(links, quadrature_nodes)?.with_quadrature(quadrature)
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
    fn adaptive_integration_counts_all_work_against_its_budget() {
        let calls = std::cell::Cell::new(0);
        let integrand = |x: f64| {
            calls.set(calls.get() + 1);
            Ok(-10_000.0 * (x - 0.13).powi(2))
        };
        let policy = FactorQuadrature {
            max_nodes: 72,
            rel_tol: 1e-12,
            ..Default::default()
        };
        assert!(adaptive_factor_integral(&integrand, &[0.0, 1.0], 25, policy).is_err());
        assert_eq!(calls.get(), 72);
        let value =
            adaptive_factor_integral(&integrand, &[0.0, 1.0], 25, FactorQuadrature::default())
                .unwrap();
        // Integral of exp(-10000(x-.13)^2) over this interval equals the full
        // Gaussian integral to far better than double precision.
        let exact = (std::f64::consts::PI / 10_000.0).sqrt().ln();
        assert!((value - exact).abs() < 1e-9);
    }

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
