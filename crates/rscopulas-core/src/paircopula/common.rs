use serde::{Deserialize, Serialize};

use crate::{
    backend::{ExecutionStrategy, Operation, parallel_try_map_range_collect, resolve_strategy},
    domain::ExecPolicy,
    errors::{BackendError, CopulaError, FitError},
    math::maximize_scalar,
    vine::{SelectionCriterion, VineFitOptions},
};

use super::{clayton, frank, gaussian, gumbel, joe, khoudraji, rotated, student_t};

/// Supported bivariate pair-copula families for vine edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairCopulaFamily {
    Independence,
    Gaussian,
    StudentT,
    Clayton,
    Frank,
    Gumbel,
    Khoudraji,
    Joe,
}

/// Rotation applied to a bivariate pair-copula kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rotation {
    R0,
    R90,
    R180,
    R270,
}

/// Structured parameterization for a Khoudraji pair-copula.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KhoudrajiParams {
    pub first: Box<PairCopulaSpec>,
    pub second: Box<PairCopulaSpec>,
    pub shape_first: f64,
    pub shape_second: f64,
}

impl KhoudrajiParams {
    pub fn new(
        first: PairCopulaSpec,
        second: PairCopulaSpec,
        shape_first: f64,
        shape_second: f64,
    ) -> Result<Self, CopulaError> {
        if !(0.0..=1.0).contains(&shape_first) || !(0.0..=1.0).contains(&shape_second) {
            return Err(FitError::Failed {
                reason: "khoudraji shape parameters must lie in [0, 1]",
            }
            .into());
        }
        if first.family == PairCopulaFamily::Khoudraji
            || second.family == PairCopulaFamily::Khoudraji
        {
            return Err(FitError::Failed {
                reason: "nested khoudraji pair copulas are not supported",
            }
            .into());
        }
        Ok(Self {
            first: Box::new(first),
            second: Box::new(second),
            shape_first,
            shape_second,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.first.parameter_count() + self.second.parameter_count() + 2
    }

    pub fn flat_values(&self) -> Vec<f64> {
        let mut values = self.first.params.flat_values();
        values.extend(self.second.params.flat_values());
        values.push(self.shape_first);
        values.push(self.shape_second);
        values
    }
}

/// Parameter storage for pair-copula families.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PairCopulaParams {
    None,
    One(f64),
    Two(f64, f64),
    Khoudraji(KhoudrajiParams),
}

impl PairCopulaParams {
    pub fn flat_values(&self) -> Vec<f64> {
        match self {
            Self::None => Vec::new(),
            Self::One(value) => vec![*value],
            Self::Two(first, second) => vec![*first, *second],
            Self::Khoudraji(params) => params.flat_values(),
        }
    }
}

/// Fully specified pair-copula family, rotation, and parameter tuple.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairCopulaSpec {
    pub family: PairCopulaFamily,
    pub rotation: Rotation,
    pub params: PairCopulaParams,
}

/// Result of fitting a single pair-copula candidate to one edge.
#[derive(Debug, Clone)]
pub struct PairFitResult {
    pub spec: PairCopulaSpec,
    pub loglik: f64,
    pub aic: f64,
    pub bic: f64,
    pub cond_on_first: Vec<f64>,
    pub cond_on_second: Vec<f64>,
}

#[derive(Debug)]
struct PairBatchEvaluation {
    log_pdf: Vec<f64>,
    cond_on_first: Vec<f64>,
    cond_on_second: Vec<f64>,
}

pub(crate) struct PairBatchBuffers<'a> {
    pub(crate) log_pdf: &'a mut [f64],
    pub(crate) cond_on_first: &'a mut [f64],
    pub(crate) cond_on_second: &'a mut [f64],
}

impl PairCopulaSpec {
    /// Returns the independence pair-copula specification.
    pub fn independence() -> Self {
        Self {
            family: PairCopulaFamily::Independence,
            rotation: Rotation::R0,
            params: PairCopulaParams::None,
        }
    }

    /// Returns a Khoudraji pair-copula specification with no outer rotation.
    pub fn khoudraji(
        first: PairCopulaSpec,
        second: PairCopulaSpec,
        shape_first: f64,
        shape_second: f64,
    ) -> Result<Self, CopulaError> {
        Ok(Self {
            family: PairCopulaFamily::Khoudraji,
            rotation: Rotation::R0,
            params: PairCopulaParams::Khoudraji(KhoudrajiParams::new(
                first,
                second,
                shape_first,
                shape_second,
            )?),
        })
    }

    /// Swaps the conditioning axis while preserving the represented copula.
    pub fn swap_axes(self) -> Self {
        let rotation = match self.rotation {
            Rotation::R90 => Rotation::R270,
            Rotation::R270 => Rotation::R90,
            other => other,
        };
        Self { rotation, ..self }
    }

    /// Returns the number of free parameters implied by `params`.
    pub fn parameter_count(&self) -> usize {
        match &self.params {
            PairCopulaParams::None => 0,
            PairCopulaParams::One(_) => 1,
            PairCopulaParams::Two(_, _) => 2,
            PairCopulaParams::Khoudraji(params) => params.parameter_count(),
        }
    }

    /// Returns a flattened numeric representation of the free parameters.
    pub fn flat_parameters(&self) -> Vec<f64> {
        self.params.flat_values()
    }

    /// Evaluates the pair-copula CDF at `(u1, u2)`.
    pub(crate) fn cdf(&self, u1: f64, u2: f64, clip_eps: f64) -> Result<f64, CopulaError> {
        let u1 = u1.clamp(clip_eps, 1.0 - clip_eps);
        let u2 = u2.clamp(clip_eps, 1.0 - clip_eps);
        match self.rotation {
            Rotation::R0 => self.base_cdf(u1, u2, clip_eps),
            Rotation::R180 => {
                Ok((u1 + u2 - 1.0 + self.base_cdf(1.0 - u1, 1.0 - u2, clip_eps)?).clamp(0.0, 1.0))
            }
            Rotation::R90 => Ok((u2 - self.base_cdf(1.0 - u1, u2, clip_eps)?).clamp(0.0, 1.0)),
            Rotation::R270 => Ok((u1 - self.base_cdf(u1, 1.0 - u2, clip_eps)?).clamp(0.0, 1.0)),
        }
    }

    /// Evaluates the pair-copula log-density at `(u1, u2)`.
    pub fn log_pdf(&self, u1: f64, u2: f64, clip_eps: f64) -> Result<f64, CopulaError> {
        let ((x1, x2), rotation) = rotated::to_base_inputs(self.rotation, u1, u2, clip_eps);
        let base = match (self.family, &self.params) {
            (PairCopulaFamily::Independence, PairCopulaParams::None) => 0.0,
            (PairCopulaFamily::Gaussian, PairCopulaParams::One(rho)) => {
                gaussian::log_pdf(x1, x2, *rho)?
            }
            (PairCopulaFamily::StudentT, PairCopulaParams::Two(rho, nu)) => {
                student_t::log_pdf(x1, x2, *rho, *nu)?
            }
            (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => {
                clayton::log_pdf(x1, x2, *theta)?
            }
            (PairCopulaFamily::Frank, PairCopulaParams::One(theta)) => {
                frank::log_pdf(x1, x2, *theta)?
            }
            (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => {
                gumbel::log_pdf(x1, x2, *theta)?
            }
            (PairCopulaFamily::Joe, PairCopulaParams::One(theta)) => {
                joe::log_pdf(x1, x2, *theta)?
            }
            (PairCopulaFamily::Khoudraji, PairCopulaParams::Khoudraji(params)) => {
                khoudraji::log_pdf(x1, x2, params, clip_eps)?
            }
            _ => {
                return Err(FitError::Failed {
                    reason: "pair-copula family/parameter combination is invalid",
                }
                .into());
            }
        };
        Ok(rotated::from_base_log_pdf(rotation, base))
    }

    /// Evaluates `h_{1|2}(u1 | u2)`.
    pub fn cond_first_given_second(
        &self,
        u1: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        match self.rotation {
            Rotation::R0 => self.base_cond_first_given_second(u1, u2, clip_eps),
            Rotation::R180 => {
                Ok(1.0 - self.base_cond_first_given_second(1.0 - u1, 1.0 - u2, clip_eps)?)
            }
            Rotation::R90 => Ok(1.0 - self.base_cond_first_given_second(1.0 - u1, u2, clip_eps)?),
            Rotation::R270 => self.base_cond_first_given_second(u1, 1.0 - u2, clip_eps),
        }
        .map(|value| value.clamp(clip_eps, 1.0 - clip_eps))
    }

    /// Evaluates `h_{2|1}(u2 | u1)`.
    pub fn cond_second_given_first(
        &self,
        u1: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        match self.rotation {
            Rotation::R0 => self.base_cond_second_given_first(u1, u2, clip_eps),
            Rotation::R180 => {
                Ok(1.0 - self.base_cond_second_given_first(1.0 - u1, 1.0 - u2, clip_eps)?)
            }
            Rotation::R90 => self.base_cond_second_given_first(1.0 - u1, u2, clip_eps),
            Rotation::R270 => Ok(1.0 - self.base_cond_second_given_first(u1, 1.0 - u2, clip_eps)?),
        }
        .map(|value| value.clamp(clip_eps, 1.0 - clip_eps))
    }

    /// Evaluates the inverse h-function for the first margin conditional on the second.
    pub fn inv_first_given_second(
        &self,
        p: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        let p = p.clamp(clip_eps, 1.0 - clip_eps);
        let u2 = u2.clamp(clip_eps, 1.0 - clip_eps);
        match self.rotation {
            Rotation::R0 => self.base_inv_first_given_second(p, u2, clip_eps),
            Rotation::R180 => {
                Ok(1.0 - self.base_inv_first_given_second(1.0 - p, 1.0 - u2, clip_eps)?)
            }
            Rotation::R90 => Ok(1.0 - self.base_inv_first_given_second(1.0 - p, u2, clip_eps)?),
            Rotation::R270 => self.base_inv_first_given_second(p, 1.0 - u2, clip_eps),
        }
        .map(|value| value.clamp(clip_eps, 1.0 - clip_eps))
    }

    /// Evaluates the inverse h-function for the second margin conditional on the first.
    pub fn inv_second_given_first(
        &self,
        u1: f64,
        p: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        let p = p.clamp(clip_eps, 1.0 - clip_eps);
        let u1 = u1.clamp(clip_eps, 1.0 - clip_eps);
        match self.rotation {
            Rotation::R0 => self.base_inv_second_given_first(u1, p, clip_eps),
            Rotation::R180 => {
                Ok(1.0 - self.base_inv_second_given_first(1.0 - u1, 1.0 - p, clip_eps)?)
            }
            Rotation::R90 => self.base_inv_second_given_first(1.0 - u1, p, clip_eps),
            Rotation::R270 => Ok(1.0 - self.base_inv_second_given_first(u1, 1.0 - p, clip_eps)?),
        }
        .map(|value| value.clamp(clip_eps, 1.0 - clip_eps))
    }

    fn base_cond_first_given_second(
        &self,
        u1: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        let u1 = u1.clamp(clip_eps, 1.0 - clip_eps);
        let u2 = u2.clamp(clip_eps, 1.0 - clip_eps);
        match (self.family, &self.params) {
            (PairCopulaFamily::Independence, PairCopulaParams::None) => Ok(u1),
            (PairCopulaFamily::Gaussian, PairCopulaParams::One(rho)) => {
                gaussian::cond_first_given_second(u1, u2, *rho)
            }
            (PairCopulaFamily::StudentT, PairCopulaParams::Two(rho, nu)) => {
                student_t::cond_first_given_second(u1, u2, *rho, *nu)
            }
            (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => {
                clayton::cond_first_given_second(u1, u2, *theta)
            }
            (PairCopulaFamily::Frank, PairCopulaParams::One(theta)) => {
                frank::cond_first_given_second(u1, u2, *theta)
            }
            (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => {
                gumbel::cond_first_given_second(u1, u2, *theta, clip_eps)
            }
            (PairCopulaFamily::Joe, PairCopulaParams::One(theta)) => {
                joe::cond_first_given_second(u1, u2, *theta)
            }
            (PairCopulaFamily::Khoudraji, PairCopulaParams::Khoudraji(params)) => {
                khoudraji::cond_first_given_second(u1, u2, params, clip_eps)
            }
            _ => Err(FitError::Failed {
                reason: "pair-copula family/parameter combination is invalid",
            }
            .into()),
        }
    }

    fn base_cond_second_given_first(
        &self,
        u1: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        let u1 = u1.clamp(clip_eps, 1.0 - clip_eps);
        let u2 = u2.clamp(clip_eps, 1.0 - clip_eps);
        match (self.family, &self.params) {
            (PairCopulaFamily::Independence, PairCopulaParams::None) => Ok(u2),
            (PairCopulaFamily::Gaussian, PairCopulaParams::One(rho)) => {
                gaussian::cond_second_given_first(u1, u2, *rho)
            }
            (PairCopulaFamily::StudentT, PairCopulaParams::Two(rho, nu)) => {
                student_t::cond_second_given_first(u1, u2, *rho, *nu)
            }
            (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => {
                clayton::cond_second_given_first(u1, u2, *theta)
            }
            (PairCopulaFamily::Frank, PairCopulaParams::One(theta)) => {
                frank::cond_second_given_first(u1, u2, *theta)
            }
            (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => {
                gumbel::cond_second_given_first(u1, u2, *theta, clip_eps)
            }
            (PairCopulaFamily::Joe, PairCopulaParams::One(theta)) => {
                joe::cond_second_given_first(u1, u2, *theta)
            }
            (PairCopulaFamily::Khoudraji, PairCopulaParams::Khoudraji(params)) => {
                khoudraji::cond_second_given_first(u1, u2, params, clip_eps)
            }
            _ => Err(FitError::Failed {
                reason: "pair-copula family/parameter combination is invalid",
            }
            .into()),
        }
    }

    fn base_inv_first_given_second(
        &self,
        p: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        match (self.family, &self.params) {
            (PairCopulaFamily::Independence, PairCopulaParams::None) => Ok(p),
            (PairCopulaFamily::Gaussian, PairCopulaParams::One(rho)) => {
                gaussian::inv_first_given_second(p, u2, *rho)
            }
            (PairCopulaFamily::StudentT, PairCopulaParams::Two(rho, nu)) => {
                student_t::inv_first_given_second(p, u2, *rho, *nu)
            }
            (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => {
                clayton::inv_first_given_second(p, u2, *theta)
            }
            (PairCopulaFamily::Frank, PairCopulaParams::One(theta)) => {
                frank::inv_first_given_second(p, u2, *theta)
            }
            (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => {
                gumbel::inv_first_given_second(p, u2, *theta, clip_eps)
            }
            (PairCopulaFamily::Joe, PairCopulaParams::One(theta)) => {
                joe::inv_first_given_second(p, u2, *theta, clip_eps)
            }
            (PairCopulaFamily::Khoudraji, PairCopulaParams::Khoudraji(params)) => {
                khoudraji::inv_first_given_second(p, u2, params, clip_eps)
            }
            _ => Err(FitError::Failed {
                reason: "pair-copula family/parameter combination is invalid",
            }
            .into()),
        }
    }

    fn base_inv_second_given_first(
        &self,
        u1: f64,
        p: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        match (self.family, &self.params) {
            (PairCopulaFamily::Independence, PairCopulaParams::None) => Ok(p),
            (PairCopulaFamily::Gaussian, PairCopulaParams::One(rho)) => {
                gaussian::inv_second_given_first(u1, p, *rho)
            }
            (PairCopulaFamily::StudentT, PairCopulaParams::Two(rho, nu)) => {
                student_t::inv_second_given_first(u1, p, *rho, *nu)
            }
            (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => {
                clayton::inv_second_given_first(u1, p, *theta)
            }
            (PairCopulaFamily::Frank, PairCopulaParams::One(theta)) => {
                frank::inv_second_given_first(u1, p, *theta)
            }
            (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => {
                gumbel::inv_second_given_first(u1, p, *theta, clip_eps)
            }
            (PairCopulaFamily::Joe, PairCopulaParams::One(theta)) => {
                joe::inv_second_given_first(u1, p, *theta, clip_eps)
            }
            (PairCopulaFamily::Khoudraji, PairCopulaParams::Khoudraji(params)) => {
                khoudraji::inv_second_given_first(u1, p, params, clip_eps)
            }
            _ => Err(FitError::Failed {
                reason: "pair-copula family/parameter combination is invalid",
            }
            .into()),
        }
    }

    fn base_cdf(&self, u1: f64, u2: f64, clip_eps: f64) -> Result<f64, CopulaError> {
        match (self.family, &self.params) {
            (PairCopulaFamily::Independence, PairCopulaParams::None) => {
                Ok((u1 * u2).clamp(0.0, 1.0))
            }
            (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => {
                let sum = u1.powf(-*theta) + u2.powf(-*theta) - 1.0;
                Ok(sum.max(0.0).powf(-1.0 / *theta).clamp(0.0, 1.0))
            }
            (PairCopulaFamily::Frank, PairCopulaParams::One(theta)) => {
                Ok(frank_cdf_stable(u1, u2, *theta).clamp(0.0, 1.0))
            }
            (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => {
                let term = ((-u1.ln()).powf(*theta) + (-u2.ln()).powf(*theta)).powf(1.0 / *theta);
                Ok(f64::exp(-term).clamp(0.0, 1.0))
            }
            (PairCopulaFamily::Joe, PairCopulaParams::One(theta)) => {
                // C(u1, u2) = 1 - ((1-u1)^θ + (1-u2)^θ - (1-u1)^θ(1-u2)^θ)^(1/θ)
                let a = (1.0 - u1).powf(*theta);
                let b = (1.0 - u2).powf(*theta);
                let s = a + b - a * b;
                Ok((1.0 - s.max(0.0).powf(1.0 / *theta)).clamp(0.0, 1.0))
            }
            (PairCopulaFamily::Gaussian, PairCopulaParams::One(_))
            | (PairCopulaFamily::StudentT, PairCopulaParams::Two(_, _))
            | (PairCopulaFamily::Khoudraji, PairCopulaParams::Khoudraji(_)) => {
                integrate_cdf_from_h(self, u1, u2, clip_eps)
            }
            _ => Err(FitError::Failed {
                reason: "pair-copula family/parameter combination is invalid",
            }
            .into()),
        }
    }
}

// Numerically stable Frank bivariate CDF. The naive evaluation suffers the
// same cancellation as the legacy density: for large θ and u, v away from 0
// all exponentials underflow to 1, and the ratio of differences collapses.
// We rewrite the argument of the log in terms of (T1 + T2)/(1 - e^{-θ})
// using the same identity employed by the density.
fn frank_cdf_stable(u1: f64, u2: f64, theta: f64) -> f64 {
    if !theta.is_finite() || theta <= 0.0 {
        return u1 * u2;
    }
    // log(1 - e^{-x}) helper — inlined to avoid extra module boundary.
    let log_one_minus_exp_neg = |x: f64| -> f64 {
        if x.is_nan() || x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if x > std::f64::consts::LN_2 {
            (-((-x).exp())).ln_1p()
        } else {
            (-((-x).exp_m1())).ln()
        }
    };
    let logsumexp2 = |a: f64, b: f64| -> f64 {
        if a == f64::NEG_INFINITY {
            return b;
        }
        if b == f64::NEG_INFINITY {
            return a;
        }
        let m = a.max(b);
        m + ((a - m).exp() + (b - m).exp()).ln()
    };
    let log_t1 = -theta * u1 + log_one_minus_exp_neg(theta * u2);
    let log_t2 = -theta * u2 + log_one_minus_exp_neg(theta * (1.0 - u2));
    let log_den = logsumexp2(log_t1, log_t2);
    let log_d = log_one_minus_exp_neg(theta);
    (log_d - log_den) / theta
}

fn integrate_cdf_from_h(
    spec: &PairCopulaSpec,
    u1: f64,
    u2: f64,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let upper = u2.clamp(clip_eps, 1.0 - clip_eps);
    let lower = clip_eps;
    let baseline = lower * spec.cond_first_given_second(u1, lower, clip_eps)?;
    if upper <= lower {
        return Ok(baseline.clamp(0.0, 1.0));
    }

    let steps = 32usize;
    let step = (upper - lower) / steps as f64;
    let mut total = spec.cond_first_given_second(u1, lower, clip_eps)?
        + spec.cond_first_given_second(u1, upper, clip_eps)?;
    for idx in 1..steps {
        let value = lower + idx as f64 * step;
        let weight = if idx % 2 == 0 { 2.0 } else { 4.0 };
        total += weight * spec.cond_first_given_second(u1, value, clip_eps)?;
    }
    Ok((baseline + total * step / 3.0).clamp(0.0, 1.0))
}

/// Fits the best pair-copula specification for one bivariate edge.
pub fn fit_pair_copula(
    u1: &[f64],
    u2: &[f64],
    options: &VineFitOptions,
) -> Result<PairFitResult, CopulaError> {
    if u1.len() != u2.len() || u1.is_empty() {
        return Err(FitError::Failed {
            reason: "pair-copula fit requires equally sized non-empty inputs",
        }
        .into());
    }

    let tau = crate::stats::kendall_tau_bivariate(u1, u2)?;
    if options
        .independence_threshold
        .is_some_and(|threshold| tau.abs() <= threshold)
    {
        return finalize_pair_fit(PairCopulaSpec::independence(), u1, u2, options);
    }

    let mut candidates = Vec::new();
    for family in &options.family_set {
        for rotation in candidate_rotations(*family, options.include_rotations, tau) {
            let spec = match fit_family_with_rotation(
                *family,
                *rotation,
                u1,
                u2,
                tau,
                options.base.max_iter,
            ) {
                Ok(spec) => spec,
                Err(_) => continue,
            };
            candidates.push(spec);
        }
    }

    let strategy = resolve_strategy(
        options.base.exec,
        Operation::PairFitScoring,
        candidates.len(),
    )?;
    let fits = parallel_try_map_range_collect(candidates.len(), strategy, |idx| {
        finalize_pair_fit(candidates[idx].clone(), u1, u2, options)
    })?;
    let best = fits.into_iter().min_by(|left, right| {
        criterion_value(left, options.criterion)
            .total_cmp(&criterion_value(right, options.criterion))
    });

    best.ok_or(
        FitError::Failed {
            reason: "pair-copula selection produced no candidate",
        }
        .into(),
    )
}

fn finalize_pair_fit(
    spec: PairCopulaSpec,
    u1: &[f64],
    u2: &[f64],
    options: &VineFitOptions,
) -> Result<PairFitResult, CopulaError> {
    let batch = evaluate_pair_batch(
        spec.clone(),
        u1,
        u2,
        options.base.clip_eps,
        options.base.exec,
    )?;

    // If any per-observation log-density evaluates to a non-finite value
    // (e.g. a floating-point overflow in a pair kernel at extreme
    // parameters), the sum below would poison AIC/BIC to ±∞ and the
    // argmin-based selector would deterministically pick this degenerate
    // candidate. We instead score such candidates as the worst possible so
    // they are never selected, while keeping the raw likelihood numerically
    // meaningful (-∞ rather than +∞) for downstream diagnostics.
    let any_non_finite = batch.log_pdf.iter().any(|value| !value.is_finite());
    let loglik = if any_non_finite {
        f64::NEG_INFINITY
    } else {
        batch.log_pdf.iter().sum::<f64>()
    };

    let k = spec.parameter_count() as f64;
    let n = u1.len() as f64;
    let (aic, bic) = if loglik.is_finite() {
        (2.0 * k - 2.0 * loglik, k * n.ln() - 2.0 * loglik)
    } else {
        (f64::INFINITY, f64::INFINITY)
    };
    Ok(PairFitResult {
        spec,
        loglik,
        aic,
        bic,
        cond_on_first: batch.cond_on_first,
        cond_on_second: batch.cond_on_second,
    })
}

fn criterion_value(fit: &PairFitResult, criterion: SelectionCriterion) -> f64 {
    match criterion {
        SelectionCriterion::Aic => fit.aic,
        SelectionCriterion::Bic => fit.bic,
    }
}

fn candidate_rotations(
    family: PairCopulaFamily,
    include_rotations: bool,
    tau: f64,
) -> &'static [Rotation] {
    use PairCopulaFamily as Family;
    use Rotation as Rot;

    match family {
        Family::Independence | Family::Gaussian | Family::StudentT | Family::Frank => &[Rot::R0],
        Family::Clayton | Family::Gumbel | Family::Joe | Family::Khoudraji
            if include_rotations && tau >= 0.0 =>
        {
            &[Rot::R0, Rot::R180]
        }
        Family::Clayton | Family::Gumbel | Family::Joe | Family::Khoudraji if include_rotations => {
            &[Rot::R90, Rot::R270]
        }
        Family::Clayton | Family::Gumbel | Family::Joe | Family::Khoudraji => &[Rot::R0],
    }
}

fn fit_family_with_rotation(
    family: PairCopulaFamily,
    rotation: Rotation,
    u1: &[f64],
    u2: &[f64],
    _tau: f64,
    max_iter: usize,
) -> Result<PairCopulaSpec, CopulaError> {
    if family == PairCopulaFamily::Khoudraji {
        return fit_khoudraji_with_rotation(rotation, u1, u2, _tau, max_iter);
    }

    let transformed = rotated::transform_sample(rotation, u1, u2);
    let x1 = &transformed.0;
    let x2 = &transformed.1;
    let transformed_tau = rotated_tau(rotation, _tau);

    fit_simple_family(family, transformed_tau, x1, x2, max_iter).map(|params| PairCopulaSpec {
        family,
        rotation,
        params,
    })
}

fn fit_simple_family(
    family: PairCopulaFamily,
    tau: f64,
    x1: &[f64],
    x2: &[f64],
    max_iter: usize,
) -> Result<PairCopulaParams, CopulaError> {
    let search_iterations = max_iter.clamp(16, 64);
    let params = match family {
        PairCopulaFamily::Independence => PairCopulaParams::None,
        PairCopulaFamily::Gaussian => {
            let rho = gaussian::tau_to_rho(tau);
            PairCopulaParams::One(rho.clamp(-0.98, 0.98))
        }
        PairCopulaFamily::StudentT => {
            let mut best = None;
            let mut best_loglik = f64::NEG_INFINITY;
            let rho_seed = gaussian::tau_to_rho(tau).clamp(-0.95, 0.95);
            for nu in student_t::candidate_nus() {
                let rho = maximize_scalar(-0.98, 0.98, search_iterations, |rho| {
                    x1.iter()
                        .zip(x2.iter())
                        .map(|(&u, &v)| {
                            student_t::log_pdf(u, v, rho, nu).unwrap_or(f64::NEG_INFINITY)
                        })
                        .sum::<f64>()
                });
                let rho = if rho.is_finite() { rho } else { rho_seed };
                let loglik = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(&u, &v)| student_t::log_pdf(u, v, rho, nu).unwrap_or(f64::NEG_INFINITY))
                    .sum::<f64>();
                if loglik > best_loglik {
                    best_loglik = loglik;
                    best = Some((rho, nu));
                }
            }
            let (rho, nu) = best.ok_or(FitError::Failed {
                reason: "student t pair fit failed",
            })?;
            PairCopulaParams::Two(rho, nu)
        }
        PairCopulaFamily::Clayton => {
            let init = clayton::theta_from_tau(tau)?;
            let upper = (init * 4.0 + 2.0).max(20.0);
            let theta = maximize_scalar(1e-6, upper, search_iterations, |theta| {
                x1.iter()
                    .zip(x2.iter())
                    .map(|(&u, &v)| clayton::log_pdf(u, v, theta).unwrap_or(f64::NEG_INFINITY))
                    .sum::<f64>()
            });
            PairCopulaParams::One(theta)
        }
        PairCopulaFamily::Frank => {
            let init = frank::theta_from_tau(tau)?;
            let upper = (init * 4.0 + 2.0).max(20.0);
            let theta = maximize_scalar(1e-6, upper, search_iterations, |theta| {
                x1.iter()
                    .zip(x2.iter())
                    .map(|(&u, &v)| frank::log_pdf(u, v, theta).unwrap_or(f64::NEG_INFINITY))
                    .sum::<f64>()
            });
            PairCopulaParams::One(theta)
        }
        PairCopulaFamily::Gumbel => {
            let init = gumbel::theta_from_tau(tau)?;
            let upper = (init * 4.0 + 2.0).max(20.0);
            let theta = maximize_scalar(1.0 + 1e-6, upper, search_iterations, |theta| {
                x1.iter()
                    .zip(x2.iter())
                    .map(|(&u, &v)| gumbel::log_pdf(u, v, theta).unwrap_or(f64::NEG_INFINITY))
                    .sum::<f64>()
            });
            PairCopulaParams::One(theta)
        }
        PairCopulaFamily::Joe => {
            let init = joe::theta_from_tau(tau)?;
            let upper = (init * 4.0 + 2.0).max(20.0);
            let theta = maximize_scalar(1.0 + 1e-6, upper, search_iterations, |theta| {
                x1.iter()
                    .zip(x2.iter())
                    .map(|(&u, &v)| joe::log_pdf(u, v, theta).unwrap_or(f64::NEG_INFINITY))
                    .sum::<f64>()
            });
            PairCopulaParams::One(theta)
        }
        PairCopulaFamily::Khoudraji => {
            return Err(FitError::Failed {
                reason: "khoudraji must be fitted via fit_khoudraji_with_rotation",
            }
            .into());
        }
    };
    Ok(params)
}

fn fit_khoudraji_with_rotation(
    rotation: Rotation,
    u1: &[f64],
    u2: &[f64],
    tau: f64,
    max_iter: usize,
) -> Result<PairCopulaSpec, CopulaError> {
    let base_fit_iterations = max_iter.clamp(8, 16);
    let shape_iterations = max_iter.clamp(8, 12);
    let transformed = rotated::transform_sample(rotation, u1, u2);
    let x1 = &transformed.0;
    let x2 = &transformed.1;
    let tau = rotated_tau(rotation, tau);
    let base_families = [
        PairCopulaFamily::Independence,
        PairCopulaFamily::Gaussian,
        PairCopulaFamily::Clayton,
        PairCopulaFamily::Frank,
        PairCopulaFamily::Gumbel,
    ];
    let mut base_specs = Vec::new();
    for family in base_families {
        let params = fit_simple_family(family, tau, x1, x2, base_fit_iterations)?;
        base_specs.push(PairCopulaSpec {
            family,
            rotation: Rotation::R0,
            params,
        });
    }

    let mut best_spec = None;
    let mut best_loglik = f64::NEG_INFINITY;
    for first in &base_specs {
        for second in &base_specs {
            let (shape_first, shape_second, loglik) =
                optimize_khoudraji_shapes(first, second, u1, u2, rotation, shape_iterations)?;
            let spec = PairCopulaSpec {
                family: PairCopulaFamily::Khoudraji,
                rotation,
                params: PairCopulaParams::Khoudraji(KhoudrajiParams::new(
                    first.clone(),
                    second.clone(),
                    shape_first,
                    shape_second,
                )?),
            };
            if loglik > best_loglik {
                best_loglik = loglik;
                best_spec = Some(spec);
            }
        }
    }

    best_spec.ok_or(
        FitError::Failed {
            reason: "khoudraji pair fit failed",
        }
        .into(),
    )
}

fn rotated_tau(rotation: Rotation, tau: f64) -> f64 {
    match rotation {
        Rotation::R0 | Rotation::R180 => tau,
        Rotation::R90 | Rotation::R270 => -tau,
    }
}

fn optimize_khoudraji_shapes(
    first: &PairCopulaSpec,
    second: &PairCopulaSpec,
    u1: &[f64],
    u2: &[f64],
    rotation: Rotation,
    max_iter: usize,
) -> Result<(f64, f64, f64), CopulaError> {
    let seeds = [(0.2, 0.8), (0.5, 0.5)];
    let mut best = None;
    let mut best_loglik = f64::NEG_INFINITY;

    for (mut shape_first, mut shape_second) in seeds {
        for _ in 0..4 {
            shape_first = maximize_scalar(0.0, 1.0, max_iter.max(8), |candidate| {
                khoudraji_loglik(first, second, candidate, shape_second, u1, u2, rotation)
                    .unwrap_or(f64::NEG_INFINITY)
            });
            shape_second = maximize_scalar(0.0, 1.0, max_iter.max(8), |candidate| {
                khoudraji_loglik(first, second, shape_first, candidate, u1, u2, rotation)
                    .unwrap_or(f64::NEG_INFINITY)
            });
        }

        let loglik = khoudraji_loglik(first, second, shape_first, shape_second, u1, u2, rotation)?;
        if loglik > best_loglik {
            best_loglik = loglik;
            best = Some((shape_first, shape_second, loglik));
        }
    }

    best.ok_or(
        FitError::Failed {
            reason: "khoudraji shape optimization failed",
        }
        .into(),
    )
}

fn khoudraji_loglik(
    first: &PairCopulaSpec,
    second: &PairCopulaSpec,
    shape_first: f64,
    shape_second: f64,
    u1: &[f64],
    u2: &[f64],
    rotation: Rotation,
) -> Result<f64, CopulaError> {
    let spec = PairCopulaSpec {
        family: PairCopulaFamily::Khoudraji,
        rotation,
        params: PairCopulaParams::Khoudraji(KhoudrajiParams::new(
            first.clone(),
            second.clone(),
            shape_first,
            shape_second,
        )?),
    };
    pair_loglik(&spec, u1, u2, 1e-12)
}

fn pair_loglik(
    spec: &PairCopulaSpec,
    u1: &[f64],
    u2: &[f64],
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let mut total = 0.0;
    for (&left, &right) in u1.iter().zip(u2.iter()) {
        let value = spec.log_pdf(left, right, clip_eps)?;
        // Any non-finite per-observation log-density indicates a parameter
        // combination where the pair kernel overflows; treating the whole
        // fit as having -∞ log-likelihood prevents inner optimizers (e.g.
        // the Khoudraji shape search) from latching onto pathological
        // extrema that would later poison AIC/BIC selection.
        if !value.is_finite() {
            return Ok(f64::NEG_INFINITY);
        }
        total += value;
    }
    Ok(total)
}

fn evaluate_pair_batch(
    spec: PairCopulaSpec,
    u1: &[f64],
    u2: &[f64],
    clip_eps: f64,
    exec: crate::domain::ExecPolicy,
) -> Result<PairBatchEvaluation, CopulaError> {
    let mut log_pdf = vec![0.0; u1.len()];
    let mut cond_on_first = vec![0.0; u1.len()];
    let mut cond_on_second = vec![0.0; u1.len()];
    let mut outputs = PairBatchBuffers {
        log_pdf: &mut log_pdf,
        cond_on_first: &mut cond_on_first,
        cond_on_second: &mut cond_on_second,
    };
    evaluate_pair_batch_into(&spec, u1, u2, clip_eps, exec, &mut outputs)?;

    Ok(PairBatchEvaluation {
        log_pdf,
        cond_on_first,
        cond_on_second,
    })
}

pub(crate) fn evaluate_pair_batch_into(
    spec: &PairCopulaSpec,
    u1: &[f64],
    u2: &[f64],
    clip_eps: f64,
    exec: ExecPolicy,
    outputs: &mut PairBatchBuffers<'_>,
) -> Result<(), CopulaError> {
    let output_lengths = [
        outputs.log_pdf.len(),
        outputs.cond_on_first.len(),
        outputs.cond_on_second.len(),
    ];
    validate_batch_buffers(u1, u2, &output_lengths)?;
    let strategy = resolve_strategy(exec, Operation::PairBatchEval, u1.len())?;
    match strategy {
        ExecutionStrategy::CpuSerial | ExecutionStrategy::CpuParallel => {
            fill_pair_batch_cpu(spec, u1, u2, clip_eps, strategy, outputs)
        }
        ExecutionStrategy::Cuda(ordinal) => match gaussian_pair_request(spec, u1, u2, clip_eps) {
            Some(request) => {
                let batch = crate::accel::evaluate_gaussian_pair_batch(
                    crate::accel::Device::Cuda(ordinal),
                    request,
                )
                .map_err(|err| BackendError::Failed {
                    backend: "cuda",
                    reason: err.to_string(),
                })?;
                copy_gaussian_batch(batch, outputs);
                Ok(())
            }
            None => fill_pair_batch_cpu(
                spec,
                u1,
                u2,
                clip_eps,
                ExecutionStrategy::CpuParallel,
                outputs,
            ),
        },
        ExecutionStrategy::Metal => match gaussian_pair_request(spec, u1, u2, clip_eps) {
            Some(request) => {
                let batch = crate::accel::evaluate_gaussian_pair_batch(
                    crate::accel::Device::Metal,
                    request,
                )
                .map_err(|err| BackendError::Failed {
                    backend: "metal",
                    reason: err.to_string(),
                })?;
                copy_gaussian_batch(batch, outputs);
                Ok(())
            }
            None => fill_pair_batch_cpu(
                spec,
                u1,
                u2,
                clip_eps,
                ExecutionStrategy::CpuParallel,
                outputs,
            ),
        },
    }
}

pub(crate) fn inverse_second_given_first_batch_into(
    spec: &PairCopulaSpec,
    u1: &[f64],
    u2: &[f64],
    clip_eps: f64,
    strategy: ExecutionStrategy,
    out: &mut [f64],
) -> Result<(), CopulaError> {
    validate_batch_buffers(u1, u2, &[out.len()])?;
    fill_unary_pair_batch(spec, u1, u2, strategy, out, |spec, left, right| {
        spec.inv_second_given_first(left, right, clip_eps)
    })
}

pub(crate) fn cond_first_given_second_batch_into(
    spec: &PairCopulaSpec,
    u1: &[f64],
    u2: &[f64],
    clip_eps: f64,
    strategy: ExecutionStrategy,
    out: &mut [f64],
) -> Result<(), CopulaError> {
    validate_batch_buffers(u1, u2, &[out.len()])?;
    fill_unary_pair_batch(spec, u1, u2, strategy, out, |spec, left, right| {
        spec.cond_first_given_second(left, right, clip_eps)
    })
}

pub(crate) fn cond_second_given_first_batch_into(
    spec: &PairCopulaSpec,
    u1: &[f64],
    u2: &[f64],
    clip_eps: f64,
    strategy: ExecutionStrategy,
    out: &mut [f64],
) -> Result<(), CopulaError> {
    validate_batch_buffers(u1, u2, &[out.len()])?;
    fill_unary_pair_batch(spec, u1, u2, strategy, out, |spec, left, right| {
        spec.cond_second_given_first(left, right, clip_eps)
    })
}

fn gaussian_pair_request<'a>(
    spec: &PairCopulaSpec,
    u1: &'a [f64],
    u2: &'a [f64],
    clip_eps: f64,
) -> Option<crate::accel::GaussianPairBatchRequest<'a>> {
    match (spec.family, spec.rotation, &spec.params) {
        (PairCopulaFamily::Gaussian, Rotation::R0, PairCopulaParams::One(rho)) => {
            Some(crate::accel::GaussianPairBatchRequest {
                u1,
                u2,
                rho: *rho,
                clip_eps,
            })
        }
        _ => None,
    }
}

fn fill_pair_batch_cpu(
    spec: &PairCopulaSpec,
    u1: &[f64],
    u2: &[f64],
    clip_eps: f64,
    strategy: ExecutionStrategy,
    outputs: &mut PairBatchBuffers<'_>,
) -> Result<(), CopulaError> {
    match strategy {
        ExecutionStrategy::CpuSerial => {
            for idx in 0..u1.len() {
                outputs.log_pdf[idx] = spec.log_pdf(u1[idx], u2[idx], clip_eps)?;
                outputs.cond_on_first[idx] =
                    spec.cond_first_given_second(u1[idx], u2[idx], clip_eps)?;
                outputs.cond_on_second[idx] =
                    spec.cond_second_given_first(u1[idx], u2[idx], clip_eps)?;
            }
            Ok(())
        }
        ExecutionStrategy::CpuParallel => {
            let rows = parallel_try_map_range_collect(u1.len(), strategy, |idx| {
                Ok((
                    spec.log_pdf(u1[idx], u2[idx], clip_eps)?,
                    spec.cond_first_given_second(u1[idx], u2[idx], clip_eps)?,
                    spec.cond_second_given_first(u1[idx], u2[idx], clip_eps)?,
                ))
            })?;
            for (idx, (log_value, cond_first, cond_second)) in rows.into_iter().enumerate() {
                outputs.log_pdf[idx] = log_value;
                outputs.cond_on_first[idx] = cond_first;
                outputs.cond_on_second[idx] = cond_second;
            }
            Ok(())
        }
        ExecutionStrategy::Cuda(_) | ExecutionStrategy::Metal => {
            unreachable!("GPU pair batches must be handled before CPU filling")
        }
    }
}

fn fill_unary_pair_batch<F>(
    spec: &PairCopulaSpec,
    u1: &[f64],
    u2: &[f64],
    strategy: ExecutionStrategy,
    out: &mut [f64],
    f: F,
) -> Result<(), CopulaError>
where
    F: Fn(&PairCopulaSpec, f64, f64) -> Result<f64, CopulaError> + Sync + Send,
{
    match strategy {
        ExecutionStrategy::CpuSerial => {
            for idx in 0..u1.len() {
                out[idx] = f(spec, u1[idx], u2[idx])?;
            }
            Ok(())
        }
        ExecutionStrategy::CpuParallel => {
            let values = parallel_try_map_range_collect(u1.len(), strategy, |idx| {
                f(spec, u1[idx], u2[idx])
            })?;
            out.copy_from_slice(&values);
            Ok(())
        }
        ExecutionStrategy::Cuda(_) | ExecutionStrategy::Metal => {
            unreachable!("sampling helpers currently support CPU strategies only")
        }
    }
}

fn validate_batch_buffers(
    u1: &[f64],
    u2: &[f64],
    output_lens: &[usize],
) -> Result<(), CopulaError> {
    if u1.len() != u2.len() || output_lens.iter().any(|&len| len != u1.len()) {
        return Err(FitError::Failed {
            reason: "pair batch helpers require equally sized input and output buffers",
        }
        .into());
    }
    Ok(())
}

fn copy_gaussian_batch(
    batch: crate::accel::GaussianPairBatchResult,
    outputs: &mut PairBatchBuffers<'_>,
) {
    outputs.log_pdf.copy_from_slice(&batch.log_pdf);
    outputs.cond_on_first.copy_from_slice(&batch.cond_on_first);
    outputs
        .cond_on_second
        .copy_from_slice(&batch.cond_on_second);
}
