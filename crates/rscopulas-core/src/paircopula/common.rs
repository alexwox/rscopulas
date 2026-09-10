use serde::{Deserialize, Serialize};

use statrs::distribution::{ContinuousCDF, Normal};

use crate::{
    backend::{ExecutionStrategy, Operation, parallel_try_map_range_collect, resolve_strategy},
    domain::ExecPolicy,
    errors::{BackendError, CopulaError, FitError},
    math::{NelderMeadOptions, maximize_scalar_brent, nelder_mead_maximize},
    vine::{SelectionCriterion, VineFitOptions},
};

pub use super::tll::{TllOrder, TllParams};
use super::{
    bb1, bb6, bb7, bb8, clayton, frank, gaussian, gumbel, joe, khoudraji,
    polish::{decode_fit_params, encode_fit_params, fit_brackets, inv_logit, logit},
    rotated, student_t, tawn, tll,
};

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
    Bb1,
    Bb6,
    Bb7,
    Bb8,
    Tawn1,
    Tawn2,
    Tll,
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
        first.validate()?;
        second.validate()?;
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
    Tll(TllParams),
}

impl PairCopulaParams {
    pub fn flat_values(&self) -> Vec<f64> {
        match self {
            Self::None => Vec::new(),
            Self::One(value) => vec![*value],
            Self::Two(first, second) => vec![*first, *second],
            Self::Khoudraji(params) => params.flat_values(),
            // Tll's parameter state is a grid, not scalar values — we report
            // the stored effective degrees of freedom so BIC-style scoring
            // still sees a meaningful number of parameters. Serde round-trips
            // the full TllParams payload independently.
            Self::Tll(params) => vec![params.effective_df],
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
    /// Factor quadrature carries both tails in log form, avoiding loss of
    /// relative precision when a latent probability rounds close to one.
    pub(crate) fn log_pdf_from_log_probabilities(
        &self,
        mut log_u: [f64; 2],
        log_survival: [f64; 2],
    ) -> Option<f64> {
        let mut log_s = log_survival;
        if matches!(self.rotation, Rotation::R90 | Rotation::R180) {
            std::mem::swap(&mut log_u[0], &mut log_s[0]);
        }
        if matches!(self.rotation, Rotation::R270 | Rotation::R180) {
            std::mem::swap(&mut log_u[1], &mut log_s[1]);
        }
        match (&self.family, &self.params) {
            (PairCopulaFamily::Clayton, PairCopulaParams::One(theta)) => {
                Some(clayton::log_pdf_from_logs(log_u[0], log_u[1], *theta))
            }
            (PairCopulaFamily::Bb1, PairCopulaParams::Two(theta, delta)) => {
                Some(bb1::log_pdf_from_logs(log_u[0], log_u[1], *theta, *delta))
            }
            (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => {
                Some(gumbel::log_pdf_from_logs(log_u[0], log_u[1], *theta))
            }
            (PairCopulaFamily::Joe, PairCopulaParams::One(theta)) => {
                Some(joe::log_pdf_from_log_survival(log_s[0], log_s[1], *theta))
            }
            _ => None,
        }
    }
    /// Validates the family and parameter domain before constructing a model.
    pub fn validate(&self) -> Result<(), CopulaError> {
        use PairCopulaFamily as F;
        use PairCopulaParams as P;
        let valid = match (&self.family, &self.params) {
            (F::Independence, P::None) => true,
            (F::Gaussian, P::One(rho)) => rho.is_finite() && rho.abs() < 1.0,
            (F::StudentT, P::Two(rho, nu)) => {
                rho.is_finite() && rho.abs() < 1.0 && nu.is_finite() && *nu > 0.0
            }
            (F::Clayton, P::One(theta)) => theta.is_finite() && *theta > 0.0,
            // Frank covers the whole real line; θ = 0 is independence and has
            // its own family.
            (F::Frank, P::One(theta)) => theta.is_finite() && *theta != 0.0,
            (F::Gumbel | F::Joe, P::One(theta)) => theta.is_finite() && *theta >= 1.0,
            (F::Bb1, P::Two(theta, delta)) => {
                theta.is_finite() && *theta > 0.0 && delta.is_finite() && *delta >= 1.0
            }
            (F::Bb6, P::Two(theta, delta)) => {
                theta.is_finite() && *theta >= 1.0 && delta.is_finite() && *delta >= 1.0
            }
            (F::Bb7, P::Two(theta, delta)) => {
                theta.is_finite() && *theta >= 1.0 && delta.is_finite() && *delta > 0.0
            }
            (F::Bb8, P::Two(theta, delta)) => {
                theta.is_finite()
                    && *theta >= 1.0
                    && delta.is_finite()
                    && *delta > 0.0
                    && *delta <= 1.0
            }
            (F::Tawn1 | F::Tawn2, P::Two(theta, shape)) => {
                theta.is_finite()
                    && *theta >= 1.0
                    && shape.is_finite()
                    && (0.0..=1.0).contains(shape)
            }
            (F::Khoudraji, P::Khoudraji(p)) => {
                p.first.validate()?;
                p.second.validate()?;
                p.first.family != F::Khoudraji
                    && p.second.family != F::Khoudraji
                    && (0.0..=1.0).contains(&p.shape_first)
                    && (0.0..=1.0).contains(&p.shape_second)
            }
            (F::Tll, P::Tll(p)) => {
                p.grid_min.is_finite()
                    && p.grid_max.is_finite()
                    && p.grid_min < p.grid_max
                    && p.log_density.dim() == (tll::GRID_SIZE, tll::GRID_SIZE)
                    && p.bandwidth.is_finite()
                    && p.bandwidth > 0.0
                    && p.effective_df.is_finite()
                    && p.effective_df >= 0.0
            }
            _ => false,
        };
        if !valid {
            return Err(FitError::Failed {
                reason: "pair-copula family or parameters are invalid",
            }
            .into());
        }
        Ok(())
    }

    fn validate_evaluation(
        &self,
        first: f64,
        second: f64,
        clip_eps: f64,
    ) -> Result<(), CopulaError> {
        crate::data::validate_clip_eps(clip_eps)?;
        crate::data::validate_probability(first)?;
        crate::data::validate_probability(second)?;
        self.validate()
    }
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
    pub fn swap_axes(mut self) -> Self {
        let rotation = match self.rotation {
            Rotation::R90 => Rotation::R270,
            Rotation::R270 => Rotation::R90,
            other => other,
        };
        self.rotation = rotation;
        self.family = match self.family {
            PairCopulaFamily::Tawn1 => PairCopulaFamily::Tawn2,
            PairCopulaFamily::Tawn2 => PairCopulaFamily::Tawn1,
            other => other,
        };
        self.params = match self.params {
            PairCopulaParams::Khoudraji(p) => PairCopulaParams::Khoudraji(KhoudrajiParams {
                first: Box::new(p.first.swap_axes()),
                second: Box::new(p.second.swap_axes()),
                shape_first: p.shape_second,
                shape_second: p.shape_first,
            }),
            PairCopulaParams::Tll(mut p) => {
                p.transpose();
                PairCopulaParams::Tll(p)
            }
            other => other,
        };
        self
    }

    /// Returns the number of free parameters implied by `params`.
    pub fn parameter_count(&self) -> usize {
        match &self.params {
            PairCopulaParams::None => 0,
            PairCopulaParams::One(_) => 1,
            PairCopulaParams::Two(_, _) => 2,
            PairCopulaParams::Khoudraji(params) => params.parameter_count(),
            // Nonparametric fit — report the effective degrees of freedom
            // estimated at fit time, rounded up to the nearest integer.
            PairCopulaParams::Tll(params) => params.effective_df.ceil().max(1.0) as usize,
        }
    }

    /// Returns a flattened numeric representation of the free parameters.
    pub fn flat_parameters(&self) -> Vec<f64> {
        self.params.flat_values()
    }

    /// Evaluates the pair-copula CDF at `(u1, u2)`.
    pub(crate) fn cdf(&self, u1: f64, u2: f64, clip_eps: f64) -> Result<f64, CopulaError> {
        self.validate_evaluation(u1, u2, clip_eps)?;
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
        self.validate_evaluation(u1, u2, clip_eps)?;
        self.log_pdf_validated(u1, u2, clip_eps)
    }

    /// [`log_pdf`](Self::log_pdf) without the per-call input and parameter
    /// validation. The pair-fit optimisers validate the sample once and the
    /// candidate spec per objective evaluation, so they skip the per-row
    /// checks here.
    fn log_pdf_validated(&self, u1: f64, u2: f64, clip_eps: f64) -> Result<f64, CopulaError> {
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
            (PairCopulaFamily::Joe, PairCopulaParams::One(theta)) => joe::log_pdf(x1, x2, *theta)?,
            (PairCopulaFamily::Bb1, PairCopulaParams::Two(theta, delta)) => {
                bb1::log_pdf(x1, x2, *theta, *delta)?
            }
            (PairCopulaFamily::Bb6, PairCopulaParams::Two(theta, delta)) => {
                bb6::log_pdf(x1, x2, *theta, *delta)?
            }
            (PairCopulaFamily::Bb7, PairCopulaParams::Two(theta, delta)) => {
                bb7::log_pdf(x1, x2, *theta, *delta)?
            }
            (PairCopulaFamily::Bb8, PairCopulaParams::Two(theta, delta)) => {
                bb8::log_pdf(x1, x2, *theta, *delta)?
            }
            (PairCopulaFamily::Tawn1, PairCopulaParams::Two(theta, alpha)) => {
                tawn::log_pdf(x1, x2, *theta, *alpha, 1.0)?
            }
            (PairCopulaFamily::Tawn2, PairCopulaParams::Two(theta, beta)) => {
                tawn::log_pdf(x1, x2, *theta, 1.0, *beta)?
            }
            (PairCopulaFamily::Tll, PairCopulaParams::Tll(params)) => tll::log_pdf(x1, x2, params)?,
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
        // A zero density is a valid row result. NaN and +infinity indicate a
        // numerical failure and must not be silently converted into zero mass.
        if base.is_nan() || base == f64::INFINITY {
            return Err(crate::errors::NumericalError::Failed {
                reason: "pair density is not finite",
            }
            .into());
        }
        Ok(rotated::from_base_log_pdf(rotation, base))
    }

    /// Evaluates `h_{1|2}(u1 | u2)`.
    pub fn cond_first_given_second(
        &self,
        u1: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        self.validate_evaluation(u1, u2, clip_eps)?;
        match self.rotation {
            Rotation::R0 => self.base_cond_first_given_second(u1, u2, clip_eps),
            Rotation::R180 => {
                Ok(1.0 - self.base_cond_first_given_second(1.0 - u1, 1.0 - u2, clip_eps)?)
            }
            Rotation::R90 => Ok(1.0 - self.base_cond_first_given_second(1.0 - u1, u2, clip_eps)?),
            Rotation::R270 => self.base_cond_first_given_second(u1, 1.0 - u2, clip_eps),
        }
        .and_then(|value| {
            if !value.is_finite() {
                return Err(crate::errors::NumericalError::Failed {
                    reason: "pair conditional is not finite",
                }
                .into());
            }
            Ok(value.clamp(clip_eps, 1.0 - clip_eps))
        })
    }

    /// Evaluates `h_{2|1}(u2 | u1)`.
    pub fn cond_second_given_first(
        &self,
        u1: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        self.validate_evaluation(u1, u2, clip_eps)?;
        match self.rotation {
            Rotation::R0 => self.base_cond_second_given_first(u1, u2, clip_eps),
            Rotation::R180 => {
                Ok(1.0 - self.base_cond_second_given_first(1.0 - u1, 1.0 - u2, clip_eps)?)
            }
            Rotation::R90 => self.base_cond_second_given_first(1.0 - u1, u2, clip_eps),
            Rotation::R270 => Ok(1.0 - self.base_cond_second_given_first(u1, 1.0 - u2, clip_eps)?),
        }
        .and_then(|value| {
            if !value.is_finite() {
                return Err(crate::errors::NumericalError::Failed {
                    reason: "pair conditional is not finite",
                }
                .into());
            }
            Ok(value.clamp(clip_eps, 1.0 - clip_eps))
        })
    }

    /// Evaluates the inverse h-function for the first margin conditional on the second.
    pub fn inv_first_given_second(
        &self,
        p: f64,
        u2: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        self.validate_evaluation(p, u2, clip_eps)?;
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
        .and_then(|value| {
            if !value.is_finite() {
                return Err(crate::errors::NumericalError::Failed {
                    reason: "pair conditional is not finite",
                }
                .into());
            }
            Ok(value.clamp(clip_eps, 1.0 - clip_eps))
        })
    }

    /// Evaluates the inverse h-function for the second margin conditional on the first.
    pub fn inv_second_given_first(
        &self,
        u1: f64,
        p: f64,
        clip_eps: f64,
    ) -> Result<f64, CopulaError> {
        self.validate_evaluation(u1, p, clip_eps)?;
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
        .and_then(|value| {
            if !value.is_finite() {
                return Err(crate::errors::NumericalError::Failed {
                    reason: "pair conditional is not finite",
                }
                .into());
            }
            Ok(value.clamp(clip_eps, 1.0 - clip_eps))
        })
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
            (PairCopulaFamily::Bb1, PairCopulaParams::Two(theta, delta)) => {
                bb1::cond_first_given_second(u1, u2, *theta, *delta)
            }
            (PairCopulaFamily::Bb6, PairCopulaParams::Two(theta, delta)) => {
                bb6::cond_first_given_second(u1, u2, *theta, *delta)
            }
            (PairCopulaFamily::Bb7, PairCopulaParams::Two(theta, delta)) => {
                bb7::cond_first_given_second(u1, u2, *theta, *delta)
            }
            (PairCopulaFamily::Bb8, PairCopulaParams::Two(theta, delta)) => {
                bb8::cond_first_given_second(u1, u2, *theta, *delta)
            }
            (PairCopulaFamily::Tawn1, PairCopulaParams::Two(theta, alpha)) => {
                tawn::cond_first_given_second(u1, u2, *theta, *alpha, 1.0)
            }
            (PairCopulaFamily::Tawn2, PairCopulaParams::Two(theta, beta)) => {
                tawn::cond_first_given_second(u1, u2, *theta, 1.0, *beta)
            }
            (PairCopulaFamily::Tll, PairCopulaParams::Tll(params)) => {
                tll::cond_first_given_second(u1, u2, params)
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
            (PairCopulaFamily::Bb1, PairCopulaParams::Two(theta, delta)) => {
                bb1::cond_second_given_first(u1, u2, *theta, *delta)
            }
            (PairCopulaFamily::Bb6, PairCopulaParams::Two(theta, delta)) => {
                bb6::cond_second_given_first(u1, u2, *theta, *delta)
            }
            (PairCopulaFamily::Bb7, PairCopulaParams::Two(theta, delta)) => {
                bb7::cond_second_given_first(u1, u2, *theta, *delta)
            }
            (PairCopulaFamily::Bb8, PairCopulaParams::Two(theta, delta)) => {
                bb8::cond_second_given_first(u1, u2, *theta, *delta)
            }
            (PairCopulaFamily::Tawn1, PairCopulaParams::Two(theta, alpha)) => {
                tawn::cond_second_given_first(u1, u2, *theta, *alpha, 1.0)
            }
            (PairCopulaFamily::Tawn2, PairCopulaParams::Two(theta, beta)) => {
                tawn::cond_second_given_first(u1, u2, *theta, 1.0, *beta)
            }
            (PairCopulaFamily::Tll, PairCopulaParams::Tll(params)) => {
                tll::cond_second_given_first(u1, u2, params)
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
            (PairCopulaFamily::Bb1, PairCopulaParams::Two(theta, delta)) => {
                bb1::inv_first_given_second(p, u2, *theta, *delta, clip_eps)
            }
            (PairCopulaFamily::Bb6, PairCopulaParams::Two(theta, delta)) => {
                bb6::inv_first_given_second(p, u2, *theta, *delta, clip_eps)
            }
            (PairCopulaFamily::Bb7, PairCopulaParams::Two(theta, delta)) => {
                bb7::inv_first_given_second(p, u2, *theta, *delta, clip_eps)
            }
            (PairCopulaFamily::Bb8, PairCopulaParams::Two(theta, delta)) => {
                bb8::inv_first_given_second(p, u2, *theta, *delta, clip_eps)
            }
            (PairCopulaFamily::Tawn1, PairCopulaParams::Two(theta, alpha)) => {
                tawn::inv_first_given_second(p, u2, *theta, *alpha, 1.0, clip_eps)
            }
            (PairCopulaFamily::Tawn2, PairCopulaParams::Two(theta, beta)) => {
                tawn::inv_first_given_second(p, u2, *theta, 1.0, *beta, clip_eps)
            }
            (PairCopulaFamily::Tll, PairCopulaParams::Tll(params)) => {
                tll::inv_first_given_second(p, u2, params, clip_eps)
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
            (PairCopulaFamily::Bb1, PairCopulaParams::Two(theta, delta)) => {
                bb1::inv_second_given_first(u1, p, *theta, *delta, clip_eps)
            }
            (PairCopulaFamily::Bb6, PairCopulaParams::Two(theta, delta)) => {
                bb6::inv_second_given_first(u1, p, *theta, *delta, clip_eps)
            }
            (PairCopulaFamily::Bb7, PairCopulaParams::Two(theta, delta)) => {
                bb7::inv_second_given_first(u1, p, *theta, *delta, clip_eps)
            }
            (PairCopulaFamily::Bb8, PairCopulaParams::Two(theta, delta)) => {
                bb8::inv_second_given_first(u1, p, *theta, *delta, clip_eps)
            }
            (PairCopulaFamily::Tawn1, PairCopulaParams::Two(theta, alpha)) => {
                tawn::inv_second_given_first(u1, p, *theta, *alpha, 1.0, clip_eps)
            }
            (PairCopulaFamily::Tawn2, PairCopulaParams::Two(theta, beta)) => {
                tawn::inv_second_given_first(u1, p, *theta, 1.0, *beta, clip_eps)
            }
            (PairCopulaFamily::Tll, PairCopulaParams::Tll(params)) => {
                tll::inv_second_given_first(u1, p, params, clip_eps)
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
                clayton::cdf(u1, u2, *theta)
            }
            (PairCopulaFamily::Frank, PairCopulaParams::One(theta)) => frank::cdf(u1, u2, *theta),
            (PairCopulaFamily::Gumbel, PairCopulaParams::One(theta)) => gumbel::cdf(u1, u2, *theta),
            (PairCopulaFamily::Joe, PairCopulaParams::One(theta)) => joe::cdf(u1, u2, *theta),
            (PairCopulaFamily::Bb1, PairCopulaParams::Two(theta, delta)) => {
                bb1::cdf(u1, u2, *theta, *delta)
            }
            (PairCopulaFamily::Bb6, PairCopulaParams::Two(theta, delta)) => {
                bb6::cdf(u1, u2, *theta, *delta)
            }
            (PairCopulaFamily::Bb7, PairCopulaParams::Two(theta, delta)) => {
                bb7::cdf(u1, u2, *theta, *delta)
            }
            (PairCopulaFamily::Bb8, PairCopulaParams::Two(theta, delta)) => {
                bb8::cdf(u1, u2, *theta, *delta)
            }
            (PairCopulaFamily::Tawn1, PairCopulaParams::Two(theta, alpha)) => {
                tawn::cdf(u1, u2, *theta, *alpha, 1.0)
            }
            (PairCopulaFamily::Tawn2, PairCopulaParams::Two(theta, beta)) => {
                tawn::cdf(u1, u2, *theta, 1.0, *beta)
            }
            (PairCopulaFamily::Tll, PairCopulaParams::Tll(params)) => tll::cdf(u1, u2, params),
            (PairCopulaFamily::Gaussian, PairCopulaParams::One(rho)) => gaussian::cdf(u1, u2, *rho),
            (PairCopulaFamily::StudentT, PairCopulaParams::Two(rho, nu)) => {
                student_t::cdf(u1, u2, *rho, *nu)
            }
            (PairCopulaFamily::Khoudraji, PairCopulaParams::Khoudraji(params)) => {
                khoudraji::cdf(u1, u2, params, clip_eps)
            }
            _ => Err(FitError::Failed {
                reason: "pair-copula family/parameter combination is invalid",
            }
            .into()),
        }
    }
}

pub(super) fn integrate_1d<F: Fn(f64) -> Result<f64, CopulaError>>(
    f: &F,
    lower: f64,
    upper: f64,
    tol: f64,
    depth: usize,
) -> Result<f64, CopulaError> {
    use std::sync::OnceLock;
    type Rule = (Vec<f64>, Vec<f64>);
    static RULES: OnceLock<(Rule, Rule)> = OnceLock::new();
    let (small, large) = RULES.get_or_init(|| {
        (
            crate::math::gauss_legendre_01(8),
            crate::math::gauss_legendre_01(16),
        )
    });
    let integrate = |rule: &Rule| -> Result<f64, CopulaError> {
        let mut total = 0.0;
        for (&x, &w) in rule.0.iter().zip(&rule.1) {
            total += w * f(lower + (upper - lower) * x)?;
        }
        Ok((upper - lower) * total)
    };
    let coarse = integrate(small)?;
    let fine = integrate(large)?;
    if (fine - coarse).abs() <= tol + 1e-11 * fine.abs() {
        return Ok(fine);
    }
    if depth == 0 {
        return Err(crate::errors::NumericalError::Failed {
            reason: "pair CDF quadrature did not converge",
        }
        .into());
    }
    let mid = (lower + upper) * 0.5;
    Ok(integrate_1d(f, lower, mid, tol * 0.5, depth - 1)?
        + integrate_1d(f, mid, upper, tol * 0.5, depth - 1)?)
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

    options.base.validate()?;
    let tau = crate::stats::kendall_tau_bivariate(u1, u2)?;
    if let Some(threshold) = options.independence_threshold {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(FitError::Failed {
                reason: "invalid independence threshold",
            }
            .into());
        }
        let dependence = match options.tree_criterion {
            crate::vine::TreeCriterion::Tau => tau,
            crate::vine::TreeCriterion::Rho => crate::stats::spearman_rho_bivariate(u1, u2)?,
            crate::vine::TreeCriterion::Hoeffding => crate::stats::hoeffding_d_bivariate(u1, u2)?,
        };
        if dependence.abs() <= threshold {
            return finalize_pair_fit(PairCopulaSpec::independence(), u1, u2, options);
        }
    }
    if let Some(alpha) = options.independence_test_level {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(FitError::Failed {
                reason: "invalid independence test level",
            }
            .into());
        }
        if !crate::stats::kendall_tau_rejects_independence(tau, u1.len(), alpha) {
            return finalize_pair_fit(PairCopulaSpec::independence(), u1, u2, options);
        }
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
                options.base.clip_eps,
                options.base.max_iter,
                options.criterion,
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
        match finalize_pair_fit(candidates[idx].clone(), u1, u2, options) {
            Ok(fit) if fit.loglik.is_finite() => Ok(Some(fit)),
            // A numerically unusable family must not prevent another valid
            // candidate from winning. Input and backend failures still surface.
            Ok(_) | Err(CopulaError::Numerical(_)) => Ok(None),
            Err(error) => Err(error),
        }
    })?;
    let best = fits.into_iter().flatten().min_by(|left, right| {
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
        // mBICV is a *tree-level* penalty, not defined per edge. Fall back
        // to BIC for the pair-copula family comparison, matching vinecopulib's
        // behaviour (`tools_select.ipp` uses BIC inside `select_bicop`).
        SelectionCriterion::Bic | SelectionCriterion::Mbicv { .. } => fit.bic,
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
        Family::Independence
        | Family::Gaussian
        | Family::StudentT
        | Family::Frank
        | Family::Tll => &[Rot::R0],
        Family::Clayton
        | Family::Gumbel
        | Family::Joe
        | Family::Bb1
        | Family::Bb6
        | Family::Bb7
        | Family::Bb8
        | Family::Tawn1
        | Family::Tawn2
        | Family::Khoudraji
            if include_rotations && tau >= 0.0 =>
        {
            &[Rot::R0, Rot::R180]
        }
        Family::Clayton
        | Family::Gumbel
        | Family::Joe
        | Family::Bb1
        | Family::Bb6
        | Family::Bb7
        | Family::Bb8
        | Family::Tawn1
        | Family::Tawn2
        | Family::Khoudraji
            if include_rotations =>
        {
            &[Rot::R90, Rot::R270]
        }
        Family::Clayton
        | Family::Gumbel
        | Family::Joe
        | Family::Bb1
        | Family::Bb6
        | Family::Bb7
        | Family::Bb8
        | Family::Tawn1
        | Family::Tawn2
        | Family::Khoudraji => &[Rot::R0],
    }
}

/// Absolute tolerance (in natural parameter units) for the scalar searches
/// that finish a one-parameter fit.
const SCALAR_TOL: f64 = 1e-8;
/// Looser tolerance for the inner searches of a two-parameter grid warm
/// start. The joint polish that follows refines the result, so the warm
/// start only needs to land in the right basin.
const WARM_START_TOL: f64 = 1e-4;
/// Largest |ρ| the Gaussian MLE may return. Keeps `1 − ρ²` away from zero
/// while still representing near-comonotone data.
const GAUSSIAN_RHO_MAX: f64 = 1.0 - 1e-9;
/// |ρ| bracket for the Student-t grid warm start; the joint polish may move
/// closer to ±1 afterwards.
const STUDENT_T_RHO_MAX: f64 = 0.9999;
/// Below this |τ| Frank's τ inversion is numerically meaningless, so the fit
/// starts from a small positive θ instead.
const FRANK_TAU_FLOOR: f64 = 1e-10;

#[allow(clippy::too_many_arguments)]
fn fit_family_with_rotation(
    family: PairCopulaFamily,
    rotation: Rotation,
    u1: &[f64],
    u2: &[f64],
    tau: f64,
    clip_eps: f64,
    max_iter: usize,
    criterion: SelectionCriterion,
) -> Result<PairCopulaSpec, CopulaError> {
    if family == PairCopulaFamily::Khoudraji {
        return fit_khoudraji_with_rotation(rotation, u1, u2, tau, clip_eps, max_iter, criterion);
    }

    let (x1, x2) = clipped_transform(rotation, u1, u2, clip_eps);
    let transformed_tau = rotated_tau(rotation, tau);

    fit_simple_family(family, transformed_tau, &x1, &x2, clip_eps, max_iter).map(|params| {
        PairCopulaSpec {
            family,
            rotation,
            params,
        }
    })
}

/// Rotates the sample into the base orientation and clips it exactly as the
/// scoring pass in `finalize_pair_fit` does, so the optimisers and the
/// reported log-likelihood evaluate the same objective.
fn clipped_transform(
    rotation: Rotation,
    u1: &[f64],
    u2: &[f64],
    clip_eps: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (mut x1, mut x2) = rotated::transform_sample(rotation, u1, u2);
    for value in x1.iter_mut().chain(x2.iter_mut()) {
        *value = value.clamp(clip_eps, 1.0 - clip_eps);
    }
    (x1, x2)
}

/// Log-likelihood of a rotation-free candidate on an already rotated and
/// clipped sample. Any non-finite row makes the whole objective `-inf`,
/// mirroring how `finalize_pair_fit` scores candidates, so the optimisers
/// never trade a valid fit for a numerically broken one.
struct PairObjective<'a> {
    family: PairCopulaFamily,
    x1: &'a [f64],
    x2: &'a [f64],
    clip_eps: f64,
}

impl PairObjective<'_> {
    fn loglik(&self, params: PairCopulaParams) -> f64 {
        let spec = PairCopulaSpec {
            family: self.family,
            rotation: Rotation::R0,
            params,
        };
        if spec.validate().is_err() {
            return f64::NEG_INFINITY;
        }
        let mut total = 0.0;
        for (&u, &v) in self.x1.iter().zip(self.x2) {
            match spec.log_pdf_validated(u, v, self.clip_eps) {
                Ok(value) if value.is_finite() => total += value,
                _ => return f64::NEG_INFINITY,
            }
        }
        total
    }
}

fn fit_simple_family(
    family: PairCopulaFamily,
    tau: f64,
    x1: &[f64],
    x2: &[f64],
    clip_eps: f64,
    max_iter: usize,
) -> Result<PairCopulaParams, CopulaError> {
    let objective = PairObjective {
        family,
        x1,
        x2,
        clip_eps,
    };
    let params = match family {
        PairCopulaFamily::Independence => PairCopulaParams::None,
        PairCopulaFamily::Gaussian => fit_gaussian_mle(tau, x1, x2, max_iter),
        PairCopulaFamily::StudentT => {
            // Grid warm start over ν. The t-quantiles depend on ν only, so
            // each grid point inverts the CDF once and runs a cheap scalar
            // search over ρ on the cached quantiles. The joint (ρ, ν) polish
            // then makes ν continuous.
            let rho_seed = gaussian::tau_to_rho(tau).clamp(-0.95, 0.95);
            let mut best: Option<(f64, f64, f64)> = None;
            for nu in student_t::candidate_nus() {
                let Ok(terms) = student_t::NuTerms::new(x1, x2, nu) else {
                    continue;
                };
                let search = maximize_scalar_brent(
                    -STUDENT_T_RHO_MAX,
                    STUDENT_T_RHO_MAX,
                    Some(rho_seed),
                    SCALAR_TOL,
                    max_iter,
                    |rho| terms.loglik(rho),
                );
                if search.value.is_finite() && best.is_none_or(|(_, _, value)| search.value > value)
                {
                    best = Some((search.x, nu, search.value));
                }
            }
            let (rho, nu, _) = best.ok_or(FitError::Failed {
                reason: "student t pair fit failed",
            })?;
            polish_two_parameter(&objective, PairCopulaParams::Two(rho, nu), max_iter)
        }
        PairCopulaFamily::Clayton => {
            let init = clayton::theta_from_tau(tau)?;
            let upper = (init * 4.0 + 2.0).max(20.0);
            one_parameter_mle(&objective, 1e-6, upper, init, max_iter)?
        }
        PairCopulaFamily::Frank => {
            // θ and τ share their sign, so the search runs on the half-line
            // matching τ and never crosses the excluded θ = 0.
            let init = if tau.abs() < FRANK_TAU_FLOOR {
                1e-3
            } else {
                frank::theta_from_tau(tau)?
            };
            let upper = (init.abs() * 4.0 + 2.0).max(20.0);
            if init < 0.0 {
                one_parameter_mle(&objective, -upper, -1e-6, init, max_iter)?
            } else {
                one_parameter_mle(&objective, 1e-6, upper, init, max_iter)?
            }
        }
        PairCopulaFamily::Gumbel => {
            let init = gumbel::theta_from_tau(tau)?;
            let upper = (init * 4.0 + 2.0).max(20.0);
            one_parameter_mle(&objective, 1.0 + 1e-6, upper, init, max_iter)?
        }
        PairCopulaFamily::Joe => {
            let init = joe::theta_from_tau(tau)?;
            let upper = (init * 4.0 + 2.0).max(20.0);
            one_parameter_mle(&objective, 1.0 + 1e-6, upper, init, max_iter)?
        }
        PairCopulaFamily::Bb1 => {
            // 2-parameter family (θ > 0, δ ≥ 1). Coarse outer grid over δ
            // (the tail-dependence ingredient) with an inner 1-D search over
            // θ (the Clayton ingredient), initialised from the closed-form τ
            // relation τ = 1 - 2 / (δ(θ+2)) so each δ starts near a
            // reasonable θ, followed by the joint polish.
            let start = two_parameter_warm_start(
                &objective,
                &[1.05_f64, 1.25, 1.5, 2.0, 3.0, 5.0],
                |delta| {
                    let init_theta = bb1::params_from_tau(tau, delta).unwrap_or(1.0);
                    (1e-4, (init_theta * 4.0 + 2.0).max(10.0), Some(init_theta))
                },
                max_iter,
                "bb1 pair fit failed",
            )?;
            polish_two_parameter(&objective, start, max_iter)
        }
        PairCopulaFamily::Bb6 => {
            // θ ≥ 1, δ ≥ 1 — Joe-Gumbel blend. τ has no closed form so we
            // just coarse-grid δ and search θ in [1+ε, 20] before polishing.
            let start = two_parameter_warm_start(
                &objective,
                &[1.05_f64, 1.25, 1.5, 2.0, 3.0, 5.0],
                |_| (1.0 + 1e-6, 20.0, None),
                max_iter,
                "bb6 pair fit failed",
            )?;
            polish_two_parameter(&objective, start, max_iter)
        }
        PairCopulaFamily::Bb7 => {
            // θ ≥ 1, δ > 0 — Joe-Clayton blend.
            let start = two_parameter_warm_start(
                &objective,
                &[0.25_f64, 0.5, 1.0, 2.0, 4.0, 8.0],
                |_| (1.0 + 1e-6, 20.0, None),
                max_iter,
                "bb7 pair fit failed",
            )?;
            polish_two_parameter(&objective, start, max_iter)
        }
        PairCopulaFamily::Bb8 => {
            // θ ≥ 1, δ ∈ (0, 1] — Joe-Frank blend.
            let start = two_parameter_warm_start(
                &objective,
                &[0.1_f64, 0.3, 0.5, 0.7, 0.9, 1.0 - 1e-6],
                |_| (1.0 + 1e-6, 20.0, None),
                max_iter,
                "bb8 pair fit failed",
            )?;
            polish_two_parameter(&objective, start, max_iter)
        }
        PairCopulaFamily::Tll => {
            // Nonparametric — no scalar optimisation. Fit directly from the
            // (already rotation-transformed but since Tll is rotationless
            // that's the identity) sample.
            let params = tll::fit(x1, x2, tll::TllOrder::Constant)?;
            PairCopulaParams::Tll(params)
        }
        PairCopulaFamily::Tawn1 | PairCopulaFamily::Tawn2 => {
            // Tawn1 has β = 1 fixed; Tawn2 has α = 1 fixed. In both cases the
            // free parameters are (θ, ψ) with θ ≥ 1 and ψ ∈ [0, 1]. Outer
            // grid over ψ, inner search over θ — same pattern as BB6/7/8.
            let start = two_parameter_warm_start(
                &objective,
                &[0.1_f64, 0.3, 0.5, 0.7, 0.9, 1.0 - 1e-6],
                |_| (1.0 + 1e-6, 20.0, None),
                max_iter,
                "tawn pair fit failed",
            )?;
            polish_two_parameter(&objective, start, max_iter)
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

/// Gaussian pair MLE. The summed log-density collapses to a function of the
/// three sufficient statistics `Σz₁²`, `Σz₂²`, `Σz₁z₂`, so the profile is
/// essentially free to evaluate; it is maximised over `atanh ρ` with Brent's
/// method, warm-started at the Kendall-τ inversion `ρ = sin(πτ/2)`. The
/// result is never worse than that start.
fn fit_gaussian_mle(tau: f64, x1: &[f64], x2: &[f64], max_iter: usize) -> PairCopulaParams {
    let normal = Normal::new(0.0, 1.0).expect("standard normal parameters should be valid");
    let (mut s11, mut s22, mut s12) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&u, &v) in x1.iter().zip(x2) {
        let z1 = normal.inverse_cdf(u);
        let z2 = normal.inverse_cdf(v);
        s11 += z1 * z1;
        s22 += z2 * z2;
        s12 += z1 * z2;
    }
    let n = x1.len() as f64;
    let profile = |t: f64| {
        let rho = t.tanh();
        let one_minus = 1.0 - rho * rho;
        if one_minus.is_nan() || one_minus <= 0.0 {
            return f64::NEG_INFINITY;
        }
        -0.5 * n * one_minus.ln() - (rho * rho * (s11 + s22) - 2.0 * rho * s12) / (2.0 * one_minus)
    };

    let rho_start = gaussian::tau_to_rho(tau).clamp(-GAUSSIAN_RHO_MAX, GAUSSIAN_RHO_MAX);
    let t_start = rho_start.atanh();
    let t_max = GAUSSIAN_RHO_MAX.atanh();
    let search = maximize_scalar_brent(-t_max, t_max, Some(t_start), 1e-10, max_iter, profile);
    let rho = if search.value.is_finite() && search.value >= profile(t_start) {
        search.x.tanh()
    } else {
        rho_start
    };
    PairCopulaParams::One(rho.clamp(-GAUSSIAN_RHO_MAX, GAUSSIAN_RHO_MAX))
}

/// Bracketed scalar MLE for a one-parameter family, warm-started at the
/// moment-based `init` and stopped by tolerance or `max_iter`.
fn one_parameter_mle(
    objective: &PairObjective<'_>,
    low: f64,
    high: f64,
    init: f64,
    max_iter: usize,
) -> Result<PairCopulaParams, CopulaError> {
    let search = maximize_scalar_brent(low, high, Some(init), SCALAR_TOL, max_iter, |theta| {
        objective.loglik(PairCopulaParams::One(theta))
    });
    if !search.value.is_finite() {
        return Err(FitError::Failed {
            reason: "pair-copula likelihood is not finite at any candidate",
        }
        .into());
    }
    Ok(PairCopulaParams::One(search.x))
}

/// Coarse warm start for a two-parameter family: for every value of the
/// second parameter in `grid`, run a loosely converged scalar search over the
/// first parameter inside the `(low, high, start)` bracket returned by
/// `bracket`, and keep the best pair.
fn two_parameter_warm_start<B>(
    objective: &PairObjective<'_>,
    grid: &[f64],
    bracket: B,
    max_iter: usize,
    failure: &'static str,
) -> Result<PairCopulaParams, CopulaError>
where
    B: Fn(f64) -> (f64, f64, Option<f64>),
{
    let mut best: Option<(f64, f64, f64)> = None;
    for &second in grid {
        let (low, high, start) = bracket(second);
        let search = maximize_scalar_brent(low, high, start, WARM_START_TOL, max_iter, |first| {
            objective.loglik(PairCopulaParams::Two(first, second))
        });
        if search.value.is_finite() && best.is_none_or(|(_, _, value)| search.value > value) {
            best = Some((search.x, second, search.value));
        }
    }
    let (first, second, _) = best.ok_or(FitError::Failed { reason: failure })?;
    Ok(PairCopulaParams::Two(first, second))
}

/// Joint maximisation of a two-parameter family from a warm start, run with
/// Nelder–Mead in the unconstrained space of `polish.rs`. Returns the warm
/// start unchanged if the polish does not improve on it or produces an
/// invalid spec, so the fitted log-likelihood never regresses.
fn polish_two_parameter(
    objective: &PairObjective<'_>,
    start: PairCopulaParams,
    max_iter: usize,
) -> PairCopulaParams {
    let template = PairCopulaSpec {
        family: objective.family,
        rotation: Rotation::R0,
        params: start.clone(),
    };
    let x0 = encode_fit_params(&template);
    let bounds = fit_brackets(objective.family);
    if x0.len() != 2 || bounds.len() != 2 {
        return start;
    }
    let start_value = objective.loglik(start.clone());
    if !start_value.is_finite() {
        return start;
    }
    let options = NelderMeadOptions {
        initial_step: 0.2,
        ftol: 1e-8 * (1.0 + start_value.abs()),
        xtol: 1e-6,
        max_iter,
        restarts: 1,
    };
    let result = nelder_mead_maximize(&x0, &bounds, &options, |x| {
        objective.loglik(decode_fit_params(&template, x).params)
    });
    let polished = decode_fit_params(&template, &result.x);
    if result.value.is_finite() && result.value >= start_value && polished.validate().is_ok() {
        polished.params
    } else {
        start
    }
}

/// Number of base-pair combinations that receive the joint Khoudraji polish.
/// The warm start ranks pairs by their penalised score, which is only a rough
/// proxy for the polished optimum, so a few runners-up are polished as well.
const KHOUDRAJI_POLISH_CANDIDATES: usize = 3;
/// Logit-space box for the Khoudraji shapes during the joint polish;
/// `inv_logit(±12) ≈ 6e-6` keeps both power exponents strictly inside (0, 1).
const KHOUDRAJI_SHAPE_LOGIT_BOUND: f64 = 12.0;

/// One base-pair combination of the Khoudraji fit together with its shapes
/// and log-likelihood on the caller's (unrotated) sample.
struct KhoudrajiCandidate {
    first: PairCopulaSpec,
    second: PairCopulaSpec,
    shape_first: f64,
    shape_second: f64,
    loglik: f64,
}

impl KhoudrajiCandidate {
    fn parameter_count(&self) -> usize {
        self.first.parameter_count() + self.second.parameter_count() + 2
    }

    fn into_spec(self, rotation: Rotation) -> Result<PairCopulaSpec, CopulaError> {
        Ok(PairCopulaSpec {
            family: PairCopulaFamily::Khoudraji,
            rotation,
            params: PairCopulaParams::Khoudraji(KhoudrajiParams::new(
                self.first,
                self.second,
                self.shape_first,
                self.shape_second,
            )?),
        })
    }
}

/// Information-criterion score used to rank Khoudraji base-pair candidates:
/// AIC under `SelectionCriterion::Aic`, BIC otherwise (mBICV is a tree-level
/// penalty and falls back to BIC per edge, exactly as in `criterion_value`).
/// Ranking by a penalised score rather than raw log-likelihood makes the
/// simplest of several equivalent representations win — e.g.
/// `Independence ⊗ Clayton` over `Clayton(≈0) ⊗ Clayton` at the same optimum.
fn penalised_score(loglik: f64, k: f64, n: f64, criterion: SelectionCriterion) -> f64 {
    match criterion {
        SelectionCriterion::Aic => 2.0 * k - 2.0 * loglik,
        SelectionCriterion::Bic | SelectionCriterion::Mbicv { .. } => k * n.ln() - 2.0 * loglik,
    }
}

/// Khoudraji fit in three stages: (1) fit each base family to the rotated
/// sample and run the shape search for every *unordered* base pair — since
/// `(C₁, C₂, a, b)` and `(C₂, C₁, 1 − a, 1 − b)` are the same copula, ordered
/// pairs would only double the work; (2) jointly polish the leading pairs
/// over (base parameters, shapes) with Nelder–Mead in the unconstrained space
/// of `polish.rs`; (3) select the pair by the caller's information criterion.
fn fit_khoudraji_with_rotation(
    rotation: Rotation,
    u1: &[f64],
    u2: &[f64],
    tau: f64,
    clip_eps: f64,
    max_iter: usize,
    criterion: SelectionCriterion,
) -> Result<PairCopulaSpec, CopulaError> {
    let (x1, x2) = clipped_transform(rotation, u1, u2, clip_eps);
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
        let params = fit_simple_family(family, tau, &x1, &x2, clip_eps, max_iter)?;
        base_specs.push(PairCopulaSpec {
            family,
            rotation: Rotation::R0,
            params,
        });
    }

    let n = u1.len() as f64;
    let score = |candidate: &KhoudrajiCandidate| {
        penalised_score(
            candidate.loglik,
            candidate.parameter_count() as f64,
            n,
            criterion,
        )
    };
    let mut candidates = Vec::with_capacity(base_specs.len() * (base_specs.len() + 1) / 2);
    for (index, first) in base_specs.iter().enumerate() {
        for second in &base_specs[index..] {
            candidates.push(warm_start_khoudraji_candidate(
                first, second, u1, u2, rotation, clip_eps, max_iter,
            )?);
        }
    }
    candidates.sort_by(|left, right| score(left).total_cmp(&score(right)));

    for candidate in candidates.iter_mut().take(KHOUDRAJI_POLISH_CANDIDATES) {
        if let Some(polished) =
            polish_khoudraji_jointly(candidate, u1, u2, rotation, clip_eps, max_iter)
        {
            *candidate = polished;
        }
    }

    let best = candidates
        .into_iter()
        .filter(|candidate| candidate.loglik.is_finite())
        .min_by(|left, right| score(left).total_cmp(&score(right)))
        .ok_or(FitError::Failed {
            reason: "khoudraji pair fit failed",
        })?;
    best.into_spec(rotation)
}

/// Joint maximisation over (base-1 parameters, base-2 parameters, shapes)
/// from the two-stage warm start. Base parameters use the pair-fit encoding,
/// shapes use the logit. Returns `None` unless the polished point validates
/// and strictly improves the log-likelihood, so the warm start is never
/// replaced by something worse.
fn polish_khoudraji_jointly(
    candidate: &KhoudrajiCandidate,
    u1: &[f64],
    u2: &[f64],
    rotation: Rotation,
    clip_eps: f64,
    max_iter: usize,
) -> Option<KhoudrajiCandidate> {
    if !candidate.loglik.is_finite() {
        return None;
    }
    let first_coords = encode_fit_params(&candidate.first);
    let second_coords = encode_fit_params(&candidate.second);
    let (n_first, n_second) = (first_coords.len(), second_coords.len());
    let mut x0 = first_coords;
    x0.extend(second_coords);
    x0.push(logit(candidate.shape_first));
    x0.push(logit(candidate.shape_second));
    let mut bounds = fit_brackets(candidate.first.family);
    bounds.extend(fit_brackets(candidate.second.family));
    bounds.push((-KHOUDRAJI_SHAPE_LOGIT_BOUND, KHOUDRAJI_SHAPE_LOGIT_BOUND));
    bounds.push((-KHOUDRAJI_SHAPE_LOGIT_BOUND, KHOUDRAJI_SHAPE_LOGIT_BOUND));

    let decode = |x: &[f64]| {
        let first = decode_fit_params(&candidate.first, &x[..n_first]);
        let second = decode_fit_params(&candidate.second, &x[n_first..n_first + n_second]);
        let shape_first = inv_logit(x[n_first + n_second]);
        let shape_second = inv_logit(x[n_first + n_second + 1]);
        (first, second, shape_first, shape_second)
    };
    // Khoudraji parameters are only ever reported to a few decimals and the
    // objective is a full pass over the sample per evaluation, so the polish
    // stops at 1e-4 in unconstrained space (about four significant digits of
    // each base parameter and shape).
    let options = NelderMeadOptions {
        initial_step: 0.2,
        ftol: 1e-6 * (1.0 + candidate.loglik.abs()),
        xtol: 1e-4,
        max_iter,
        restarts: 1,
    };
    let result = nelder_mead_maximize(&x0, &bounds, &options, |x| {
        let (first, second, shape_first, shape_second) = decode(x);
        khoudraji_loglik(
            &first,
            &second,
            shape_first,
            shape_second,
            u1,
            u2,
            rotation,
            clip_eps,
        )
        .unwrap_or(f64::NEG_INFINITY)
    });
    if !result.value.is_finite() || result.value <= candidate.loglik {
        return None;
    }
    let (first, second, shape_first, shape_second) = decode(&result.x);
    let spec = PairCopulaSpec {
        family: PairCopulaFamily::Khoudraji,
        rotation,
        params: PairCopulaParams::Khoudraji(
            KhoudrajiParams::new(first.clone(), second.clone(), shape_first, shape_second).ok()?,
        ),
    };
    spec.validate().ok()?;
    Some(KhoudrajiCandidate {
        first,
        second,
        shape_first,
        shape_second,
        loglik: result.value,
    })
}

fn rotated_tau(rotation: Rotation, tau: f64) -> f64 {
    match rotation {
        Rotation::R0 | Rotation::R180 => tau,
        Rotation::R90 | Rotation::R270 => -tau,
    }
}

/// Shape seeds for the Khoudraji block coordinate ascent. Only unordered base
/// pairs are enumerated, so the seeds cover both orientations of a pair.
const KHOUDRAJI_SHAPE_SEEDS: [(f64, f64); 3] = [(0.5, 0.5), (0.2, 0.8), (0.8, 0.2)];
/// Maximum sweeps of the block coordinate ascent (base parameters given the
/// shapes, then shapes given the base parameters) per seed. A seed stops
/// early once a sweep no longer improves the log-likelihood materially.
const KHOUDRAJI_SWEEPS: usize = 3;
/// Tolerance of the scalar searches inside the Khoudraji warm start (shapes
/// and unconstrained base coordinates); the joint polish refines the result.
const KHOUDRAJI_WARM_START_TOL: f64 = 1e-2;
/// Relative log-likelihood improvement below which a warm-start sweep is
/// considered converged.
const KHOUDRAJI_SWEEP_REL_TOL: f64 = 1e-4;

/// Two-stage warm start for one base pair: block coordinate ascent that
/// alternates the base parameters given the shapes with the shapes given the
/// base parameters, from each seed in [`KHOUDRAJI_SHAPE_SEEDS`].
///
/// Refitting the bases first matters. The base copulas arrive fitted to the
/// raw sample, where their dependence is "diluted" by the shape powers, so
/// optimising the shapes at those parameters collapses onto a single base
/// copula (shapes at 0 or 1) — a flat corner of the parameter space that a
/// joint polish cannot escape.
fn warm_start_khoudraji_candidate(
    first: &PairCopulaSpec,
    second: &PairCopulaSpec,
    u1: &[f64],
    u2: &[f64],
    rotation: Rotation,
    clip_eps: f64,
    max_iter: usize,
) -> Result<KhoudrajiCandidate, CopulaError> {
    let mut best: Option<KhoudrajiCandidate> = None;
    for (shape_first, shape_second) in KHOUDRAJI_SHAPE_SEEDS {
        let mut candidate = KhoudrajiCandidate {
            first: first.clone(),
            second: second.clone(),
            shape_first,
            shape_second,
            loglik: f64::NEG_INFINITY,
        };
        let mut previous = f64::NEG_INFINITY;
        for _ in 0..KHOUDRAJI_SWEEPS {
            refine_khoudraji_base(&mut candidate, true, u1, u2, rotation, clip_eps, max_iter);
            refine_khoudraji_base(&mut candidate, false, u1, u2, rotation, clip_eps, max_iter);

            let (base_first, base_second) = (&candidate.first, &candidate.second);
            let fixed_second = candidate.shape_second;
            candidate.shape_first = maximize_scalar_brent(
                0.0,
                1.0,
                Some(candidate.shape_first),
                KHOUDRAJI_WARM_START_TOL,
                max_iter,
                |shape| {
                    khoudraji_loglik(
                        base_first,
                        base_second,
                        shape,
                        fixed_second,
                        u1,
                        u2,
                        rotation,
                        clip_eps,
                    )
                    .unwrap_or(f64::NEG_INFINITY)
                },
            )
            .x;
            let fixed_first = candidate.shape_first;
            let search = maximize_scalar_brent(
                0.0,
                1.0,
                Some(candidate.shape_second),
                KHOUDRAJI_WARM_START_TOL,
                max_iter,
                |shape| {
                    khoudraji_loglik(
                        base_first,
                        base_second,
                        fixed_first,
                        shape,
                        u1,
                        u2,
                        rotation,
                        clip_eps,
                    )
                    .unwrap_or(f64::NEG_INFINITY)
                },
            );
            candidate.shape_second = search.x;
            if search.value.is_finite()
                && search.value - previous <= KHOUDRAJI_SWEEP_REL_TOL * (1.0 + search.value.abs())
            {
                break;
            }
            previous = search.value;
        }
        candidate.loglik = khoudraji_loglik(
            &candidate.first,
            &candidate.second,
            candidate.shape_first,
            candidate.shape_second,
            u1,
            u2,
            rotation,
            clip_eps,
        )?;
        if best
            .as_ref()
            .is_none_or(|incumbent| candidate.loglik > incumbent.loglik)
        {
            best = Some(candidate);
        }
    }
    best.ok_or(
        FitError::Failed {
            reason: "khoudraji shape optimization failed",
        }
        .into(),
    )
}

/// Refits one base copula of `candidate` at the current shapes: a bounded
/// scalar search per unconstrained coordinate of the base's pair-fit
/// encoding, warm-started at the current value. Independence has nothing to
/// refit.
fn refine_khoudraji_base(
    candidate: &mut KhoudrajiCandidate,
    refine_first: bool,
    u1: &[f64],
    u2: &[f64],
    rotation: Rotation,
    clip_eps: f64,
    max_iter: usize,
) {
    let template = if refine_first {
        candidate.first.clone()
    } else {
        candidate.second.clone()
    };
    let mut coords = encode_fit_params(&template);
    if coords.is_empty() {
        return;
    }
    let bounds = fit_brackets(template.family);
    for index in 0..coords.len() {
        let (low, high) = bounds[index];
        let search = maximize_scalar_brent(
            low,
            high,
            Some(coords[index]),
            KHOUDRAJI_WARM_START_TOL,
            max_iter,
            |value| {
                let mut trial = coords.clone();
                trial[index] = value;
                let refined = decode_fit_params(&template, &trial);
                let (first, second) = if refine_first {
                    (&refined, &candidate.second)
                } else {
                    (&candidate.first, &refined)
                };
                khoudraji_loglik(
                    first,
                    second,
                    candidate.shape_first,
                    candidate.shape_second,
                    u1,
                    u2,
                    rotation,
                    clip_eps,
                )
                .unwrap_or(f64::NEG_INFINITY)
            },
        );
        if search.value.is_finite() {
            coords[index] = search.x;
        }
    }
    let refined = decode_fit_params(&template, &coords);
    if refined.validate().is_ok() {
        if refine_first {
            candidate.first = refined;
        } else {
            candidate.second = refined;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn khoudraji_loglik(
    first: &PairCopulaSpec,
    second: &PairCopulaSpec,
    shape_first: f64,
    shape_second: f64,
    u1: &[f64],
    u2: &[f64],
    rotation: Rotation,
    clip_eps: f64,
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
    pair_loglik(&spec, u1, u2, clip_eps)
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

#[cfg(test)]
mod khoudraji_fit_tests {
    use super::*;

    fn reference_fixture() -> (Vec<f64>, Vec<f64>, f64, (f64, f64)) {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/reference/r-copula/v1_1_3/khoudraji_fit_case01.json"
        );
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).expect("fixture should exist"))
                .expect("fixture should deserialize");
        let rows = fixture["input_pobs"].as_array().expect("input_pobs");
        let u1 = rows.iter().map(|row| row[0].as_f64().unwrap()).collect();
        let u2 = rows.iter().map(|row| row[1].as_f64().unwrap()).collect();
        let theta = fixture["expected_theta"].as_f64().unwrap();
        let shapes = (
            fixture["expected_shape_1"].as_f64().unwrap(),
            fixture["expected_shape_2"].as_f64().unwrap(),
        );
        (u1, u2, theta, shapes)
    }

    /// Runs the Khoudraji warm start and joint polish for one fixed base
    /// pair, exactly as `fit_khoudraji_with_rotation` does before selecting
    /// among pairs.
    fn fit_fixed_pair(
        first: PairCopulaSpec,
        second: PairCopulaSpec,
        u1: &[f64],
        u2: &[f64],
        max_iter: usize,
    ) -> KhoudrajiCandidate {
        let candidate =
            warm_start_khoudraji_candidate(&first, &second, u1, u2, Rotation::R0, 1e-12, max_iter)
                .expect("warm start should succeed");
        polish_khoudraji_jointly(&candidate, u1, u2, Rotation::R0, 1e-12, max_iter)
            .unwrap_or(candidate)
    }

    #[test]
    fn independence_clayton_pair_matches_r_joint_mle_on_the_reference_fixture() {
        // `copula::fitCopula` on the same 96 rows for the same base pair
        // gives theta = 4.927, shapes = (0.761, 0.415), loglik 11.528 in this
        // kernel. Two implementations of one MLE on one dataset must agree
        // within a fraction of a standard error (SE(theta) ~ 1.6 and
        // SE(shape) ~ 0.05-0.1 at n = 96).
        let (u1, u2, theta, shapes) = reference_fixture();
        let r_solution = PairCopulaSpec::khoudraji(
            PairCopulaSpec::independence(),
            PairCopulaSpec {
                family: PairCopulaFamily::Clayton,
                rotation: Rotation::R0,
                params: PairCopulaParams::One(theta),
            },
            shapes.0,
            shapes.1,
        )
        .expect("R solution should validate");
        let r_loglik = pair_loglik(&r_solution, &u1, &u2, 1e-12).expect("R loglik");

        // The raw-sample base fit the enumeration starts from.
        let tau = crate::stats::kendall_tau_bivariate(&u1, &u2).expect("tau");
        let clayton = PairCopulaSpec {
            family: PairCopulaFamily::Clayton,
            rotation: Rotation::R0,
            params: fit_simple_family(PairCopulaFamily::Clayton, tau, &u1, &u2, 1e-12, 500)
                .expect("clayton base fit"),
        };

        for max_iter in [500usize, 8] {
            let fitted = fit_fixed_pair(
                PairCopulaSpec::independence(),
                clayton.clone(),
                &u1,
                &u2,
                max_iter,
            );
            let PairCopulaParams::One(fitted_theta) = fitted.second.params else {
                panic!("clayton base keeps one parameter");
            };
            println!(
                "max_iter {max_iter}: theta {fitted_theta} shapes ({}, {}) loglik {} vs R theta {theta} shapes {shapes:?} loglik {r_loglik}",
                fitted.shape_first, fitted.shape_second, fitted.loglik
            );
            if max_iter == 500 {
                assert!(fitted.loglik >= r_loglik - 1e-3, "loglik {}", fitted.loglik);
                assert!((fitted_theta - theta).abs() < 1.0, "theta {fitted_theta}");
                assert!((fitted.shape_first - shapes.0).abs() < 0.1);
                assert!((fitted.shape_second - shapes.1).abs() < 0.1);
            }
        }
    }

    #[test]
    fn swapped_base_pair_is_the_same_copula_with_complementary_shapes() {
        // (C1, C2, a, b) and (C2, C1, 1 - a, 1 - b) describe one copula, which
        // is why the fitter enumerates unordered base pairs only.
        let clayton = PairCopulaSpec {
            family: PairCopulaFamily::Clayton,
            rotation: Rotation::R0,
            params: PairCopulaParams::One(2.5),
        };
        let gumbel = PairCopulaSpec {
            family: PairCopulaFamily::Gumbel,
            rotation: Rotation::R0,
            params: PairCopulaParams::One(1.8),
        };
        let direct = PairCopulaSpec::khoudraji(clayton.clone(), gumbel.clone(), 0.3, 0.8).unwrap();
        let swapped = PairCopulaSpec::khoudraji(gumbel, clayton, 0.7, 0.2).unwrap();
        for (u, v) in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.3), (0.35, 0.95)] {
            let a = direct.log_pdf(u, v, 1e-12).unwrap();
            let b = swapped.log_pdf(u, v, 1e-12).unwrap();
            assert!((a - b).abs() < 1e-10, "({u}, {v}): {a} vs {b}");
            let a = direct.cond_second_given_first(u, v, 1e-12).unwrap();
            let b = swapped.cond_second_given_first(u, v, 1e-12).unwrap();
            assert!((a - b).abs() < 1e-10);
        }
    }
}
