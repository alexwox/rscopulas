use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{data::PseudoObs, errors::CopulaError};

/// Execution target requested by the caller.
///
/// `Cuda` is the first `f64`-native GPU target. `Metal` is intentionally
/// bounded to mixed-precision kernels whose tolerance budget has been reviewed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(u32),
    Metal,
}

/// Policy for selecting which execution backend to use.
///
/// `Auto` currently means CPU reference / CPU parallel dispatch only.
/// `Force(Device::Cuda(_))` and `Force(Device::Metal)` opt into the narrow
/// GPU-aware paths that exist today:
/// - Gaussian pair batch evaluation used by pair-fit scoring
/// - Gaussian vine log-density evaluation built from those pair batches
///
/// Other operations still return a backend error instead of pretending they are
/// accelerated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecPolicy {
    Auto,
    Force(Device),
}

/// Options shared by fitting routines in the core crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitOptions {
    /// Backend policy used for dependence estimation and pair scoring.
    ///
    /// Pair-family selection remains CPU-driven, but forced GPU backends can
    /// accelerate the Gaussian pair batch substeps inside that scoring loop.
    pub exec: ExecPolicy,
    /// Probability clipping applied by fitters that evaluate densities internally.
    pub clip_eps: f64,
    /// Upper bound for iterative search routines used during fitting.
    pub max_iter: usize,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            exec: ExecPolicy::Auto,
            clip_eps: 1e-12,
            max_iter: 500,
        }
    }
}

/// Options shared by density evaluation routines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalOptions {
    /// Backend policy used for batch density evaluation.
    ///
    /// GPU-backed evaluation is currently limited to Gaussian vine log-density.
    /// Single-family copula density evaluation remains on CPU.
    pub exec: ExecPolicy,
    /// Clamp pseudo-observations into `[clip_eps, 1 - clip_eps]` before inversion.
    pub clip_eps: f64,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self {
            exec: ExecPolicy::Auto,
            clip_eps: 1e-12,
        }
    }
}

/// Options shared by sampling routines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleOptions {
    /// Backend policy used for batch sampling.
    ///
    /// Sampling is still CPU-only; forced GPU sampling requests remain
    /// unsupported until dedicated kernels are added.
    pub exec: ExecPolicy,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            exec: ExecPolicy::Auto,
        }
    }
}

/// Common diagnostics returned alongside fitted models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitDiagnostics {
    pub loglik: f64,
    pub aic: f64,
    pub bic: f64,
    pub converged: bool,
    pub n_iter: usize,
}

/// Supported copula families exposed by the high-level API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CopulaFamily {
    Gaussian,
    StudentT,
    Clayton,
    Frank,
    Gumbel,
    HierarchicalArchimedean,
    Vine,
}

/// Shared interface implemented by all fitted copula models.
pub trait CopulaModel {
    /// Returns the model family identifier.
    fn family(&self) -> CopulaFamily;

    /// Returns the model dimension.
    fn dim(&self) -> usize;

    /// Evaluates the copula log-density for each observation in `data`.
    fn log_pdf(&self, data: &PseudoObs, options: &EvalOptions) -> Result<Vec<f64>, CopulaError>;

    /// Draws `n` pseudo-observations from the fitted model.
    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError>;
}

#[allow(clippy::large_enum_variant)]
/// Convenience sum type for callers that want to store different fitted models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Copula {
    Gaussian(super::GaussianCopula),
    StudentT(super::StudentTCopula),
    Clayton(super::ClaytonCopula),
    Frank(super::FrankCopula),
    Gumbel(super::GumbelHougaardCopula),
    HierarchicalArchimedean(super::HierarchicalArchimedeanCopula),
    Vine(super::VineCopula),
}
