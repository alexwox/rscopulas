use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{data::PseudoObs, errors::CopulaError};

/// Options shared by fitting routines in the core crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitOptions {
    /// Probability clipping applied by fitters that evaluate densities internally.
    pub clip_eps: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self { clip_eps: 1e-12 }
    }
}

/// Options shared by density evaluation routines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalOptions {
    /// Clamp pseudo-observations into `[clip_eps, 1 - clip_eps]` before inversion.
    pub clip_eps: f64,
}

impl Default for EvalOptions {
    fn default() -> Self {
        Self { clip_eps: 1e-12 }
    }
}

/// Options shared by sampling routines.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SampleOptions;

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
    Vine(super::VineCopula),
}
