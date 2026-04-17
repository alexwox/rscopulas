use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{data::PseudoObs, errors::CopulaError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(u32),
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecPolicy {
    Auto,
    Force(Device),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitOptions {
    pub exec: ExecPolicy,
    pub clip_eps: f64,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalOptions {
    pub exec: ExecPolicy,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleOptions {
    pub exec: ExecPolicy,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            exec: ExecPolicy::Auto,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitDiagnostics {
    pub loglik: f64,
    pub aic: f64,
    pub bic: f64,
    pub converged: bool,
    pub n_iter: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CopulaFamily {
    Gaussian,
    StudentT,
    Clayton,
    Frank,
    Gumbel,
    Vine,
}

pub trait CopulaModel {
    fn family(&self) -> CopulaFamily;
    fn dim(&self) -> usize;
    fn log_pdf(&self, data: &PseudoObs, options: &EvalOptions) -> Result<Vec<f64>, CopulaError>;
    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Copula {
    Gaussian(super::GaussianCopula),
    StudentT(super::StudentTCopula),
    Clayton(super::ClaytonCopula),
    Frank(super::FrankCopula),
    Gumbel(super::GumbelHougaardCopula),
    Vine(super::VineCopula),
}
