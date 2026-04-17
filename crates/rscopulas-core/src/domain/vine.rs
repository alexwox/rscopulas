use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    data::PseudoObs,
    errors::{CopulaError, FitError},
    fit::FitResult,
};

use super::{CopulaFamily, CopulaModel, EvalOptions, FitOptions, SampleOptions};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VineCopula {
    dim: usize,
    truncation_level: Option<usize>,
}

impl VineCopula {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            truncation_level: None,
        }
    }

    pub fn fit_c_vine(
        _data: &PseudoObs,
        _options: &FitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        Err(FitError::NotImplemented.into())
    }

    pub fn fit_d_vine(
        _data: &PseudoObs,
        _options: &FitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        Err(FitError::NotImplemented.into())
    }

    pub fn fit_r_vine(
        _data: &PseudoObs,
        _options: &FitOptions,
    ) -> Result<FitResult<Self>, CopulaError> {
        Err(FitError::NotImplemented.into())
    }

    pub fn truncation_level(&self) -> Option<usize> {
        self.truncation_level
    }
}

impl CopulaModel for VineCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::Vine
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn log_pdf(&self, _data: &PseudoObs, _options: &EvalOptions) -> Result<Vec<f64>, CopulaError> {
        Err(FitError::NotImplemented.into())
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        _n: usize,
        _rng: &mut R,
        _options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        Err(FitError::NotImplemented.into())
    }
}
