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
pub struct GaussianCopula {
    dim: usize,
    correlation: Array2<f64>,
}

impl GaussianCopula {
    pub fn new(correlation: Array2<f64>) -> Self {
        let dim = correlation.ncols();
        Self { dim, correlation }
    }

    pub fn fit(data: &PseudoObs, _options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        let _ = data;
        Err(FitError::NotImplemented.into())
    }

    pub fn correlation(&self) -> &Array2<f64> {
        &self.correlation
    }
}

impl CopulaModel for GaussianCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::Gaussian
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentTCopula {
    dim: usize,
    correlation: Array2<f64>,
    degrees_of_freedom: f64,
}

impl StudentTCopula {
    pub fn new(correlation: Array2<f64>, degrees_of_freedom: f64) -> Self {
        let dim = correlation.ncols();
        Self {
            dim,
            correlation,
            degrees_of_freedom,
        }
    }

    pub fn fit(_data: &PseudoObs, _options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        Err(FitError::NotImplemented.into())
    }

    pub fn correlation(&self) -> &Array2<f64> {
        &self.correlation
    }

    pub fn degrees_of_freedom(&self) -> f64 {
        self.degrees_of_freedom
    }
}

impl CopulaModel for StudentTCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::StudentT
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
