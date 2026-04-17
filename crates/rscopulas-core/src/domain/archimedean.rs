use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    data::PseudoObs,
    errors::{CopulaError, FitError},
    fit::FitResult,
};

use super::{CopulaFamily, CopulaModel, EvalOptions, FitOptions, SampleOptions};

macro_rules! impl_archimedean_family {
    ($name:ident, $family:expr) => {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct $name {
            dim: usize,
            theta: f64,
        }

        impl $name {
            pub fn new(dim: usize, theta: f64) -> Self {
                Self { dim, theta }
            }

            pub fn fit(
                _data: &PseudoObs,
                _options: &FitOptions,
            ) -> Result<FitResult<Self>, CopulaError> {
                Err(FitError::NotImplemented.into())
            }

            pub fn theta(&self) -> f64 {
                self.theta
            }
        }

        impl CopulaModel for $name {
            fn family(&self) -> CopulaFamily {
                $family
            }

            fn dim(&self) -> usize {
                self.dim
            }

            fn log_pdf(
                &self,
                _data: &PseudoObs,
                _options: &EvalOptions,
            ) -> Result<Vec<f64>, CopulaError> {
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
    };
}

impl_archimedean_family!(ClaytonCopula, CopulaFamily::Clayton);
impl_archimedean_family!(FrankCopula, CopulaFamily::Frank);
impl_archimedean_family!(GumbelHougaardCopula, CopulaFamily::Gumbel);
