use ndarray::Array2;
use rand::Rng;

pub use crate::vine::{
    SelectionCriterion, VineCopula, VineEdge, VineFitOptions, VineStructure, VineStructureKind,
    VineTree,
};

use crate::{data::PseudoObs, errors::CopulaError};

use super::{CopulaFamily, CopulaModel, EvalOptions, SampleOptions};

impl CopulaModel for VineCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::Vine
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn log_pdf(&self, data: &PseudoObs, options: &EvalOptions) -> Result<Vec<f64>, CopulaError> {
        self.log_pdf_internal(data, options.clip_eps)
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        _options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        self.sample_internal(n, rng, 1e-12)
    }
}
