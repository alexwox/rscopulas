use ndarray::Array2;
use rand::Rng;

use crate::{domain::SampleOptions, errors::CopulaError};

use super::{
    VineCopula,
    rosenblatt::{DEFAULT_CLIP_EPS, draw_uniform_matrix},
};

impl VineCopula {
    /// Unconditional sampling from the fitted vine. Implementation-wise this
    /// is just `inverse_rosenblatt(U)` applied to a freshly-drawn uniform
    /// matrix; the RNG call sequence is identical to the pre-Rosenblatt
    /// sampler so fixed-seed draws remain bit-identical.
    pub(crate) fn sample_internal<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        let u = draw_uniform_matrix(rng, n, &self.variable_order, DEFAULT_CLIP_EPS);
        self.inverse_rosenblatt(u.view(), options)
    }
}
