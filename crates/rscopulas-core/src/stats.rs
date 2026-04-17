use ndarray::Array2;

use crate::data::PseudoObs;

pub fn kendall_tau_matrix(data: &PseudoObs) -> Array2<f64> {
    Array2::eye(data.dim())
}
