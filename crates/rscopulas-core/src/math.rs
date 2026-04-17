use ndarray::Array2;

use crate::errors::NumericalError;

pub fn identity_correlation(dim: usize) -> Array2<f64> {
    Array2::eye(dim)
}

pub fn validate_correlation_matrix(matrix: &Array2<f64>) -> Result<(), NumericalError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(NumericalError::InvalidCorrelationMatrix);
    }

    for i in 0..matrix.nrows() {
        if (matrix[(i, i)] - 1.0).abs() > 1e-12 {
            return Err(NumericalError::InvalidCorrelationMatrix);
        }
    }

    Ok(())
}
