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

        for j in 0..matrix.ncols() {
            let value = matrix[(i, j)];
            if !value.is_finite() {
                return Err(NumericalError::InvalidCorrelationMatrix);
            }

            if (value - matrix[(j, i)]).abs() > 1e-12 {
                return Err(NumericalError::InvalidCorrelationMatrix);
            }
        }
    }

    cholesky(matrix)?;
    Ok(())
}

pub fn cholesky(matrix: &Array2<f64>) -> Result<Array2<f64>, NumericalError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(NumericalError::InvalidCorrelationMatrix);
    }

    let dim = matrix.nrows();
    let mut lower = Array2::zeros((dim, dim));

    for i in 0..dim {
        for j in 0..=i {
            let mut value = matrix[(i, j)];
            for k in 0..j {
                value -= lower[(i, k)] * lower[(j, k)];
            }

            if i == j {
                if value <= 0.0 {
                    return Err(NumericalError::DecompositionFailed);
                }
                lower[(i, j)] = value.sqrt();
            } else {
                let pivot = lower[(j, j)];
                if pivot == 0.0 {
                    return Err(NumericalError::DecompositionFailed);
                }
                lower[(i, j)] = value / pivot;
            }
        }
    }

    Ok(lower)
}

pub fn log_determinant_from_cholesky(lower: &Array2<f64>) -> f64 {
    2.0 * (0..lower.nrows()).map(|i| lower[(i, i)].ln()).sum::<f64>()
}

pub fn solve_lower_triangular(
    lower: &Array2<f64>,
    rhs: &[f64],
) -> Result<Vec<f64>, NumericalError> {
    if lower.nrows() != lower.ncols() || lower.nrows() != rhs.len() {
        return Err(NumericalError::InvalidCorrelationMatrix);
    }

    let mut solution = vec![0.0; rhs.len()];
    for i in 0..rhs.len() {
        let mut value = rhs[i];
        for j in 0..i {
            value -= lower[(i, j)] * solution[j];
        }

        let pivot = lower[(i, i)];
        if pivot == 0.0 {
            return Err(NumericalError::DecompositionFailed);
        }
        solution[i] = value / pivot;
    }

    Ok(solution)
}

pub fn quadratic_form_from_cholesky(
    lower: &Array2<f64>,
    vector: &[f64],
) -> Result<f64, NumericalError> {
    let whitened = solve_lower_triangular(lower, vector)?;
    Ok(whitened.iter().map(|value| value * value).sum())
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{cholesky, log_determinant_from_cholesky, quadratic_form_from_cholesky};

    #[test]
    fn cholesky_factorizes_spd_matrix() {
        let matrix = array![[1.0, 0.7], [0.7, 1.0]];
        let lower = cholesky(&matrix).expect("matrix should be SPD");

        let reconstructed = array![
            [lower[(0, 0)] * lower[(0, 0)], lower[(0, 0)] * lower[(1, 0)],],
            [
                lower[(1, 0)] * lower[(0, 0)],
                lower[(1, 0)] * lower[(1, 0)] + lower[(1, 1)] * lower[(1, 1)],
            ]
        ];

        for ((row, col), expected) in matrix.indexed_iter() {
            assert!((reconstructed[(row, col)] - expected).abs() < 1e-12);
        }

        let log_det = log_determinant_from_cholesky(&lower);
        assert!((log_det - (1.0_f64 - 0.49).ln()).abs() < 1e-12);

        let quad = quadratic_form_from_cholesky(&lower, &[1.0, -1.0]).expect("solve should work");
        assert!(quad > 0.0);
    }
}
