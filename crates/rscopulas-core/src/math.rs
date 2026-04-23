use ndarray::Array2;

use crate::errors::NumericalError;

pub fn identity_correlation(dim: usize) -> Array2<f64> {
    Array2::eye(dim)
}

pub fn make_spd_correlation(matrix: &Array2<f64>) -> Result<Array2<f64>, NumericalError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(NumericalError::InvalidCorrelationMatrix);
    }

    let dim = matrix.nrows();
    let mut sym = Array2::zeros((dim, dim));
    for row in 0..dim {
        sym[(row, row)] = 1.0;
        for col in (row + 1)..dim {
            let value = 0.5 * (matrix[(row, col)] + matrix[(col, row)]);
            sym[(row, col)] = value;
            sym[(col, row)] = value;
        }
    }

    if validate_correlation_matrix(&sym).is_ok() {
        return Ok(sym);
    }

    for step in 0..500 {
        let shrink = 1.0 - (step as f64 + 1.0) / 600.0;
        let mut candidate = Array2::eye(dim);
        for row in 0..dim {
            for col in (row + 1)..dim {
                let value = sym[(row, col)] * shrink;
                candidate[(row, col)] = value;
                candidate[(col, row)] = value;
            }
        }

        if validate_correlation_matrix(&candidate).is_ok() {
            return Ok(candidate);
        }
    }

    Err(NumericalError::InvalidCorrelationMatrix)
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

pub fn inverse(matrix: &Array2<f64>) -> Result<Array2<f64>, NumericalError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(NumericalError::InvalidCorrelationMatrix);
    }

    let dim = matrix.nrows();
    let mut augmented = Array2::zeros((dim, 2 * dim));
    for row in 0..dim {
        for col in 0..dim {
            augmented[(row, col)] = matrix[(row, col)];
        }
        augmented[(row, dim + row)] = 1.0;
    }

    for pivot in 0..dim {
        let mut max_row = pivot;
        let mut max_value = augmented[(pivot, pivot)].abs();
        for row in (pivot + 1)..dim {
            let value = augmented[(row, pivot)].abs();
            if value > max_value {
                max_value = value;
                max_row = row;
            }
        }

        if max_value < 1e-14 {
            return Err(NumericalError::DecompositionFailed);
        }

        if max_row != pivot {
            for col in 0..(2 * dim) {
                let tmp = augmented[(pivot, col)];
                augmented[(pivot, col)] = augmented[(max_row, col)];
                augmented[(max_row, col)] = tmp;
            }
        }

        let pivot_value = augmented[(pivot, pivot)];
        for col in 0..(2 * dim) {
            augmented[(pivot, col)] /= pivot_value;
        }

        for row in 0..dim {
            if row == pivot {
                continue;
            }
            let factor = augmented[(row, pivot)];
            for col in 0..(2 * dim) {
                augmented[(row, col)] -= factor * augmented[(pivot, col)];
            }
        }
    }

    let mut result = Array2::zeros((dim, dim));
    for row in 0..dim {
        for col in 0..dim {
            result[(row, col)] = augmented[(row, dim + col)];
        }
    }

    Ok(result)
}

/// Returns the (nodes, weights) for an `n`-point Gauss–Legendre quadrature
/// rule on `[0, 1]`.
///
/// The rule is derived on the reference interval `[-1, 1]` and linearly
/// rescaled; weights absorb the Jacobian (factor of 1/2). The reference-
/// interval roots are found by Newton iteration on the `n`-th Legendre
/// polynomial, started from Tricomi's asymptotic approximation.
///
/// This is the primitive used by the factor-copula log-density, which
/// integrates the product of link densities against the latent factor's
/// uniform distribution on `[0, 1]`. Twenty-five nodes match Joe's
/// `CopulaModel` R package default and are accurate to ~1e-14 for the
/// integrands that arise from well-behaved link families.
pub fn gauss_legendre_01(n: usize) -> (Vec<f64>, Vec<f64>) {
    debug_assert!(n >= 1, "gauss_legendre_01 requires at least one node");

    let nf = n as f64;
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];

    // Legendre polynomial roots are symmetric about 0; we compute the (n+1)/2
    // non-positive ones and mirror.
    let half = n.div_ceil(2);
    for i in 0..half {
        // Tricomi's initial guess for the i-th (1-indexed) root on [-1, 1].
        let theta = std::f64::consts::PI * ((i + 1) as f64 - 0.25) / (nf + 0.5);
        let mut x = (1.0 - (nf - 1.0) / (8.0 * nf.powi(3))) * theta.cos();

        // Newton iteration: x ← x - P_n(x) / P_n'(x).
        let mut pp;
        loop {
            let (p, p_prime) = legendre_p_and_p_prime(n, x);
            pp = p_prime;
            let dx = p / p_prime;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }

        // Map [-1, 1] → [0, 1]: node' = (1 - x)/2, weight' = 1 / ((1 - x²) P_n'(x)²)
        // (the standard weight 2 / ((1-x²) P'²) picks up a factor of 1/2 from the
        // Jacobian of the linear map). We emit ascending nodes in [0, 1], so the
        // root closest to -1 (Tricomi's first guess) lands at the smallest x₀₁.
        let node_01 = 0.5 * (1.0 - x);
        let weight_01 = 1.0 / ((1.0 - x * x) * pp * pp);

        nodes[i] = node_01;
        weights[i] = weight_01;

        // Mirror root at -x (which maps to 1 - node_01); this is the symmetric
        // partner in [0, 1]. Skip if n is odd and x == 0 (the middle root).
        let mirror = n - 1 - i;
        if mirror != i {
            nodes[mirror] = 1.0 - node_01;
            weights[mirror] = weight_01;
        }
    }

    (nodes, weights)
}

/// Evaluates `(P_n(x), P_n'(x))` via the three-term recurrence. Used by the
/// Gauss–Legendre root finder above.
fn legendre_p_and_p_prime(n: usize, x: f64) -> (f64, f64) {
    // Recurrence: P_0 = 1, P_1 = x, k P_k = (2k-1) x P_{k-1} - (k-1) P_{k-2}.
    let mut p_prev = 1.0;
    let mut p_cur = x;
    if n == 0 {
        return (1.0, 0.0);
    }
    for k in 2..=n {
        let k_f = k as f64;
        let p_next = ((2.0 * k_f - 1.0) * x * p_cur - (k_f - 1.0) * p_prev) / k_f;
        p_prev = p_cur;
        p_cur = p_next;
    }
    // P_n'(x) = n (x P_n - P_{n-1}) / (x² - 1) away from endpoints.
    let denom = x * x - 1.0;
    let p_prime = n as f64 * (x * p_cur - p_prev) / denom;
    (p_cur, p_prime)
}

pub fn maximize_scalar<F>(mut low: f64, mut high: f64, iterations: usize, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let resphi = 2.0 - phi;
    let mut x1 = low + resphi * (high - low);
    let mut x2 = high - resphi * (high - low);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    for _ in 0..iterations {
        if f1 < f2 {
            low = x1;
            x1 = x2;
            f1 = f2;
            x2 = high - resphi * (high - low);
            f2 = f(x2);
        } else {
            high = x2;
            x2 = x1;
            f2 = f1;
            x1 = low + resphi * (high - low);
            f1 = f(x1);
        }
    }

    if f1 > f2 { x1 } else { x2 }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{
        cholesky, gauss_legendre_01, log_determinant_from_cholesky, quadratic_form_from_cholesky,
    };

    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        // An n-point rule is exact for polynomials of degree up to 2n-1.
        // Test ∫₀¹ x^k dx = 1/(k+1) for a few k using 25 nodes.
        let (nodes, weights) = gauss_legendre_01(25);
        assert_eq!(nodes.len(), 25);
        assert_eq!(weights.len(), 25);

        // Weights should sum to the interval length (1).
        let w_sum: f64 = weights.iter().sum();
        assert!((w_sum - 1.0).abs() < 1e-14, "weight sum was {w_sum}");

        for k in 0usize..10 {
            let approx: f64 = nodes
                .iter()
                .zip(weights.iter())
                .map(|(x, w)| w * x.powi(k as i32))
                .sum();
            let exact = 1.0 / (k as f64 + 1.0);
            assert!(
                (approx - exact).abs() < 1e-13,
                "k={k}: approx={approx}, exact={exact}"
            );
        }
    }

    #[test]
    fn gauss_legendre_nodes_strictly_inside_unit_interval() {
        let (nodes, weights) = gauss_legendre_01(25);
        for (x, w) in nodes.iter().zip(weights.iter()) {
            assert!(*x > 0.0 && *x < 1.0, "node {x} escaped (0, 1)");
            assert!(*w > 0.0, "weight {w} not strictly positive");
        }
        // Nodes are ascending.
        for pair in nodes.windows(2) {
            assert!(pair[0] < pair[1], "nodes not ascending: {pair:?}");
        }
    }

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
