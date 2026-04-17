use ndarray::Array2;

use crate::{
    data::PseudoObs,
    errors::{CopulaError, FitError},
};

pub fn kendall_tau_matrix(data: &PseudoObs) -> Array2<f64> {
    let dim = data.dim();
    let view = data.as_view();
    let mut tau = Array2::eye(dim);

    for left in 0..dim {
        for right in (left + 1)..dim {
            let left_values = view.column(left).iter().copied().collect::<Vec<_>>();
            let right_values = view.column(right).iter().copied().collect::<Vec<_>>();
            let value = kendall_tau_bivariate(&left_values, &right_values)
                .expect("pseudo-observations should yield valid Kendall tau");
            tau[(left, right)] = value;
            tau[(right, left)] = value;
        }
    }

    tau
}

pub fn kendall_tau_bivariate(x: &[f64], y: &[f64]) -> Result<f64, CopulaError> {
    if x.len() != y.len() || x.len() < 2 {
        return Err(FitError::Failed {
            reason: "kendall tau requires equally sized inputs with at least two observations",
        }
        .into());
    }

    let mut pairs = x.iter().copied().zip(y.iter().copied()).collect::<Vec<_>>();
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut ys = pairs.iter().map(|(_, y)| *y).collect::<Vec<_>>();
    let mut sorted = ys.clone();
    sorted.sort_by(|left, right| left.total_cmp(right));
    sorted.dedup_by(|left, right| (*left - *right).abs() < 1e-15);

    for value in &mut ys {
        let rank = sorted
            .binary_search_by(|probe| probe.total_cmp(value))
            .map_err(|_| FitError::Failed {
                reason: "kendall tau rank compression failed",
            })?;
        *value = (rank + 1) as f64;
    }

    let inversions = count_inversions(&ys.iter().map(|value| *value as usize).collect::<Vec<_>>());
    let n = x.len() as f64;
    let pairs_total = n * (n - 1.0) / 2.0;
    Ok(1.0 - 2.0 * inversions as f64 / pairs_total)
}

fn count_inversions(values: &[usize]) -> usize {
    let mut fenwick = vec![0usize; values.iter().copied().max().unwrap_or(0) + 2];
    let mut inversions = 0usize;
    for (seen, &value) in values.iter().enumerate() {
        let less_or_equal = fenwick_sum(&fenwick, value);
        inversions += seen - less_or_equal;
        fenwick_add(&mut fenwick, value, 1);
    }

    inversions
}

fn fenwick_add(tree: &mut [usize], mut idx: usize, delta: usize) {
    while idx < tree.len() {
        tree[idx] += delta;
        idx += idx & (!idx + 1);
    }
}

fn fenwick_sum(tree: &[usize], mut idx: usize) -> usize {
    let mut total = 0usize;
    while idx > 0 {
        total += tree[idx];
        idx &= idx - 1;
    }
    total
}

pub fn mean_off_diagonal(matrix: &Array2<f64>) -> f64 {
    let dim = matrix.nrows();
    let mut total = 0.0;
    let mut count = 0usize;

    for row in 0..dim {
        for col in (row + 1)..dim {
            total += matrix[(row, col)];
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::PseudoObs;

    use super::kendall_tau_matrix;

    #[test]
    fn kendall_tau_detects_positive_dependence() {
        let data = PseudoObs::new(array![
            [0.1, 0.2],
            [0.2, 0.25],
            [0.3, 0.4],
            [0.4, 0.6],
            [0.5, 0.8],
        ])
        .expect("pseudo-observations should be valid");

        let tau = kendall_tau_matrix(&data);
        assert!((tau[(0, 1)] - 1.0).abs() < 1e-12);
    }
}
