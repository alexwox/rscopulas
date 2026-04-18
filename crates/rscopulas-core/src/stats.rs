use ndarray::Array2;

use crate::{
    backend::{Operation, resolve_strategy},
    data::PseudoObs,
    domain::ExecPolicy,
    errors::{CopulaError, FitError},
};

pub fn kendall_tau_matrix(data: &PseudoObs) -> Array2<f64> {
    try_kendall_tau_matrix(data, ExecPolicy::Auto)
        .expect("pseudo-observations should yield valid Kendall tau")
}

pub fn try_kendall_tau_matrix(
    data: &PseudoObs,
    exec: ExecPolicy,
) -> Result<Array2<f64>, CopulaError> {
    let dim = data.dim();
    let view = data.as_view();
    let mut tau = Array2::eye(dim);
    let strategy = resolve_strategy(
        exec,
        Operation::KendallTauMatrix,
        dim.saturating_mul(dim - 1) / 2,
    )?;
    let columns = (0..dim)
        .map(|idx| view.column(idx).iter().copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let pairs = crate::backend::parallel_try_map_range_collect(
        dim.saturating_mul(dim - 1) / 2,
        strategy,
        |pair_idx| {
            let (left, right) = decode_upper_triangle_index(pair_idx, dim);
            let value = kendall_tau_bivariate(&columns[left], &columns[right])?;
            Ok((left, right, value))
        },
    )?;

    for (left, right, value) in pairs {
        tau[(left, right)] = value;
        tau[(right, left)] = value;
    }

    Ok(tau)
}

pub fn kendall_tau_bivariate(x: &[f64], y: &[f64]) -> Result<f64, CopulaError> {
    if x.len() != y.len() || x.len() < 2 {
        return Err(FitError::Failed {
            reason: "kendall tau requires equally sized inputs with at least two observations",
        }
        .into());
    }

    let mut pairs = x
        .iter()
        .copied()
        .zip(y.iter().copied())
        .collect::<Vec<(f64, f64)>>();
    pairs.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.total_cmp(&right.1))
    });

    let mut tied_x = 0usize;
    let mut tied_xy = 0usize;
    let mut run_x = 1usize;
    let mut run_xy = 1usize;
    for idx in 1..pairs.len() {
        if pairs[idx].0 == pairs[idx - 1].0 {
            run_x += 1;
            if pairs[idx].1 == pairs[idx - 1].1 {
                run_xy += 1;
            } else if run_xy > 1 {
                tied_xy += run_xy * (run_xy - 1) / 2;
                run_xy = 1;
            }
        } else {
            if run_x > 1 {
                tied_x += run_x * (run_x - 1) / 2;
            }
            if run_xy > 1 {
                tied_xy += run_xy * (run_xy - 1) / 2;
            }
            run_x = 1;
            run_xy = 1;
        }
    }
    if run_x > 1 {
        tied_x += run_x * (run_x - 1) / 2;
    }
    if run_xy > 1 {
        tied_xy += run_xy * (run_xy - 1) / 2;
    }

    let mut scratch = pairs.clone();
    let inversions = count_inversions_by_y(&mut pairs, &mut scratch);
    let mut tied_y = 0usize;
    let mut run_y = 1usize;
    for idx in 1..pairs.len() {
        if pairs[idx].1 == pairs[idx - 1].1 {
            run_y += 1;
        } else if run_y > 1 {
            tied_y += run_y * (run_y - 1) / 2;
            run_y = 1;
        }
    }
    if run_y > 1 {
        tied_y += run_y * (run_y - 1) / 2;
    }

    let n = x.len() as f64;
    let pairs_total = n * (n - 1.0) / 2.0;
    let score =
        pairs_total - (2.0 * inversions as f64 + tied_x as f64 + tied_y as f64 - tied_xy as f64);
    let denom = ((pairs_total - tied_x as f64) * (pairs_total - tied_y as f64)).sqrt();
    if denom == 0.0 {
        Ok(0.0)
    } else {
        Ok(score / denom)
    }
}

fn decode_upper_triangle_index(mut index: usize, dim: usize) -> (usize, usize) {
    let mut left = 0usize;
    while left + 1 < dim {
        let remaining = dim - left - 1;
        if index < remaining {
            return (left, left + 1 + index);
        }
        index -= remaining;
        left += 1;
    }
    unreachable!("pair index should stay inside the strict upper triangle");
}

fn count_inversions_by_y(values: &mut [(f64, f64)], scratch: &mut [(f64, f64)]) -> usize {
    let len = values.len();
    if len <= 1 {
        return 0;
    }

    let mid = len / 2;
    let (left, right) = values.split_at_mut(mid);
    let (scratch_left, scratch_right) = scratch.split_at_mut(mid);
    let mut inversions =
        count_inversions_by_y(left, scratch_left) + count_inversions_by_y(right, scratch_right);

    let mut left_idx = 0usize;
    let mut right_idx = 0usize;
    let mut out_idx = 0usize;
    while left_idx < left.len() && right_idx < right.len() {
        if left[left_idx].1 <= right[right_idx].1 {
            scratch[out_idx] = left[left_idx];
            left_idx += 1;
        } else {
            scratch[out_idx] = right[right_idx];
            inversions += left.len() - left_idx;
            right_idx += 1;
        }
        out_idx += 1;
    }

    while left_idx < left.len() {
        scratch[out_idx] = left[left_idx];
        left_idx += 1;
        out_idx += 1;
    }
    while right_idx < right.len() {
        scratch[out_idx] = right[right_idx];
        right_idx += 1;
        out_idx += 1;
    }
    values.copy_from_slice(&scratch[..len]);
    inversions
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
