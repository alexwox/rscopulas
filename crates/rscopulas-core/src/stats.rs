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

/// Spearman's rank correlation coefficient — the Pearson correlation of the
/// mid-rank-transformed inputs. Returns a value in `[-1, 1]`.
pub fn spearman_rho_bivariate(x: &[f64], y: &[f64]) -> Result<f64, CopulaError> {
    if x.len() != y.len() || x.len() < 2 {
        return Err(FitError::Failed {
            reason: "spearman rho requires equally sized inputs with at least two observations",
        }
        .into());
    }
    let rx = midranks(x);
    let ry = midranks(y);
    Ok(pearson_correlation(&rx, &ry))
}

/// Pairwise Spearman's ρ matrix on pseudo-observation columns. Uses the
/// crate's backend to pick CPU-serial vs CPU-parallel execution depending on
/// dimension (threshold matches the Hoeffding matrix — both are per-pair
/// O(n log n)).
pub fn try_spearman_rho_matrix(
    data: &PseudoObs,
    exec: ExecPolicy,
) -> Result<Array2<f64>, CopulaError> {
    let dim = data.dim();
    let view = data.as_view();
    let mut rho = Array2::eye(dim);
    let strategy = resolve_strategy(
        exec,
        Operation::SpearmanRhoMatrix,
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
            let value = spearman_rho_bivariate(&columns[left], &columns[right])?;
            Ok((left, right, value))
        },
    )?;
    for (left, right, value) in pairs {
        rho[(left, right)] = value;
        rho[(right, left)] = value;
    }
    Ok(rho)
}

/// Hoeffding's D statistic — a rank-based measure of bivariate dependence
/// that, unlike Kendall's τ or Spearman's ρ, detects non-monotone
/// relationships. Follows Hollander & Wolfe 1973 §8:
///
/// ```text
///     D = 30 · [(n − 2)(n − 3) · D1 + D2 − 2(n − 2) · D3]
///              / [n(n − 1)(n − 2)(n − 3)(n − 4)]
/// ```
///
/// with
/// * `D1 = Σ (Q_i − 1)(Q_i − 2)`
/// * `D2 = Σ (R_i − 1)(R_i − 2)(S_i − 1)(S_i − 2)`
/// * `D3 = Σ (R_i − 2)(S_i − 2)(Q_i − 1)`
///
/// where `R_i`, `S_i` are mid-ranks in `x` and `y`, and `Q_i` is the bivariate
/// mid-rank — the number of observations with both `x_j < x_i` and
/// `y_j < y_i`, plus half the count of ties in exactly one coordinate
/// (Hmisc convention). Output range is `[-0.5, 1]` under the Hmisc / `wdm`
/// scaling; no-ties inputs land in `[0, 1]`.
pub fn hoeffding_d_bivariate(x: &[f64], y: &[f64]) -> Result<f64, CopulaError> {
    let n = x.len();
    if x.len() != y.len() || n < 5 {
        return Err(FitError::Failed {
            reason: "hoeffding D requires equally sized inputs with at least five observations",
        }
        .into());
    }
    let r = midranks(x);
    let s = midranks(y);
    let q = bivariate_midranks(x, y);

    let mut d1 = 0.0_f64;
    let mut d2 = 0.0_f64;
    let mut d3 = 0.0_f64;
    for i in 0..n {
        d1 += (q[i] - 1.0) * (q[i] - 2.0);
        d2 += (r[i] - 1.0) * (r[i] - 2.0) * (s[i] - 1.0) * (s[i] - 2.0);
        d3 += (r[i] - 2.0) * (s[i] - 2.0) * (q[i] - 1.0);
    }

    let n_f = n as f64;
    let denom = n_f * (n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0) * (n_f - 4.0);
    Ok(30.0 * ((n_f - 2.0) * (n_f - 3.0) * d1 + d2 - 2.0 * (n_f - 2.0) * d3) / denom)
}

/// Pairwise Hoeffding's D matrix across pseudo-observation columns.
pub fn try_hoeffding_d_matrix(
    data: &PseudoObs,
    exec: ExecPolicy,
) -> Result<Array2<f64>, CopulaError> {
    let dim = data.dim();
    let view = data.as_view();
    let mut out = Array2::eye(dim);
    let strategy = resolve_strategy(
        exec,
        Operation::HoeffdingDMatrix,
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
            let value = hoeffding_d_bivariate(&columns[left], &columns[right])?;
            Ok((left, right, value))
        },
    )?;
    for (left, right, value) in pairs {
        out[(left, right)] = value;
        out[(right, left)] = value;
    }
    Ok(out)
}

/// Computes the mid-rank of each element in `values`. Ties receive the
/// average of the ranks they would otherwise span. Output indexed the same
/// as `values` (not sorted).
fn midranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && indexed[j].1 == indexed[i].1 {
            j += 1;
        }
        // Tie-group [i, j). 1-based rank average = (i + 1 + j) / 2.
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Bivariate mid-rank `Q_i`: the number of observations `j` with
/// `x_j < x_i` AND `y_j < y_i`, plus ½·(count of ties in exactly one
/// coordinate) plus ¼·(count of ties in both). Defined on 1-based ranks to
/// match Hoeffding's original formulation.
fn bivariate_midranks(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut q = vec![1.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let x_lt = x[j] < x[i];
            let x_eq = x[j] == x[i];
            let y_lt = y[j] < y[i];
            let y_eq = y[j] == y[i];
            if x_lt && y_lt {
                q[i] += 1.0;
            } else if (x_lt && y_eq) || (x_eq && y_lt) {
                q[i] += 0.5;
            } else if x_eq && y_eq {
                q[i] += 0.25;
            }
        }
    }
    q
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut cov = 0.0_f64;
    let mut var_x = 0.0_f64;
    let mut var_y = 0.0_f64;
    for (a, b) in x.iter().zip(y.iter()) {
        let dx = a - mean_x;
        let dy = b - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 { 0.0 } else { cov / denom }
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

    use super::{
        hoeffding_d_bivariate, kendall_tau_matrix, spearman_rho_bivariate,
        try_hoeffding_d_matrix, try_spearman_rho_matrix,
    };

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

    #[test]
    fn spearman_rho_is_one_under_monotone_transform() {
        // rho is invariant under any strictly increasing marginal transform —
        // ranks are identical, so Pearson on ranks is 1.
        let x = [0.1_f64, 0.25, 0.4, 0.55, 0.7, 0.85];
        let y: Vec<f64> = x.iter().map(|v| v.powi(3)).collect();
        let rho = spearman_rho_bivariate(&x, &y).unwrap();
        assert!((rho - 1.0).abs() < 1e-12, "rho = {rho}");
    }

    #[test]
    fn spearman_rho_matches_scipy_reference() {
        // SciPy reference (scipy.stats.spearmanr on the inputs below) rounds
        // to -0.6; verify we match to well under 1e-6.
        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0_f64, 6.0, 7.0, 8.0, 7.0];
        // spearmanr((1,2,3,4,5), (5,6,7,8,7)) → rho ≈ 0.82078
        let rho = spearman_rho_bivariate(&x, &y).unwrap();
        assert!((rho - 0.82078268166812329).abs() < 1e-10, "rho = {rho}");
    }

    #[test]
    fn hoeffding_d_detects_u_shape() {
        // Hoeffding's D picks up non-monotone dependence where Spearman rho
        // returns ~0. Construct a noise-free U-shape: y = (x - 0.5)^2.
        let x: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
        let y: Vec<f64> = x.iter().map(|u| (u - 0.5).powi(2)).collect();
        let rho = spearman_rho_bivariate(&x, &y).unwrap();
        assert!(rho.abs() < 0.2, "expected weak Spearman under U-shape, got {rho}");
        let d = hoeffding_d_bivariate(&x, &y).unwrap();
        assert!(d > 0.1, "Hoeffding D should flag U-shape, got {d}");
    }

    #[test]
    fn hoeffding_d_is_zero_under_independence() {
        // Random uniform data → Hoeffding D ≈ 0 (within MC noise at n=200).
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(7);
        let x: Vec<f64> = (0..200).map(|_| rng.random::<f64>()).collect();
        let y: Vec<f64> = (0..200).map(|_| rng.random::<f64>()).collect();
        let d = hoeffding_d_bivariate(&x, &y).unwrap();
        assert!(d.abs() < 0.05, "independence Hoeffding D = {d}");
    }

    #[test]
    fn matrix_helpers_produce_symmetric_eye_diagonal_output() {
        let data = PseudoObs::new(array![
            [0.1, 0.2, 0.3],
            [0.2, 0.25, 0.35],
            [0.3, 0.45, 0.55],
            [0.4, 0.6, 0.65],
            [0.55, 0.75, 0.85],
            [0.7, 0.9, 0.95],
        ])
        .unwrap();
        let rho = try_spearman_rho_matrix(&data, crate::domain::ExecPolicy::Auto).unwrap();
        let d = try_hoeffding_d_matrix(&data, crate::domain::ExecPolicy::Auto).unwrap();
        for k in 0..3 {
            assert!((rho[(k, k)] - 1.0).abs() < 1e-12);
            assert!((d[(k, k)] - 1.0).abs() < 1e-12);
        }
        for i in 0..3 {
            for j in 0..3 {
                assert!((rho[(i, j)] - rho[(j, i)]).abs() < 1e-12);
                assert!((d[(i, j)] - d[(j, i)]).abs() < 1e-12);
            }
        }
    }
}
