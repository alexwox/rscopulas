use ndarray::Array2;

use crate::data::PseudoObs;

pub fn kendall_tau_matrix(data: &PseudoObs) -> Array2<f64> {
    let dim = data.dim();
    let n_obs = data.n_obs();
    let view = data.as_view();
    let mut tau = Array2::eye(dim);

    for left in 0..dim {
        for right in (left + 1)..dim {
            let mut concordant = 0_i64;
            let mut discordant = 0_i64;

            for i in 0..n_obs {
                for j in (i + 1)..n_obs {
                    let left_diff = view[(i, left)] - view[(j, left)];
                    let right_diff = view[(i, right)] - view[(j, right)];
                    let product = left_diff * right_diff;

                    if product > 0.0 {
                        concordant += 1;
                    } else if product < 0.0 {
                        discordant += 1;
                    }
                }
            }

            let pairs = (n_obs * (n_obs - 1) / 2) as f64;
            let value = (concordant - discordant) as f64 / pairs;
            tau[(left, right)] = value;
            tau[(right, left)] = value;
        }
    }

    tau
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
