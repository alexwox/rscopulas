pub(crate) mod frank {
    use crate::errors::{CopulaError, FitError};

    pub(crate) fn generator(t: f64, theta: f64) -> f64 {
        let scale = 1.0 - (-theta).exp();
        -(1.0 - scale * (-t).exp()).ln() / theta
    }

    pub(crate) fn log_abs_generator_derivative(dim: usize, q: f64, theta: f64) -> f64 {
        let numerator_poly = derivative_polynomial(dim, q);
        q.ln() + numerator_poly.abs().ln() - (dim as f64) * (1.0 - q).ln() - theta.ln()
    }

    pub(crate) fn invert_tau(
        target_tau: f64,
        bracket_error: &'static str,
    ) -> Result<f64, CopulaError> {
        let mut low = 1e-6;
        let mut high = 1.0;
        while tau(high) < target_tau && high < 1e6 {
            high *= 2.0;
        }

        if tau(high) < target_tau {
            return Err(FitError::Failed {
                reason: bracket_error,
            }
            .into());
        }

        for _ in 0..80 {
            let mid = 0.5 * (low + high);
            if tau(mid) < target_tau {
                low = mid;
            } else {
                high = mid;
            }
        }

        Ok(0.5 * (low + high))
    }

    fn derivative_polynomial(dim: usize, q: f64) -> f64 {
        let mut coeffs = vec![1.0];
        for n in 1..dim {
            let mut deriv = vec![0.0; coeffs.len().saturating_sub(1)];
            for idx in 1..coeffs.len() {
                deriv[idx - 1] = idx as f64 * coeffs[idx];
            }

            let mut next = vec![0.0; coeffs.len() + 1];
            for (idx, coeff) in coeffs.iter().enumerate() {
                next[idx] += coeff;
                next[idx + 1] += (n as f64 - 1.0) * coeff;
            }
            for (idx, coeff) in deriv.iter().enumerate() {
                next[idx + 1] += coeff;
                next[idx + 2] -= coeff;
            }
            coeffs = next;
        }

        coeffs
            .iter()
            .enumerate()
            .map(|(idx, coeff)| coeff * q.powi(idx as i32))
            .sum()
    }

    fn debye_1(theta: f64) -> f64 {
        if theta.abs() < 1e-6 {
            return 1.0 - theta / 4.0 + theta * theta / 36.0;
        }

        let intervals = 1024usize;
        let step = theta / intervals as f64;
        let integrand = |x: f64| -> f64 { if x == 0.0 { 1.0 } else { x / x.exp_m1() } };

        let mut total = integrand(0.0) + integrand(theta);
        for idx in 1..intervals {
            let x = idx as f64 * step;
            total += if idx % 2 == 0 {
                2.0 * integrand(x)
            } else {
                4.0 * integrand(x)
            };
        }

        (step / 3.0) * total / theta
    }

    fn tau(theta: f64) -> f64 {
        1.0 - 4.0 / theta + 4.0 * debye_1(theta) / theta
    }
}

pub(crate) mod gumbel {
    pub(crate) fn log_abs_generator_derivative(dim: usize, t: f64, alpha: f64) -> f64 {
        let derivatives = (1..=dim)
            .map(|order| g_derivative(order, t, alpha))
            .collect::<Vec<_>>();
        let bell = complete_bell_polynomial(&derivatives);
        -t.powf(alpha) + bell.abs().ln()
    }

    fn g_derivative(order: usize, t: f64, alpha: f64) -> f64 {
        let falling = (0..order).fold(1.0, |acc, idx| acc * (alpha - idx as f64));
        -falling * t.powf(alpha - order as f64)
    }

    fn complete_bell_polynomial(derivatives: &[f64]) -> f64 {
        let n = derivatives.len();
        let mut bell = vec![0.0; n + 1];
        bell[0] = 1.0;

        for order in 1..=n {
            let mut total = 0.0;
            for k in 1..=order {
                total += binomial(order - 1, k - 1) * derivatives[k - 1] * bell[order - k];
            }
            bell[order] = total;
        }

        bell[n]
    }

    fn binomial(n: usize, k: usize) -> f64 {
        if k == 0 || k == n {
            return 1.0;
        }
        let k = k.min(n - k);
        let mut value = 1.0;
        for idx in 0..k {
            value *= (n - idx) as f64 / (idx + 1) as f64;
        }
        value
    }
}
