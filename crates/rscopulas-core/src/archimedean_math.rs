pub(crate) mod frank {
    use crate::errors::{CopulaError, FitError};

    pub(crate) fn generator(t: f64, theta: f64) -> f64 {
        generator_from_log_argument(t.ln(), theta)
    }

    pub(crate) fn generator_from_log_argument(log_t: f64, theta: f64) -> f64 {
        if theta < 1e-10 {
            return (-log_t.exp()).exp();
        }
        let log_complement = crate::math::logaddexp(
            -theta,
            crate::math::log1mexp(-theta) + crate::math::log1mexp_neg_exp(log_t),
        );
        (-log_complement / theta).clamp(1e-12, 1.0 - 1e-12)
    }

    /// Logarithmic-series frailty via the constant-time Kemp mixture.
    /// Keeping log(N) avoids integer overflow and exp(theta) overflow.
    /// Reference: Kemp (1981), Applied Statistics 30(3), 249-253.
    pub(crate) fn sample_log_frailty<R: rand::Rng + ?Sized>(rng: &mut R, theta: f64) -> f64 {
        use rand::distr::Open01;
        let selector: f64 = rng.sample(Open01);
        if selector > -(-theta).exp_m1() {
            return 0.0;
        }
        let mixture: f64 = rng.sample(Open01);
        let exponent = -theta * mixture;
        let log_q = crate::math::log1mexp(exponent);
        if selector.ln() < 2.0 * log_q {
            let log_ratio = (-selector.ln()).ln() - crate::math::log_neg_log1mexp(exponent);
            if log_ratio > 36.0 {
                log_ratio
            } else {
                (log_ratio.exp().floor() + 1.0).ln()
            }
        } else if selector.ln() > log_q {
            0.0
        } else {
            2.0_f64.ln()
        }
    }

    /// `coefficients` must come from [`derivative_coefficients`] for `dim`.
    pub(crate) fn log_abs_generator_derivative(
        dim: usize,
        coefficients: &[f64],
        log_q: f64,
        theta: f64,
    ) -> f64 {
        let q = log_q.exp();
        let numerator_poly: f64 = coefficients
            .iter()
            .enumerate()
            .map(|(idx, coeff)| coeff * q.powi(idx as i32))
            .sum();
        log_q + numerator_poly.abs().ln() - (dim as f64) * crate::math::log1mexp(log_q) - theta.ln()
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

    /// Coefficients (ascending powers of `q`) of the polynomial in the
    /// numerator of the `dim`-th derivative of the Frank generator. They
    /// depend on the dimension only, so a density pass builds them once.
    pub(crate) fn derivative_coefficients(dim: usize) -> Vec<f64> {
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
    }

    /// First Debye function `D₁(θ) = (1/θ) ∫₀^θ x / (eˣ − 1) dx`.
    ///
    /// The integrand is analytic with its nearest singularities at `±2πi`,
    /// so a 64-point Gauss–Legendre rule on `[0, min(θ, 40)]` is exact to
    /// double precision; beyond 40 the integrand is below `2e-16` and the
    /// remainder is dropped.
    fn debye_1(theta: f64) -> f64 {
        use std::sync::OnceLock;
        if theta.abs() < 1e-6 {
            return 1.0 - theta / 4.0 + theta * theta / 36.0;
        }
        static RULE: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
        let (nodes, weights) = RULE.get_or_init(|| crate::math::gauss_legendre_01(64));
        let upper = theta.min(40.0);
        let mut total = 0.0;
        for (node, weight) in nodes.iter().zip(weights) {
            let x = upper * node;
            total += weight * x / x.exp_m1();
        }
        upper * total / theta
    }

    /// Composite Simpson evaluation of `D₁`, the rule used before the
    /// Gauss–Legendre one; kept as an independent check.
    #[cfg(test)]
    fn debye_1_simpson(theta: f64) -> f64 {
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

    #[cfg(test)]
    mod tests {
        #[test]
        fn debye_gauss_legendre_matches_simpson_and_the_large_theta_limit() {
            for theta in [1e-3, 0.1, 1.0, 4.5, 12.0, 30.0] {
                let (fast, reference) = (super::debye_1(theta), super::debye_1_simpson(theta));
                // Simpson with 1024 panels is itself good to ~1e-10 at θ = 30.
                assert!(
                    (fast - reference).abs() < 2e-10,
                    "theta {theta}: {fast} vs {reference}"
                );
            }
            // D₁(θ) → π²/(6θ) as θ → ∞.
            for theta in [60.0, 1e3, 1e6] {
                let limit = std::f64::consts::PI.powi(2) / (6.0 * theta);
                let value = super::debye_1(theta);
                assert!(
                    (value - limit).abs() < 1e-14 * limit,
                    "theta {theta}: {value}"
                );
            }
        }
    }
}

pub(crate) mod joe {
    use crate::errors::{CopulaError, FitError};

    use statrs::function::gamma::digamma;

    /// First four derivatives of the digamma function at 2:
    /// `π²/6 − 1`, `2 − 2ζ(3)`, `π⁴/15 − 6`, `24 − 24ζ(5)`.
    const POLYGAMMA_AT_2: [f64; 4] = [
        0.644_934_066_848_226_4,
        -0.404_113_806_319_188_6,
        0.493_939_402_266_829_1,
        -0.886_266_123_440_878_3,
    ];

    /// Kendall's tau of the Joe copula in closed form,
    ///
    /// ```text
    /// τ(θ) = 1 + 2 [ψ(2) − ψ(2/θ + 1)] / (2 − θ),
    /// ```
    ///
    /// the sum of the series `1 − 4 Σ_k 1 / [k (θk + 2)(θ(k−1) + 2)]`. Near
    /// the removable singularity at `θ = 2` the difference quotient is
    /// replaced by its Taylor expansion in `x = 2/θ − 1`. At `θ = 1` Joe
    /// reduces to independence and `τ = 0`.
    ///
    /// Accuracy is about `1e-9` (limited by the digamma implementation close
    /// to `θ = 2`), ample for its use as the starting point of a likelihood
    /// search.
    pub(crate) fn tau(theta: f64) -> f64 {
        if theta <= 1.0 {
            return 0.0;
        }
        let gap = 2.0 - theta;
        if gap.abs() < 1e-2 {
            let x = gap / theta;
            let [d1, d2, d3, d4] = POLYGAMMA_AT_2;
            return 1.0 - 2.0 / theta * (d1 + x * (d2 / 2.0 + x * (d3 / 6.0 + x * d4 / 24.0)));
        }
        1.0 + 2.0 * (digamma(2.0) - digamma(2.0 / theta + 1.0)) / gap
    }

    /// The series the closed form sums, kept as an independent check.
    #[cfg(test)]
    fn tau_series(theta: f64) -> f64 {
        let mut sum = 0.0;
        for k in 1..2_000_000 {
            let kf = k as f64;
            sum += 1.0 / (kf * (theta * kf + 2.0) * (theta * (kf - 1.0) + 2.0));
        }
        1.0 - 4.0 * sum
    }

    pub(crate) fn invert_tau(
        target_tau: f64,
        bracket_error: &'static str,
    ) -> Result<f64, CopulaError> {
        let mut low = 1.0;
        let mut high = 30.0;
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

    #[cfg(test)]
    mod tests {
        #[test]
        fn closed_form_tau_matches_the_series() {
            for theta in [
                1.0001, 1.2, 1.7, 1.985, 1.9905, 1.9995, 2.0, 2.0005, 2.0095, 2.015, 2.4, 3.0, 8.0,
                30.0, 400.0,
            ] {
                let closed = super::tau(theta);
                let series = super::tau_series(theta);
                assert!(
                    (closed - series).abs() < 2e-9,
                    "theta {theta}: {closed} vs {series}"
                );
            }
            assert_eq!(super::tau(1.0), 0.0);
        }
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
