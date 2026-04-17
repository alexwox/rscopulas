use crate::errors::{CopulaError, FitError};

pub fn theta_from_tau(tau: f64) -> Result<f64, CopulaError> {
    if !tau.is_finite() || tau <= 0.0 || tau >= 1.0 {
        return Err(FitError::Failed {
            reason: "gumbel pair fit requires tau in (0, 1)",
        }
        .into());
    }
    Ok(1.0 / (1.0 - tau))
}

pub fn log_pdf(u1: f64, u2: f64, theta: f64) -> Result<f64, CopulaError> {
    if !theta.is_finite() || theta < 1.0 {
        return Err(FitError::Failed {
            reason: "gumbel pair theta must be at least 1",
        }
        .into());
    }
    let alpha = 1.0 / theta;
    let t = (-u1.ln()).powf(theta) + (-u2.ln()).powf(theta);
    let log_abs_derivative = gumbel_log_abs_generator_derivative(2, t, alpha);
    let log_phi = theta.ln()
        + (theta - 1.0) * (-u1.ln()).ln()
        - u1.ln()
        + theta.ln()
        + (theta - 1.0) * (-u2.ln()).ln()
        - u2.ln();
    Ok(log_abs_derivative + log_phi)
}

pub fn cond_first_given_second(
    u1: f64,
    u2: f64,
    theta: f64,
    _clip_eps: f64,
) -> Result<f64, CopulaError> {
    let x = (-u1.ln()).powf(theta);
    let y = (-u2.ln()).powf(theta);
    let t = (x + y).powf(1.0 / theta);
    Ok((-t).exp() * (x + y).powf(1.0 / theta - 1.0) * (-u2.ln()).powf(theta - 1.0) / u2)
}

pub fn cond_second_given_first(
    u1: f64,
    u2: f64,
    theta: f64,
    _clip_eps: f64,
) -> Result<f64, CopulaError> {
    let x = (-u1.ln()).powf(theta);
    let y = (-u2.ln()).powf(theta);
    let t = (x + y).powf(1.0 / theta);
    Ok((-t).exp() * (x + y).powf(1.0 / theta - 1.0) * (-u1.ln()).powf(theta - 1.0) / u1)
}

pub fn inv_first_given_second(
    p: f64,
    u2: f64,
    theta: f64,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let mut low = clip_eps;
    let mut high = 1.0 - clip_eps;
    for _ in 0..90 {
        let mid = 0.5 * (low + high);
        if cond_first_given_second(mid, u2, theta, clip_eps)? < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(0.5 * (low + high))
}

pub fn inv_second_given_first(
    u1: f64,
    p: f64,
    theta: f64,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let mut low = clip_eps;
    let mut high = 1.0 - clip_eps;
    for _ in 0..90 {
        let mid = 0.5 * (low + high);
        if cond_second_given_first(u1, mid, theta, clip_eps)? < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(0.5 * (low + high))
}

fn gumbel_log_abs_generator_derivative(dim: usize, t: f64, alpha: f64) -> f64 {
    let derivatives = (1..=dim)
        .map(|order| gumbel_g_derivative(order, t, alpha))
        .collect::<Vec<_>>();
    let bell = complete_bell_polynomial(&derivatives);
    -t.powf(alpha) + bell.abs().ln()
}

fn gumbel_g_derivative(order: usize, t: f64, alpha: f64) -> f64 {
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
