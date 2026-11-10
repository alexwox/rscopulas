//! Deterministic input data for the golden-snapshot tests and the perf suite.
//!
//! Nothing in here calls into `rscopulas`: the generator is a SplitMix64
//! stream pushed through closed-form transforms and a rank transform, so the
//! inputs stay bit-identical no matter how the library's own samplers,
//! quantile functions, or fitters change. That independence is what lets a
//! snapshot diff be read as "the library's output changed".
#![allow(dead_code)]

use ndarray::Array2;

/// SplitMix64 — tiny, portable, and independent of the `rand` crate version.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on the open interval (0, 1).
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal via Box–Muller (one draw per call; the second is
    /// discarded to keep the stream trivially reproducible).
    pub fn normal(&mut self) -> f64 {
        let u = self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Shape of the dependence injected between the columns.
#[derive(Debug, Clone, Copy)]
pub enum Dependence {
    /// One Gaussian latent factor; column `j` loads on it with a weight that
    /// cycles through `loadings`.
    Gaussian,
    /// Latent factor times a shared heavy-tailed scale (tail dependence in
    /// both corners, like a Student t copula).
    HeavyTail,
    /// Shared positive shock added on the exponential scale: strong
    /// lower-tail dependence (Clayton-like).
    LowerTail,
    /// Mirror image of [`Dependence::LowerTail`] (Gumbel-like).
    UpperTail,
    /// No dependence at all.
    Independent,
}

const LOADINGS: [f64; 6] = [0.85, 0.7, -0.6, 0.5, 0.75, 0.3];

/// Mid-rank-free rank transform to pseudo-observations `rank / (n + 1)`.
/// Inputs are continuous, so ties do not occur.
pub fn rank_transform(column: &[f64]) -> Vec<f64> {
    let n = column.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| column[a].total_cmp(&column[b]));
    let mut out = vec![0.0; n];
    for (rank, &idx) in order.iter().enumerate() {
        out[idx] = (rank + 1) as f64 / (n + 1) as f64;
    }
    out
}

/// `n × dim` pseudo-observations with the requested dependence shape.
pub fn pseudo_obs(n: usize, dim: usize, dependence: Dependence, seed: u64) -> Array2<f64> {
    let mut rng = SplitMix64::new(seed);
    let mut raw = vec![vec![0.0; n]; dim];
    for obs in 0..n {
        let z = rng.normal();
        let scale = {
            // Inverse-gamma-ish scale mixing variable with a heavy right tail.
            let g = rng.normal().powi(2) + rng.normal().powi(2) + rng.normal().powi(2);
            (3.0 / g.max(1e-12)).sqrt()
        };
        let shock = -rng.uniform().ln();
        for (j, column) in raw.iter_mut().enumerate() {
            let a = LOADINGS[j % LOADINGS.len()];
            let e = rng.normal();
            let base = a * z + (1.0 - a * a).sqrt() * e;
            column[obs] = match dependence {
                Dependence::Gaussian => base,
                Dependence::HeavyTail => base * scale,
                // Exponential over a shared frailty: a small shared `shock`
                // inflates every column at once (Clayton's construction).
                Dependence::LowerTail => rng.uniform().ln() / shock,
                Dependence::UpperTail => -rng.uniform().ln() / shock,
                Dependence::Independent => e,
            };
        }
    }
    let mut out = Array2::zeros((n, dim));
    for (j, column) in raw.iter().enumerate() {
        for (obs, value) in rank_transform(column).into_iter().enumerate() {
            out[(obs, j)] = value;
        }
    }
    out
}

/// Two pseudo-observation columns with the requested dependence shape.
pub fn pair(n: usize, dependence: Dependence, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let data = pseudo_obs(n, 2, dependence, seed);
    (data.column(0).to_vec(), data.column(1).to_vec())
}

/// Plain uniforms on (0, 1): inputs for kernels and Rosenblatt transforms
/// that do not need to be rank-based.
pub fn uniforms(n: usize, dim: usize, seed: u64) -> Array2<f64> {
    let mut rng = SplitMix64::new(seed);
    Array2::from_shape_fn((n, dim), |_| rng.uniform())
}

/// Equicorrelation-like SPD correlation matrix with varied off-diagonals.
pub fn correlation(dim: usize) -> Array2<f64> {
    // R = L Lᵀ rescaled to unit diagonal, with L from a fixed loading vector
    // plus identity noise — always SPD.
    let mut matrix = Array2::zeros((dim, dim));
    for i in 0..dim {
        for j in 0..dim {
            let a = LOADINGS[i % LOADINGS.len()];
            let b = LOADINGS[j % LOADINGS.len()];
            matrix[(i, j)] = if i == j { 1.0 } else { a * b };
        }
    }
    matrix
}
