use ndarray::Array2;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::errors::{CopulaError, FitError};

// TLL (Transformation Local Likelihood) nonparametric bivariate copula.
//
// Implementation notes:
//   * Only the *constant*-order variant (TLL0, equivalently Gaussian-kernel
//     density estimation on Φ⁻¹-transformed inputs) is implemented in this
//     phase. `TllOrder::Linear` and `TllOrder::Quadratic` are part of the
//     type surface for future local-polynomial extensions but currently
//     return an error from `fit`.
//   * The density is stored as a 30×30 grid of log-density values on the
//     z-scale, covering z ∈ [-3.5, 3.5]² — which captures ≈ 99.95% of the
//     standard-normal mass on each axis. Evaluation uses bilinear
//     interpolation in log-space.
//   * h-functions integrate the interpolated density via Simpson's rule on
//     100 nodes, then renormalise by the marginal integral so h(1|v) = 1 is
//     approximately satisfied even when the KDE's marginal integral drifts.
//   * Inverse h-functions: 90-iteration bisection (same as Gumbel/Joe).
//   * Rotations: Tll is rotationless — only Rotation::R0 is supported. The
//     dispatch layer keeps Tll in its own single-rotation bucket.

pub(crate) const GRID_SIZE: usize = 30;
const GRID_MIN: f64 = -3.5;
const GRID_MAX: f64 = 3.5;
const GRID_CLIP: f64 = 1e-12;
const SIMPSON_NODES: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TllOrder {
    Constant,
    Linear,
    Quadratic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TllParams {
    pub method: TllOrder,
    pub grid_min: f64,
    pub grid_max: f64,
    /// log-density values on the z-scale grid (shape `GRID_SIZE × GRID_SIZE`).
    pub log_density: Array2<f64>,
    pub bandwidth: f64,
    pub effective_df: f64,
}

fn standard_normal() -> Normal {
    Normal::new(0.0, 1.0).expect("standard normal parameters should be valid")
}

fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.39894228040143267793994605993438;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

fn log_normal_pdf(x: f64) -> f64 {
    const HALF_LN_2PI: f64 = 0.91893853320467274178032973640562;
    -0.5 * x * x - HALF_LN_2PI
}

fn sample_std(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return 1.0;
    }
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

fn grid_step(params: &TllParams) -> f64 {
    (params.grid_max - params.grid_min) / (GRID_SIZE - 1) as f64
}

pub fn fit(u1: &[f64], u2: &[f64], method: TllOrder) -> Result<TllParams, CopulaError> {
    if u1.len() != u2.len() || u1.is_empty() {
        return Err(FitError::Failed {
            reason: "tll pair fit requires equally sized non-empty inputs",
        }
        .into());
    }

    let normal = standard_normal();
    let z1: Vec<f64> = u1
        .iter()
        .map(|&u| normal.inverse_cdf(u.clamp(GRID_CLIP, 1.0 - GRID_CLIP)))
        .collect();
    let z2: Vec<f64> = u2
        .iter()
        .map(|&u| normal.inverse_cdf(u.clamp(GRID_CLIP, 1.0 - GRID_CLIP)))
        .collect();

    let n = z1.len() as f64;
    let sigma = 0.5 * (sample_std(&z1) + sample_std(&z2));
    // Bandwidth: vinecopulib's rule is `1.5 · n^(−1/(2p+1))` for polynomial
    // order p (TLL0 → p=0, TLL1 → p=1, TLL2 → p=2). We keep the sample-std
    // rescaling so the bandwidth remains robust when the Φ⁻¹-transformed
    // inputs drift from unit variance.
    let exponent = match method {
        TllOrder::Constant => -1.0 / 6.0, // classic Silverman 1.0·n^{-1/6} (p=0 in our convention)
        TllOrder::Linear => -1.0 / 3.0,   // p=1, so -1/(2p+1) = -1/3
        TllOrder::Quadratic => -1.0 / 5.0, // p=2, so -1/5
    };
    let scale = match method {
        TllOrder::Constant => 1.0,
        // vinecopulib's `1.5` multiplier for local-polynomial orders; keeps
        // enough smoothing to counter the extra bias-variance that higher-
        // order corrections would otherwise introduce.
        TllOrder::Linear | TllOrder::Quadratic => 1.5,
    };
    let bandwidth = scale * sigma.max(1e-3) * n.powf(exponent);

    let step = (GRID_MAX - GRID_MIN) / (GRID_SIZE - 1) as f64;

    // Vinecopulib's closed-form local-polynomial density estimator — direct
    // port of `tll.ipp:85-148`. At each grid point `z_k` the estimator runs
    // the data through `irB = B^{-1/2}` to get decorrelated shifts `zz`,
    // computes the Gaussian kernel `K` at those shifts, forms weighted
    // moments `f0 = mean(K)` (premultiplied by `det_irB`) and, for the
    // polynomial orders, a *second* application of `irB` to get `zz_tilde =
    // irB · zz`. The linear estimator then uses `exp(−½ · bᵀ · B · b)` with
    // `S = B` (its scope-level default). The quadratic estimator additionally
    // rebuilds `S` from the kernel-weighted covariance of `zz_tilde` and
    // multiplies by `sqrt(det S) / det_irB`.
    //
    // For isotropic `B = h² · I` the Cholesky root is `irB = (1/h) · I` with
    // `det_irB = 1/h²`. Specialising on this keeps the 2×2 matrix algebra
    // inline and avoids a generic linalg dependency.
    let inv_h = 1.0 / bandwidth;
    let det_irb = inv_h * inv_h;
    let b_mat = [[bandwidth * bandwidth, 0.0], [0.0, bandwidth * bandwidth]];

    let mut log_density = Array2::<f64>::zeros((GRID_SIZE, GRID_SIZE));
    for i in 0..GRID_SIZE {
        let gx = GRID_MIN + i as f64 * step;
        for j in 0..GRID_SIZE {
            let gy = GRID_MIN + j as f64 * step;

            // `zz` is already once-decorrelated: `(z_data - z_grid) / h`.
            // Kernel weights evaluated here include the `det_irB` factor
            // from vinecopulib's `kernels = gaussian_kernel_2d(zz) * det_irB`.
            let mut k_sum = 0.0_f64;
            // `zz_tilde = irB · zz = zz / h` for isotropic B — i.e. the data
            // *twice*-decorrelated. Accumulated weighted moments use this
            // second decorrelation, matching the `zz = (irB * zz.transpose())
            // .transpose()` step inside vinecopulib's `if != "constant"`.
            let mut zt_k_sum = [0.0_f64; 2];
            let mut zt_outer_k_sum = [[0.0_f64; 2]; 2];
            for k in 0..z1.len() {
                let zz = [(z1[k] - gx) * inv_h, (z2[k] - gy) * inv_h];
                let weight = normal_pdf(zz[0]) * normal_pdf(zz[1]) * det_irb;
                k_sum += weight;
                let zt = [zz[0] * inv_h, zz[1] * inv_h];
                zt_k_sum[0] += zt[0] * weight;
                zt_k_sum[1] += zt[1] * weight;
                zt_outer_k_sum[0][0] += zt[0] * zt[0] * weight;
                zt_outer_k_sum[0][1] += zt[0] * zt[1] * weight;
                zt_outer_k_sum[1][0] += zt[1] * zt[0] * weight;
                zt_outer_k_sum[1][1] += zt[1] * zt[1] * weight;
            }
            // `f0` already carries `det_irB`, so the bare-log-density is
            // `log f0` — no separate `+ log det_irB` additive is needed.
            let f0 = k_sum / n;
            if f0 <= 0.0 || !f0.is_finite() {
                log_density[[i, j]] = (1e-300_f64).ln();
                continue;
            }

            let mut res_log = f0.ln();
            let f1 = [zt_k_sum[0] / n, zt_k_sum[1] / n];
            let b_decorr = [f1[0] / f0, f1[1] / f0];

            match method {
                TllOrder::Constant => {}
                TllOrder::Linear => {
                    // S defaults to B at function scope in vinecopulib — so
                    // for the linear estimator `bᵀ·S·b` evaluates as
                    // `bᵀ·B·b = h² · (b[0]² + b[1]²)` (isotropic B).
                    let quad = b_mat[0][0] * b_decorr[0] * b_decorr[0]
                        + b_mat[1][1] * b_decorr[1] * b_decorr[1];
                    res_log -= 0.5 * quad;
                }
                TllOrder::Quadratic => {
                    // `zz2 = zz_tilde · K / (f0 · n)` ⇒ `zz.T · zz2` equals
                    // `zt_outer_k_sum / (f0 · n)` pointwise.
                    let denom = f0 * n;
                    let mut zz_cov = [[0.0_f64; 2]; 2];
                    for a in 0..2 {
                        for b in 0..2 {
                            zz_cov[a][b] = zt_outer_k_sum[a][b] / denom;
                        }
                    }
                    // `b = B · b` — remap into original-parameter scale.
                    let b_prime = [
                        b_mat[0][0] * b_decorr[0],
                        b_mat[1][1] * b_decorr[1],
                    ];
                    // S_inv = B · zz_cov · B − b' · b'ᵀ.
                    let mut s_inv = [[0.0_f64; 2]; 2];
                    for a in 0..2 {
                        for b in 0..2 {
                            s_inv[a][b] = b_mat[a][a] * zz_cov[a][b] * b_mat[b][b]
                                - b_prime[a] * b_prime[b];
                        }
                    }
                    let fallback_to_linear = |res_log: f64| -> f64 {
                        let quad = b_mat[0][0] * b_decorr[0] * b_decorr[0]
                            + b_mat[1][1] * b_decorr[1] * b_decorr[1];
                        res_log - 0.5 * quad
                    };
                    let Some((s_mat, det_s_inv)) = invert_2x2(&s_inv) else {
                        log_density[[i, j]] = fallback_to_linear(res_log).max(1e-300_f64.ln());
                        continue;
                    };
                    if det_s_inv <= 0.0 || !det_s_inv.is_finite() {
                        log_density[[i, j]] = fallback_to_linear(res_log).max(1e-300_f64.ln());
                        continue;
                    }
                    // `res(k) *= sqrt(det S) / det_irB` (vinecopulib). Since
                    // `det S = 1 / det S_inv`, use the reciprocal identity:
                    // `0.5 · ln(det S) = −0.5 · ln(det S_inv)`.
                    let det_s = 1.0 / det_s_inv;
                    res_log += 0.5 * det_s.ln() - det_irb.ln();
                    // `exp(−½ · bᵀ·S·b)` with the remapped b.
                    let sb = [
                        s_mat[0][0] * b_prime[0] + s_mat[0][1] * b_prime[1],
                        s_mat[1][0] * b_prime[0] + s_mat[1][1] * b_prime[1],
                    ];
                    let quad = b_prime[0] * sb[0] + b_prime[1] * sb[1];
                    res_log -= 0.5 * quad;
                }
            }
            log_density[[i, j]] = res_log.max(1e-300_f64.ln());
        }
    }

    // Effective degrees of freedom. We use vinecopulib's trace-of-hat-matrix
    // definition: per-grid-point influence `infl(z_k) = K(0) · det_irB ·
    // [M⁻¹]_{0,0} · 1/n` where `M` is the local Gram matrix over the
    // polynomial basis evaluated under Gaussian kernel weights — 1×1 for
    // TLL0 (just `f0`), 3×3 for TLL1, 6×6 for TLL2. The reported
    // `effective_df` is the sum of influence values interpolated at each
    // data point, clipped to `[−0.2, 1.3]` (vinecopulib's safety band) and
    // floored at 1.0.
    let mut infl_grid = Array2::<f64>::zeros((GRID_SIZE, GRID_SIZE));
    let kernel_zero = normal_pdf(0.0) * normal_pdf(0.0);
    for i in 0..GRID_SIZE {
        let gx = GRID_MIN + i as f64 * step;
        for j in 0..GRID_SIZE {
            let gy = GRID_MIN + j as f64 * step;
            let m_inv_00 = local_gram_m_inv_00(gx, gy, &z1, &z2, bandwidth, method);
            let infl = kernel_zero * det_irb * m_inv_00 / n;
            infl_grid[[i, j]] = infl.clamp(-0.2, 1.3);
        }
    }
    let mut infl_sum = 0.0_f64;
    let infl_params = TllParams {
        method: TllOrder::Constant,
        grid_min: GRID_MIN,
        grid_max: GRID_MAX,
        log_density: infl_grid.mapv(f64::ln),
        bandwidth,
        effective_df: 0.0,
    };
    for k in 0..z1.len() {
        // Re-use the bilinear log-density helper against the (already
        // log-transformed) influence grid; exponentiating undoes the ln so
        // we sum in the original influence units.
        let interp = bilinear_log_density(&infl_params, z1[k], z2[k]).exp();
        infl_sum += interp;
    }
    let effective_df = infl_sum.max(1.0).min(n);

    Ok(TllParams {
        method,
        grid_min: GRID_MIN,
        grid_max: GRID_MAX,
        log_density,
        bandwidth,
        effective_df,
    })
}

/// Returns the `(0, 0)` entry of the inverse of the local Gram matrix `M`
/// built from kernel-weighted polynomial basis evaluations at grid point
/// `(gx, gy)`. The basis size matches the TLL order: 1 for `Constant` (→
/// `M⁻¹[0,0] = 1/f0`), 3 for `Linear`, 6 for `Quadratic`.
fn local_gram_m_inv_00(
    gx: f64,
    gy: f64,
    z1: &[f64],
    z2: &[f64],
    bandwidth: f64,
    method: TllOrder,
) -> f64 {
    let n = z1.len() as f64;
    let basis_size = match method {
        TllOrder::Constant => 1,
        TllOrder::Linear => 3,
        TllOrder::Quadratic => 6,
    };
    let mut m = vec![0.0_f64; basis_size * basis_size];
    let inv_h = 1.0 / bandwidth;
    let det_irb = inv_h * inv_h;

    for k in 0..z1.len() {
        let u = (z1[k] - gx) * inv_h;
        let v = (z2[k] - gy) * inv_h;
        // Match vinecopulib: the kernel vector that builds `M` already
        // carries the `det_irB` factor (same premultiplication as `kernels`
        // in the density loop). Omitting it here leaves `[M⁻¹]_{0,0}` too
        // large and drives the influence formula past the `1.3` clamp for
        // every TLL1 grid point.
        let w = normal_pdf(u) * normal_pdf(v) * det_irb;
        let phi: [f64; 6] = [1.0, u, v, u * u, v * v, u * v];
        for a in 0..basis_size {
            for b in 0..basis_size {
                m[a * basis_size + b] += w * phi[a] * phi[b] / n;
            }
        }
    }

    if basis_size == 1 {
        if m[0] > 1e-300 { 1.0 / m[0] } else { 1e-300 }
    } else {
        // Invert the small dense matrix and return [M⁻¹]_{0,0}. If
        // inversion fails (rare — requires degenerate local windows in the
        // tails), fall back to the TLL0 equivalent `1/f0` so BIC scoring
        // stays finite.
        let mut b = vec![0.0_f64; basis_size];
        b[0] = 1.0;
        match solve_symmetric(&mut m.clone(), &mut b, basis_size) {
            Some(sol) => sol[0].max(0.0),
            None => {
                if m[0] > 1e-300 { 1.0 / m[0] } else { 1e-300 }
            }
        }
    }
}

fn invert_2x2(mat: &[[f64; 2]; 2]) -> Option<([[f64; 2]; 2], f64)> {
    let det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    if det.abs() < 1e-300 {
        return None;
    }
    let inv_det = 1.0 / det;
    let inv = [
        [mat[1][1] * inv_det, -mat[0][1] * inv_det],
        [-mat[1][0] * inv_det, mat[0][0] * inv_det],
    ];
    Some((inv, det))
}

/// Gauss-Jordan elimination with partial pivoting for a small dense matrix.
/// Kept for the `local_gram_m_inv_00` helper; TLL1/TLL2 density evaluation
/// uses `invert_2x2` directly.
#[allow(dead_code)]
fn fit_local_polynomial_at(
    gx: f64,
    gy: f64,
    z1: &[f64],
    z2: &[f64],
    y: &[f64],
    bandwidth: f64,
    basis_size: usize,
) -> Option<f64> {
    let mut xtwx = vec![0.0_f64; basis_size * basis_size];
    let mut xtwy = vec![0.0_f64; basis_size];
    let mut effective_n = 0.0_f64;

    for k in 0..z1.len() {
        let u = (z1[k] - gx) / bandwidth;
        let v = (z2[k] - gy) / bandwidth;
        let w = normal_pdf(u) * normal_pdf(v);
        // Skip points whose contribution is below round-off. Without this
        // guard, far-tail grid nodes would accumulate tiny-but-nonzero
        // weights that make the design matrix numerically singular.
        if w < 1e-30 {
            continue;
        }
        effective_n += w;
        let phi = match basis_size {
            3 => vec![1.0, u, v],
            6 => vec![1.0, u, v, u * u, v * v, u * v],
            _ => unreachable!(),
        };
        for a in 0..basis_size {
            xtwy[a] += w * phi[a] * y[k];
            for b in 0..basis_size {
                xtwx[a * basis_size + b] += w * phi[a] * phi[b];
            }
        }
    }

    // Demand at least `basis_size + 1` effective observations (rough rule:
    // more points than basis columns) so the regression is well-posed.
    if effective_n < basis_size as f64 + 1.0 {
        return None;
    }
    let solution = solve_symmetric(&mut xtwx, &mut xtwy, basis_size)?;
    Some(solution[0])
}

/// Gauss-Jordan elimination with partial pivoting for a small dense
/// `basis_size × basis_size` system. Small enough that the standard
/// row-ops implementation is the right tool; ndarray's `solve` isn't wired
/// for this crate.
fn solve_symmetric(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for pivot in 0..n {
        let mut max_row = pivot;
        let mut max_val = a[pivot * n + pivot].abs();
        for row in (pivot + 1)..n {
            let value = a[row * n + pivot].abs();
            if value > max_val {
                max_val = value;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        if max_row != pivot {
            for col in 0..n {
                a.swap(pivot * n + col, max_row * n + col);
            }
            b.swap(pivot, max_row);
        }
        let diag = a[pivot * n + pivot];
        for col in 0..n {
            a[pivot * n + col] /= diag;
        }
        b[pivot] /= diag;
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = a[row * n + pivot];
            if factor == 0.0 {
                continue;
            }
            for col in 0..n {
                a[row * n + col] -= factor * a[pivot * n + col];
            }
            b[row] -= factor * b[pivot];
        }
    }
    Some(b.to_vec())
}

/// Bilinear interpolation of the stored log-density at a point on the
/// z-scale. Points outside `[grid_min, grid_max]²` clamp to the edge.
fn bilinear_log_density(params: &TllParams, zx: f64, zy: f64) -> f64 {
    let step = grid_step(params);
    let zx_c = zx.clamp(params.grid_min, params.grid_max);
    let zy_c = zy.clamp(params.grid_min, params.grid_max);

    let fx = (zx_c - params.grid_min) / step;
    let fy = (zy_c - params.grid_min) / step;
    let i = (fx.floor() as usize).min(GRID_SIZE - 2);
    let j = (fy.floor() as usize).min(GRID_SIZE - 2);
    let ax = (fx - i as f64).clamp(0.0, 1.0);
    let ay = (fy - j as f64).clamp(0.0, 1.0);

    let g00 = params.log_density[[i, j]];
    let g01 = params.log_density[[i, j + 1]];
    let g10 = params.log_density[[i + 1, j]];
    let g11 = params.log_density[[i + 1, j + 1]];

    let g0 = g00 * (1.0 - ay) + g01 * ay;
    let g1 = g10 * (1.0 - ay) + g11 * ay;
    g0 * (1.0 - ax) + g1 * ax
}

pub fn log_pdf(u1: f64, u2: f64, params: &TllParams) -> Result<f64, CopulaError> {
    let normal = standard_normal();
    let zx = normal.inverse_cdf(u1.clamp(GRID_CLIP, 1.0 - GRID_CLIP));
    let zy = normal.inverse_cdf(u2.clamp(GRID_CLIP, 1.0 - GRID_CLIP));
    // log c(u, v) = log f(z1, z2) − log φ(z1) − log φ(z2).
    Ok(bilinear_log_density(params, zx, zy) - log_normal_pdf(zx) - log_normal_pdf(zy))
}

fn pdf(u1: f64, u2: f64, params: &TllParams) -> f64 {
    log_pdf(u1, u2, params).map(f64::exp).unwrap_or(0.0)
}

/// Simpson integration of `f` over `[a, b]` using `n` intervals (forced even).
fn simpson<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    if b <= a {
        return 0.0;
    }
    let n = n.max(2) & !1;
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for k in 1..n {
        let x = a + k as f64 * h;
        sum += if k % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}

pub fn cond_first_given_second(
    u1: f64,
    u2: f64,
    params: &TllParams,
) -> Result<f64, CopulaError> {
    // h_{1|2}(u | v) = P(U ≤ u | V = v). Evaluated on the z-scale (where the
    // KDE density f is bounded and smooth) rather than on the copula scale
    // (where c(u, v) has an integrable spike near the corners from the 1/φ
    // Jacobian). Change of variables u → z = Φ⁻¹(u):
    //   h_{1|2}(u | v) = ∫_{-∞}^{z_u} f(z, z_v) dz  /  ∫_{-∞}^{∞} f(z, z_v) dz.
    // The denominator is the z-scale marginal density at z_v; numerically the
    // ratio stays in [0, 1] and reaches 1 at the grid boundary exactly.
    let normal = standard_normal();
    let zv = normal.inverse_cdf(u2.clamp(GRID_CLIP, 1.0 - GRID_CLIP));
    let zu = normal.inverse_cdf(u1.clamp(GRID_CLIP, 1.0 - GRID_CLIP));
    let lower = params.grid_min;
    let upper = params.grid_max;
    let integrand = |z: f64| bilinear_log_density(params, z, zv).exp();
    let total = simpson(integrand, lower, upper, SIMPSON_NODES);
    if total <= 0.0 {
        return Ok(0.5);
    }
    let zu_bounded = zu.clamp(lower, upper);
    let partial = simpson(integrand, lower, zu_bounded, SIMPSON_NODES);
    Ok((partial / total).clamp(0.0, 1.0))
}

pub fn cond_second_given_first(
    u1: f64,
    u2: f64,
    params: &TllParams,
) -> Result<f64, CopulaError> {
    let normal = standard_normal();
    let zu = normal.inverse_cdf(u1.clamp(GRID_CLIP, 1.0 - GRID_CLIP));
    let zv = normal.inverse_cdf(u2.clamp(GRID_CLIP, 1.0 - GRID_CLIP));
    let lower = params.grid_min;
    let upper = params.grid_max;
    let integrand = |z: f64| bilinear_log_density(params, zu, z).exp();
    let total = simpson(integrand, lower, upper, SIMPSON_NODES);
    if total <= 0.0 {
        return Ok(0.5);
    }
    let zv_bounded = zv.clamp(lower, upper);
    let partial = simpson(integrand, lower, zv_bounded, SIMPSON_NODES);
    Ok((partial / total).clamp(0.0, 1.0))
}

pub fn inv_first_given_second(
    p: f64,
    u2: f64,
    params: &TllParams,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let mut low = clip_eps;
    let mut high = 1.0 - clip_eps;
    for _ in 0..90 {
        let mid = 0.5 * (low + high);
        if cond_first_given_second(mid, u2, params)? < p {
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
    params: &TllParams,
    clip_eps: f64,
) -> Result<f64, CopulaError> {
    let mut low = clip_eps;
    let mut high = 1.0 - clip_eps;
    for _ in 0..90 {
        let mid = 0.5 * (low + high);
        if cond_second_given_first(u1, mid, params)? < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(0.5 * (low + high))
}

pub fn cdf(u1: f64, u2: f64, params: &TllParams) -> Result<f64, CopulaError> {
    // C(u, v) = ∫_0^u h_{2|1}(v | s) · 1 ds is the identity we want, but
    // computing it that way requires another nested integral. Instead,
    // integrate the marginal conditional: C(u, v) ≈ ∫_0^u h_{1|2}(s | v)
    // ds — but again nested. The pragmatic choice is to compute C by
    // returning u·v as a placeholder when the density hasn't been fit yet;
    // a proper implementation would use a 2-D integral. For the current
    // test surface (density, h, inverse h) we don't need CDF, so we return
    // a Simpson-based 1-D approximation that is monotone but not exact.
    let lower = GRID_CLIP;
    let upper = 1.0 - GRID_CLIP;
    let integrand = |s: f64| {
        let h_val = cond_second_given_first(s, u2, params).unwrap_or(0.0);
        h_val.max(0.0)
    };
    let value = simpson(integrand, lower, u1.clamp(lower, upper), SIMPSON_NODES);
    Ok(value.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Generate a Gaussian-ρ sample via Cholesky on uniform pseudo-obs.
    fn gaussian_sample(n: usize, rho: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, Normal as RandNormal};
        let mut rng = StdRng::seed_from_u64(seed);
        let snorm = RandNormal::new(0.0, 1.0).unwrap();
        let cnorm = standard_normal();
        let mut u1 = Vec::with_capacity(n);
        let mut u2 = Vec::with_capacity(n);
        for _ in 0..n {
            let a = snorm.sample(&mut rng);
            let b = snorm.sample(&mut rng);
            let x = a;
            let y = rho * a + (1.0 - rho * rho).sqrt() * b;
            u1.push(cnorm.cdf(x));
            u2.push(cnorm.cdf(y));
        }
        (u1, u2)
    }

    #[test]
    fn tll_fit_recovers_moderate_gaussian_density() {
        let (u1, u2) = gaussian_sample(2000, 0.5, 7);
        let params = fit(&u1, &u2, TllOrder::Constant).expect("fit should succeed");

        // The density at the centre (u = v = 0.5) should be roughly the
        // Gaussian copula density with ρ = 0.5, which is 1 / sqrt(1 - 0.25) ≈
        // 1.1547. KDE with n = 2000 should be within 20% of the target.
        let log_c = log_pdf(0.5, 0.5, &params).expect("log_pdf ok");
        let density_est = log_c.exp();
        let density_target = 1.0 / (1.0 - 0.25_f64).sqrt();
        assert!(
            (density_est - density_target).abs() / density_target < 0.2,
            "tll density at (0.5, 0.5) = {density_est} should be within 20% of target {density_target}"
        );
    }

    #[test]
    fn tll_h_function_is_monotone_and_ends_at_one() {
        let (u1, u2) = gaussian_sample(2000, 0.5, 11);
        let params = fit(&u1, &u2, TllOrder::Constant).unwrap();

        let v = 0.4;
        let mut prev = 0.0;
        for idx in 1..=9 {
            let u = idx as f64 / 10.0;
            let h = cond_first_given_second(u, v, &params).unwrap();
            assert!(
                h >= prev - 1e-9,
                "h_{{1|2}}(u={u} | v={v}) = {h} dropped below previous {prev}"
            );
            prev = h;
        }
        // h(1|v) should be ≈ 1 by the normalisation.
        let h_one = cond_first_given_second(1.0 - 1e-9, v, &params).unwrap();
        assert!(
            (h_one - 1.0).abs() < 1e-6,
            "h_{{1|2}}(1 | v) = {h_one} should equal 1"
        );
    }

    #[test]
    fn tll_inverse_h_round_trips_within_tolerance() {
        let (u1, u2) = gaussian_sample(2000, 0.3, 17);
        let params = fit(&u1, &u2, TllOrder::Constant).unwrap();

        for v in [0.25_f64, 0.5, 0.75] {
            for u in [0.2_f64, 0.5, 0.8] {
                let h = cond_first_given_second(u, v, &params).unwrap();
                let u_back = inv_first_given_second(h, v, &params, 1e-12).unwrap();
                assert!(
                    (u - u_back).abs() < 5e-3,
                    "round-trip u = {u} v = {v}: recovered {u_back} (h = {h})"
                );
            }
        }
    }

    #[test]
    fn tll_log_pdf_is_finite_across_grid_interior() {
        let (u1, u2) = gaussian_sample(1000, 0.4, 23);
        let params = fit(&u1, &u2, TllOrder::Constant).unwrap();

        for u in [0.01_f64, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
            for v in [0.01_f64, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
                let value = log_pdf(u, v, &params).unwrap();
                assert!(value.is_finite(), "log_pdf({u}, {v}) = {value} non-finite");
            }
        }
    }
}
