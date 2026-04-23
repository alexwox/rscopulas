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
    if !matches!(method, TllOrder::Constant) {
        return Err(FitError::Failed {
            reason: "tll fit currently only supports TllOrder::Constant (linear/quadratic deferred)",
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
    // Silverman's rule of thumb for bivariate KDE: h = n^(-1/6), scaled by
    // the average sample std of the z-transformed axes. Gaussian-transformed
    // pseudo-observations are approximately unit-variance per axis, so the
    // scale factor is near 1 — keeping it in the formula just makes the
    // bandwidth robust to mis-scaled inputs.
    let sigma = 0.5 * (sample_std(&z1) + sample_std(&z2));
    let bandwidth = sigma.max(1e-3) * n.powf(-1.0 / 6.0);

    let step = (GRID_MAX - GRID_MIN) / (GRID_SIZE - 1) as f64;
    let mut log_density = Array2::<f64>::zeros((GRID_SIZE, GRID_SIZE));
    for i in 0..GRID_SIZE {
        let gx = GRID_MIN + i as f64 * step;
        for j in 0..GRID_SIZE {
            let gy = GRID_MIN + j as f64 * step;
            let mut sum = 0.0;
            for k in 0..z1.len() {
                let dx = (gx - z1[k]) / bandwidth;
                let dy = (gy - z2[k]) / bandwidth;
                sum += normal_pdf(dx) * normal_pdf(dy);
            }
            let density = sum / (n * bandwidth * bandwidth);
            log_density[[i, j]] = density.max(1e-300).ln();
        }
    }

    // Effective degrees of freedom: a rough analogue for BIC scoring. For
    // Gaussian KDE with bandwidth h on n points, tr(S) ≈ 1/(2·√π·h) per axis
    // under the product kernel. We cap at n to keep BIC finite for small
    // samples.
    let effective_df = ((1.0 / (2.0 * std::f64::consts::PI.sqrt() * bandwidth)).powi(2)).min(n);

    Ok(TllParams {
        method,
        grid_min: GRID_MIN,
        grid_max: GRID_MAX,
        log_density,
        bandwidth,
        effective_df,
    })
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
