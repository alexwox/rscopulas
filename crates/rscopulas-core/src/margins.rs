//! Marginal models with extreme-value tails.
//!
//! A copula describes dependence between uniforms. To go from raw data to a
//! joint model and back, each column also needs a marginal distribution
//! function `F` (raw value -> uniform) and its inverse (uniform -> raw
//! value). A rank transform gives the first direction only, and it cannot
//! say anything beyond the largest observation.
//!
//! [`SemiParametricMargin`] gives both directions and extrapolates:
//!
//! ```text
//!        lower tail            body                upper tail
//!   generalized Pareto | interpolated ECDF | generalized Pareto
//!                     u_L                 u_U
//! ```
//!
//! The body is the empirical distribution function `(#{x_i <= x}) / (n + 1)`,
//! linear between observations. Beyond each threshold the excesses over the
//! threshold follow a generalized Pareto distribution (GPD) fitted by maximum
//! likelihood, which is what the Pickands–Balkema–de Haan theorem says they
//! approach for a high threshold. The three pieces join continuously, and
//! the function is strictly increasing, so it has an exact inverse.
//!
//! [`Margins`] holds one margin per column and maps a raw `(n, d)` matrix to
//! [`PseudoObs`] and simulated uniforms back to the raw scale.
//!
//! [`GeneralizedExtremeValue`] is the companion model for block maxima
//! (annual maxima, say), with return levels. [`mean_excess`] and
//! [`threshold_stability`] are the two standard diagnostics for choosing a
//! GPD threshold.
//!
//! Sign convention: a positive shape `xi` is a heavy (power-law) tail,
//! `xi = 0` an exponential-type tail, and a negative shape a tail with a
//! finite end point. SciPy's `genextreme` uses `c = -xi`; its `genpareto`
//! uses `c = xi`.

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use statrs::function::gamma::gamma;

use crate::{
    PseudoObs,
    errors::{CopulaError, FitError, InputError},
    math::{
        NelderMeadOptions, QuasiNewtonOptions, inverse, maximize_scalar_brent,
        nelder_mead_maximize, numerical_hessian, quasi_newton_maximize,
    },
};

/// Below this magnitude the shape is treated as exactly zero.
const SHAPE_ZERO: f64 = 1e-12;
/// Uniforms returned by [`Margins::to_uniform`] are kept this far from 0 and 1.
const UNIFORM_CLIP: f64 = 1e-12;

fn invalid(reason: &'static str) -> CopulaError {
    InputError::InvalidArgument { reason }.into()
}

fn require_finite(values: &[f64]) -> Result<(), CopulaError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InputError::NonFiniteValue.into());
    }
    Ok(())
}

fn require_open_probability(p: f64) -> Result<(), CopulaError> {
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(invalid("probabilities must lie strictly inside (0, 1)"));
    }
    Ok(())
}

/// `ln(1 + xi z) / xi`, continuous through `xi = 0` (where it is `z`).
/// `NaN` outside the support `1 + xi z > 0`.
fn log1p_over_shape(shape: f64, z: f64) -> f64 {
    if shape.abs() < SHAPE_ZERO {
        return z;
    }
    let inner = shape * z;
    if inner <= -1.0 {
        return f64::NAN;
    }
    inner.ln_1p() / shape
}

/// `(exp(xi w) - 1) / xi`, continuous through `xi = 0` (where it is `w`).
fn expm1_over_shape(shape: f64, w: f64) -> f64 {
    if shape.abs() < SHAPE_ZERO {
        w
    } else {
        (shape * w).exp_m1() / shape
    }
}

/// Inverse of the negative Hessian of `f` at its maximum, in the caller's
/// parameters. `f` takes offsets `s` in coordinates where every parameter has
/// unit scale; `jacobian[i]` is `d parameter_i / d s_i` at `s = 0`.
fn covariance_at_maximum<F>(dim: usize, jacobian: &[f64], f: F) -> Option<Array2<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let hessian = numerical_hessian(&vec![0.0; dim], 1e-4, f);
    if hessian.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut covariance = inverse(&hessian.mapv(|value| -value)).ok()?;
    for row in 0..dim {
        for col in 0..dim {
            covariance[(row, col)] *= jacobian[row] * jacobian[col];
        }
    }
    // A maximum in the interior has a positive definite information matrix.
    if (0..dim).any(|idx| covariance[(idx, idx)].is_nan() || covariance[(idx, idx)] <= 0.0) {
        return None;
    }
    Some(covariance)
}

// ---------------------------------------------------------------------------
// Generalized Pareto
// ---------------------------------------------------------------------------

/// Generalized Pareto distribution of an excess `y >= 0` over a threshold:
/// `P(Y <= y) = 1 - (1 + xi y / sigma)^(-1/xi)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeneralizedPareto {
    scale: f64,
    shape: f64,
}

/// A fitted [`GeneralizedPareto`] with its log-likelihood.
#[derive(Debug, Clone, PartialEq)]
pub struct GpdFit {
    pub model: GeneralizedPareto,
    pub loglik: f64,
    pub n_obs: usize,
    /// Asymptotic covariance of `(scale, shape)` from the observed
    /// information. `None` when the estimate sits on the boundary
    /// `shape = -1`, where the usual theory does not apply.
    pub covariance: Option<Array2<f64>>,
}

impl GeneralizedPareto {
    /// `scale > 0`; any finite `shape`.
    pub fn new(scale: f64, shape: f64) -> Result<Self, CopulaError> {
        if !scale.is_finite() || scale <= 0.0 || !shape.is_finite() {
            return Err(invalid(
                "generalized Pareto needs a finite scale > 0 and a finite shape",
            ));
        }
        Ok(Self { scale, shape })
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// `P(Y > y)`. One for `y <= 0`; zero beyond the end point when
    /// `shape < 0`.
    pub fn survival(&self, y: f64) -> f64 {
        if y <= 0.0 {
            return 1.0;
        }
        let exponent = log1p_over_shape(self.shape, y / self.scale);
        if exponent.is_nan() {
            0.0
        } else {
            (-exponent).exp()
        }
    }

    /// `P(Y <= y)`.
    pub fn cdf(&self, y: f64) -> f64 {
        if y <= 0.0 {
            return 0.0;
        }
        let exponent = log1p_over_shape(self.shape, y / self.scale);
        if exponent.is_nan() {
            1.0
        } else {
            -(-exponent).exp_m1()
        }
    }

    /// Log-density; `-inf` outside the support.
    pub fn ln_pdf(&self, y: f64) -> f64 {
        if y < 0.0 {
            return f64::NEG_INFINITY;
        }
        let exponent = log1p_over_shape(self.shape, y / self.scale);
        if exponent.is_nan() {
            return f64::NEG_INFINITY;
        }
        -self.scale.ln() - (1.0 + self.shape) * exponent
    }

    /// The excess exceeded with probability `survival`, for
    /// `0 < survival <= 1`. Taking the survival probability keeps full
    /// precision far in the tail, where `1 - p` would round.
    pub fn quantile_from_survival(&self, survival: f64) -> f64 {
        self.scale * expm1_over_shape(self.shape, -survival.ln())
    }

    /// The `p`-quantile, `0 <= p < 1`.
    pub fn quantile(&self, p: f64) -> Result<f64, CopulaError> {
        if !p.is_finite() || !(0.0..1.0).contains(&p) {
            return Err(invalid("generalized Pareto quantiles need 0 <= p < 1"));
        }
        Ok(self.scale * expm1_over_shape(self.shape, -(-p).ln_1p()))
    }

    /// Maximum-likelihood fit to excesses over a threshold (all `>= 0`).
    ///
    /// With `tau = shape / scale` the likelihood equation for the shape has
    /// the explicit solution `shape(tau) = mean(ln(1 + tau y_i))`, which
    /// leaves a one-dimensional profile likelihood in `tau`. It is scanned on
    /// a grid over its whole domain `tau > -1 / max(y)` and the best cell is
    /// refined with Brent's method, so the result does not depend on a
    /// starting value.
    ///
    /// The likelihood is unbounded for `shape < -1`; the search is restricted
    /// to `shape >= -1`, as is conventional.
    pub fn fit(excesses: &[f64]) -> Result<GpdFit, CopulaError> {
        require_finite(excesses)?;
        if excesses.len() < 2 {
            return Err(invalid(
                "a generalized Pareto fit needs at least two excesses",
            ));
        }
        if excesses.iter().any(|value| *value < 0.0) {
            return Err(invalid(
                "excesses over a threshold must be non-negative; subtract the threshold first",
            ));
        }
        let n = excesses.len() as f64;
        let mean = excesses.iter().sum::<f64>() / n;
        let largest = excesses.iter().copied().fold(0.0, f64::max);
        if mean <= 0.0 {
            return Err(invalid("all excesses are zero"));
        }

        // Profile log-likelihood and the (scale, shape) it implies.
        let at = |tau: f64| -> (f64, f64, f64) {
            if tau.abs() * largest < 1e-9 {
                // Exponential limit.
                return (-n * (mean.ln() + 1.0), mean, 0.0);
            }
            if tau * largest <= -1.0 {
                return (f64::NEG_INFINITY, f64::NAN, f64::NAN);
            }
            let shape = excesses.iter().map(|y| (tau * y).ln_1p()).sum::<f64>() / n;
            let scale = shape / tau;
            if shape < -1.0 || scale.is_nan() || scale <= 0.0 {
                return (f64::NEG_INFINITY, scale, shape);
            }
            (-n * (scale.ln() + shape + 1.0), scale, shape)
        };

        // Grid: tau * mean from 1e-6 to 1e4 on the heavy side, and tau
        // approaching -1 / max(y) geometrically on the bounded side. Each
        // point costs one pass over the sample, so the grid is only as fine
        // as it needs to be to bracket the maximum; Brent does the rest.
        let mut grid = Vec::with_capacity(72);
        for step in (0..25).rev() {
            let gap = 10f64.powf(-12.0 * step as f64 / 24.0); // 1e-12 ..= 1
            grid.push(-(1.0 - gap) / largest);
        }
        grid.push(0.0);
        for step in 0..41 {
            grid.push(10f64.powf(-6.0 + 10.0 * step as f64 / 40.0) / mean);
        }
        grid.sort_by(f64::total_cmp);
        grid.dedup();

        let values: Vec<f64> = grid.iter().map(|tau| at(*tau).0).collect();
        let best = (0..grid.len())
            .max_by(|a, b| values[*a].total_cmp(&values[*b]))
            .expect("grid is not empty");
        let low = grid[best.saturating_sub(1)];
        let high = grid[(best + 1).min(grid.len() - 1)];
        // The likelihood is flat to second order at its maximum, so a
        // relative 1e-10 on tau leaves the log-likelihood exact to rounding.
        let refined = maximize_scalar_brent(
            low,
            high,
            Some(grid[best]),
            1e-10 * (high - low).abs(),
            80,
            |tau| at(tau).0,
        );
        let (mut loglik, mut scale, mut shape) = at(refined.x);
        // The profile only reaches the boundary shape = -1 at a scale above
        // max(y). The best point on that boundary is the uniform
        // distribution on [0, max(y)], with log-likelihood -n ln(max(y));
        // for small samples with a bounded tail it can beat every interior
        // point, so it is compared explicitly.
        let boundary = -n * largest.ln();
        if boundary > loglik || !loglik.is_finite() {
            (loglik, scale, shape) = (boundary, largest, -1.0);
        }
        if !loglik.is_finite() {
            return Err(FitError::Failed {
                reason: "generalized Pareto likelihood has no finite maximum for this sample",
            }
            .into());
        }
        let model = Self::new(scale, shape)?;

        let covariance = if shape > -1.0 + 1e-6 {
            covariance_at_maximum(2, &[scale, 1.0], |s| {
                let candidate = Self {
                    scale: scale * s[0].exp(),
                    shape: shape + s[1],
                };
                excesses.iter().map(|y| candidate.ln_pdf(*y)).sum()
            })
        } else {
            None
        };

        Ok(GpdFit {
            model,
            loglik,
            n_obs: excesses.len(),
            covariance,
        })
    }
}

// ---------------------------------------------------------------------------
// Generalized extreme value
// ---------------------------------------------------------------------------

/// Generalized extreme value distribution, the limit law of block maxima:
/// `P(M <= x) = exp(-(1 + xi (x - mu) / sigma)^(-1/xi))`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeneralizedExtremeValue {
    location: f64,
    scale: f64,
    shape: f64,
}

/// A fitted [`GeneralizedExtremeValue`] with its log-likelihood.
#[derive(Debug, Clone, PartialEq)]
pub struct GevFit {
    pub model: GeneralizedExtremeValue,
    pub loglik: f64,
    pub n_obs: usize,
    /// Asymptotic covariance of `(location, scale, shape)` from the observed
    /// information; `None` when it is not positive definite.
    pub covariance: Option<Array2<f64>>,
}

impl GeneralizedExtremeValue {
    /// `scale > 0`; any finite `location` and `shape`.
    pub fn new(location: f64, scale: f64, shape: f64) -> Result<Self, CopulaError> {
        if !location.is_finite() || !scale.is_finite() || scale <= 0.0 || !shape.is_finite() {
            return Err(invalid(
                "generalized extreme value needs finite parameters and a scale > 0",
            ));
        }
        Ok(Self {
            location,
            scale,
            shape,
        })
    }

    pub fn location(&self) -> f64 {
        self.location
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// `-ln t(x)` where `P(M <= x) = exp(-t(x))`; `NaN` outside the support.
    fn neg_ln_t(&self, x: f64) -> f64 {
        log1p_over_shape(self.shape, (x - self.location) / self.scale)
    }

    /// `P(M <= x)`.
    pub fn cdf(&self, x: f64) -> f64 {
        let exponent = self.neg_ln_t(x);
        if exponent.is_nan() {
            // Below the lower end point (xi > 0) or above the upper (xi < 0).
            return if self.shape > 0.0 { 0.0 } else { 1.0 };
        }
        (-(-exponent).exp()).exp()
    }

    /// Log-density; `-inf` outside the support.
    pub fn ln_pdf(&self, x: f64) -> f64 {
        let exponent = self.neg_ln_t(x);
        if exponent.is_nan() {
            return f64::NEG_INFINITY;
        }
        -self.scale.ln() - (1.0 + self.shape) * exponent - (-exponent).exp()
    }

    /// The `p`-quantile, `0 < p < 1`.
    pub fn quantile(&self, p: f64) -> Result<f64, CopulaError> {
        require_open_probability(p)?;
        Ok(self.quantile_from_reduced(-p.ln()))
    }

    /// Quantile at `y = -ln p`.
    fn quantile_from_reduced(&self, y: f64) -> f64 {
        self.location + self.scale * expm1_over_shape(self.shape, -y.ln())
    }

    /// The level exceeded by a block maximum once every `period` blocks on
    /// average: the `1 - 1/period` quantile. `period > 1`.
    pub fn return_level(&self, period: f64) -> Result<f64, CopulaError> {
        if !period.is_finite() || period <= 1.0 {
            return Err(invalid("return periods must be greater than 1"));
        }
        // -ln(1 - 1/T), accurate for large T.
        Ok(self.quantile_from_reduced(-(-1.0 / period).ln_1p()))
    }

    /// Maximum-likelihood fit to block maxima.
    ///
    /// Starts from the best of the probability-weighted-moment estimates of
    /// Hosking, Wallis and Wood (1985) and two moment-based guesses, and
    /// maximises the likelihood in `(location, ln scale, shape)` with a
    /// quasi-Newton search, falling back to a restarted Nelder–Mead search
    /// if that does not converge.
    /// The shape is restricted to `[-1, 5]`.
    pub fn fit(maxima: &[f64]) -> Result<GevFit, CopulaError> {
        require_finite(maxima)?;
        let n_obs = maxima.len();
        if n_obs < 3 {
            return Err(invalid(
                "a generalized extreme value fit needs at least three block maxima",
            ));
        }
        let n = n_obs as f64;
        let mean = maxima.iter().sum::<f64>() / n;
        let variance = maxima.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        if variance.is_nan() || variance <= 0.0 {
            return Err(invalid("all block maxima are equal"));
        }

        let loglik = |location: f64, log_scale: f64, shape: f64| -> f64 {
            let candidate = Self {
                location,
                scale: log_scale.exp(),
                shape,
            };
            maxima.iter().map(|x| candidate.ln_pdf(*x)).sum()
        };

        let std_dev = variance.sqrt();
        let gumbel_scale = std_dev * 6f64.sqrt() / std::f64::consts::PI;
        let mut starts = vec![[
            mean - std::f64::consts::EULER_GAMMA * gumbel_scale,
            gumbel_scale,
            0.0,
        ]];
        if let Some(pwm) = gev_pwm_start(maxima) {
            starts.push(pwm);
        }
        // The widest start always covers the sample.
        starts.push([mean, 2.0 * std_dev, 0.1]);

        // Search in standardised coordinates so the three parameters have
        // comparable scales: ((location - mean) / sd, ln(scale / sd), shape).
        let objective = |x: &[f64]| loglik(mean + std_dev * x[0], std_dev.ln() + x[1], x[2]);
        let bounds = [(-50.0, 50.0), (1e-8f64.ln(), 1e4f64.ln()), (-1.0, 5.0)];
        let mut starts: Vec<(f64, [f64; 3])> = starts
            .into_iter()
            .map(|start| {
                let x0 = [
                    (start[0] - mean) / std_dev,
                    (start[1] / std_dev).ln(),
                    start[2],
                ];
                (objective(&x0), x0)
            })
            .filter(|(value, _)| value.is_finite())
            .collect();
        starts.sort_by(|a, b| b.0.total_cmp(&a.0));

        // A quasi-Newton search from the best start needs a few dozen passes
        // over the sample. The simplex search is the fallback for a start in
        // a region where the finite-difference model fails; it costs
        // thousands of passes, so it only runs when needed.
        let newton = QuasiNewtonOptions {
            ftol: 1e-11,
            initial_step: 0.05,
            max_iter: 200,
            parallel_gradient: false,
        };
        let simplex = NelderMeadOptions {
            initial_step: 0.1,
            ftol: 1e-12,
            xtol: 1e-10,
            max_iter: 4000,
            restarts: 6,
        };
        let mut best: Option<(f64, Vec<f64>)> = None;
        for (_, x0) in &starts {
            let result = quasi_newton_maximize(x0, &bounds, &newton, objective);
            let (value, x) = if result.converged {
                (result.value, result.x)
            } else {
                let polished = nelder_mead_maximize(&result.x, &bounds, &simplex, objective);
                (polished.value, polished.x)
            };
            if best
                .as_ref()
                .is_none_or(|(incumbent, _)| value > *incumbent)
            {
                best = Some((value, x));
            }
            if result.converged {
                break;
            }
        }
        let (loglik_value, x) = best.ok_or(FitError::Failed {
            reason: "no starting value gave a finite generalized extreme value likelihood",
        })?;
        let x = [mean + std_dev * x[0], std_dev.ln() + x[1], x[2]];
        let model = Self::new(x[0], x[1].exp(), x[2])?;

        let covariance = if model.shape > -1.0 + 1e-6 && model.shape < 5.0 - 1e-6 {
            covariance_at_maximum(3, &[model.scale, model.scale, 1.0], |s| {
                loglik(
                    model.location + model.scale * s[0],
                    model.scale.ln() + s[1],
                    model.shape + s[2],
                )
            })
        } else {
            None
        };

        Ok(GevFit {
            model,
            loglik: loglik_value,
            n_obs,
            covariance,
        })
    }
}

/// Probability-weighted-moment estimates `[location, scale, shape]`, or
/// `None` when they are not usable as a likelihood start.
fn gev_pwm_start(maxima: &[f64]) -> Option<[f64; 3]> {
    let n = maxima.len() as f64;
    let mut sorted = maxima.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (mut b0, mut b1, mut b2) = (0.0, 0.0, 0.0);
    for (idx, value) in sorted.iter().enumerate() {
        let i = idx as f64;
        b0 += value;
        b1 += value * i / (n - 1.0);
        b2 += value * i * (i - 1.0) / ((n - 1.0) * (n - 2.0));
    }
    let (b0, b1, b2) = (b0 / n, b1 / n, b2 / n);
    let c = (2.0 * b1 - b0) / (3.0 * b2 - b0) - 2f64.ln() / 3f64.ln();
    // Hosking's k is the negative of the shape used here.
    let k = 7.859 * c + 2.9554 * c * c;
    if !k.is_finite() || k.abs() < 1e-6 || k <= -0.9 {
        return None;
    }
    let g = gamma(1.0 + k);
    let scale = (2.0 * b1 - b0) * k / (g * (1.0 - 2f64.powf(-k)));
    let location = b0 + scale * (g - 1.0) / k;
    (scale.is_finite() && scale > 0.0 && location.is_finite()).then_some([location, scale, -k])
}

impl GevFit {
    /// Delta-method standard error of [`GeneralizedExtremeValue::return_level`]
    /// at `period`. `None` when the fit has no covariance.
    pub fn return_level_std_error(&self, period: f64) -> Result<Option<f64>, CopulaError> {
        self.model.return_level(period)?;
        let Some(covariance) = &self.covariance else {
            return Ok(None);
        };
        let base = [self.model.location, self.model.scale, self.model.shape];
        let steps = [1e-6 * self.model.scale, 1e-6 * self.model.scale, 1e-6];
        let level = |params: [f64; 3]| -> f64 {
            GeneralizedExtremeValue {
                location: params[0],
                scale: params[1],
                shape: params[2],
            }
            .quantile_from_reduced(-(-1.0 / period).ln_1p())
        };
        let mut gradient = [0.0; 3];
        for idx in 0..3 {
            let (mut up, mut down) = (base, base);
            up[idx] += steps[idx];
            down[idx] -= steps[idx];
            gradient[idx] = (level(up) - level(down)) / (2.0 * steps[idx]);
        }
        let mut variance = 0.0;
        for row in 0..3 {
            for col in 0..3 {
                variance += gradient[row] * covariance[(row, col)] * gradient[col];
            }
        }
        Ok((variance >= 0.0).then(|| variance.sqrt()))
    }
}

/// Maxima of consecutive blocks of `block_size` observations. A final
/// incomplete block is dropped.
pub fn block_maxima(values: &[f64], block_size: usize) -> Result<Vec<f64>, CopulaError> {
    require_finite(values)?;
    if block_size == 0 {
        return Err(invalid("block_size must be at least 1"));
    }
    Ok(values
        .chunks_exact(block_size)
        .map(|block| block.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        .collect())
}

// ---------------------------------------------------------------------------
// Semi-parametric margin
// ---------------------------------------------------------------------------

/// Where a tail starts.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Threshold {
    /// Put this fraction of the sample in the tail, `0 < fraction < 0.5`.
    Fraction(f64),
    /// Start the tail at this value on the raw scale. It is moved to the
    /// nearest observation on the body side, so that the pieces join at a
    /// data point.
    Value(f64),
}

/// How [`SemiParametricMargin::fit`] treats the two tails.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MarginFitOptions {
    /// `None` leaves the lower tail empirical.
    pub lower_tail: Option<Threshold>,
    /// `None` leaves the upper tail empirical.
    pub upper_tail: Option<Threshold>,
    /// Fewest observations beyond a threshold for which a GPD is fitted.
    pub min_exceedances: usize,
}

impl Default for MarginFitOptions {
    /// The lowest and highest tenth of the sample are modelled as tails.
    fn default() -> Self {
        Self {
            lower_tail: Some(Threshold::Fraction(0.1)),
            upper_tail: Some(Threshold::Fraction(0.1)),
            min_exceedances: 10,
        }
    }
}

/// One fitted tail of a [`SemiParametricMargin`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GpdTail {
    /// Raw-scale value where the tail starts.
    pub threshold: f64,
    /// Probability mass beyond the threshold.
    pub probability: f64,
    /// Distribution of the distance beyond the threshold.
    pub gpd: GeneralizedPareto,
    /// Observations the GPD was fitted to.
    pub n_exceedances: usize,
}

/// Distribution function of one variable: interpolated empirical body with
/// generalized Pareto tails. See the [module documentation](self).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemiParametricMargin {
    /// Distinct observations of the body, increasing.
    knots: Vec<f64>,
    /// Distribution function at each knot, strictly increasing.
    probabilities: Vec<f64>,
    lower: Option<GpdTail>,
    upper: Option<GpdTail>,
    n_obs: usize,
}

impl SemiParametricMargin {
    /// Fits the margin to a sample of one variable.
    pub fn fit(values: &[f64], options: &MarginFitOptions) -> Result<Self, CopulaError> {
        require_finite(values)?;
        let n_obs = values.len();
        if n_obs < 2 {
            return Err(invalid("a margin needs at least two observations"));
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        if sorted[0] == sorted[n_obs - 1] {
            return Err(invalid("all observations are equal"));
        }

        // Thresholds are observations: the body runs from `lower_at` to
        // `upper_at` inclusive.
        let lower_at = match options.lower_tail {
            None => sorted[0],
            Some(Threshold::Fraction(fraction)) => {
                sorted[tail_count(fraction, n_obs)?.min(n_obs - 1)]
            }
            Some(Threshold::Value(value)) => {
                if !value.is_finite() {
                    return Err(InputError::NonFiniteValue.into());
                }
                let idx = sorted.partition_point(|x| *x < value);
                sorted[idx.min(n_obs - 1)]
            }
        };
        let upper_at = match options.upper_tail {
            None => sorted[n_obs - 1],
            Some(Threshold::Fraction(fraction)) => {
                sorted[n_obs - 1 - tail_count(fraction, n_obs)?.min(n_obs - 1)]
            }
            Some(Threshold::Value(value)) => {
                if !value.is_finite() {
                    return Err(InputError::NonFiniteValue.into());
                }
                let idx = sorted.partition_point(|x| *x <= value);
                sorted[idx.saturating_sub(1)]
            }
        };
        if lower_at >= upper_at {
            return Err(invalid(
                "the lower threshold must lie below the upper threshold",
            ));
        }

        // Body: distinct observations in [lower_at, upper_at] with the
        // empirical distribution function (#{x_i <= x}) / (n + 1).
        let first = sorted.partition_point(|x| *x < lower_at);
        let last = sorted.partition_point(|x| *x <= upper_at);
        let mut knots = Vec::new();
        let mut probabilities = Vec::new();
        let mut idx = first;
        while idx < last {
            let value = sorted[idx];
            let end = idx + sorted[idx..last].partition_point(|x| *x <= value);
            knots.push(value);
            probabilities.push(end as f64 / (n_obs as f64 + 1.0));
            idx = end;
        }

        let fit_tail = |excesses: Vec<f64>,
                        threshold: f64,
                        probability: f64|
         -> Result<GpdTail, CopulaError> {
            if excesses.len() < options.min_exceedances.max(2) {
                return Err(invalid(
                    "too few observations beyond the threshold to fit a tail; \
                     lower the threshold or min_exceedances, or disable that tail",
                ));
            }
            Ok(GpdTail {
                threshold,
                probability,
                gpd: GeneralizedPareto::fit(&excesses)?.model,
                n_exceedances: excesses.len(),
            })
        };
        let lower = match options.lower_tail {
            None => None,
            Some(_) => Some(fit_tail(
                sorted[..first].iter().map(|x| lower_at - x).collect(),
                lower_at,
                probabilities[0],
            )?),
        };
        let upper = match options.upper_tail {
            None => None,
            Some(_) => Some(fit_tail(
                sorted[last..].iter().map(|x| x - upper_at).collect(),
                upper_at,
                1.0 - probabilities[probabilities.len() - 1],
            )?),
        };

        Ok(Self {
            knots,
            probabilities,
            lower,
            upper,
            n_obs,
        })
    }

    /// Checks the invariants [`fit`](Self::fit) guarantees. Deserialized
    /// margins should be validated before use: the derived deserializer
    /// accepts any numbers.
    pub fn validate(&self) -> Result<(), CopulaError> {
        let bad = || invalid("margin state is inconsistent");
        let last = self.knots.len().checked_sub(1).ok_or_else(bad)?;
        if self.probabilities.len() != self.knots.len() {
            return Err(bad());
        }
        let increasing = |values: &[f64]| {
            values.iter().all(|value| value.is_finite())
                && values.windows(2).all(|pair| pair[0] < pair[1])
        };
        if !increasing(&self.knots)
            || !increasing(&self.probabilities)
            || self.probabilities[0] <= 0.0
            || self.probabilities[last] >= 1.0
        {
            return Err(bad());
        }
        let tail_ok = |tail: &GpdTail, threshold: f64, probability: f64| {
            tail.threshold == threshold
                && (tail.probability - probability).abs() < 1e-12
                && tail.gpd.scale.is_finite()
                && tail.gpd.scale > 0.0
                && tail.gpd.shape.is_finite()
        };
        if let Some(tail) = &self.lower
            && !tail_ok(tail, self.knots[0], self.probabilities[0])
        {
            return Err(bad());
        }
        if let Some(tail) = &self.upper
            && !tail_ok(tail, self.knots[last], 1.0 - self.probabilities[last])
        {
            return Err(bad());
        }
        Ok(())
    }

    /// Number of observations the margin was fitted to.
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// The fitted lower tail, if one was requested.
    pub fn lower_tail(&self) -> Option<&GpdTail> {
        self.lower.as_ref()
    }

    /// The fitted upper tail, if one was requested.
    pub fn upper_tail(&self) -> Option<&GpdTail> {
        self.upper.as_ref()
    }

    /// `P(X <= x)`, in `(0, 1)` wherever the fitted tails have mass.
    ///
    /// Without a fitted tail the function is flat beyond the sample: it
    /// returns `1 / (n + 1)` below the smallest observation and
    /// `n / (n + 1)` above the largest.
    pub fn cdf(&self, x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }
        let last = self.knots.len() - 1;
        if x < self.knots[0] {
            return match &self.lower {
                Some(tail) => tail.probability * tail.gpd.survival(tail.threshold - x),
                None => self.probabilities[0],
            };
        }
        if x > self.knots[last] {
            return match &self.upper {
                Some(tail) => 1.0 - tail.probability * tail.gpd.survival(x - tail.threshold),
                None => self.probabilities[last],
            };
        }
        // knots[idx - 1] <= x < knots[idx], or x is the last knot.
        let idx = self.knots.partition_point(|knot| *knot <= x);
        if idx > last {
            return self.probabilities[last];
        }
        let (x0, x1) = (self.knots[idx - 1], self.knots[idx]);
        let (p0, p1) = (self.probabilities[idx - 1], self.probabilities[idx]);
        p0 + (p1 - p0) * (x - x0) / (x1 - x0)
    }

    /// The `p`-quantile, `0 < p < 1`: the inverse of [`cdf`](Self::cdf).
    ///
    /// Without a fitted tail, probabilities beyond the body map to the
    /// smallest or largest observation.
    pub fn quantile(&self, p: f64) -> Result<f64, CopulaError> {
        require_open_probability(p)?;
        let last = self.knots.len() - 1;
        if p < self.probabilities[0] {
            return Ok(match &self.lower {
                Some(tail) => {
                    tail.threshold - tail.gpd.quantile_from_survival(p / tail.probability)
                }
                None => self.knots[0],
            });
        }
        if p > self.probabilities[last] {
            return Ok(match &self.upper {
                Some(tail) => {
                    tail.threshold
                        + tail
                            .gpd
                            .quantile_from_survival((1.0 - p) / tail.probability)
                }
                None => self.knots[last],
            });
        }
        let idx = self.probabilities.partition_point(|value| *value <= p);
        if idx > last {
            return Ok(self.knots[last]);
        }
        let (x0, x1) = (self.knots[idx - 1], self.knots[idx]);
        let (p0, p1) = (self.probabilities[idx - 1], self.probabilities[idx]);
        Ok(x0 + (x1 - x0) * (p - p0) / (p1 - p0))
    }

    /// The level exceeded once every `period` observations on average: the
    /// `1 - 1/period` quantile. `period > 1`.
    pub fn return_level(&self, period: f64) -> Result<f64, CopulaError> {
        if !period.is_finite() || period <= 1.0 {
            return Err(invalid("return periods must be greater than 1"));
        }
        let exceedance = 1.0 / period;
        match &self.upper {
            // Stay on the survival scale: 1 - 1/period rounds for long periods.
            Some(tail) if exceedance < tail.probability => Ok(tail.threshold
                + tail
                    .gpd
                    .quantile_from_survival(exceedance / tail.probability)),
            _ => self.quantile(1.0 - exceedance),
        }
    }
}

/// Observations in a tail that holds `fraction` of a sample of `n_obs`.
fn tail_count(fraction: f64, n_obs: usize) -> Result<usize, CopulaError> {
    if !fraction.is_finite() || fraction <= 0.0 || fraction >= 0.5 {
        return Err(invalid("tail fractions must lie strictly inside (0, 0.5)"));
    }
    Ok((fraction * n_obs as f64).floor() as usize)
}

/// One [`SemiParametricMargin`] per column of a data matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Margins {
    margins: Vec<SemiParametricMargin>,
}

impl Margins {
    /// Fits every column of an `(n, d)` matrix of raw observations with the
    /// same options.
    pub fn fit(data: ArrayView2<'_, f64>, options: &MarginFitOptions) -> Result<Self, CopulaError> {
        if data.ncols() == 0 {
            return Err(invalid("expected at least one column"));
        }
        let margins = data
            .columns()
            .into_iter()
            .map(|column| SemiParametricMargin::fit(&column.to_vec(), options))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { margins })
    }

    /// Builds the collection from margins fitted one at a time, for example
    /// with different thresholds per column.
    pub fn from_margins(margins: Vec<SemiParametricMargin>) -> Result<Self, CopulaError> {
        if margins.is_empty() {
            return Err(invalid("expected at least one margin"));
        }
        let margins = Self { margins };
        margins.validate()?;
        Ok(margins)
    }

    /// Validates every margin; see [`SemiParametricMargin::validate`].
    pub fn validate(&self) -> Result<(), CopulaError> {
        if self.margins.is_empty() {
            return Err(invalid("expected at least one margin"));
        }
        self.margins.iter().try_for_each(|margin| margin.validate())
    }

    /// Number of columns.
    pub fn dim(&self) -> usize {
        self.margins.len()
    }

    /// The margin of each column.
    pub fn margins(&self) -> &[SemiParametricMargin] {
        &self.margins
    }

    fn check_columns(&self, got: usize) -> Result<(), CopulaError> {
        if got != self.margins.len() {
            return Err(InputError::DimensionMismatch {
                expected: self.margins.len(),
                got,
            }
            .into());
        }
        Ok(())
    }

    /// Applies each column's distribution function: raw values to uniforms.
    ///
    /// Values are kept `1e-12` away from 0 and 1, so the result is always
    /// valid input for a copula (a fitted tail with negative shape assigns
    /// probability exactly 0 or 1 beyond its end point).
    pub fn cdf(&self, data: ArrayView2<'_, f64>) -> Result<Array2<f64>, CopulaError> {
        self.check_columns(data.ncols())?;
        if data.iter().any(|value| !value.is_finite()) {
            return Err(InputError::NonFiniteValue.into());
        }
        let mut uniforms = Array2::zeros(data.raw_dim());
        for (col, margin) in self.margins.iter().enumerate() {
            for row in 0..data.nrows() {
                uniforms[(row, col)] = margin
                    .cdf(data[(row, col)])
                    .clamp(UNIFORM_CLIP, 1.0 - UNIFORM_CLIP);
            }
        }
        Ok(uniforms)
    }

    /// [`cdf`](Self::cdf) wrapped as [`PseudoObs`], ready for a copula fit.
    pub fn to_pseudo_obs(&self, data: ArrayView2<'_, f64>) -> Result<PseudoObs, CopulaError> {
        Ok(PseudoObs::new(self.cdf(data)?)?)
    }

    /// Applies each column's quantile function: uniforms (for example a
    /// sample drawn from a fitted copula) back to the raw scale.
    pub fn quantile(&self, uniforms: ArrayView2<'_, f64>) -> Result<Array2<f64>, CopulaError> {
        self.check_columns(uniforms.ncols())?;
        let mut values = Array2::zeros(uniforms.raw_dim());
        for (col, margin) in self.margins.iter().enumerate() {
            for row in 0..uniforms.nrows() {
                values[(row, col)] = margin.quantile(uniforms[(row, col)])?;
            }
        }
        Ok(values)
    }
}

// ---------------------------------------------------------------------------
// Threshold diagnostics
// ---------------------------------------------------------------------------

/// Mean excess over one threshold. See [`mean_excess`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanExcess {
    pub threshold: f64,
    /// Average of `x - threshold` over observations above the threshold.
    pub mean_excess: f64,
    /// Standard error of that average (`NaN` with a single exceedance).
    pub std_error: f64,
    pub n_exceedances: usize,
}

/// Mean residual life: the average excess over each threshold.
///
/// If excesses over `u0` are generalized Pareto with shape `xi < 1`, the mean
/// excess over any `u > u0` is `(sigma + xi (u - u0)) / (1 - xi)`: linear in
/// `u`. The lowest threshold from which the plot looks linear is a
/// candidate. The slope is positive for a heavy tail, zero for an
/// exponential tail and negative for a bounded one.
///
/// Every threshold must leave at least one observation above it.
pub fn mean_excess(values: &[f64], thresholds: &[f64]) -> Result<Vec<MeanExcess>, CopulaError> {
    require_finite(values)?;
    require_finite(thresholds)?;
    thresholds
        .iter()
        .map(|&threshold| {
            let excesses: Vec<f64> = values
                .iter()
                .filter(|x| **x > threshold)
                .map(|x| x - threshold)
                .collect();
            if excesses.is_empty() {
                return Err(invalid("a threshold has no observations above it"));
            }
            let n = excesses.len() as f64;
            let mean = excesses.iter().sum::<f64>() / n;
            let std_error = if excesses.len() > 1 {
                (excesses.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt()
            } else {
                f64::NAN
            };
            Ok(MeanExcess {
                threshold,
                mean_excess: mean,
                std_error,
                n_exceedances: excesses.len(),
            })
        })
        .collect()
}

/// Generalized Pareto fit above one threshold. See [`threshold_stability`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdFit {
    pub threshold: f64,
    pub n_exceedances: usize,
    pub shape: f64,
    /// `NaN` when the fit has no covariance.
    pub shape_std_error: f64,
    pub scale: f64,
    /// `scale - shape * threshold`, which does not depend on the threshold
    /// once the generalized Pareto model holds.
    pub modified_scale: f64,
    /// Delta-method standard error of `modified_scale`; `NaN` when the fit
    /// has no covariance.
    pub modified_scale_std_error: f64,
}

/// Parameter stability: a generalized Pareto fit above each threshold.
///
/// Above a valid threshold the shape and the modified scale stay constant up
/// to sampling noise. The lowest threshold from which both are flat (within
/// their standard errors) is a candidate.
///
/// Every threshold must leave at least two observations above it.
pub fn threshold_stability(
    values: &[f64],
    thresholds: &[f64],
) -> Result<Vec<ThresholdFit>, CopulaError> {
    require_finite(values)?;
    require_finite(thresholds)?;
    thresholds
        .iter()
        .map(|&threshold| {
            let excesses: Vec<f64> = values
                .iter()
                .filter(|x| **x > threshold)
                .map(|x| x - threshold)
                .collect();
            if excesses.len() < 2 {
                return Err(invalid(
                    "a threshold has fewer than two observations above it",
                ));
            }
            let fit = GeneralizedPareto::fit(&excesses)?;
            let (scale, shape) = (fit.model.scale(), fit.model.shape());
            let (shape_std_error, modified_scale_std_error) = match &fit.covariance {
                Some(cov) => {
                    // Var(scale - shape * u) with gradient (1, -u).
                    let variance = cov[(0, 0)] - 2.0 * threshold * cov[(0, 1)]
                        + threshold * threshold * cov[(1, 1)];
                    (cov[(1, 1)].sqrt(), variance.max(0.0).sqrt())
                }
                None => (f64::NAN, f64::NAN),
            };
            Ok(ThresholdFit {
                threshold,
                n_exceedances: excesses.len(),
                shape,
                shape_std_error,
                scale,
                modified_scale: scale - shape * threshold,
                modified_scale_std_error,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_helpers_are_continuous_at_zero() {
        for z in [0.01, 1.0, 7.5] {
            assert!((log1p_over_shape(1e-13, z) - z).abs() < 1e-12);
            assert!((log1p_over_shape(1e-9, z) - z).abs() < 1e-6);
            assert!((expm1_over_shape(1e-9, z) - z).abs() < 1e-6);
        }
        assert!(log1p_over_shape(-0.5, 2.0).is_nan());
    }

    #[test]
    fn tail_count_rejects_fractions_outside_the_open_half() {
        assert_eq!(tail_count(0.1, 205).unwrap(), 20);
        for fraction in [0.0, 0.5, -0.1, f64::NAN] {
            assert!(tail_count(fraction, 100).is_err());
        }
    }
}
