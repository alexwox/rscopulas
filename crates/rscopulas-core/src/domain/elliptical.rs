use ndarray::Array2;
use rand::Rng;
use rand_distr::{ChiSquared, StandardNormal};
use serde::{Deserialize, Deserializer, Serialize};
use statrs::function::gamma::ln_gamma;

use crate::{
    backend::{ExecutionStrategy, Operation, resolve_strategy, try_fill_rows},
    data::PseudoObs,
    errors::NumericalError,
    errors::{CopulaError, FitError},
    fit::FitResult,
    math::{
        cholesky, log_determinant_from_cholesky, make_spd_correlation, validate_correlation_matrix,
    },
    special::{StudentT, norm_cdf, norm_ppf},
    stats::try_kendall_tau_matrix,
};

use super::{CopulaFamily, CopulaModel, EvalOptions, FitOptions, SampleOptions};

/// Gaussian copula parameterized by a correlation matrix.
#[derive(Debug, Clone, Serialize)]
pub struct GaussianCopula {
    dim: usize,
    correlation: Array2<f64>,
    #[serde(skip)]
    cholesky: Array2<f64>,
    log_det: f64,
}

impl GaussianCopula {
    /// Constructs a Gaussian copula from a valid correlation matrix.
    pub fn new(correlation: Array2<f64>) -> Result<Self, CopulaError> {
        validate_correlation_matrix(&correlation)?;
        let cholesky = cholesky(&correlation)?;
        let log_det = log_determinant_from_cholesky(&cholesky);
        let dim = correlation.ncols();
        Ok(Self {
            dim,
            correlation,
            cholesky,
            log_det,
        })
    }

    /// Fits a Gaussian copula by inverting the Kendall tau matrix.
    pub fn fit(data: &PseudoObs, options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        options.validate()?;
        let tau = try_kendall_tau_matrix(data, options.exec)?;
        let correlation =
            make_spd_correlation(&tau.mapv(|value| (std::f64::consts::FRAC_PI_2 * value).sin()))?;
        let model = Self::new(correlation)?;
        let loglik = model
            .log_pdf(
                data,
                &EvalOptions {
                    exec: options.exec,
                    clip_eps: options.clip_eps,
                },
            )?
            .into_iter()
            .sum::<f64>();
        let parameter_count = (model.dim * (model.dim - 1) / 2) as f64;
        let n_obs = data.n_obs() as f64;
        let diagnostics = super::FitDiagnostics {
            likelihood_kind: crate::domain::LikelihoodKind::Joint,
            loglik,
            aic: 2.0 * parameter_count - 2.0 * loglik,
            bic: parameter_count * n_obs.ln() - 2.0 * loglik,
            converged: true,
            n_iter: 0,
        };

        Ok(FitResult { model, diagnostics })
    }

    /// Returns the fitted correlation matrix.
    pub fn correlation(&self) -> &Array2<f64> {
        &self.correlation
    }
}

#[derive(Deserialize)]
struct GaussianCopulaRepr {
    #[serde(default)]
    dim: Option<usize>,
    correlation: Array2<f64>,
    #[serde(default)]
    log_det: Option<f64>,
}

impl<'de> Deserialize<'de> for GaussianCopula {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = GaussianCopulaRepr::deserialize(deserializer)?;
        let model = Self::new(repr.correlation).map_err(serde::de::Error::custom)?;
        if repr.dim.is_some_and(|dim| dim != model.dim) {
            return Err(serde::de::Error::custom(
                "serialized GaussianCopula dim does not match correlation",
            ));
        }
        if repr.log_det.is_some_and(|log_det| !log_det.is_finite()) {
            return Err(serde::de::Error::custom(
                "serialized GaussianCopula log_det must be finite",
            ));
        }
        Ok(model)
    }
}

impl CopulaModel for GaussianCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::Gaussian
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn log_pdf(&self, data: &PseudoObs, options: &EvalOptions) -> Result<Vec<f64>, CopulaError> {
        options.validate()?;
        crate::data::validate_dim(self.dim, data.dim())?;

        let strategy = resolve_strategy(options.exec, Operation::DensityEval, data.n_obs())?;
        let view = data.as_view();
        let whitening = Whitening::new(&self.cholesky);
        let (low, high) = (options.clip_eps, 1.0 - options.clip_eps);
        let mut out = vec![0.0; data.n_obs()];
        try_fill_rows(
            &mut out,
            strategy,
            || RowScratch::new(self.dim),
            |row_idx, scratch| {
                let mut euclidean = 0.0;
                for (slot, value) in scratch.z.iter_mut().zip(view.row(row_idx)) {
                    let z = norm_ppf(value.clamp(low, high));
                    *slot = z;
                    euclidean += z * z;
                }
                let quadratic = whitening.quadratic(&scratch.z, &mut scratch.work)?;
                Ok(-0.5 * self.log_det - 0.5 * (quadratic - euclidean))
            },
        )?;
        Ok(out)
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        resolve_strategy(options.exec, Operation::Sample, n)?;
        let whitening = Whitening::new(&self.cholesky);
        let dim = self.dim;
        // The RNG stream is consumed row by row exactly as before; only the
        // deterministic transform to uniforms runs in parallel.
        let mut values = vec![0.0; n * dim];
        let mut epsilon = vec![0.0; dim];
        for row in values.chunks_exact_mut(dim) {
            for slot in epsilon.iter_mut() {
                *slot = rng.sample::<f64, _>(StandardNormal);
            }
            whitening.correlate(&epsilon, row);
        }
        map_in_place(&mut values, n, norm_cdf);
        Ok(Array2::from_shape_vec((n, dim), values)
            .expect("sample buffer should match the requested shape"))
    }
}

/// Row-major copy of a lower Cholesky factor: the triangular solve and
/// product in the row kernels read it sequentially instead of through 2-D
/// indexing.
struct Whitening {
    dim: usize,
    lower: Vec<f64>,
}

impl Whitening {
    fn new(cholesky: &Array2<f64>) -> Self {
        let dim = cholesky.nrows();
        let mut lower = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..=i {
                lower[i * dim + j] = cholesky[(i, j)];
            }
        }
        Self { dim, lower }
    }

    /// `zᵀ Σ⁻¹ z = ‖L⁻¹ z‖²` by forward substitution into `work`.
    #[inline]
    fn quadratic(&self, z: &[f64], work: &mut [f64]) -> Result<f64, CopulaError> {
        let dim = self.dim;
        let mut total = 0.0;
        for i in 0..dim {
            let row = &self.lower[i * dim..i * dim + i + 1];
            let mut value = z[i];
            for (l, w) in row[..i].iter().zip(&work[..i]) {
                value -= l * w;
            }
            let pivot = row[i];
            if pivot == 0.0 {
                return Err(NumericalError::DecompositionFailed.into());
            }
            let solved = value / pivot;
            work[i] = solved;
            total += solved * solved;
        }
        Ok(total)
    }

    /// `out = L ε`.
    #[inline]
    fn correlate(&self, epsilon: &[f64], out: &mut [f64]) {
        let dim = self.dim;
        for (i, slot) in out.iter_mut().enumerate() {
            let mut value = 0.0;
            for (l, e) in self.lower[i * dim..i * dim + i + 1].iter().zip(epsilon) {
                value += l * e;
            }
            *slot = value;
        }
    }
}

/// Per-task buffers of the row kernels.
struct RowScratch {
    z: Vec<f64>,
    work: Vec<f64>,
}

impl RowScratch {
    fn new(dim: usize) -> Self {
        Self {
            z: vec![0.0; dim],
            work: vec![0.0; dim],
        }
    }
}

/// Applies `f` to every element, in parallel once there are enough rows for
/// it to pay off (same threshold as the other sampling paths).
fn map_in_place(values: &mut [f64], n_rows: usize, f: impl Fn(f64) -> f64 + Sync + Send) {
    if n_rows >= 128 {
        use rayon::prelude::*;
        values.par_chunks_mut(1024).for_each(|chunk| {
            for value in chunk {
                *value = f(*value);
            }
        });
    } else {
        for value in values {
            *value = f(*value);
        }
    }
}

/// Student t copula parameterized by a correlation matrix and degrees of freedom.
#[derive(Debug, Clone, Serialize)]
pub struct StudentTCopula {
    dim: usize,
    correlation: Array2<f64>,
    degrees_of_freedom: f64,
    #[serde(skip)]
    cholesky: Array2<f64>,
    log_det: f64,
}

impl StudentTCopula {
    /// Constructs a Student t copula from a valid correlation matrix and `nu > 0`.
    pub fn new(correlation: Array2<f64>, degrees_of_freedom: f64) -> Result<Self, CopulaError> {
        if !degrees_of_freedom.is_finite() || degrees_of_freedom <= 0.0 {
            return Err(FitError::Failed {
                reason: "degrees of freedom must be positive",
            }
            .into());
        }

        validate_correlation_matrix(&correlation)?;
        let cholesky = cholesky(&correlation)?;
        let log_det = log_determinant_from_cholesky(&cholesky);
        let dim = correlation.ncols();
        Ok(Self {
            dim,
            correlation,
            degrees_of_freedom,
            cholesky,
            log_det,
        })
    }

    /// Fits a Student t copula by Kendall tau inversion plus a grid search over `nu`.
    pub fn fit(data: &PseudoObs, options: &FitOptions) -> Result<FitResult<Self>, CopulaError> {
        options.validate()?;
        let tau = try_kendall_tau_matrix(data, options.exec)?;
        let correlation =
            make_spd_correlation(&tau.mapv(|value| (std::f64::consts::FRAC_PI_2 * value).sin()))?;

        // Every candidate ν shares the correlation matrix, so it also shares
        // the Cholesky factor and the set of probabilities to invert.
        let template = Self::new(correlation, 4.0)?;
        let strategy = resolve_strategy(options.exec, Operation::DensityEval, data.n_obs())?;
        let plan = QuantilePlan::new(data, options.clip_eps);
        let candidates = candidate_degrees_of_freedom();
        let evaluate = |nu: f64| template.loglik_at(nu, &plan);
        let logliks: Vec<f64> = match strategy {
            ExecutionStrategy::CpuParallel => {
                use rayon::prelude::*;
                candidates
                    .par_iter()
                    .map(|&nu| evaluate(nu))
                    .collect::<Result<_, _>>()?
            }
            _ => candidates
                .iter()
                .map(|&nu| evaluate(nu))
                .collect::<Result<_, _>>()?,
        };

        let mut best: Option<(f64, f64)> = None;
        for (&nu, &loglik) in candidates.iter().zip(&logliks) {
            if loglik > best.map_or(f64::NEG_INFINITY, |(_, value)| value) {
                best = Some((nu, loglik));
            }
        }
        let (nu, best_loglik) = best.ok_or(FitError::Failed {
            reason: "failed to select degrees of freedom",
        })?;
        let model = Self {
            degrees_of_freedom: nu,
            ..template
        };
        let parameter_count = (model.dim * (model.dim - 1) / 2 + 1) as f64;
        let n_obs = data.n_obs() as f64;
        let diagnostics = super::FitDiagnostics {
            likelihood_kind: crate::domain::LikelihoodKind::Joint,
            loglik: best_loglik,
            aic: 2.0 * parameter_count - 2.0 * best_loglik,
            bic: parameter_count * n_obs.ln() - 2.0 * best_loglik,
            converged: true,
            n_iter: candidates.len(),
        };

        Ok(FitResult { model, diagnostics })
    }

    /// `ln Γ((ν+d)/2) − ln Γ(ν/2) − ½ ln|Σ| − (d/2) ln(νπ)` for `nu`.
    fn log_constant(&self, nu: f64) -> f64 {
        let dim = self.dim as f64;
        ln_gamma((nu + dim) / 2.0)
            - ln_gamma(nu / 2.0)
            - 0.5 * self.log_det
            - 0.5 * dim * (nu * std::f64::consts::PI).ln()
    }

    /// Log-likelihood of `plan`'s data under this correlation matrix with
    /// `nu` degrees of freedom. Equal to summing `log_pdf` over the rows, but
    /// each distinct probability is inverted once.
    fn loglik_at(&self, nu: f64, plan: &QuantilePlan) -> Result<f64, CopulaError> {
        let dist = StudentT::new(nu);
        let quantiles: Vec<f64> = plan.unique.iter().map(|&u| dist.ppf(u)).collect();
        let log_marginals: Vec<f64> = quantiles.iter().map(|&x| dist.ln_pdf(x)).collect();
        let whitening = Whitening::new(&self.cholesky);
        let dim = self.dim;
        let constant = self.log_constant(nu);
        let half_power = 0.5 * (nu + dim as f64);
        let mut scratch = RowScratch::new(dim);
        let mut total = 0.0;
        for row in plan.index.chunks_exact(dim) {
            let mut marginal = 0.0;
            for (slot, &idx) in scratch.z.iter_mut().zip(row) {
                *slot = quantiles[idx as usize];
                marginal += log_marginals[idx as usize];
            }
            let quadratic = whitening.quadratic(&scratch.z, &mut scratch.work)?;
            total += constant - half_power * (1.0 + quadratic / nu).ln() - marginal;
        }
        Ok(total)
    }

    /// Returns the fitted correlation matrix.
    pub fn correlation(&self) -> &Array2<f64> {
        &self.correlation
    }

    /// Returns the fitted degrees of freedom.
    pub fn degrees_of_freedom(&self) -> f64 {
        self.degrees_of_freedom
    }

    fn univariate_t(&self) -> StudentT {
        StudentT::new(self.degrees_of_freedom)
    }
}

#[derive(Deserialize)]
struct StudentTCopulaRepr {
    #[serde(default)]
    dim: Option<usize>,
    correlation: Array2<f64>,
    degrees_of_freedom: f64,
    #[serde(default)]
    log_det: Option<f64>,
}

impl<'de> Deserialize<'de> for StudentTCopula {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = StudentTCopulaRepr::deserialize(deserializer)?;
        let model = Self::new(repr.correlation, repr.degrees_of_freedom)
            .map_err(serde::de::Error::custom)?;
        if repr.dim.is_some_and(|dim| dim != model.dim) {
            return Err(serde::de::Error::custom(
                "serialized StudentTCopula dim does not match correlation",
            ));
        }
        if repr.log_det.is_some_and(|log_det| !log_det.is_finite()) {
            return Err(serde::de::Error::custom(
                "serialized StudentTCopula log_det must be finite",
            ));
        }
        Ok(model)
    }
}

impl CopulaModel for StudentTCopula {
    fn family(&self) -> CopulaFamily {
        CopulaFamily::StudentT
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn log_pdf(&self, data: &PseudoObs, options: &EvalOptions) -> Result<Vec<f64>, CopulaError> {
        options.validate()?;
        crate::data::validate_dim(self.dim, data.dim())?;

        let dist = self.univariate_t();
        let nu = self.degrees_of_freedom;
        let constant = self.log_constant(nu);
        let half_power = 0.5 * (nu + self.dim as f64);
        let whitening = Whitening::new(&self.cholesky);
        let (low, high) = (options.clip_eps, 1.0 - options.clip_eps);

        let strategy = resolve_strategy(options.exec, Operation::DensityEval, data.n_obs())?;
        let view = data.as_view();
        let mut out = vec![0.0; data.n_obs()];
        try_fill_rows(
            &mut out,
            strategy,
            || RowScratch::new(self.dim),
            |row_idx, scratch| {
                let mut marginal = 0.0;
                for (slot, value) in scratch.z.iter_mut().zip(view.row(row_idx)) {
                    let x = dist.ppf(value.clamp(low, high));
                    *slot = x;
                    marginal += dist.ln_pdf(x);
                }
                let quadratic = whitening.quadratic(&scratch.z, &mut scratch.work)?;
                Ok(constant - half_power * (1.0 + quadratic / nu).ln() - marginal)
            },
        )?;
        Ok(out)
    }

    fn sample<R: Rng + ?Sized>(
        &self,
        n: usize,
        rng: &mut R,
        options: &SampleOptions,
    ) -> Result<Array2<f64>, CopulaError> {
        resolve_strategy(options.exec, Operation::Sample, n)?;
        let dist = self.univariate_t();
        let chi_squared =
            ChiSquared::new(self.degrees_of_freedom).map_err(|_| FitError::Failed {
                reason: "degrees of freedom must be positive",
            })?;
        let whitening = Whitening::new(&self.cholesky);
        let dim = self.dim;
        // RNG consumption per row is unchanged: `dim` normals, then the
        // chi-square mixing draw.
        let mut values = vec![0.0; n * dim];
        let mut epsilon = vec![0.0; dim];
        for row in values.chunks_exact_mut(dim) {
            for slot in epsilon.iter_mut() {
                *slot = rng.sample::<f64, _>(StandardNormal);
            }
            let scale = (self.degrees_of_freedom / rng.sample::<f64, _>(chi_squared)).sqrt();
            whitening.correlate(&epsilon, row);
            for value in row.iter_mut() {
                *value *= scale;
            }
        }
        map_in_place(&mut values, n, |x| dist.cdf(x));
        Ok(Array2::from_shape_vec((n, dim), values)
            .expect("sample buffer should match the requested shape"))
    }
}

/// The distinct clipped probabilities of a data matrix and, for every cell
/// (row-major), which of them it holds.
///
/// Rank-based pseudo-observations take the same `n` values in every column,
/// so a quantile transform that is evaluated per distinct value does `1/d` of
/// the work of one evaluated per cell.
struct QuantilePlan {
    unique: Vec<f64>,
    index: Vec<u32>,
}

impl QuantilePlan {
    fn new(data: &PseudoObs, clip_eps: f64) -> Self {
        let view = data.as_view();
        let (low, high) = (clip_eps, 1.0 - clip_eps);
        let cells: Vec<f64> = view
            .rows()
            .into_iter()
            .flat_map(|row| row.into_iter().map(move |value| value.clamp(low, high)))
            .collect();
        let mut order: Vec<u32> = (0..cells.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| cells[a as usize].total_cmp(&cells[b as usize]));
        let mut unique = Vec::new();
        let mut index = vec![0u32; cells.len()];
        for &cell in &order {
            let value = cells[cell as usize];
            if unique
                .last()
                .is_none_or(|last: &f64| last.to_bits() != value.to_bits())
            {
                unique.push(value);
            }
            index[cell as usize] = (unique.len() - 1) as u32;
        }
        Self { unique, index }
    }
}

fn candidate_degrees_of_freedom() -> Vec<f64> {
    let min = 2.1_f64.ln();
    let max = 50.0_f64.ln();
    let steps = 40usize;

    (0..steps)
        .map(|idx| {
            let fraction = idx as f64 / (steps - 1) as f64;
            (min + (max - min) * fraction).exp()
        })
        .collect()
}
