//! The binding logic, free of JavaScript types.
//!
//! Everything here takes and returns plain Rust values and reports failures
//! as `String`, so it runs under `cargo test` on any target. `lib.rs` wraps
//! these functions for `wasm-bindgen` and adds nothing of its own.
//!
//! Matrices cross the boundary as a flat row-major `f64` slice plus the
//! number of columns: row `i`, column `j` is `data[i * dim + j]`.

use std::fmt::Display;

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use rscopulas::{
    ClaytonCopula, CopulaModel, EvalOptions, ExecPolicy, FitDiagnostics, FitOptions, FrankCopula,
    GaussianCopula, GumbelHougaardCopula, PairCopulaFamily, PairCopulaParams, PairCopulaSpec,
    PseudoObs, Rotation, SampleOptions, SelectionCriterion, StudentTCopula, VineCopula,
    VineFitOptions, paircopula::fit_pair_copula, stats::try_kendall_tau_matrix,
};
use serde::Deserialize;

pub type ApiResult<T> = Result<T, String>;

fn text(error: impl Display) -> String {
    error.to_string()
}

/// Largest integer a JavaScript number holds exactly.
const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;

// ---------------------------------------------------------------------------
// Matrices, seeds
// ---------------------------------------------------------------------------

/// Reads a flat row-major slice as an `(n, dim)` matrix.
pub fn matrix(data: &[f64], dim: usize) -> ApiResult<Array2<f64>> {
    if dim == 0 {
        return Err("dim must be at least 1".into());
    }
    if data.is_empty() {
        return Err("expected at least one observation".into());
    }
    if !data.len().is_multiple_of(dim) {
        return Err(format!(
            "data has {} values, which is not a multiple of dim = {dim}",
            data.len()
        ));
    }
    Array2::from_shape_vec((data.len() / dim, dim), data.to_vec()).map_err(text)
}

fn pseudo_obs(data: &[f64], dim: usize) -> ApiResult<PseudoObs> {
    PseudoObs::new(matrix(data, dim)?).map_err(text)
}

/// Flattens a matrix row by row.
pub fn flatten(values: &Array2<f64>) -> Vec<f64> {
    values.iter().copied().collect()
}

/// A JavaScript seed: absent, or a non-negative integer up to `2^53 - 1`.
pub fn seed(value: Option<f64>) -> ApiResult<Option<u64>> {
    match value {
        None => Ok(None),
        Some(value)
            if value.is_finite()
                && value >= 0.0
                && value.fract() == 0.0
                && value <= MAX_SAFE_INTEGER =>
        {
            Ok(Some(value as u64))
        }
        Some(value) => Err(format!(
            "seed must be a non-negative integer up to 2^53 - 1, got {value}"
        )),
    }
}

/// The generator the Python package uses for the same seed, so that seeded
/// samples agree across languages. Without a seed, entropy comes from the
/// host (`crypto.getRandomValues` in a browser or Node).
fn rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    }
}

/// Rank transform of every column: `rank / (n + 1)`, ties sharing the
/// average rank. Matches `rscopulas.to_pseudo_obs` in Python.
pub fn rank_pseudo_obs(data: &[f64], dim: usize) -> ApiResult<Vec<f64>> {
    let values = matrix(data, dim)?;
    if values.iter().any(|value| value.is_nan()) {
        return Err("cannot rank NaN values; drop or impute them first".into());
    }
    let n = values.nrows();
    let mut ranks = Array2::<f64>::zeros(values.raw_dim());
    for col in 0..dim {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|a, b| values[(*a, col)].total_cmp(&values[(*b, col)]));
        let mut start = 0;
        while start < n {
            let mut end = start + 1;
            while end < n && values[(order[end], col)] == values[(order[start], col)] {
                end += 1;
            }
            // Sorted positions start..end hold ranks start+1 ..= end.
            let shared = (start + 1 + end) as f64 / 2.0;
            for position in start..end {
                ranks[(order[position], col)] = shared / (n as f64 + 1.0);
            }
            start = end;
        }
    }
    Ok(flatten(&ranks))
}

/// Kendall's tau for every pair of columns, as a flat `(dim, dim)` matrix.
pub fn kendall_tau(data: &[f64], dim: usize) -> ApiResult<Vec<f64>> {
    let data = pseudo_obs(data, dim)?;
    try_kendall_tau_matrix(&data, ExecPolicy::Auto)
        .map(|tau| flatten(&tau))
        .map_err(text)
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

pub fn pair_family(name: &str) -> ApiResult<PairCopulaFamily> {
    use PairCopulaFamily as F;
    Ok(match name.trim().to_ascii_lowercase().as_str() {
        "independence" => F::Independence,
        "gaussian" => F::Gaussian,
        "student_t" | "student-t" | "studentt" | "student" => F::StudentT,
        "clayton" => F::Clayton,
        "frank" => F::Frank,
        "gumbel" => F::Gumbel,
        "joe" => F::Joe,
        "bb1" => F::Bb1,
        "bb6" => F::Bb6,
        "bb7" => F::Bb7,
        "bb8" => F::Bb8,
        "tawn1" => F::Tawn1,
        "tawn2" => F::Tawn2,
        "tll" => F::Tll,
        "khoudraji" => F::Khoudraji,
        other => {
            return Err(format!(
                "unsupported pair family '{other}'; expected one of independence, gaussian, \
                 student_t, clayton, frank, gumbel, joe, bb1, bb6, bb7, bb8, tawn1, tawn2, tll, \
                 khoudraji"
            ));
        }
    })
}

pub fn pair_family_name(family: PairCopulaFamily) -> &'static str {
    use PairCopulaFamily as F;
    match family {
        F::Independence => "independence",
        F::Gaussian => "gaussian",
        F::StudentT => "student_t",
        F::Clayton => "clayton",
        F::Frank => "frank",
        F::Gumbel => "gumbel",
        F::Joe => "joe",
        F::Bb1 => "bb1",
        F::Bb6 => "bb6",
        F::Bb7 => "bb7",
        F::Bb8 => "bb8",
        F::Tawn1 => "tawn1",
        F::Tawn2 => "tawn2",
        F::Tll => "tll",
        F::Khoudraji => "khoudraji",
    }
}

pub fn rotation(name: &str) -> ApiResult<Rotation> {
    Ok(match name.trim().to_ascii_uppercase().as_str() {
        "R0" => Rotation::R0,
        "R90" => Rotation::R90,
        "R180" => Rotation::R180,
        "R270" => Rotation::R270,
        other => {
            return Err(format!(
                "unsupported rotation '{other}'; expected one of R0, R90, R180, R270"
            ));
        }
    })
}

pub fn rotation_name(rotation: Rotation) -> &'static str {
    match rotation {
        Rotation::R0 => "R0",
        Rotation::R90 => "R90",
        Rotation::R180 => "R180",
        Rotation::R270 => "R270",
    }
}

fn criterion(name: &str) -> ApiResult<SelectionCriterion> {
    let lower = name.trim().to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("mbicv") {
        let psi0 = match rest.strip_prefix(':') {
            None if rest.is_empty() => 0.9,
            Some(value) => value
                .parse::<f64>()
                .map_err(|err| format!("invalid psi0 in mbicv criterion: {err}"))?,
            None => return Err(format!("unsupported selection criterion '{lower}'")),
        };
        if !(0.0 < psi0 && psi0 < 1.0) {
            return Err(format!("mbicv psi0 must lie in (0, 1), got {psi0}"));
        }
        return Ok(SelectionCriterion::Mbicv { psi0 });
    }
    match lower.as_str() {
        "aic" => Ok(SelectionCriterion::Aic),
        "bic" => Ok(SelectionCriterion::Bic),
        other => Err(format!(
            "unsupported selection criterion '{other}'; expected 'aic', 'bic', 'mbicv' or \
             'mbicv:<psi0>'"
        )),
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Options for vine and pair fitting, as the JSON form of the JavaScript
/// options object. Every field is optional; unknown fields are an error so
/// that a misspelt option is not silently ignored.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FitConfig {
    /// `"r"` (default), `"c"` or `"d"`. Vine fits only.
    pub kind: Option<String>,
    /// Pair families to choose from.
    pub family_set: Option<Vec<String>>,
    /// Also try the 90/180/270 degree rotations of each family.
    pub include_rotations: Option<bool>,
    /// `"aic"` (default), `"bic"`, `"mbicv"` or `"mbicv:<psi0>"`.
    pub criterion: Option<String>,
    /// Keep this many trees; the rest are independence.
    pub truncation_level: Option<usize>,
    /// Significance level of the Kendall tau independence pre-test.
    pub independence_test_level: Option<f64>,
    /// Variable order for `"c"` and `"d"` vines.
    pub order: Option<Vec<usize>>,
}

impl FitConfig {
    /// Parses the JSON text of an options object; `None` means defaults.
    pub fn parse(json: Option<&str>) -> ApiResult<Self> {
        match json {
            None => Ok(Self::default()),
            Some(json) => {
                serde_json::from_str(json).map_err(|err| format!("invalid options: {err}"))
            }
        }
    }

    fn vine_options(&self) -> ApiResult<VineFitOptions> {
        let mut options = VineFitOptions::default();
        if let Some(families) = &self.family_set {
            options.family_set = families
                .iter()
                .map(|name| pair_family(name))
                .collect::<ApiResult<Vec<_>>>()?;
        }
        if let Some(include) = self.include_rotations {
            options.include_rotations = include;
        }
        if let Some(name) = &self.criterion {
            options.criterion = criterion(name)?;
        }
        options.truncation_level = self.truncation_level;
        if let Some(alpha) = self.independence_test_level {
            if !(alpha.is_finite() && 0.0 < alpha && alpha < 1.0) {
                return Err(format!(
                    "independenceTestLevel must lie in (0, 1), got {alpha}"
                ));
            }
            options.independence_test_level = Some(alpha);
        }
        Ok(options)
    }
}

// ---------------------------------------------------------------------------
// Multivariate models
// ---------------------------------------------------------------------------

/// The models reachable from JavaScript.
// One model per JavaScript object: boxing the large variants would only add
// an allocation.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum Kind {
    Gaussian(GaussianCopula),
    StudentT(StudentTCopula),
    Clayton(ClaytonCopula),
    Frank(FrankCopula),
    Gumbel(GumbelHougaardCopula),
    Vine(VineCopula),
}

/// Runs the same expression on whichever model is held.
macro_rules! with_model {
    ($kind:expr, $model:ident => $body:expr) => {
        match $kind {
            Kind::Gaussian($model) => $body,
            Kind::StudentT($model) => $body,
            Kind::Clayton($model) => $body,
            Kind::Frank($model) => $body,
            Kind::Gumbel($model) => $body,
            Kind::Vine($model) => $body,
        }
    };
}

/// A model, with the diagnostics of the fit that produced it (if any).
#[derive(Debug, Clone)]
pub struct Model {
    pub kind: Kind,
    pub diagnostics: Option<FitDiagnostics>,
}

impl Model {
    fn fitted<M>(result: rscopulas::fit::FitResult<M>, wrap: fn(M) -> Kind) -> Self {
        Self {
            kind: wrap(result.model),
            diagnostics: Some(result.diagnostics),
        }
    }

    fn built(kind: Kind) -> Self {
        Self {
            kind,
            diagnostics: None,
        }
    }

    pub fn fit_gaussian(data: &[f64], dim: usize) -> ApiResult<Self> {
        let data = pseudo_obs(data, dim)?;
        GaussianCopula::fit(&data, &FitOptions::default())
            .map(|result| Self::fitted(result, Kind::Gaussian))
            .map_err(text)
    }

    pub fn fit_student_t(data: &[f64], dim: usize) -> ApiResult<Self> {
        let data = pseudo_obs(data, dim)?;
        StudentTCopula::fit(&data, &FitOptions::default())
            .map(|result| Self::fitted(result, Kind::StudentT))
            .map_err(text)
    }

    pub fn fit_clayton(data: &[f64], dim: usize) -> ApiResult<Self> {
        let data = pseudo_obs(data, dim)?;
        ClaytonCopula::fit(&data, &FitOptions::default())
            .map(|result| Self::fitted(result, Kind::Clayton))
            .map_err(text)
    }

    pub fn fit_frank(data: &[f64], dim: usize) -> ApiResult<Self> {
        let data = pseudo_obs(data, dim)?;
        FrankCopula::fit(&data, &FitOptions::default())
            .map(|result| Self::fitted(result, Kind::Frank))
            .map_err(text)
    }

    pub fn fit_gumbel(data: &[f64], dim: usize) -> ApiResult<Self> {
        let data = pseudo_obs(data, dim)?;
        GumbelHougaardCopula::fit(&data, &FitOptions::default())
            .map(|result| Self::fitted(result, Kind::Gumbel))
            .map_err(text)
    }

    pub fn fit_vine(data: &[f64], dim: usize, config: &FitConfig) -> ApiResult<Self> {
        let data = pseudo_obs(data, dim)?;
        let options = config.vine_options()?;
        let kind = config
            .kind
            .as_deref()
            .unwrap_or("r")
            .trim()
            .to_ascii_lowercase();
        let result = match (kind.as_str(), &config.order) {
            ("r", None) => VineCopula::fit_r_vine(&data, &options),
            ("r", Some(_)) => return Err("'order' applies to 'c' and 'd' vines only".into()),
            ("c", None) => VineCopula::fit_c_vine(&data, &options),
            ("c", Some(order)) => VineCopula::fit_c_vine_with_order(&data, order, &options),
            ("d", None) => VineCopula::fit_d_vine(&data, &options),
            ("d", Some(order)) => VineCopula::fit_d_vine_with_order(&data, order, &options),
            (other, _) => {
                return Err(format!(
                    "unsupported vine kind '{other}'; expected 'r', 'c' or 'd'"
                ));
            }
        };
        result
            .map(|result| Self::fitted(result, Kind::Vine))
            .map_err(text)
    }

    pub fn gaussian(correlation: &[f64], dim: usize) -> ApiResult<Self> {
        let correlation = square(correlation, dim)?;
        GaussianCopula::new(correlation)
            .map(|model| Self::built(Kind::Gaussian(model)))
            .map_err(text)
    }

    pub fn student_t(correlation: &[f64], dim: usize, degrees_of_freedom: f64) -> ApiResult<Self> {
        let correlation = square(correlation, dim)?;
        StudentTCopula::new(correlation, degrees_of_freedom)
            .map(|model| Self::built(Kind::StudentT(model)))
            .map_err(text)
    }

    pub fn clayton(dim: usize, theta: f64) -> ApiResult<Self> {
        ClaytonCopula::new(dim, theta)
            .map(|model| Self::built(Kind::Clayton(model)))
            .map_err(text)
    }

    pub fn frank(dim: usize, theta: f64) -> ApiResult<Self> {
        FrankCopula::new(dim, theta)
            .map(|model| Self::built(Kind::Frank(model)))
            .map_err(text)
    }

    pub fn gumbel(dim: usize, theta: f64) -> ApiResult<Self> {
        GumbelHougaardCopula::new(dim, theta)
            .map(|model| Self::built(Kind::Gumbel(model)))
            .map_err(text)
    }

    /// Loads a model from the JSON that `to_json` here, or `to_json()` on the
    /// matching Python class, produced.
    pub fn from_json(family: &str, payload: &str) -> ApiResult<Self> {
        fn parse<T: serde::de::DeserializeOwned>(payload: &str, label: &str) -> ApiResult<T> {
            serde_json::from_str(payload)
                .map_err(|err| format!("failed to deserialize {label}: {err}"))
        }
        let rebuilt = |result: Result<Kind, rscopulas::CopulaError>, label: &str| {
            result.map_err(|err| format!("failed to deserialize {label}: {err}"))
        };
        let kind = match family.trim().to_ascii_lowercase().as_str() {
            "gaussian" => Kind::Gaussian(parse(payload, "Gaussian copula")?),
            "student_t" => Kind::StudentT(parse(payload, "Student t copula")?),
            // The derived deserializers do not check `theta`; go through the
            // checked constructors.
            "clayton" => {
                let parsed: ClaytonCopula = parse(payload, "Clayton copula")?;
                rebuilt(
                    ClaytonCopula::new(parsed.dim(), parsed.theta()).map(Kind::Clayton),
                    "Clayton copula",
                )?
            }
            "frank" => {
                let parsed: FrankCopula = parse(payload, "Frank copula")?;
                rebuilt(
                    FrankCopula::new(parsed.dim(), parsed.theta()).map(Kind::Frank),
                    "Frank copula",
                )?
            }
            "gumbel" => {
                let parsed: GumbelHougaardCopula = parse(payload, "Gumbel copula")?;
                rebuilt(
                    GumbelHougaardCopula::new(parsed.dim(), parsed.theta()).map(Kind::Gumbel),
                    "Gumbel copula",
                )?
            }
            "vine" => Kind::Vine(parse(payload, "vine copula")?),
            other => {
                return Err(format!(
                    "unsupported family '{other}'; expected one of gaussian, student_t, clayton, \
                     frank, gumbel, vine"
                ));
            }
        };
        Ok(Self::built(kind))
    }

    pub fn to_json(&self) -> ApiResult<String> {
        with_model!(&self.kind, model => serde_json::to_string(model)).map_err(text)
    }

    pub fn family(&self) -> &'static str {
        match &self.kind {
            Kind::Gaussian(_) => "gaussian",
            Kind::StudentT(_) => "student_t",
            Kind::Clayton(_) => "clayton",
            Kind::Frank(_) => "frank",
            Kind::Gumbel(_) => "gumbel",
            Kind::Vine(_) => "vine",
        }
    }

    pub fn dim(&self) -> usize {
        with_model!(&self.kind, model => model.dim())
    }

    /// Log-density of every row.
    pub fn log_pdf(&self, data: &[f64]) -> ApiResult<Vec<f64>> {
        let data = pseudo_obs(data, self.dim())?;
        with_model!(&self.kind, model => model.log_pdf(&data, &EvalOptions::default()))
            .map_err(text)
    }

    /// `n` draws, as a flat `(n, dim)` matrix.
    pub fn sample(&self, n: usize, seed: Option<u64>) -> ApiResult<Vec<f64>> {
        if n == 0 {
            return Err("n must be at least 1".into());
        }
        let mut rng = rng(seed);
        with_model!(&self.kind, model => model.sample(n, &mut rng, &SampleOptions::default()))
            .map(|sample| flatten(&sample))
            .map_err(text)
    }

    fn vine(&self) -> ApiResult<&VineCopula> {
        match &self.kind {
            Kind::Vine(vine) => Ok(vine),
            _ => Err(format!(
                "Rosenblatt transforms are available for vine copulas only, not '{}'",
                self.family()
            )),
        }
    }

    /// Vine only: dependent uniforms to independent uniforms.
    pub fn rosenblatt(&self, data: &[f64]) -> ApiResult<Vec<f64>> {
        let vine = self.vine()?;
        let data = matrix(data, self.dim())?;
        vine.rosenblatt(data.view(), &SampleOptions::default())
            .map(|values| flatten(&values))
            .map_err(text)
    }

    /// Vine only: independent uniforms to a draw from the vine.
    pub fn inverse_rosenblatt(&self, data: &[f64]) -> ApiResult<Vec<f64>> {
        let vine = self.vine()?;
        let data = matrix(data, self.dim())?;
        vine.inverse_rosenblatt(data.view(), &SampleOptions::default())
            .map(|values| flatten(&values))
            .map_err(text)
    }
}

fn square(values: &[f64], dim: usize) -> ApiResult<Array2<f64>> {
    if dim == 0 || values.len() != dim * dim {
        return Err(format!(
            "expected a {dim} x {dim} correlation matrix ({} values), got {}",
            dim * dim,
            values.len()
        ));
    }
    Array2::from_shape_vec((dim, dim), values.to_vec()).map_err(text)
}

// ---------------------------------------------------------------------------
// Pair copulas
// ---------------------------------------------------------------------------

/// A bivariate pair copula, with the scores of the fit that chose it (if any).
#[derive(Debug, Clone)]
pub struct Pair {
    pub spec: PairCopulaSpec,
    /// `(loglik, aic, bic)`.
    pub scores: Option<(f64, f64, f64)>,
}

const PAIR_CLIP: f64 = 1e-12;

impl Pair {
    pub fn new(family: &str, parameters: &[f64], rotation_name: Option<&str>) -> ApiResult<Self> {
        use PairCopulaFamily as F;
        let family = pair_family(family)?;
        let params = match (family, parameters) {
            (F::Independence, []) => PairCopulaParams::None,
            (F::Gaussian | F::Clayton | F::Frank | F::Gumbel | F::Joe, [value]) => {
                PairCopulaParams::One(*value)
            }
            (
                F::StudentT | F::Bb1 | F::Bb6 | F::Bb7 | F::Bb8 | F::Tawn1 | F::Tawn2,
                [first, second],
            ) => PairCopulaParams::Two(*first, *second),
            (F::Khoudraji | F::Tll, _) => {
                return Err(format!(
                    "'{}' pair copulas cannot be built from a parameter list; fit them or load \
                     them with fromJson",
                    pair_family_name(family)
                ));
            }
            (_, values) => {
                return Err(format!(
                    "wrong number of parameters for '{}' (got {})",
                    pair_family_name(family),
                    values.len()
                ));
            }
        };
        let spec = PairCopulaSpec {
            family,
            rotation: rotation(rotation_name.unwrap_or("R0"))?,
            params,
        };
        spec.validate().map_err(text)?;
        Ok(Self { spec, scores: None })
    }

    /// Chooses the best family for one pair of columns by the criterion in
    /// `config` (`kind`, `truncationLevel` and `order` do not apply).
    pub fn fit(first: &[f64], second: &[f64], config: &FitConfig) -> ApiResult<Self> {
        let options = config.vine_options()?;
        let result = fit_pair_copula(first, second, &options).map_err(text)?;
        Ok(Self {
            spec: result.spec,
            scores: Some((result.loglik, result.aic, result.bic)),
        })
    }

    pub fn from_json(payload: &str) -> ApiResult<Self> {
        let spec: PairCopulaSpec = serde_json::from_str(payload)
            .map_err(|err| format!("failed to deserialize pair copula: {err}"))?;
        spec.validate()
            .map_err(|err| format!("failed to deserialize pair copula: {err}"))?;
        Ok(Self { spec, scores: None })
    }

    pub fn to_json(&self) -> ApiResult<String> {
        serde_json::to_string(&self.spec).map_err(text)
    }

    pub fn parameters(&self) -> Vec<f64> {
        self.spec.flat_parameters()
    }

    fn each(
        &self,
        first: &[f64],
        second: &[f64],
        f: impl Fn(&PairCopulaSpec, f64, f64, f64) -> Result<f64, rscopulas::CopulaError>,
    ) -> ApiResult<Vec<f64>> {
        if first.len() != second.len() {
            return Err(format!(
                "the two inputs must have the same length (got {} and {})",
                first.len(),
                second.len()
            ));
        }
        first
            .iter()
            .zip(second)
            .map(|(a, b)| f(&self.spec, *a, *b, PAIR_CLIP))
            .collect::<Result<Vec<_>, _>>()
            .map_err(text)
    }

    pub fn log_pdf(&self, u1: &[f64], u2: &[f64]) -> ApiResult<Vec<f64>> {
        self.each(u1, u2, PairCopulaSpec::log_pdf)
    }

    /// `P(U1 <= u1 | U2 = u2)`.
    pub fn cond_first_given_second(&self, u1: &[f64], u2: &[f64]) -> ApiResult<Vec<f64>> {
        self.each(u1, u2, PairCopulaSpec::cond_first_given_second)
    }

    /// `P(U2 <= u2 | U1 = u1)`.
    pub fn cond_second_given_first(&self, u1: &[f64], u2: &[f64]) -> ApiResult<Vec<f64>> {
        self.each(u1, u2, PairCopulaSpec::cond_second_given_first)
    }

    /// The `u1` with `P(U1 <= u1 | U2 = u2) = p`.
    pub fn inv_first_given_second(&self, p: &[f64], u2: &[f64]) -> ApiResult<Vec<f64>> {
        self.each(p, u2, PairCopulaSpec::inv_first_given_second)
    }

    /// The `u2` with `P(U2 <= u2 | U1 = u1) = p`.
    pub fn inv_second_given_first(&self, u1: &[f64], p: &[f64]) -> ApiResult<Vec<f64>> {
        self.each(u1, p, PairCopulaSpec::inv_second_given_first)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DATA: [f64; 16] = [
        0.12, 0.18, 0.21, 0.25, 0.27, 0.22, 0.35, 0.42, 0.48, 0.51, 0.56, 0.49, 0.68, 0.73, 0.82,
        0.79,
    ];

    #[test]
    fn matrices_are_row_major() {
        let values = matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3).unwrap();
        assert_eq!(values.dim(), (2, 3));
        assert_eq!(values[(1, 0)], 4.0);
        assert_eq!(flatten(&values), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(matrix(&[1.0, 2.0, 3.0], 2).is_err());
        assert!(matrix(&[], 2).is_err());
        assert!(matrix(&[1.0], 0).is_err());
    }

    #[test]
    fn seeds_must_be_exact_non_negative_integers() {
        assert_eq!(seed(None).unwrap(), None);
        assert_eq!(seed(Some(0.0)).unwrap(), Some(0));
        assert_eq!(
            seed(Some(MAX_SAFE_INTEGER)).unwrap(),
            Some(9_007_199_254_740_991)
        );
        for bad in [-1.0, 0.5, f64::NAN, f64::INFINITY, MAX_SAFE_INTEGER + 2.0] {
            assert!(seed(Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn ranks_share_ties_and_stay_inside_the_unit_interval() {
        // Column 0: 3, 1, 2, 2 -> ranks 4, 1, 2.5, 2.5 over n + 1 = 5.
        let ranks = rank_pseudo_obs(&[3.0, 9.0, 1.0, 8.0, 2.0, 7.0, 2.0, 6.0], 2).unwrap();
        assert_eq!(ranks, vec![0.8, 0.8, 0.2, 0.6, 0.5, 0.4, 0.5, 0.2]);
        assert!(rank_pseudo_obs(&[1.0, f64::NAN], 1).is_err());
    }

    #[test]
    fn options_parse_from_camel_case_and_reject_unknown_fields() {
        let config = FitConfig::parse(Some(
            r#"{"kind":"d","familySet":["gaussian","clayton"],"includeRotations":false,
                "criterion":"mbicv:0.8","truncationLevel":1,"independenceTestLevel":0.05}"#,
        ))
        .unwrap();
        let options = config.vine_options().unwrap();
        assert_eq!(
            options.family_set,
            vec![PairCopulaFamily::Gaussian, PairCopulaFamily::Clayton]
        );
        assert!(!options.include_rotations);
        assert_eq!(options.truncation_level, Some(1));
        assert_eq!(options.independence_test_level, Some(0.05));
        assert!(matches!(options.criterion, SelectionCriterion::Mbicv { psi0 } if psi0 == 0.8));

        assert!(FitConfig::parse(None).unwrap().kind.is_none());
        assert!(FitConfig::parse(Some(r#"{"truncationLvl":1}"#)).is_err());
        let bad = |json: &str| {
            FitConfig::parse(Some(json))
                .unwrap()
                .vine_options()
                .is_err()
        };
        assert!(bad(r#"{"criterion":"mbicv:2"}"#));
        assert!(bad(r#"{"criterion":"r2"}"#));
        assert!(bad(r#"{"familySet":["nope"]}"#));
        assert!(bad(r#"{"independenceTestLevel":1.5}"#));
    }

    #[test]
    fn models_fit_evaluate_sample_and_round_trip() {
        type Fit = fn(&[f64], usize) -> ApiResult<Model>;
        let fits: [Fit; 5] = [
            Model::fit_gaussian,
            Model::fit_student_t,
            Model::fit_clayton,
            Model::fit_frank,
            Model::fit_gumbel,
        ];
        for fit in fits {
            let model = fit(&DATA, 2).unwrap();
            assert_eq!(model.dim(), 2);
            assert!(model.diagnostics.as_ref().unwrap().loglik.is_finite());
            let log_pdf = model.log_pdf(&DATA).unwrap();
            assert_eq!(log_pdf.len(), 8);
            let sample = model.sample(5, Some(3)).unwrap();
            assert_eq!(sample.len(), 10);
            assert_eq!(sample, model.sample(5, Some(3)).unwrap());

            let restored = Model::from_json(model.family(), &model.to_json().unwrap()).unwrap();
            assert!(restored.diagnostics.is_none());
            assert_eq!(restored.log_pdf(&DATA).unwrap(), log_pdf);
            assert!(model.rosenblatt(&DATA).is_err());
        }
        assert!(Model::from_json("clayton", r#"{"dim":2,"theta":-1.0}"#).is_err());
        assert!(Model::from_json("unknown", "{}").is_err());
        assert!(Model::fit_gaussian(&[0.5, 1.5], 2).is_err());
    }

    #[test]
    fn vines_fit_in_every_structure_and_invert_their_rosenblatt_transform() {
        let truth = Model::clayton(3, 2.0).unwrap();
        let data = truth.sample(300, Some(1)).unwrap();
        for kind in ["r", "c", "d"] {
            let config =
                FitConfig::parse(Some(&format!(r#"{{"kind":"{kind}","criterion":"bic"}}"#)))
                    .unwrap();
            let vine = Model::fit_vine(&data, 3, &config).unwrap();
            assert_eq!(vine.family(), "vine");
            let rows = &data[..30];
            let independent = vine.rosenblatt(rows).unwrap();
            let back = vine.inverse_rosenblatt(&independent).unwrap();
            for (got, expected) in back.iter().zip(rows) {
                assert!((got - expected).abs() < 1e-8, "{kind}: {got} vs {expected}");
            }
            let restored = Model::from_json("vine", &vine.to_json().unwrap()).unwrap();
            assert_eq!(restored.log_pdf(rows).unwrap(), vine.log_pdf(rows).unwrap());
        }
        let ordered = FitConfig::parse(Some(r#"{"kind":"c","order":[2,0,1]}"#)).unwrap();
        assert!(Model::fit_vine(&data, 3, &ordered).is_ok());
        let misplaced = FitConfig::parse(Some(r#"{"kind":"r","order":[2,0,1]}"#)).unwrap();
        assert!(Model::fit_vine(&data, 3, &misplaced).is_err());
        let unknown = FitConfig::parse(Some(r#"{"kind":"x"}"#)).unwrap();
        assert!(Model::fit_vine(&data, 3, &unknown).is_err());
    }

    #[test]
    fn pair_copulas_build_evaluate_and_invert() {
        let pair = Pair::new("clayton", &[2.0], Some("R180")).unwrap();
        assert_eq!(pair.parameters(), vec![2.0]);
        let (u1, u2) = ([0.2, 0.7], [0.4, 0.9]);
        let h = pair.cond_first_given_second(&u1, &u2).unwrap();
        let back = pair.inv_first_given_second(&h, &u2).unwrap();
        assert!((back[0] - 0.2).abs() < 1e-10 && (back[1] - 0.7).abs() < 1e-10);
        assert_eq!(
            Pair::from_json(&pair.to_json().unwrap())
                .unwrap()
                .log_pdf(&u1, &u2)
                .unwrap(),
            pair.log_pdf(&u1, &u2).unwrap()
        );

        assert!(Pair::new("clayton", &[1.0, 2.0], None).is_err());
        assert!(Pair::new("clayton", &[-2.0], None).is_err());
        assert!(Pair::new("clayton", &[2.0], Some("R45")).is_err());
        assert!(Pair::new("tll", &[], None).is_err());
        assert!(pair.log_pdf(&u1, &[0.5]).is_err());
        assert_eq!(
            Pair::new("independence", &[], None)
                .unwrap()
                .log_pdf(&u1, &u2)
                .unwrap(),
            vec![0.0, 0.0]
        );
    }
}
