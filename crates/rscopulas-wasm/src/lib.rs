//! WebAssembly bindings for `rscopulas`.
//!
//! The same Rust core that backs the Python package, compiled to
//! WebAssembly for browsers and Node.js. See the README of this crate for
//! the JavaScript API; the logic lives in [`api`] and this file only adapts
//! it to `wasm-bindgen`.
//!
//! Conventions on the JavaScript side:
//!
//! - A matrix is a row-major `Float64Array` plus its number of columns.
//! - Failures throw an `Error` whose message is the core's message.
//! - A seed is a non-negative integer; the same seed gives the same sample
//!   as the Python package.

pub mod api;

use wasm_bindgen::prelude::*;

use api::{ApiResult, FitConfig, Model, Pair};

#[wasm_bindgen(typescript_custom_section)]
const TYPES: &'static str = r#"
/** Options for `Copula.fitVine` and `PairCopula.fit`. Every field is optional. */
export interface FitOptions {
  /** Vine structure: "r" (default), "c" or "d". Vine fits only. */
  kind?: "r" | "c" | "d";
  /** Pair families to choose from, for example ["gaussian", "clayton", "gumbel"]. */
  familySet?: string[];
  /** Also try the 90, 180 and 270 degree rotations of each family. Default true. */
  includeRotations?: boolean;
  /** "aic" (default), "bic", "mbicv" or "mbicv:<psi0>". */
  criterion?: string;
  /** Keep this many trees; deeper trees are independence. Vine fits only. */
  truncationLevel?: number;
  /** Significance level of the Kendall tau independence pre-test, in (0, 1). */
  independenceTestLevel?: number;
  /** Variable order for "c" and "d" vines. */
  order?: number[];
}
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "FitOptions")]
    pub type FitOptions;

    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(message: &str);
}

/// Reports Rust panics on the console instead of a bare `unreachable`.
#[wasm_bindgen(start)]
fn start() {
    std::panic::set_hook(Box::new(|info| {
        console_error(&format!("rscopulas: {info}"))
    }));
}

fn js<T>(result: ApiResult<T>) -> Result<T, JsError> {
    result.map_err(|message| JsError::new(&message))
}

/// Reads an optional JavaScript options object through its JSON form.
fn config(options: Option<FitOptions>) -> Result<FitConfig, JsError> {
    let value: JsValue = match options {
        Some(options) => options.into(),
        None => return Ok(FitConfig::default()),
    };
    if value.is_undefined() || value.is_null() {
        return Ok(FitConfig::default());
    }
    let json = js_sys::JSON::stringify(&value)
        .map_err(|_| JsError::new("options must be a plain object"))?;
    let json: String = json.into();
    js(FitConfig::parse(Some(&json)))
}

/// Package version, the same as the Rust crate and the Python package.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Rank-transforms every column of a raw `(n, dim)` matrix into
/// pseudo-observations: `rank / (n + 1)`, ties sharing the average rank.
#[wasm_bindgen(js_name = toPseudoObs)]
pub fn to_pseudo_obs(data: &[f64], dim: usize) -> Result<Vec<f64>, JsError> {
    js(api::rank_pseudo_obs(data, dim))
}

/// Kendall's tau for every pair of columns: a flat `(dim, dim)` matrix.
#[wasm_bindgen(js_name = kendallTauMatrix)]
pub fn kendall_tau_matrix(data: &[f64], dim: usize) -> Result<Vec<f64>, JsError> {
    js(api::kendall_tau(data, dim))
}

/// A multivariate copula: Gaussian, Student t, Clayton, Frank, Gumbel or a
/// vine.
#[wasm_bindgen]
pub struct Copula {
    inner: Model,
}

impl From<Model> for Copula {
    fn from(inner: Model) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen]
impl Copula {
    /// Fits a Gaussian copula to pseudo-observations.
    #[wasm_bindgen(js_name = fitGaussian)]
    pub fn fit_gaussian(data: &[f64], dim: usize) -> Result<Copula, JsError> {
        js(Model::fit_gaussian(data, dim)).map(Into::into)
    }

    /// Fits a Student t copula to pseudo-observations.
    #[wasm_bindgen(js_name = fitStudentT)]
    pub fn fit_student_t(data: &[f64], dim: usize) -> Result<Copula, JsError> {
        js(Model::fit_student_t(data, dim)).map(Into::into)
    }

    /// Fits a Clayton copula to pseudo-observations.
    #[wasm_bindgen(js_name = fitClayton)]
    pub fn fit_clayton(data: &[f64], dim: usize) -> Result<Copula, JsError> {
        js(Model::fit_clayton(data, dim)).map(Into::into)
    }

    /// Fits a Frank copula to pseudo-observations.
    #[wasm_bindgen(js_name = fitFrank)]
    pub fn fit_frank(data: &[f64], dim: usize) -> Result<Copula, JsError> {
        js(Model::fit_frank(data, dim)).map(Into::into)
    }

    /// Fits a Gumbel copula to pseudo-observations.
    #[wasm_bindgen(js_name = fitGumbel)]
    pub fn fit_gumbel(data: &[f64], dim: usize) -> Result<Copula, JsError> {
        js(Model::fit_gumbel(data, dim)).map(Into::into)
    }

    /// Fits a vine copula, choosing a pair family for every edge.
    #[wasm_bindgen(js_name = fitVine)]
    pub fn fit_vine(
        data: &[f64],
        dim: usize,
        options: Option<FitOptions>,
    ) -> Result<Copula, JsError> {
        let config = config(options)?;
        js(Model::fit_vine(data, dim, &config)).map(Into::into)
    }

    /// A Gaussian copula with the given `(dim, dim)` correlation matrix.
    pub fn gaussian(correlation: &[f64], dim: usize) -> Result<Copula, JsError> {
        js(Model::gaussian(correlation, dim)).map(Into::into)
    }

    /// A Student t copula with the given correlation matrix and degrees of
    /// freedom.
    #[wasm_bindgen(js_name = studentT)]
    pub fn student_t(
        correlation: &[f64],
        dim: usize,
        degrees_of_freedom: f64,
    ) -> Result<Copula, JsError> {
        js(Model::student_t(correlation, dim, degrees_of_freedom)).map(Into::into)
    }

    /// A Clayton copula, `theta > 0`.
    pub fn clayton(dim: usize, theta: f64) -> Result<Copula, JsError> {
        js(Model::clayton(dim, theta)).map(Into::into)
    }

    /// A Frank copula.
    pub fn frank(dim: usize, theta: f64) -> Result<Copula, JsError> {
        js(Model::frank(dim, theta)).map(Into::into)
    }

    /// A Gumbel copula, `theta >= 1`.
    pub fn gumbel(dim: usize, theta: f64) -> Result<Copula, JsError> {
        js(Model::gumbel(dim, theta)).map(Into::into)
    }

    /// Loads a model saved with `toJson()`, or with `to_json()` on the
    /// matching class of the Python package. `family` is one of `gaussian`,
    /// `student_t`, `clayton`, `frank`, `gumbel`, `vine`.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(family: &str, payload: &str) -> Result<Copula, JsError> {
        js(Model::from_json(family, payload)).map(Into::into)
    }

    /// The model as JSON, readable by `fromJson` here and by `from_json` on
    /// the matching Python class.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsError> {
        js(self.inner.to_json())
    }

    /// `gaussian`, `student_t`, `clayton`, `frank`, `gumbel` or `vine`.
    #[wasm_bindgen(getter)]
    pub fn family(&self) -> String {
        self.inner.family().to_string()
    }

    /// Number of variables.
    #[wasm_bindgen(getter)]
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Log-likelihood of the fit; `undefined` for a model that was built
    /// from parameters or loaded from JSON.
    #[wasm_bindgen(getter)]
    pub fn loglik(&self) -> Option<f64> {
        self.inner.diagnostics.as_ref().map(|d| d.loglik)
    }

    /// Akaike information criterion of the fit, or `undefined`.
    #[wasm_bindgen(getter)]
    pub fn aic(&self) -> Option<f64> {
        self.inner.diagnostics.as_ref().map(|d| d.aic)
    }

    /// Bayesian information criterion of the fit, or `undefined`.
    #[wasm_bindgen(getter)]
    pub fn bic(&self) -> Option<f64> {
        self.inner.diagnostics.as_ref().map(|d| d.bic)
    }

    /// Log-density of every row of an `(n, dim)` matrix of
    /// pseudo-observations.
    #[wasm_bindgen(js_name = logPdf)]
    pub fn log_pdf(&self, data: &[f64]) -> Result<Vec<f64>, JsError> {
        js(self.inner.log_pdf(data))
    }

    /// `n` draws as a flat `(n, dim)` matrix. Pass an integer `seed` for a
    /// reproducible sample; without one the host's entropy source is used.
    pub fn sample(&self, n: usize, seed: Option<f64>) -> Result<Vec<f64>, JsError> {
        let seed = js(api::seed(seed))?;
        js(self.inner.sample(n, seed))
    }

    /// Vine only: maps dependent uniforms to independent uniforms.
    pub fn rosenblatt(&self, data: &[f64]) -> Result<Vec<f64>, JsError> {
        js(self.inner.rosenblatt(data))
    }

    /// Vine only: maps independent uniforms to a draw from the vine.
    #[wasm_bindgen(js_name = inverseRosenblatt)]
    pub fn inverse_rosenblatt(&self, data: &[f64]) -> Result<Vec<f64>, JsError> {
        js(self.inner.inverse_rosenblatt(data))
    }
}

/// A bivariate pair copula: density, conditional distributions (h-functions)
/// and their inverses.
#[wasm_bindgen]
pub struct PairCopula {
    inner: Pair,
}

impl From<Pair> for PairCopula {
    fn from(inner: Pair) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen]
impl PairCopula {
    /// `family` is a name such as `clayton` or `student_t`; `parameters`
    /// holds its one or two parameters; `rotation` is `R0` (default), `R90`,
    /// `R180` or `R270`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        family: &str,
        parameters: &[f64],
        rotation: Option<String>,
    ) -> Result<PairCopula, JsError> {
        js(Pair::new(family, parameters, rotation.as_deref())).map(Into::into)
    }

    /// Chooses and fits the best family for two columns of
    /// pseudo-observations.
    pub fn fit(u1: &[f64], u2: &[f64], options: Option<FitOptions>) -> Result<PairCopula, JsError> {
        let config = config(options)?;
        js(Pair::fit(u1, u2, &config)).map(Into::into)
    }

    /// Loads a pair copula saved with `toJson()` or with the Python package.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(payload: &str) -> Result<PairCopula, JsError> {
        js(Pair::from_json(payload)).map(Into::into)
    }

    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsError> {
        js(self.inner.to_json())
    }

    #[wasm_bindgen(getter)]
    pub fn family(&self) -> String {
        api::pair_family_name(self.inner.spec.family).to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn rotation(&self) -> String {
        api::rotation_name(self.inner.spec.rotation).to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn parameters(&self) -> Vec<f64> {
        self.inner.parameters()
    }

    /// Log-likelihood of the fit, or `undefined`.
    #[wasm_bindgen(getter)]
    pub fn loglik(&self) -> Option<f64> {
        self.inner.scores.map(|scores| scores.0)
    }

    /// Akaike information criterion of the fit, or `undefined`.
    #[wasm_bindgen(getter)]
    pub fn aic(&self) -> Option<f64> {
        self.inner.scores.map(|scores| scores.1)
    }

    /// Bayesian information criterion of the fit, or `undefined`.
    #[wasm_bindgen(getter)]
    pub fn bic(&self) -> Option<f64> {
        self.inner.scores.map(|scores| scores.2)
    }

    /// Log-density at each pair `(u1[i], u2[i])`.
    #[wasm_bindgen(js_name = logPdf)]
    pub fn log_pdf(&self, u1: &[f64], u2: &[f64]) -> Result<Vec<f64>, JsError> {
        js(self.inner.log_pdf(u1, u2))
    }

    /// `P(U1 <= u1 | U2 = u2)`.
    #[wasm_bindgen(js_name = condFirstGivenSecond)]
    pub fn cond_first_given_second(&self, u1: &[f64], u2: &[f64]) -> Result<Vec<f64>, JsError> {
        js(self.inner.cond_first_given_second(u1, u2))
    }

    /// `P(U2 <= u2 | U1 = u1)`.
    #[wasm_bindgen(js_name = condSecondGivenFirst)]
    pub fn cond_second_given_first(&self, u1: &[f64], u2: &[f64]) -> Result<Vec<f64>, JsError> {
        js(self.inner.cond_second_given_first(u1, u2))
    }

    /// The `u1` with `P(U1 <= u1 | U2 = u2) = p`.
    #[wasm_bindgen(js_name = invFirstGivenSecond)]
    pub fn inv_first_given_second(&self, p: &[f64], u2: &[f64]) -> Result<Vec<f64>, JsError> {
        js(self.inner.inv_first_given_second(p, u2))
    }

    /// The `u2` with `P(U2 <= u2 | U1 = u1) = p`.
    #[wasm_bindgen(js_name = invSecondGivenFirst)]
    pub fn inv_second_given_first(&self, u1: &[f64], p: &[f64]) -> Result<Vec<f64>, JsError> {
        js(self.inner.inv_second_given_first(u1, p))
    }
}
