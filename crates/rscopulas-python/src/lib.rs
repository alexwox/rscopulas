use std::any::Any;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    Bound, create_exception,
    exceptions::{PyException, PyTypeError, PyValueError},
    prelude::*,
    sync::PyOnceLock,
    types::{PyDict, PyList, PyModule, PyTuple, PyType},
};
use rand::{Rng, SeedableRng, random, rngs::StdRng};
use rscopulas::{
    ClaytonCopula, CopulaError, CopulaFamily, CopulaModel, EvalOptions, ExecPolicy, FactorCopula,
    FactorFitOptions, FactorFitResult, FactorLayout, FactorQuadrature, FitDiagnostics, FitError,
    FitOptions, FrankCopula, GaussianCopula, GumbelHougaardCopula, HacFamily, HacFitMethod,
    HacFitOptions, HacNode, HacStructureMethod, HacTree, HierarchicalArchimedeanCopula,
    KhoudrajiParams, PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation, SampleOptions,
    SelectionCriterion, StudentTCopula, TreeAlgorithm, TreeCriterion, VineCopula, VineEdge,
    VineFitOptions, VineStructureKind, VineTree,
};

// ---------------------------------------------------------------------------
// Exception hierarchy
//
//   RscopulasError (Exception)
//   ├── InvalidInputError (also a ValueError)
//   │   └── NonPrefixConditioningError
//   ├── ModelFitError
//   ├── NumericalError
//   ├── BackendError
//   └── InternalError
// ---------------------------------------------------------------------------

create_exception!(
    rscopulas,
    RscopulasError,
    PyException,
    "Base class for every exception raised by rscopulas."
);
create_exception!(
    rscopulas,
    ModelFitError,
    RscopulasError,
    "Raised when a fitting or estimation routine fails to produce a valid model."
);
create_exception!(
    rscopulas,
    NumericalError,
    RscopulasError,
    "Raised when a numerical routine produces an invalid result."
);
create_exception!(
    rscopulas,
    BackendError,
    RscopulasError,
    "Raised when the requested execution backend is unavailable or fails."
);
create_exception!(
    rscopulas,
    InternalError,
    RscopulasError,
    "Raised when an internal Rust panic is caught at the Python boundary; please report it."
);

/// `InvalidInputError` derives from both `RscopulasError` and the builtin
/// `ValueError` so it can be caught either way. `create_exception!` only
/// supports a single base class, so the type object is assembled by hand
/// and cached exactly like the macro-generated exception types.
#[repr(transparent)]
pub struct InvalidInputError(PyAny);

pyo3::impl_exception_boilerplate!(InvalidInputError);
pyo3::pyobject_native_type_named!(InvalidInputError);

const INVALID_INPUT_DOC: &str = "Raised when input data or arguments fail validation. \
Subclass of both RscopulasError and ValueError.";

fn build_invalid_input_error_type(py: Python<'_>) -> PyResult<Py<PyType>> {
    let bases = PyTuple::new(
        py,
        [
            py.get_type::<RscopulasError>(),
            py.get_type::<PyValueError>(),
        ],
    )?;
    let namespace = PyDict::new(py);
    namespace.set_item("__module__", "rscopulas")?;
    namespace.set_item("__doc__", INVALID_INPUT_DOC)?;
    let type_builder = py.import("builtins")?.getattr("type")?;
    let created = type_builder.call1(("InvalidInputError", bases, namespace))?;
    Ok(created.cast_into::<PyType>()?.unbind())
}

#[allow(deprecated)]
unsafe impl pyo3::type_object::PyTypeInfo for InvalidInputError {
    const NAME: &'static str = "InvalidInputError";
    const MODULE: Option<&'static str> = Some("rscopulas");

    fn type_object_raw(py: Python<'_>) -> *mut pyo3::ffi::PyTypeObject {
        static TYPE_OBJECT: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE_OBJECT
            .get_or_init(py, || {
                build_invalid_input_error_type(py)
                    .expect("failed to initialize rscopulas.InvalidInputError")
            })
            .as_ptr()
            .cast()
    }
}

create_exception!(
    rscopulas,
    NonPrefixConditioningError,
    InvalidInputError,
    "Raised by VineCopula.sample_conditional when the known columns are not a diagonal prefix of variable_order."
);

fn is_dimension_mismatch(reason: &str) -> bool {
    reason.contains("dimension does not match")
}

fn to_pyerr(error: CopulaError) -> PyErr {
    match error {
        CopulaError::InvalidInput(inner) => InvalidInputError::new_err(inner.to_string()),
        // HAC and factor models still classify evaluation-time dimension
        // mismatches as fit failures inside the core; surface them as input
        // errors so that every model behaves the same at the Python boundary.
        CopulaError::FitFailed(FitError::Failed { reason }) if is_dimension_mismatch(reason) => {
            InvalidInputError::new_err(reason.to_string())
        }
        CopulaError::FitFailed(inner) => ModelFitError::new_err(inner.to_string()),
        CopulaError::Numerical(inner) => NumericalError::new_err(inner.to_string()),
        CopulaError::Backend(inner) => BackendError::new_err(inner.to_string()),
    }
}

fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(message) => (*message).to_string(),
            Err(_) => "panic without message".to_string(),
        },
    }
}

/// Converts a Rust panic inside a binding into `InternalError` instead of
/// letting PyO3 raise `pyo3_runtime.PanicException` (a `BaseException`).
fn catch_internal_panic<T, F>(f: F) -> PyResult<T>
where
    F: FnOnce() -> PyResult<T>,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => Err(InternalError::new_err(format!(
            "internal panic: {}",
            panic_payload_message(payload)
        ))),
    }
}

/// Runs a compute-bound core call with the GIL released.
///
/// The closure must only touch owned Rust data: the `Send` bound rejects
/// captured Python objects at compile time. Panics propagate through
/// `Python::detach`, which re-attaches before unwinding, so callers can
/// still wrap the result in `catch_internal_panic`.
fn detached<T, F>(py: Python<'_>, f: F) -> PyResult<T>
where
    T: Send,
    F: FnOnce() -> Result<T, CopulaError> + Send,
{
    py.detach(f).map_err(to_pyerr)
}

/// Every model that is evaluated with the GIL released must be `Send + Sync`,
/// because several Python threads may call into the same object at once.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GaussianCopula>();
    assert_send_sync::<StudentTCopula>();
    assert_send_sync::<ClaytonCopula>();
    assert_send_sync::<FrankCopula>();
    assert_send_sync::<GumbelHougaardCopula>();
    assert_send_sync::<PairCopulaSpec>();
    assert_send_sync::<VineCopula>();
    assert_send_sync::<HierarchicalArchimedeanCopula>();
    assert_send_sync::<FactorCopula>();
};

fn fit_options(clip_eps: f64, max_iter: usize) -> FitOptions {
    FitOptions {
        exec: ExecPolicy::Auto,
        clip_eps,
        max_iter,
    }
}

fn eval_options(clip_eps: f64) -> EvalOptions {
    EvalOptions {
        exec: ExecPolicy::Auto,
        clip_eps,
    }
}

fn sample_options() -> SampleOptions {
    SampleOptions {
        exec: ExecPolicy::Auto,
    }
}

/// Clipping applied to raw uniforms before they enter an inverse Rosenblatt
/// pass; mirrors the clipping used by the core samplers.
const UNIFORM_CLIP_EPS: f64 = 1e-12;

fn rng_from_seed(seed: Option<u64>) -> StdRng {
    StdRng::seed_from_u64(seed.unwrap_or_else(random))
}

/// Validates a Python seed value: `None`, or an integer in `[0, 2**64)`.
///
/// Out-of-range integers raise `InvalidInputError` (instead of the
/// `OverflowError` that a bare `u64` extraction would produce) and
/// non-integers raise `TypeError`.
fn seed_from_py(seed: Option<&Bound<'_, PyAny>>) -> PyResult<Option<u64>> {
    let Some(seed) = seed else {
        return Ok(None);
    };
    if seed.is_none() {
        return Ok(None);
    }
    match seed.extract::<u64>() {
        Ok(value) => Ok(Some(value)),
        Err(_) if seed.hasattr("__index__")? => Err(InvalidInputError::new_err(format!(
            "seed must be an integer in [0, 2**64), got {seed}"
        ))),
        Err(_) => Err(PyTypeError::new_err(format!(
            "seed must be an int or None, got {}",
            seed.get_type().name()?
        ))),
    }
}

fn positive_count(value: usize, name: &str) -> PyResult<()> {
    if value == 0 {
        return Err(InvalidInputError::new_err(format!(
            "{name} must be a positive integer, got 0"
        )));
    }
    Ok(())
}

macro_rules! json_string {
    ($value:expr, $label:expr) => {
        serde_json::to_string($value)
            .map_err(|err| InternalError::new_err(format!("failed to serialize {}: {err}", $label)))
    };
}

macro_rules! parse_json {
    ($ty:ty, $payload:expr, $label:expr) => {
        serde_json::from_str::<$ty>($payload).map_err(|err| {
            InvalidInputError::new_err(format!("failed to deserialize {}: {err}", $label))
        })
    };
}

/// Structural equality through the serialized representation.
macro_rules! json_equal {
    ($left:expr, $right:expr) => {
        match (serde_json::to_value($left), serde_json::to_value($right)) {
            (Ok(left), Ok(right)) => Ok(left == right),
            (Err(err), _) | (_, Err(err)) => Err(InternalError::new_err(format!(
                "failed to compare models: {err}"
            ))),
        }
    };
}

const REPR_LIMIT: usize = 6;

fn fmt_scalar(value: f64) -> String {
    format!("{value:?}")
}

fn fmt_vector(values: &[f64]) -> String {
    let shown: Vec<String> = values
        .iter()
        .take(REPR_LIMIT)
        .map(|value| format!("{value:.4}"))
        .collect();
    if values.len() > REPR_LIMIT {
        format!("[{}, ...]", shown.join(", "))
    } else {
        format!("[{}]", shown.join(", "))
    }
}

fn fmt_matrix(matrix: &Array2<f64>) -> String {
    const ROW_LIMIT: usize = 3;
    let rows: Vec<String> = matrix
        .rows()
        .into_iter()
        .take(ROW_LIMIT)
        .map(|row| fmt_vector(&row.to_vec()))
        .collect();
    if matrix.nrows() > ROW_LIMIT {
        format!("[{}, ...]", rows.join(", "))
    } else {
        format!("[{}]", rows.join(", "))
    }
}

fn fmt_names<'a>(names: impl Iterator<Item = &'a str>, total: usize) -> String {
    let shown: Vec<String> = names
        .take(REPR_LIMIT)
        .map(|name| format!("'{name}'"))
        .collect();
    if total > REPR_LIMIT {
        format!("[{}, ...]", shown.join(", "))
    } else {
        format!("[{}]", shown.join(", "))
    }
}

fn pseudo_obs_from_py(data: PyReadonlyArray2<'_, f64>) -> PyResult<rscopulas::PseudoObs> {
    rscopulas::PseudoObs::from_view(data.as_array())
        .map_err(|err| InvalidInputError::new_err(err.to_string()))
}

fn matrix_from_py(data: PyReadonlyArray2<'_, f64>) -> Array2<f64> {
    data.as_array().to_owned()
}

/// Shared `log_pdf` binding: copy the input, release the GIL, evaluate.
fn model_log_pdf<'py, M>(
    py: Python<'py>,
    model: &M,
    data: PyReadonlyArray2<'_, f64>,
    clip_eps: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>>
where
    M: CopulaModel + Sync,
{
    catch_internal_panic(|| {
        let data = pseudo_obs_from_py(data)?;
        let options = eval_options(clip_eps);
        let values = detached(py, || model.log_pdf(&data, &options))?;
        Ok(values.into_pyarray(py))
    })
}

/// Shared `sample` binding: validate `n` and `seed`, release the GIL, draw.
fn model_sample<'py, M>(
    py: Python<'py>,
    model: &M,
    n: usize,
    seed: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>>
where
    M: CopulaModel + Sync,
{
    catch_internal_panic(|| {
        positive_count(n, "n")?;
        let mut rng = rng_from_seed(seed_from_py(seed)?);
        let options = sample_options();
        let values = detached(py, move || model.sample(n, &mut rng, &options))?;
        Ok(values.into_pyarray(py))
    })
}

fn pair_family_from_name(name: &str) -> PyResult<PairCopulaFamily> {
    match name.trim().to_ascii_lowercase().as_str() {
        "independence" => Ok(PairCopulaFamily::Independence),
        "gaussian" => Ok(PairCopulaFamily::Gaussian),
        "student_t" | "student-t" | "studentt" | "student" => Ok(PairCopulaFamily::StudentT),
        "clayton" => Ok(PairCopulaFamily::Clayton),
        "frank" => Ok(PairCopulaFamily::Frank),
        "gumbel" => Ok(PairCopulaFamily::Gumbel),
        "joe" => Ok(PairCopulaFamily::Joe),
        "bb1" => Ok(PairCopulaFamily::Bb1),
        "bb6" => Ok(PairCopulaFamily::Bb6),
        "bb7" => Ok(PairCopulaFamily::Bb7),
        "bb8" => Ok(PairCopulaFamily::Bb8),
        "tawn1" | "tawn_1" | "tawn-1" => Ok(PairCopulaFamily::Tawn1),
        "tawn2" | "tawn_2" | "tawn-2" => Ok(PairCopulaFamily::Tawn2),
        "tll" => Ok(PairCopulaFamily::Tll),
        "khoudraji" => Ok(PairCopulaFamily::Khoudraji),
        other => Err(InvalidInputError::new_err(format!(
            "unsupported pair family '{other}'; expected one of independence, gaussian, student_t, clayton, frank, gumbel, joe, bb1, bb6, bb7, bb8, tawn1, tawn2, tll, khoudraji"
        ))),
    }
}

fn rotation_from_name(name: &str) -> PyResult<Rotation> {
    match name.trim().to_ascii_uppercase().as_str() {
        "R0" => Ok(Rotation::R0),
        "R90" => Ok(Rotation::R90),
        "R180" => Ok(Rotation::R180),
        "R270" => Ok(Rotation::R270),
        other => Err(InvalidInputError::new_err(format!(
            "unsupported rotation '{other}'; expected one of R0, R90, R180, R270"
        ))),
    }
}

fn pair_params_from_values(
    family: PairCopulaFamily,
    parameters: Vec<f64>,
) -> PyResult<PairCopulaParams> {
    match (family, parameters.as_slice()) {
        (PairCopulaFamily::Independence, []) => Ok(PairCopulaParams::None),
        (PairCopulaFamily::Gaussian, [value])
        | (PairCopulaFamily::Clayton, [value])
        | (PairCopulaFamily::Frank, [value])
        | (PairCopulaFamily::Gumbel, [value])
        | (PairCopulaFamily::Joe, [value]) => Ok(PairCopulaParams::One(*value)),
        (PairCopulaFamily::StudentT, [rho, nu]) => Ok(PairCopulaParams::Two(*rho, *nu)),
        (PairCopulaFamily::Bb1, [theta, delta])
        | (PairCopulaFamily::Bb6, [theta, delta])
        | (PairCopulaFamily::Bb7, [theta, delta])
        | (PairCopulaFamily::Bb8, [theta, delta])
        | (PairCopulaFamily::Tawn1, [theta, delta])
        | (PairCopulaFamily::Tawn2, [theta, delta]) => Ok(PairCopulaParams::Two(*theta, *delta)),
        (PairCopulaFamily::Khoudraji, _) => Err(InvalidInputError::new_err(
            "khoudraji pair copulas require structured base_copula_1/base_copula_2 and shape_1/shape_2 inputs",
        )),
        (PairCopulaFamily::Independence, values) => Err(InvalidInputError::new_err(format!(
            "independence pair copulas do not take parameters (got {})",
            values.len()
        ))),
        (PairCopulaFamily::StudentT, values) => Err(InvalidInputError::new_err(format!(
            "student_t pair copulas require exactly two parameters (got {})",
            values.len()
        ))),
        (PairCopulaFamily::Bb1, values)
        | (PairCopulaFamily::Bb6, values)
        | (PairCopulaFamily::Bb7, values)
        | (PairCopulaFamily::Bb8, values) => Err(InvalidInputError::new_err(format!(
            "BB pair copulas require exactly two parameters (theta, delta) (got {})",
            values.len()
        ))),
        (PairCopulaFamily::Tawn1, values) | (PairCopulaFamily::Tawn2, values) => {
            Err(InvalidInputError::new_err(format!(
                "Tawn pair copulas require exactly two parameters (theta, psi) (got {})",
                values.len()
            )))
        }
        (PairCopulaFamily::Tll, _) => Err(InvalidInputError::new_err(
            "tll pair copulas are nonparametric; use PairCopula.fit_tll(u1, u2) to fit from data",
        )),
        (_, values) => Err(InvalidInputError::new_err(format!(
            "pair family requires exactly one parameter (got {})",
            values.len()
        ))),
    }
}

fn pair_spec_from_values(
    family: &str,
    rotation: &str,
    parameters: Vec<f64>,
) -> PyResult<PairCopulaSpec> {
    let family = pair_family_from_name(family)?;
    let spec = PairCopulaSpec {
        family,
        rotation: rotation_from_name(rotation)?,
        params: pair_params_from_values(family, parameters)?,
    };
    spec.validate().map_err(to_pyerr)?;
    Ok(spec)
}

fn pair_spec_from_py_dict(dict: &Bound<'_, PyDict>) -> PyResult<PairCopulaSpec> {
    let family = dict
        .get_item("family")?
        .ok_or_else(|| InvalidInputError::new_err("pair spec dictionaries require 'family'"))?
        .extract::<String>()?;
    let rotation = dict
        .get_item("rotation")?
        .map(|value| value.extract::<String>())
        .transpose()?
        .unwrap_or_else(|| "R0".to_string());
    let parsed_family = pair_family_from_name(&family)?;
    if parsed_family == PairCopulaFamily::Tll {
        let state = dict.get_item("state")?.ok_or_else(|| {
            InvalidInputError::new_err("TLL specs require fitted 'state'; use PairCopula.fit_tll")
        })?;
        let json: String = dict
            .py()
            .import("json")?
            .call_method1("dumps", (state,))?
            .extract()?;
        let params = parse_json!(rscopulas::TllParams, &json, "TLL state")?;
        let spec = PairCopulaSpec {
            family: parsed_family,
            rotation: rotation_from_name(&rotation)?,
            params: PairCopulaParams::Tll(params),
        };
        spec.validate().map_err(to_pyerr)?;
        return Ok(spec);
    }
    if parsed_family == PairCopulaFamily::Khoudraji {
        let base_first_value = dict
            .get_item("base_copula_1")?
            .ok_or_else(|| InvalidInputError::new_err("khoudraji specs require 'base_copula_1'"))?;
        let base_first = base_first_value.cast::<PyDict>()?;
        let base_second_value = dict
            .get_item("base_copula_2")?
            .ok_or_else(|| InvalidInputError::new_err("khoudraji specs require 'base_copula_2'"))?;
        let base_second = base_second_value.cast::<PyDict>()?;
        let shape_first = dict
            .get_item("shape_1")?
            .ok_or_else(|| InvalidInputError::new_err("khoudraji specs require 'shape_1'"))?
            .extract::<f64>()?;
        let shape_second = dict
            .get_item("shape_2")?
            .ok_or_else(|| InvalidInputError::new_err("khoudraji specs require 'shape_2'"))?
            .extract::<f64>()?;
        return Ok(PairCopulaSpec {
            family: PairCopulaFamily::Khoudraji,
            rotation: rotation_from_name(&rotation)?,
            params: PairCopulaParams::Khoudraji(
                KhoudrajiParams::new(
                    pair_spec_from_py_dict(base_first)?,
                    pair_spec_from_py_dict(base_second)?,
                    shape_first,
                    shape_second,
                )
                .map_err(to_pyerr)?,
            ),
        });
    }

    let parameters = if let Some(value) = dict.get_item("parameters")? {
        value.extract::<Vec<f64>>()?
    } else if let Some(value) = dict.get_item("params")? {
        value.extract::<Vec<f64>>()?
    } else {
        Vec::new()
    };
    pair_spec_from_values(&family, &rotation, parameters)
}

fn vine_kind_from_name(name: &str) -> PyResult<VineStructureKind> {
    match name.trim().to_ascii_lowercase().as_str() {
        "c" => Ok(VineStructureKind::C),
        "d" => Ok(VineStructureKind::D),
        "r" => Ok(VineStructureKind::R),
        other => Err(InvalidInputError::new_err(format!(
            "unsupported vine kind '{other}'; expected one of c, d, r"
        ))),
    }
}

fn vector_from_py(data: PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    data.as_array().iter().copied().collect()
}

fn paired_vectors_from_py(
    left: PyReadonlyArray1<'_, f64>,
    right: PyReadonlyArray1<'_, f64>,
    left_name: &str,
    right_name: &str,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let left_values = vector_from_py(left);
    let right_values = vector_from_py(right);
    if left_values.len() != right_values.len() {
        return Err(InvalidInputError::new_err(format!(
            "{left_name} and {right_name} must have the same length"
        )));
    }
    Ok((left_values, right_values))
}

fn criterion_from_name(name: &str) -> PyResult<SelectionCriterion> {
    let lower = name.trim().to_ascii_lowercase();
    // Accept either plain "mbicv" (defaults ψ₀ = 0.9, matching vinecopulib)
    // or "mbicv:0.95" to pin a custom prior. Splitting on ':' keeps the
    // string API tidy while giving power users a knob.
    if let Some(rest) = lower.strip_prefix("mbicv") {
        let psi0 = if rest.is_empty() {
            0.9
        } else {
            rest.strip_prefix(':')
                .ok_or_else(|| {
                    InvalidInputError::new_err(format!(
                        "unsupported selection criterion '{lower}'; expected 'mbicv' or 'mbicv:<psi0>'"
                    ))
                })?
                .parse::<f64>()
                .map_err(|err| {
                    InvalidInputError::new_err(format!("invalid psi0 in mbicv criterion: {err}"))
                })?
        };
        if !(0.0 < psi0 && psi0 < 1.0) {
            return Err(InvalidInputError::new_err(format!(
                "mbicv psi0 must lie in (0, 1), got {psi0}"
            )));
        }
        return Ok(SelectionCriterion::Mbicv { psi0 });
    }
    match lower.as_str() {
        "aic" => Ok(SelectionCriterion::Aic),
        "bic" => Ok(SelectionCriterion::Bic),
        other => Err(InvalidInputError::new_err(format!(
            "unsupported selection criterion '{other}'; expected 'aic', 'bic', 'mbicv', or 'mbicv:<psi0>'"
        ))),
    }
}

fn tree_algorithm_from_name(name: &str) -> PyResult<TreeAlgorithm> {
    match name.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "kruskal" | "mst" | "mst_kruskal" => Ok(TreeAlgorithm::Kruskal),
        "prim" | "mst_prim" => Ok(TreeAlgorithm::Prim),
        "random_weighted" | "weighted" => Ok(TreeAlgorithm::RandomWeighted),
        "random_unweighted" | "unweighted" | "wilson" => Ok(TreeAlgorithm::RandomUnweighted),
        other => Err(InvalidInputError::new_err(format!(
            "unsupported tree_algorithm '{other}'; expected 'kruskal', 'prim', 'random_weighted', or 'random_unweighted'"
        ))),
    }
}

fn tree_criterion_from_name(name: &str) -> PyResult<TreeCriterion> {
    match name.trim().to_ascii_lowercase().as_str() {
        "tau" | "kendall" => Ok(TreeCriterion::Tau),
        "rho" | "spearman" => Ok(TreeCriterion::Rho),
        "hoeffd" | "hoeffding" | "d" => Ok(TreeCriterion::Hoeffding),
        other => Err(InvalidInputError::new_err(format!(
            "unsupported tree_criterion '{other}'; expected 'tau', 'rho', or 'hoeffding'"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn vine_fit_options(
    family_set: Option<Vec<String>>,
    include_rotations: bool,
    criterion: &str,
    truncation_level: Option<usize>,
    independence_threshold: Option<f64>,
    independence_test_level: Option<f64>,
    clip_eps: f64,
    max_iter: usize,
    tree_algorithm: &str,
    tree_criterion: &str,
    select_trunc_lvl: bool,
    rng_seed: Option<u64>,
) -> PyResult<VineFitOptions> {
    if let Some(alpha) = independence_test_level
        && !(alpha.is_finite() && 0.0 < alpha && alpha < 1.0)
    {
        return Err(InvalidInputError::new_err(format!(
            "independence_test_level must lie in (0, 1), got {alpha}"
        )));
    }
    let mut options = VineFitOptions {
        base: fit_options(clip_eps, max_iter),
        include_rotations,
        criterion: criterion_from_name(criterion)?,
        truncation_level,
        independence_threshold,
        independence_test_level,
        tree_algorithm: tree_algorithm_from_name(tree_algorithm)?,
        tree_criterion: tree_criterion_from_name(tree_criterion)?,
        select_trunc_lvl,
        rng_seed,
        ..VineFitOptions::default()
    };
    if let Some(families) = family_set {
        options.family_set = families
            .iter()
            .map(|family| pair_family_from_name(family))
            .collect::<PyResult<Vec<_>>>()?;
    }
    Ok(options)
}

fn family_name(family: CopulaFamily) -> &'static str {
    match family {
        CopulaFamily::Gaussian => "gaussian",
        CopulaFamily::StudentT => "student_t",
        CopulaFamily::Clayton => "clayton",
        CopulaFamily::Frank => "frank",
        CopulaFamily::Gumbel => "gumbel",
        CopulaFamily::HierarchicalArchimedean => "hierarchical_archimedean",
        CopulaFamily::Vine => "vine",
        CopulaFamily::Factor => "factor",
    }
}

fn factor_layout_name(layout: FactorLayout) -> &'static str {
    match layout {
        FactorLayout::Basic1F => "basic_1f",
    }
}

fn factor_layout_from_name(name: &str) -> PyResult<FactorLayout> {
    match name.trim().to_ascii_lowercase().as_str() {
        "basic_1f" | "basic1f" | "basic-1f" | "basic" => Ok(FactorLayout::Basic1F),
        other => Err(InvalidInputError::new_err(format!(
            "unsupported factor layout '{other}'; expected 'basic_1f'"
        ))),
    }
}

fn hac_family_name(family: HacFamily) -> &'static str {
    match family {
        HacFamily::Clayton => "clayton",
        HacFamily::Frank => "frank",
        HacFamily::Gumbel => "gumbel",
    }
}

fn hac_family_from_name(name: &str) -> PyResult<HacFamily> {
    match name.trim().to_ascii_lowercase().as_str() {
        "clayton" => Ok(HacFamily::Clayton),
        "frank" => Ok(HacFamily::Frank),
        "gumbel" => Ok(HacFamily::Gumbel),
        other => Err(InvalidInputError::new_err(format!(
            "unsupported HAC family '{other}'; expected one of clayton, frank, gumbel"
        ))),
    }
}

fn hac_structure_method_name(method: HacStructureMethod) -> &'static str {
    match method {
        HacStructureMethod::GivenTree => "given_tree",
        HacStructureMethod::AgglomerativeTau => "agglomerative_tau",
        HacStructureMethod::AgglomerativeTauThenCollapse => "agglomerative_tau_then_collapse",
    }
}

fn hac_structure_method_from_name(name: &str) -> PyResult<HacStructureMethod> {
    match name.trim().to_ascii_lowercase().as_str() {
        "given_tree" | "given" => Ok(HacStructureMethod::GivenTree),
        "agglomerative_tau" | "agglomerative" => Ok(HacStructureMethod::AgglomerativeTau),
        "agglomerative_tau_then_collapse" | "agglomerative_then_collapse" | "collapse" => {
            Ok(HacStructureMethod::AgglomerativeTauThenCollapse)
        }
        other => Err(InvalidInputError::new_err(format!(
            "unsupported HAC structure method '{other}'; expected one of given_tree, agglomerative_tau, agglomerative_tau_then_collapse"
        ))),
    }
}

fn hac_fit_method_name(method: HacFitMethod) -> &'static str {
    match method {
        HacFitMethod::TauInit => "tau_init",
        HacFitMethod::CompositeMle => "composite_mle",
        HacFitMethod::RecursiveMle => "recursive_mle",
        HacFitMethod::FullMle => "full_mle",
        HacFitMethod::Smle => "smle",
        HacFitMethod::Dmle => "dmle",
    }
}

fn hac_fit_method_from_name(name: &str) -> PyResult<HacFitMethod> {
    match name.trim().to_ascii_lowercase().as_str() {
        "tau_init" | "tau" => Ok(HacFitMethod::TauInit),
        "composite_mle" | "composite" => Ok(HacFitMethod::CompositeMle),
        "recursive_mle" | "recursive" => Ok(HacFitMethod::RecursiveMle),
        "full_mle" | "full" => Ok(HacFitMethod::FullMle),
        "smle" => Ok(HacFitMethod::Smle),
        "dmle" => Ok(HacFitMethod::Dmle),
        other => Err(InvalidInputError::new_err(format!(
            "unsupported HAC fit method '{other}'; expected one of tau_init, composite_mle, recursive_mle, full_mle, smle, dmle"
        ))),
    }
}

fn hac_tree_from_py(value: &Bound<'_, PyAny>) -> PyResult<HacTree> {
    if let Ok(index) = value.extract::<usize>() {
        return Ok(HacTree::Leaf(index));
    }
    let dict = value.cast::<PyDict>()?;
    let family = dict
        .get_item("family")?
        .ok_or_else(|| InvalidInputError::new_err("HAC node dictionaries require a 'family' key"))?
        .extract::<String>()?;
    let theta = dict
        .get_item("theta")?
        .ok_or_else(|| InvalidInputError::new_err("HAC node dictionaries require a 'theta' key"))?
        .extract::<f64>()?;
    let children_value = dict.get_item("children")?.ok_or_else(|| {
        InvalidInputError::new_err("HAC node dictionaries require a 'children' key")
    })?;
    let children = children_value.cast::<PyList>()?;
    let parsed_children = children
        .iter()
        .map(|child| hac_tree_from_py(&child))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(HacTree::Node(HacNode::new(
        hac_family_from_name(&family)?,
        theta,
        parsed_children,
    )))
}

fn hac_tree_to_py<'py>(py: Python<'py>, tree: &HacTree) -> PyResult<Py<PyAny>> {
    match tree {
        HacTree::Leaf(index) => Ok(index.into_pyobject(py)?.unbind().into()),
        HacTree::Node(node) => {
            let dict = PyDict::new(py);
            dict.set_item("family", hac_family_name(node.family))?;
            dict.set_item("theta", node.theta)?;
            let children = PyList::empty(py);
            for child in &node.children {
                children.append(hac_tree_to_py(py, child)?)?;
            }
            dict.set_item("children", children)?;
            Ok(dict.unbind().into())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn hac_fit_options(
    family_set: Option<Vec<String>>,
    structure_method: &str,
    fit_method: &str,
    collapse_eps: f64,
    mc_samples: usize,
    allow_experimental: bool,
    clip_eps: f64,
    max_iter: usize,
) -> PyResult<HacFitOptions> {
    let mut options = HacFitOptions {
        base: fit_options(clip_eps, max_iter),
        structure_method: hac_structure_method_from_name(structure_method)?,
        fit_method: hac_fit_method_from_name(fit_method)?,
        collapse_eps,
        mc_samples,
        allow_experimental,
        ..HacFitOptions::default()
    };
    if let Some(families) = family_set {
        options.family_set = families
            .iter()
            .map(|family| hac_family_from_name(family))
            .collect::<PyResult<Vec<_>>>()?;
    }
    Ok(options)
}

fn vine_kind_name(kind: VineStructureKind) -> &'static str {
    match kind {
        VineStructureKind::C => "c",
        VineStructureKind::D => "d",
        VineStructureKind::R => "r",
    }
}

fn pair_family_name(family: PairCopulaFamily) -> &'static str {
    match family {
        PairCopulaFamily::Independence => "independence",
        PairCopulaFamily::Gaussian => "gaussian",
        PairCopulaFamily::StudentT => "student_t",
        PairCopulaFamily::Clayton => "clayton",
        PairCopulaFamily::Frank => "frank",
        PairCopulaFamily::Gumbel => "gumbel",
        PairCopulaFamily::Joe => "joe",
        PairCopulaFamily::Bb1 => "bb1",
        PairCopulaFamily::Bb6 => "bb6",
        PairCopulaFamily::Bb7 => "bb7",
        PairCopulaFamily::Bb8 => "bb8",
        PairCopulaFamily::Tawn1 => "tawn1",
        PairCopulaFamily::Tawn2 => "tawn2",
        PairCopulaFamily::Tll => "tll",
        PairCopulaFamily::Khoudraji => "khoudraji",
    }
}

fn rotation_name(rotation: Rotation) -> &'static str {
    match rotation {
        Rotation::R0 => "R0",
        Rotation::R90 => "R90",
        Rotation::R180 => "R180",
        Rotation::R270 => "R270",
    }
}

fn params_to_vec(params: &PairCopulaParams) -> Vec<f64> {
    match params {
        PairCopulaParams::None => Vec::new(),
        PairCopulaParams::One(value) => vec![*value],
        PairCopulaParams::Two(first, second) => vec![*first, *second],
        PairCopulaParams::Khoudraji(params) => params.flat_values(),
        // TLL's state is a full interpolation grid; the only scalar summary
        // worth flattening at the Python surface is the effective dof used
        // for BIC scoring.
        PairCopulaParams::Tll(params) => vec![params.effective_df()],
    }
}

fn attach_pair_components<'py>(
    py: Python<'py>,
    dict: &Bound<'py, PyDict>,
    spec: &PairCopulaSpec,
) -> PyResult<()> {
    if let PairCopulaParams::Tll(params) = &spec.params {
        let state = json_string!(params, "TLL state")?;
        dict.set_item("state", py.import("json")?.call_method1("loads", (state,))?)?;
    }
    if let PairCopulaParams::Khoudraji(params) = &spec.params {
        dict.set_item("shape_1", params.shape_first)?;
        dict.set_item("shape_2", params.shape_second)?;
        dict.set_item("base_copula_1", pair_spec_to_py(py, &params.first)?)?;
        dict.set_item("base_copula_2", pair_spec_to_py(py, &params.second)?)?;
    }
    Ok(())
}

fn pair_spec_to_py<'py>(py: Python<'py>, spec: &PairCopulaSpec) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("family", pair_family_name(spec.family))?;
    dict.set_item("rotation", rotation_name(spec.rotation))?;
    dict.set_item("parameters", params_to_vec(&spec.params))?;
    attach_pair_components(py, &dict, spec)?;
    Ok(dict)
}

fn vine_edge_from_py(value: &Bound<'_, PyAny>, level: usize) -> PyResult<VineEdge> {
    let dict = value.cast::<PyDict>()?;
    let conditioned = dict
        .get_item("conditioned")?
        .ok_or_else(|| InvalidInputError::new_err("vine edge dictionaries require 'conditioned'"))?
        .extract::<(usize, usize)>()?;
    let conditioning = match dict.get_item("conditioning")? {
        Some(value) => value.extract::<Vec<usize>>()?,
        None => Vec::new(),
    };
    Ok(VineEdge {
        tree: level,
        conditioned,
        conditioning,
        copula: pair_spec_from_py_dict(dict)?,
    })
}

fn vine_tree_from_py(value: &Bound<'_, PyAny>) -> PyResult<VineTree> {
    let dict = value.cast::<PyDict>()?;
    let level = dict
        .get_item("level")?
        .ok_or_else(|| InvalidInputError::new_err("vine tree dictionaries require 'level'"))?
        .extract::<usize>()?;
    let edges = dict
        .get_item("edges")?
        .ok_or_else(|| InvalidInputError::new_err("vine tree dictionaries require 'edges'"))?
        .cast::<PyList>()?
        .iter()
        .map(|edge| vine_edge_from_py(&edge, level))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(VineTree { level, edges })
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_FitDiagnostics",
    frozen
)]
#[derive(Clone)]
struct PyFitDiagnostics {
    loglik: f64,
    aic: f64,
    bic: f64,
    converged: bool,
    n_iter: usize,
    likelihood_kind: &'static str,
}

impl From<FitDiagnostics> for PyFitDiagnostics {
    fn from(value: FitDiagnostics) -> Self {
        Self {
            loglik: value.loglik,
            aic: value.aic,
            bic: value.bic,
            converged: value.converged,
            n_iter: value.n_iter,
            likelihood_kind: match value.likelihood_kind {
                rscopulas::LikelihoodKind::Joint => "joint",
                rscopulas::LikelihoodKind::Composite => "composite",
            },
        }
    }
}

#[pymethods]
impl PyFitDiagnostics {
    #[getter]
    fn likelihood_kind(&self) -> &'static str {
        self.likelihood_kind
    }
    #[getter]
    fn loglik(&self) -> f64 {
        self.loglik
    }

    #[getter]
    fn aic(&self) -> f64 {
        self.aic
    }

    #[getter]
    fn bic(&self) -> f64 {
        self.bic
    }

    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    #[getter]
    fn n_iter(&self) -> usize {
        self.n_iter
    }

    fn __repr__(&self) -> String {
        format!(
            "_FitDiagnostics(loglik={}, aic={}, bic={}, converged={}, n_iter={}, likelihood_kind='{}')",
            fmt_scalar(self.loglik),
            fmt_scalar(self.aic),
            fmt_scalar(self.bic),
            if self.converged { "True" } else { "False" },
            self.n_iter,
            self.likelihood_kind
        )
    }
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_GaussianCopula",
    frozen
)]
#[derive(Clone)]
struct PyGaussianCopula {
    inner: GaussianCopula,
}

#[pymethods]
impl PyGaussianCopula {
    /// Rebuild from the JSON produced by `to_json`; this is the constructor
    /// that `pickle` calls through `__reduce__`.
    #[new]
    fn new(payload: &str) -> PyResult<Self> {
        Self::from_json(payload)
    }

    #[staticmethod]
    fn from_params(correlation: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        catch_internal_panic(|| {
            GaussianCopula::new(matrix_from_py(correlation))
                .map(|inner| Self { inner })
                .map_err(to_pyerr)
        })
    }

    #[staticmethod]
    #[pyo3(signature = (data, clip_eps=1e-12, max_iter=500))]
    fn fit(
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
        max_iter: usize,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let options = fit_options(clip_eps, max_iter);
            let result = detached(py, || GaussianCopula::fit(&data, &options))?;
            Ok((
                Self {
                    inner: result.model,
                },
                result.diagnostics.into(),
            ))
        })
    }

    #[staticmethod]
    fn from_json(payload: &str) -> PyResult<Self> {
        catch_internal_panic(|| {
            Ok(Self {
                inner: parse_json!(GaussianCopula, payload, "Gaussian copula")?,
            })
        })
    }

    fn to_json(&self) -> PyResult<String> {
        json_string!(&self.inner, "Gaussian copula")
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn family(&self) -> &'static str {
        family_name(self.inner.family())
    }

    #[getter]
    fn correlation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.correlation().clone().into_pyarray(py)
    }

    #[pyo3(signature = (data, clip_eps=1e-12))]
    fn log_pdf<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        model_log_pdf(py, &self.inner, data, clip_eps)
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        model_sample(py, &self.inner, n, seed)
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyType>, (String,))> {
        Ok((slf.get_type(), (slf.get().to_json()?,)))
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn __eq__(&self, other: PyRef<'_, Self>) -> PyResult<bool> {
        json_equal!(&self.inner, &other.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "_GaussianCopula(dim={}, correlation={})",
            self.inner.dim(),
            fmt_matrix(self.inner.correlation())
        )
    }
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_StudentTCopula",
    frozen
)]
#[derive(Clone)]
struct PyStudentTCopula {
    inner: StudentTCopula,
}

#[pymethods]
impl PyStudentTCopula {
    #[new]
    fn new(payload: &str) -> PyResult<Self> {
        Self::from_json(payload)
    }

    #[staticmethod]
    fn from_params(
        correlation: PyReadonlyArray2<'_, f64>,
        degrees_of_freedom: f64,
    ) -> PyResult<Self> {
        catch_internal_panic(|| {
            StudentTCopula::new(matrix_from_py(correlation), degrees_of_freedom)
                .map(|inner| Self { inner })
                .map_err(to_pyerr)
        })
    }

    #[staticmethod]
    #[pyo3(signature = (data, clip_eps=1e-12, max_iter=500))]
    fn fit(
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
        max_iter: usize,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let options = fit_options(clip_eps, max_iter);
            let result = detached(py, || StudentTCopula::fit(&data, &options))?;
            Ok((
                Self {
                    inner: result.model,
                },
                result.diagnostics.into(),
            ))
        })
    }

    #[staticmethod]
    fn from_json(payload: &str) -> PyResult<Self> {
        catch_internal_panic(|| {
            Ok(Self {
                inner: parse_json!(StudentTCopula, payload, "Student t copula")?,
            })
        })
    }

    fn to_json(&self) -> PyResult<String> {
        json_string!(&self.inner, "Student t copula")
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn family(&self) -> &'static str {
        family_name(self.inner.family())
    }

    #[getter]
    fn correlation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.correlation().clone().into_pyarray(py)
    }

    #[getter]
    fn degrees_of_freedom(&self) -> f64 {
        self.inner.degrees_of_freedom()
    }

    #[pyo3(signature = (data, clip_eps=1e-12))]
    fn log_pdf<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        model_log_pdf(py, &self.inner, data, clip_eps)
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        model_sample(py, &self.inner, n, seed)
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyType>, (String,))> {
        Ok((slf.get_type(), (slf.get().to_json()?,)))
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn __eq__(&self, other: PyRef<'_, Self>) -> PyResult<bool> {
        json_equal!(&self.inner, &other.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "_StudentTCopula(dim={}, degrees_of_freedom={}, correlation={})",
            self.inner.dim(),
            fmt_scalar(self.inner.degrees_of_freedom()),
            fmt_matrix(self.inner.correlation())
        )
    }
}

/// The three one-parameter Archimedean families share an identical binding
/// surface; the macro keeps them in lock-step.
///
/// `rustfmt` does not format multi-line attributes inside macro bodies
/// idempotently, so the macro is skipped; its body follows rustfmt style.
#[rustfmt::skip]
macro_rules! archimedean_pyclass {
    ($py_ty:ident, $core:ty, $py_name:literal, $label:literal) => {
        #[pyclass(skip_from_py_object, module = "rscopulas._rscopulas", name = $py_name, frozen)]
        #[derive(Clone)]
        struct $py_ty {
            inner: $core,
        }

        #[pymethods]
        impl $py_ty {
            #[new]
            fn new(payload: &str) -> PyResult<Self> {
                Self::from_json(payload)
            }

            #[staticmethod]
            fn from_params(dim: usize, theta: f64) -> PyResult<Self> {
                catch_internal_panic(|| {
                    <$core>::new(dim, theta)
                        .map(|inner| Self { inner })
                        .map_err(to_pyerr)
                })
            }

            #[staticmethod]
            #[pyo3(signature = (data, clip_eps=1e-12, max_iter=500))]
            fn fit(
                py: Python<'_>,
                data: PyReadonlyArray2<'_, f64>,
                clip_eps: f64,
                max_iter: usize,
            ) -> PyResult<(Self, PyFitDiagnostics)> {
                catch_internal_panic(|| {
                    let data = pseudo_obs_from_py(data)?;
                    let options = fit_options(clip_eps, max_iter);
                    let result = detached(py, || <$core>::fit(&data, &options))?;
                    Ok((
                        Self {
                            inner: result.model,
                        },
                        result.diagnostics.into(),
                    ))
                })
            }

            #[staticmethod]
            fn from_json(payload: &str) -> PyResult<Self> {
                catch_internal_panic(|| {
                    // The derived deserializer does not validate `theta`, so
                    // rebuild the model through the checked constructor.
                    let parsed = parse_json!($core, payload, $label)?;
                    <$core>::new(parsed.dim(), parsed.theta())
                        .map(|inner| Self { inner })
                        .map_err(|err| {
                            InvalidInputError::new_err(format!(
                                "failed to deserialize {}: {err}",
                                $label
                            ))
                        })
                })
            }

            fn to_json(&self) -> PyResult<String> {
                json_string!(&self.inner, $label)
            }

            #[getter]
            fn dim(&self) -> usize {
                self.inner.dim()
            }

            #[getter]
            fn family(&self) -> &'static str {
                family_name(self.inner.family())
            }

            #[getter]
            fn theta(&self) -> f64 {
                self.inner.theta()
            }

            #[pyo3(signature = (data, clip_eps=1e-12))]
            fn log_pdf<'py>(
                &self,
                py: Python<'py>,
                data: PyReadonlyArray2<'_, f64>,
                clip_eps: f64,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {
                model_log_pdf(py, &self.inner, data, clip_eps)
            }

            #[pyo3(signature = (n, seed=None))]
            fn sample<'py>(
                &self,
                py: Python<'py>,
                n: usize,
                seed: Option<&Bound<'py, PyAny>>,
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                model_sample(py, &self.inner, n, seed)
            }

            fn __reduce__<'py>(
                slf: &Bound<'py, Self>,
            ) -> PyResult<(Bound<'py, PyType>, (String,))> {
                Ok((slf.get_type(), (slf.get().to_json()?,)))
            }

            fn __copy__(&self) -> Self {
                self.clone()
            }

            fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
                self.clone()
            }

            fn __eq__(&self, other: PyRef<'_, Self>) -> PyResult<bool> {
                json_equal!(&self.inner, &other.inner)
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}(dim={}, theta={})",
                    $py_name,
                    self.inner.dim(),
                    fmt_scalar(self.inner.theta())
                )
            }
        }
    };
}

archimedean_pyclass!(
    PyClaytonCopula,
    ClaytonCopula,
    "_ClaytonCopula",
    "Clayton copula"
);
archimedean_pyclass!(PyFrankCopula, FrankCopula, "_FrankCopula", "Frank copula");
archimedean_pyclass!(
    PyGumbelCopula,
    GumbelHougaardCopula,
    "_GumbelCopula",
    "Gumbel copula"
);

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_PairCopula",
    frozen
)]
#[derive(Clone)]
struct PyPairCopula {
    inner: PairCopulaSpec,
}

impl PyPairCopula {
    #[allow(clippy::too_many_arguments)]
    fn eval_pair_batch<'py, F>(
        &self,
        py: Python<'py>,
        left: PyReadonlyArray1<'_, f64>,
        right: PyReadonlyArray1<'_, f64>,
        clip_eps: f64,
        left_name: &str,
        right_name: &str,
        callback: F,
    ) -> PyResult<Bound<'py, PyArray1<f64>>>
    where
        F: Fn(&PairCopulaSpec, f64, f64, f64) -> Result<f64, CopulaError> + Send,
    {
        catch_internal_panic(|| {
            let (left_values, right_values) =
                paired_vectors_from_py(left, right, left_name, right_name)?;
            let spec = &self.inner;
            let values = detached(py, move || {
                left_values
                    .into_iter()
                    .zip(right_values)
                    .map(|(first, second)| callback(spec, first, second, clip_eps))
                    .collect::<Result<Vec<_>, CopulaError>>()
            })?;
            Ok(values.into_pyarray(py))
        })
    }
}

#[pymethods]
impl PyPairCopula {
    #[new]
    fn new(payload: &str) -> PyResult<Self> {
        Self::from_json(payload)
    }

    #[staticmethod]
    #[pyo3(signature = (family, parameters=None, rotation="R0", state=None))]
    fn from_spec(
        py: Python<'_>,
        family: &str,
        parameters: Option<Vec<f64>>,
        rotation: &str,
        state: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        catch_internal_panic(|| {
            if let Some(state) = state {
                if pair_family_from_name(family)? != PairCopulaFamily::Tll {
                    return Err(InvalidInputError::new_err(
                        "'state' is only valid for TLL specifications",
                    ));
                }
                let spec = PyDict::new(py);
                spec.set_item("family", family)?;
                spec.set_item("rotation", rotation)?;
                spec.set_item("state", state)?;
                return Ok(Self {
                    inner: pair_spec_from_py_dict(&spec)?,
                });
            }
            if pair_family_from_name(family)? == PairCopulaFamily::Khoudraji {
                return Err(InvalidInputError::new_err(
                    "use PairCopula.from_khoudraji(...) for khoudraji specifications",
                ));
            }
            Ok(Self {
                inner: pair_spec_from_values(family, rotation, parameters.unwrap_or_default())?,
            })
        })
    }

    /// Fit a nonparametric TLL (Transformation Local Likelihood) pair
    /// copula using constant, linear, or quadratic local likelihood.
    #[staticmethod]
    #[pyo3(signature = (u1, u2, method="constant"))]
    fn fit_tll(
        py: Python<'_>,
        u1: PyReadonlyArray1<'_, f64>,
        u2: PyReadonlyArray1<'_, f64>,
        method: &str,
    ) -> PyResult<Self> {
        catch_internal_panic(|| {
            let order = match method.trim().to_ascii_lowercase().as_str() {
                "constant" | "tll0" => rscopulas::TllOrder::Constant,
                "linear" | "tll1" => rscopulas::TllOrder::Linear,
                "quadratic" | "tll2" => rscopulas::TllOrder::Quadratic,
                other => {
                    return Err(InvalidInputError::new_err(format!(
                        "unsupported tll method '{other}'; expected one of constant, linear, quadratic"
                    )));
                }
            };
            let (u1_values, u2_values) = paired_vectors_from_py(u1, u2, "u1", "u2")?;
            let tll_params = detached(py, || rscopulas::tll_fit(&u1_values, &u2_values, order))?;
            Ok(Self {
                inner: PairCopulaSpec {
                    family: PairCopulaFamily::Tll,
                    rotation: Rotation::R0,
                    params: PairCopulaParams::Tll(tll_params),
                },
            })
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        first_family,
        second_family,
        shape_1,
        shape_2,
        first_parameters=None,
        second_parameters=None,
        rotation="R0",
        first_rotation="R0",
        second_rotation="R0"
    ))]
    fn from_khoudraji(
        first_family: &str,
        second_family: &str,
        shape_1: f64,
        shape_2: f64,
        first_parameters: Option<Vec<f64>>,
        second_parameters: Option<Vec<f64>>,
        rotation: &str,
        first_rotation: &str,
        second_rotation: &str,
    ) -> PyResult<Self> {
        catch_internal_panic(|| {
            Ok(Self {
                inner: PairCopulaSpec {
                    family: PairCopulaFamily::Khoudraji,
                    rotation: rotation_from_name(rotation)?,
                    params: PairCopulaParams::Khoudraji(
                        KhoudrajiParams::new(
                            pair_spec_from_values(
                                first_family,
                                first_rotation,
                                first_parameters.unwrap_or_default(),
                            )?,
                            pair_spec_from_values(
                                second_family,
                                second_rotation,
                                second_parameters.unwrap_or_default(),
                            )?,
                            shape_1,
                            shape_2,
                        )
                        .map_err(to_pyerr)?,
                    ),
                },
            })
        })
    }

    #[staticmethod]
    fn from_json(payload: &str) -> PyResult<Self> {
        catch_internal_panic(|| {
            let inner = parse_json!(PairCopulaSpec, payload, "pair copula")?;
            inner.validate().map_err(|err| {
                InvalidInputError::new_err(format!("failed to deserialize pair copula: {err}"))
            })?;
            Ok(Self { inner })
        })
    }

    fn to_json(&self) -> PyResult<String> {
        json_string!(&self.inner, "pair copula")
    }

    #[getter]
    fn family(&self) -> &'static str {
        pair_family_name(self.inner.family)
    }

    #[getter]
    fn rotation(&self) -> &'static str {
        rotation_name(self.inner.rotation)
    }

    #[getter]
    fn parameters(&self) -> Vec<f64> {
        params_to_vec(&self.inner.params)
    }

    #[getter]
    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        catch_internal_panic(|| pair_spec_to_py(py, &self.inner))
    }

    #[getter]
    fn dim(&self) -> usize {
        2
    }

    #[pyo3(signature = (u1, u2, clip_eps=1e-12))]
    fn log_pdf<'py>(
        &self,
        py: Python<'py>,
        u1: PyReadonlyArray1<'_, f64>,
        u2: PyReadonlyArray1<'_, f64>,
        clip_eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.eval_pair_batch(
            py,
            u1,
            u2,
            clip_eps,
            "u1",
            "u2",
            |spec, left, right, eps| spec.log_pdf(left, right, eps),
        )
    }

    #[pyo3(signature = (u1, u2, clip_eps=1e-12))]
    fn cond_first_given_second<'py>(
        &self,
        py: Python<'py>,
        u1: PyReadonlyArray1<'_, f64>,
        u2: PyReadonlyArray1<'_, f64>,
        clip_eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.eval_pair_batch(
            py,
            u1,
            u2,
            clip_eps,
            "u1",
            "u2",
            |spec, left, right, eps| spec.cond_first_given_second(left, right, eps),
        )
    }

    #[pyo3(signature = (u1, u2, clip_eps=1e-12))]
    fn cond_second_given_first<'py>(
        &self,
        py: Python<'py>,
        u1: PyReadonlyArray1<'_, f64>,
        u2: PyReadonlyArray1<'_, f64>,
        clip_eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.eval_pair_batch(
            py,
            u1,
            u2,
            clip_eps,
            "u1",
            "u2",
            |spec, left, right, eps| spec.cond_second_given_first(left, right, eps),
        )
    }

    #[pyo3(signature = (p, u2, clip_eps=1e-12))]
    fn inv_first_given_second<'py>(
        &self,
        py: Python<'py>,
        p: PyReadonlyArray1<'_, f64>,
        u2: PyReadonlyArray1<'_, f64>,
        clip_eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.eval_pair_batch(py, p, u2, clip_eps, "p", "u2", |spec, left, right, eps| {
            spec.inv_first_given_second(left, right, eps)
        })
    }

    #[pyo3(signature = (u1, p, clip_eps=1e-12))]
    fn inv_second_given_first<'py>(
        &self,
        py: Python<'py>,
        u1: PyReadonlyArray1<'_, f64>,
        p: PyReadonlyArray1<'_, f64>,
        clip_eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.eval_pair_batch(py, u1, p, clip_eps, "u1", "p", |spec, left, right, eps| {
            spec.inv_second_given_first(left, right, eps)
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyType>, (String,))> {
        Ok((slf.get_type(), (slf.get().to_json()?,)))
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn __eq__(&self, other: PyRef<'_, Self>) -> PyResult<bool> {
        // `PairCopulaSpec: PartialEq` also compares the lazily built TLL
        // grid cache, so compare the serialized state instead.
        json_equal!(&self.inner, &other.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "_PairCopula(family='{}', rotation='{}', parameters={})",
            pair_family_name(self.inner.family),
            rotation_name(self.inner.rotation),
            fmt_vector(&params_to_vec(&self.inner.params))
        )
    }
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_VineCopula",
    frozen
)]
#[derive(Clone)]
struct PyVineCopula {
    inner: VineCopula,
}

impl PyVineCopula {
    fn from_fit_result(result: rscopulas::fit::FitResult<VineCopula>) -> (Self, PyFitDiagnostics) {
        (
            Self {
                inner: result.model,
            },
            result.diagnostics.into(),
        )
    }
}

#[pymethods]
impl PyVineCopula {
    #[new]
    fn new(payload: &str) -> PyResult<Self> {
        Self::from_json(payload)
    }

    #[staticmethod]
    fn from_trees(
        py: Python<'_>,
        kind: &str,
        trees: &Bound<'_, PyAny>,
        truncation_level: Option<usize>,
    ) -> PyResult<Self> {
        catch_internal_panic(|| {
            let kind = vine_kind_from_name(kind)?;
            let trees = trees
                .cast::<PyList>()?
                .iter()
                .map(|tree| vine_tree_from_py(&tree))
                .collect::<PyResult<Vec<_>>>()?;
            let inner = detached(py, move || {
                VineCopula::from_trees(kind, trees, truncation_level)
            })?;
            Ok(Self { inner })
        })
    }

    #[staticmethod]
    fn gaussian_c_vine(
        py: Python<'_>,
        order: Vec<usize>,
        correlation: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        catch_internal_panic(|| {
            let correlation = matrix_from_py(correlation);
            let inner = detached(py, move || VineCopula::gaussian_c_vine(order, correlation))?;
            Ok(Self { inner })
        })
    }

    #[staticmethod]
    fn gaussian_d_vine(
        py: Python<'_>,
        order: Vec<usize>,
        correlation: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        catch_internal_panic(|| {
            let correlation = matrix_from_py(correlation);
            let inner = detached(py, move || VineCopula::gaussian_d_vine(order, correlation))?;
            Ok(Self { inner })
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (data, family_set=None, include_rotations=true, criterion="aic", truncation_level=None, independence_threshold=None, independence_test_level=None, clip_eps=1e-12, max_iter=500, order=None, tree_algorithm="kruskal", tree_criterion="tau", select_trunc_lvl=false, rng_seed=None))]
    fn fit_c(
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f64>,
        family_set: Option<Vec<String>>,
        include_rotations: bool,
        criterion: &str,
        truncation_level: Option<usize>,
        independence_threshold: Option<f64>,
        independence_test_level: Option<f64>,
        clip_eps: f64,
        max_iter: usize,
        order: Option<Vec<usize>>,
        tree_algorithm: &str,
        tree_criterion: &str,
        select_trunc_lvl: bool,
        rng_seed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let options = vine_fit_options(
                family_set,
                include_rotations,
                criterion,
                truncation_level,
                independence_threshold,
                independence_test_level,
                clip_eps,
                max_iter,
                tree_algorithm,
                tree_criterion,
                select_trunc_lvl,
                seed_from_py(rng_seed)?,
            )?;
            let result = detached(py, || match order {
                Some(order) => VineCopula::fit_c_vine_with_order(&data, &order, &options),
                None => VineCopula::fit_c_vine(&data, &options),
            })?;
            Ok(Self::from_fit_result(result))
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (data, family_set=None, include_rotations=true, criterion="aic", truncation_level=None, independence_threshold=None, independence_test_level=None, clip_eps=1e-12, max_iter=500, order=None, tree_algorithm="kruskal", tree_criterion="tau", select_trunc_lvl=false, rng_seed=None))]
    fn fit_d(
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f64>,
        family_set: Option<Vec<String>>,
        include_rotations: bool,
        criterion: &str,
        truncation_level: Option<usize>,
        independence_threshold: Option<f64>,
        independence_test_level: Option<f64>,
        clip_eps: f64,
        max_iter: usize,
        order: Option<Vec<usize>>,
        tree_algorithm: &str,
        tree_criterion: &str,
        select_trunc_lvl: bool,
        rng_seed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let options = vine_fit_options(
                family_set,
                include_rotations,
                criterion,
                truncation_level,
                independence_threshold,
                independence_test_level,
                clip_eps,
                max_iter,
                tree_algorithm,
                tree_criterion,
                select_trunc_lvl,
                seed_from_py(rng_seed)?,
            )?;
            let result = detached(py, || match order {
                Some(order) => VineCopula::fit_d_vine_with_order(&data, &order, &options),
                None => VineCopula::fit_d_vine(&data, &options),
            })?;
            Ok(Self::from_fit_result(result))
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (data, family_set=None, include_rotations=true, criterion="aic", truncation_level=None, independence_threshold=None, independence_test_level=None, clip_eps=1e-12, max_iter=500, tree_algorithm="kruskal", tree_criterion="tau", select_trunc_lvl=false, rng_seed=None))]
    fn fit_r(
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f64>,
        family_set: Option<Vec<String>>,
        include_rotations: bool,
        criterion: &str,
        truncation_level: Option<usize>,
        independence_threshold: Option<f64>,
        independence_test_level: Option<f64>,
        clip_eps: f64,
        max_iter: usize,
        tree_algorithm: &str,
        tree_criterion: &str,
        select_trunc_lvl: bool,
        rng_seed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let options = vine_fit_options(
                family_set,
                include_rotations,
                criterion,
                truncation_level,
                independence_threshold,
                independence_test_level,
                clip_eps,
                max_iter,
                tree_algorithm,
                tree_criterion,
                select_trunc_lvl,
                seed_from_py(rng_seed)?,
            )?;
            let result = detached(py, || VineCopula::fit_r_vine(&data, &options))?;
            Ok(Self::from_fit_result(result))
        })
    }

    /// Rebuild a vine from `to_json` output. The payload carries a
    /// `format_version`; unversioned or foreign payloads are rejected.
    #[staticmethod]
    fn from_json(payload: &str) -> PyResult<Self> {
        catch_internal_panic(|| {
            Ok(Self {
                inner: parse_json!(VineCopula, payload, "vine copula")?,
            })
        })
    }

    fn to_json(&self) -> PyResult<String> {
        json_string!(&self.inner, "vine copula")
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn family(&self) -> &'static str {
        family_name(self.inner.family())
    }

    #[getter]
    fn structure_kind(&self) -> &'static str {
        vine_kind_name(self.inner.structure())
    }

    #[getter]
    fn truncation_level(&self) -> Option<usize> {
        self.inner.truncation_level()
    }

    fn order(&self) -> PyResult<Vec<usize>> {
        catch_internal_panic(|| Ok(self.inner.order()))
    }

    fn pair_parameters(&self) -> PyResult<Vec<f64>> {
        catch_internal_panic(|| Ok(self.inner.pair_parameters()))
    }

    fn structure_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        catch_internal_panic(|| {
            let info = self.inner.structure_info();
            let dict = PyDict::new(py);
            dict.set_item("kind", vine_kind_name(info.kind))?;
            dict.set_item("matrix", info.matrix.clone().into_pyarray(py))?;
            dict.set_item("truncation_level", info.truncation_level)?;
            Ok(dict)
        })
    }

    fn trees<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        catch_internal_panic(|| {
            let trees = PyList::empty(py);
            for tree in self.inner.trees() {
                let tree_dict = PyDict::new(py);
                tree_dict.set_item("level", tree.level)?;
                let edges = PyList::empty(py);
                for edge in &tree.edges {
                    let edge_dict = pair_spec_to_py(py, &edge.copula)?;
                    edge_dict.set_item("tree", edge.tree)?;
                    edge_dict.set_item("conditioned", (edge.conditioned.0, edge.conditioned.1))?;
                    edge_dict.set_item("conditioning", edge.conditioning.clone())?;
                    edges.append(edge_dict)?;
                }
                tree_dict.set_item("edges", edges)?;
                trees.append(tree_dict)?;
            }
            Ok(trees)
        })
    }

    #[pyo3(signature = (data, clip_eps=1e-12))]
    fn log_pdf<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        model_log_pdf(py, &self.inner, data, clip_eps)
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        model_sample(py, &self.inner, n, seed)
    }

    /// Diagonal ordering used by the Rosenblatt transform.
    ///
    /// `variable_order[0]` is the Rosenblatt anchor: its input uniform is
    /// passed through unchanged.
    fn variable_order(&self) -> Vec<usize> {
        self.inner.variable_order().to_vec()
    }

    /// Rosenblatt transform `U = F(V)` indexed by original variable label.
    fn rosenblatt<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        catch_internal_panic(|| {
            let data = matrix_from_py(data);
            let options = sample_options();
            let model = &self.inner;
            let values = detached(py, || model.rosenblatt(data.view(), &options))?;
            Ok(values.into_pyarray(py))
        })
    }

    /// Inverse Rosenblatt transform `V = F^{-1}(U)` indexed by original
    /// variable label. This is the primitive behind `sample` and
    /// `sample_conditional`.
    fn inverse_rosenblatt<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        catch_internal_panic(|| {
            let data = matrix_from_py(data);
            let options = sample_options();
            let model = &self.inner;
            let values = detached(py, || model.inverse_rosenblatt(data.view(), &options))?;
            Ok(values.into_pyarray(py))
        })
    }

    /// Partial forward Rosenblatt that only emits the first `col_limit`
    /// diagonal positions. The returned array has shape `(n, col_limit)`
    /// and is indexed **by diagonal position**: column `idx` of the output
    /// is the Rosenblatt uniform for variable `variable_order()[idx]`.
    fn rosenblatt_prefix<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'_, f64>,
        col_limit: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        catch_internal_panic(|| {
            let data = matrix_from_py(data);
            let options = sample_options();
            let model = &self.inner;
            let values = detached(py, || {
                model.rosenblatt_prefix(data.view(), col_limit, &options)
            })?;
            Ok(values.into_pyarray(py))
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyType>, (String,))> {
        Ok((slf.get_type(), (slf.get().to_json()?,)))
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn __eq__(&self, other: PyRef<'_, Self>) -> PyResult<bool> {
        json_equal!(&self.inner, &other.inner)
    }

    fn __repr__(&self) -> String {
        let truncation = match self.inner.truncation_level() {
            Some(level) => level.to_string(),
            None => "None".to_string(),
        };
        format!(
            "_VineCopula(kind='{}', dim={}, truncation_level={}, n_trees={})",
            vine_kind_name(self.inner.structure()),
            self.inner.dim(),
            truncation,
            self.inner.trees().len()
        )
    }
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_HierarchicalArchimedeanCopula",
    frozen
)]
#[derive(Clone)]
struct PyHierarchicalArchimedeanCopula {
    inner: HierarchicalArchimedeanCopula,
}

#[pymethods]
impl PyHierarchicalArchimedeanCopula {
    #[new]
    fn new(payload: &str) -> PyResult<Self> {
        Self::from_json(payload)
    }

    #[staticmethod]
    fn from_tree(tree: &Bound<'_, PyAny>) -> PyResult<Self> {
        catch_internal_panic(|| {
            let tree = hac_tree_from_py(tree)?;
            HierarchicalArchimedeanCopula::new(tree)
                .map(|inner| Self { inner })
                .map_err(to_pyerr)
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (data, tree=None, family_set=None, structure_method="agglomerative_tau_then_collapse", fit_method="composite_mle", collapse_eps=0.05, mc_samples=0, allow_experimental=true, clip_eps=1e-12, max_iter=500))]
    fn fit(
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f64>,
        tree: Option<&Bound<'_, PyAny>>,
        family_set: Option<Vec<String>>,
        structure_method: &str,
        fit_method: &str,
        collapse_eps: f64,
        mc_samples: usize,
        allow_experimental: bool,
        clip_eps: f64,
        max_iter: usize,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let options = hac_fit_options(
                family_set,
                structure_method,
                fit_method,
                collapse_eps,
                mc_samples,
                allow_experimental,
                clip_eps,
                max_iter,
            )?;
            let parsed_tree = tree.map(hac_tree_from_py).transpose()?;
            let result = detached(py, move || match parsed_tree {
                Some(parsed_tree) => {
                    HierarchicalArchimedeanCopula::fit_with_tree(&data, parsed_tree, &options)
                }
                None => HierarchicalArchimedeanCopula::fit(&data, &options),
            })?;
            Ok((
                Self {
                    inner: result.model,
                },
                result.diagnostics.into(),
            ))
        })
    }

    #[staticmethod]
    fn from_json(payload: &str) -> PyResult<Self> {
        catch_internal_panic(|| {
            Ok(Self {
                inner: parse_json!(HierarchicalArchimedeanCopula, payload, "HAC copula")?,
            })
        })
    }

    fn to_json(&self) -> PyResult<String> {
        json_string!(&self.inner, "HAC copula")
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn family(&self) -> &'static str {
        family_name(self.inner.family())
    }

    #[getter]
    fn is_exact(&self) -> bool {
        self.inner.is_exact()
    }

    #[getter]
    fn exact_loglik(&self) -> bool {
        self.inner.exact_loglik()
    }

    #[getter]
    fn used_smle(&self) -> bool {
        self.inner.used_smle()
    }

    #[getter]
    fn mc_samples(&self) -> usize {
        self.inner.mc_samples()
    }

    #[getter]
    fn structure_method(&self) -> &'static str {
        hac_structure_method_name(self.inner.structure_method())
    }

    #[getter]
    fn fit_method(&self) -> &'static str {
        hac_fit_method_name(self.inner.fit_method())
    }

    fn tree<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        catch_internal_panic(|| hac_tree_to_py(py, self.inner.tree()))
    }

    fn leaf_order(&self) -> PyResult<Vec<usize>> {
        catch_internal_panic(|| Ok(self.inner.leaf_order()))
    }

    fn parameters(&self) -> PyResult<Vec<f64>> {
        catch_internal_panic(|| Ok(self.inner.parameters()))
    }

    fn families(&self) -> PyResult<Vec<String>> {
        catch_internal_panic(|| {
            Ok(self
                .inner
                .families()
                .into_iter()
                .map(|family| hac_family_name(family).to_string())
                .collect())
        })
    }

    #[pyo3(signature = (data, clip_eps=1e-12))]
    fn log_pdf<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        model_log_pdf(py, &self.inner, data, clip_eps)
    }

    #[pyo3(signature = (data, clip_eps=1e-12))]
    fn composite_log_pdf<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let options = eval_options(clip_eps);
            let model = &self.inner;
            let values = detached(py, || model.composite_log_pdf(&data, &options))?;
            Ok(values.into_pyarray(py))
        })
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        model_sample(py, &self.inner, n, seed)
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyType>, (String,))> {
        Ok((slf.get_type(), (slf.get().to_json()?,)))
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn __eq__(&self, other: PyRef<'_, Self>) -> PyResult<bool> {
        json_equal!(&self.inner, &other.inner)
    }

    fn __repr__(&self) -> String {
        let families = self.inner.families();
        format!(
            "_HierarchicalArchimedeanCopula(dim={}, families={}, parameters={})",
            self.inner.dim(),
            fmt_names(
                families.iter().map(|family| hac_family_name(*family)),
                families.len()
            ),
            fmt_vector(&self.inner.parameters())
        )
    }
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_FactorCopula",
    frozen
)]
#[derive(Clone)]
struct PyFactorCopula {
    inner: FactorCopula,
}

#[pymethods]
impl PyFactorCopula {
    #[new]
    fn new(payload: &str) -> PyResult<Self> {
        Self::from_json(payload)
    }

    /// Construct a `Basic1F` factor copula from a list of link dictionaries.
    /// Each dict follows the same schema used elsewhere in the Python API —
    /// `{"family": str, "rotation": str, "parameters": [floats]}` — so users
    /// can hand-build a model for simulation studies or unit tests.
    #[staticmethod]
    #[pyo3(signature = (links, quadrature_nodes=25, *, adaptive_quadrature=true, quadrature_max_nodes=4096, quadrature_rel_tol=1e-7))]
    fn from_links(
        py: Python<'_>,
        links: &Bound<'_, PyList>,
        quadrature_nodes: usize,
        adaptive_quadrature: bool,
        quadrature_max_nodes: usize,
        quadrature_rel_tol: f64,
    ) -> PyResult<Self> {
        catch_internal_panic(|| {
            let specs: Vec<PairCopulaSpec> = links
                .iter()
                .map(|item| pair_spec_from_py_dict(item.cast::<PyDict>()?))
                .collect::<PyResult<Vec<_>>>()?;
            let quadrature = FactorQuadrature {
                adaptive: adaptive_quadrature,
                max_nodes: quadrature_max_nodes,
                rel_tol: quadrature_rel_tol,
            };
            let inner = detached(py, move || {
                FactorCopula::basic_1f(specs, quadrature_nodes)
                    .and_then(|model| model.with_quadrature(quadrature))
            })?;
            Ok(Self { inner })
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        data,
        family_set=None,
        include_rotations=true,
        criterion="aic",
        quadrature_nodes=25,
        refine_iterations=2,
        joint_polish_cycles=5,
        joint_polish_rel_tol=1e-6,
        layout="basic_1f",
        clip_eps=1e-12,
        max_iter=500,
        adaptive_quadrature=true,
        quadrature_max_nodes=4096,
        quadrature_rel_tol=1e-7
    ))]
    fn fit(
        py: Python<'_>,
        data: PyReadonlyArray2<'_, f64>,
        family_set: Option<Vec<String>>,
        include_rotations: bool,
        criterion: &str,
        quadrature_nodes: usize,
        refine_iterations: usize,
        joint_polish_cycles: usize,
        joint_polish_rel_tol: f64,
        layout: &str,
        clip_eps: f64,
        max_iter: usize,
        adaptive_quadrature: bool,
        quadrature_max_nodes: usize,
        quadrature_rel_tol: f64,
    ) -> PyResult<(Self, PyFitDiagnostics, Vec<f64>)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let mut options = FactorFitOptions {
                base: fit_options(clip_eps, max_iter),
                layout: factor_layout_from_name(layout)?,
                include_rotations,
                criterion: criterion_from_name(criterion)?,
                quadrature_nodes,
                quadrature: FactorQuadrature {
                    adaptive: adaptive_quadrature,
                    max_nodes: quadrature_max_nodes,
                    rel_tol: quadrature_rel_tol,
                },
                refine_iterations,
                joint_polish_cycles,
                joint_polish_rel_tol,
                ..FactorFitOptions::default()
            };
            if let Some(families) = family_set {
                options.family_set = families
                    .iter()
                    .map(|family| pair_family_from_name(family))
                    .collect::<PyResult<Vec<_>>>()?;
            }
            let FactorFitResult {
                model,
                diagnostics,
                std_errors,
            } = detached(py, || FactorCopula::fit(&data, &options))?;
            Ok((Self { inner: model }, diagnostics.into(), std_errors))
        })
    }

    /// Round-trip via JSON. The serialized form is exactly what `serde_json`
    /// produces for the underlying `FactorCopula` struct — stable enough for
    /// cross-version comparison provided both sides are on matching rscopulas
    /// minor versions.
    fn to_json(&self) -> PyResult<String> {
        json_string!(&self.inner, "factor copula")
    }

    #[staticmethod]
    fn from_json(payload: &str) -> PyResult<Self> {
        catch_internal_panic(|| {
            Ok(Self {
                inner: parse_json!(FactorCopula, payload, "factor copula")?,
            })
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn family(&self) -> &'static str {
        family_name(self.inner.family())
    }

    #[getter]
    fn num_factors(&self) -> usize {
        self.inner.num_factors()
    }

    #[getter]
    fn quadrature_nodes(&self) -> usize {
        self.inner.quadrature_nodes()
    }

    #[getter]
    fn adaptive_quadrature(&self) -> bool {
        self.inner.quadrature().adaptive
    }
    #[getter]
    fn quadrature_max_nodes(&self) -> usize {
        self.inner.quadrature().max_nodes
    }
    #[getter]
    fn quadrature_rel_tol(&self) -> f64 {
        self.inner.quadrature().rel_tol
    }

    #[getter]
    fn layout(&self) -> &'static str {
        factor_layout_name(self.inner.layout())
    }

    /// Returns the per-variable link specifications as a list of dictionaries.
    /// Layout matches the vine/HAC conventions so downstream tooling can share
    /// serialization code.
    fn links<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        catch_internal_panic(|| {
            let list = PyList::empty(py);
            for link in self.inner.links() {
                list.append(pair_spec_to_py(py, link)?)?;
            }
            Ok(list)
        })
    }

    #[pyo3(signature = (data, clip_eps=1e-12))]
    fn log_pdf<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        model_log_pdf(py, &self.inner, data, clip_eps)
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        model_sample(py, &self.inner, n, seed)
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyType>, (String,))> {
        Ok((slf.get_type(), (slf.get().to_json()?,)))
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn __eq__(&self, other: PyRef<'_, Self>) -> PyResult<bool> {
        json_equal!(&self.inner, &other.inner)
    }

    fn __repr__(&self) -> String {
        let links = self.inner.links();
        format!(
            "_FactorCopula(dim={}, layout='{}', links={})",
            self.inner.dim(),
            factor_layout_name(self.inner.layout()),
            fmt_names(
                links.iter().map(|link| pair_family_name(link.family)),
                links.len()
            )
        )
    }
}

/// Draws an `(n, d)` matrix of uniforms from the same seeded Rust generator
/// that `sample` uses, clipped away from 0 and 1 exactly like the core
/// samplers. `VineCopula.sample_conditional` uses this so that a seed gives
/// the same draws independently of NumPy's global random state.
#[pyfunction]
#[pyo3(signature = (n, d, seed=None))]
fn uniform_matrix<'py>(
    py: Python<'py>,
    n: usize,
    d: usize,
    seed: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    catch_internal_panic(|| {
        positive_count(n, "n")?;
        let mut rng = rng_from_seed(seed_from_py(seed)?);
        let values = py.detach(move || {
            let mut matrix = Array2::<f64>::zeros((n, d));
            for value in matrix.iter_mut() {
                *value = rng
                    .random::<f64>()
                    .clamp(UNIFORM_CLIP_EPS, 1.0 - UNIFORM_CLIP_EPS);
            }
            matrix
        });
        Ok(values.into_pyarray(py))
    })
}

#[pymodule]
fn _rscopulas(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;

    module.add("RscopulasError", py.get_type::<RscopulasError>())?;
    module.add("InvalidInputError", py.get_type::<InvalidInputError>())?;
    module.add("ModelFitError", py.get_type::<ModelFitError>())?;
    module.add("NumericalError", py.get_type::<NumericalError>())?;
    module.add("BackendError", py.get_type::<BackendError>())?;
    module.add("InternalError", py.get_type::<InternalError>())?;
    module.add(
        "NonPrefixConditioningError",
        py.get_type::<NonPrefixConditioningError>(),
    )?;

    module.add_class::<PyFitDiagnostics>()?;
    module.add_class::<PyGaussianCopula>()?;
    module.add_class::<PyStudentTCopula>()?;
    module.add_class::<PyClaytonCopula>()?;
    module.add_class::<PyFrankCopula>()?;
    module.add_class::<PyGumbelCopula>()?;
    module.add_class::<PyPairCopula>()?;
    module.add_class::<PyVineCopula>()?;
    module.add_class::<PyHierarchicalArchimedeanCopula>()?;
    module.add_class::<PyFactorCopula>()?;

    module.add_function(wrap_pyfunction!(uniform_matrix, module)?)?;
    Ok(())
}
