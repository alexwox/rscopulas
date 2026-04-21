use std::any::Any;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    Bound, create_exception,
    exceptions::{PyException, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyModule},
};
use rand::{SeedableRng, random, rngs::StdRng};
use rscopulas::{
    ClaytonCopula, CopulaError, CopulaFamily, CopulaModel, EvalOptions, ExecPolicy, FitDiagnostics,
    FitOptions, FrankCopula, GaussianCopula, GumbelHougaardCopula, HacFamily, HacFitMethod,
    HacFitOptions, HacNode, HacStructureMethod, HacTree, HierarchicalArchimedeanCopula,
    KhoudrajiParams, PairCopulaFamily, PairCopulaParams, Rotation, SampleOptions,
    SelectionCriterion, StudentTCopula, VineCopula, VineEdge, VineFitOptions, VineStructureKind,
    VineTree,
};

create_exception!(rscopulas, RscopulasError, PyException);
create_exception!(rscopulas, InvalidInputError, RscopulasError);
create_exception!(rscopulas, ModelFitError, RscopulasError);
create_exception!(rscopulas, NumericalError, RscopulasError);
create_exception!(rscopulas, BackendError, RscopulasError);
create_exception!(rscopulas, InternalError, RscopulasError);
create_exception!(rscopulas, NonPrefixConditioningError, InvalidInputError);

fn to_pyerr(error: CopulaError) -> PyErr {
    match error {
        CopulaError::InvalidInput(inner) => InvalidInputError::new_err(inner.to_string()),
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

fn rng_from_seed(seed: Option<u64>) -> StdRng {
    StdRng::seed_from_u64(seed.unwrap_or_else(random))
}

fn pseudo_obs_from_py(data: PyReadonlyArray2<'_, f64>) -> PyResult<rscopulas::PseudoObs> {
    rscopulas::PseudoObs::from_view(data.as_array())
        .map_err(|err| InvalidInputError::new_err(err.to_string()))
}

fn matrix_from_py(data: PyReadonlyArray2<'_, f64>) -> Array2<f64> {
    data.as_array().to_owned()
}

fn pair_family_from_name(name: &str) -> PyResult<PairCopulaFamily> {
    match name.trim().to_ascii_lowercase().as_str() {
        "independence" => Ok(PairCopulaFamily::Independence),
        "gaussian" => Ok(PairCopulaFamily::Gaussian),
        "student_t" | "student-t" | "studentt" | "student" => Ok(PairCopulaFamily::StudentT),
        "clayton" => Ok(PairCopulaFamily::Clayton),
        "frank" => Ok(PairCopulaFamily::Frank),
        "gumbel" => Ok(PairCopulaFamily::Gumbel),
        "khoudraji" => Ok(PairCopulaFamily::Khoudraji),
        other => Err(PyValueError::new_err(format!(
            "unsupported pair family '{other}'; expected one of independence, gaussian, student_t, clayton, frank, gumbel, khoudraji"
        ))),
    }
}

fn rotation_from_name(name: &str) -> PyResult<Rotation> {
    match name.trim().to_ascii_uppercase().as_str() {
        "R0" => Ok(Rotation::R0),
        "R90" => Ok(Rotation::R90),
        "R180" => Ok(Rotation::R180),
        "R270" => Ok(Rotation::R270),
        other => Err(PyValueError::new_err(format!(
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
        | (PairCopulaFamily::Gumbel, [value]) => Ok(PairCopulaParams::One(*value)),
        (PairCopulaFamily::StudentT, [rho, nu]) => Ok(PairCopulaParams::Two(*rho, *nu)),
        (PairCopulaFamily::Khoudraji, _) => Err(PyValueError::new_err(
            "khoudraji pair copulas require structured base_copula_1/base_copula_2 and shape_1/shape_2 inputs",
        )),
        (PairCopulaFamily::Independence, values) => Err(PyValueError::new_err(format!(
            "independence pair copulas do not take parameters (got {})",
            values.len()
        ))),
        (PairCopulaFamily::StudentT, values) => Err(PyValueError::new_err(format!(
            "student_t pair copulas require exactly two parameters (got {})",
            values.len()
        ))),
        (_, values) => Err(PyValueError::new_err(format!(
            "pair family requires exactly one parameter (got {})",
            values.len()
        ))),
    }
}

fn pair_spec_from_values(
    family: &str,
    rotation: &str,
    parameters: Vec<f64>,
) -> PyResult<rscopulas::PairCopulaSpec> {
    let family = pair_family_from_name(family)?;
    Ok(rscopulas::PairCopulaSpec {
        family,
        rotation: rotation_from_name(rotation)?,
        params: pair_params_from_values(family, parameters)?,
    })
}

fn pair_spec_from_py_dict(dict: &Bound<'_, PyDict>) -> PyResult<rscopulas::PairCopulaSpec> {
    let family = dict
        .get_item("family")?
        .ok_or_else(|| PyValueError::new_err("pair spec dictionaries require 'family'"))?
        .extract::<String>()?;
    let rotation = dict
        .get_item("rotation")?
        .map(|value| value.extract::<String>())
        .transpose()?
        .unwrap_or_else(|| "R0".to_string());
    let parsed_family = pair_family_from_name(&family)?;
    if parsed_family == PairCopulaFamily::Khoudraji {
        let base_first_value = dict
            .get_item("base_copula_1")?
            .ok_or_else(|| PyValueError::new_err("khoudraji specs require 'base_copula_1'"))?;
        let base_first = base_first_value.cast::<PyDict>()?;
        let base_second_value = dict
            .get_item("base_copula_2")?
            .ok_or_else(|| PyValueError::new_err("khoudraji specs require 'base_copula_2'"))?;
        let base_second = base_second_value.cast::<PyDict>()?;
        let shape_first = dict
            .get_item("shape_1")?
            .ok_or_else(|| PyValueError::new_err("khoudraji specs require 'shape_1'"))?
            .extract::<f64>()?;
        let shape_second = dict
            .get_item("shape_2")?
            .ok_or_else(|| PyValueError::new_err("khoudraji specs require 'shape_2'"))?
            .extract::<f64>()?;
        return Ok(rscopulas::PairCopulaSpec {
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
        other => Err(PyValueError::new_err(format!(
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
        return Err(PyValueError::new_err(format!(
            "{left_name} and {right_name} must have the same length"
        )));
    }
    Ok((left_values, right_values))
}

fn criterion_from_name(name: &str) -> PyResult<SelectionCriterion> {
    match name.trim().to_ascii_lowercase().as_str() {
        "aic" => Ok(SelectionCriterion::Aic),
        "bic" => Ok(SelectionCriterion::Bic),
        other => Err(PyValueError::new_err(format!(
            "unsupported selection criterion '{other}'; expected 'aic' or 'bic'"
        ))),
    }
}

fn vine_fit_options(
    family_set: Option<Vec<String>>,
    include_rotations: bool,
    criterion: &str,
    truncation_level: Option<usize>,
    independence_threshold: Option<f64>,
    clip_eps: f64,
    max_iter: usize,
) -> PyResult<VineFitOptions> {
    let mut options = VineFitOptions {
        base: fit_options(clip_eps, max_iter),
        include_rotations,
        criterion: criterion_from_name(criterion)?,
        truncation_level,
        independence_threshold,
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
        other => Err(PyValueError::new_err(format!(
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
        other => Err(PyValueError::new_err(format!(
            "unsupported HAC structure method '{other}'"
        ))),
    }
}

fn hac_fit_method_name(method: HacFitMethod) -> &'static str {
    match method {
        HacFitMethod::TauInit => "tau_init",
        HacFitMethod::RecursiveMle => "recursive_mle",
        HacFitMethod::FullMle => "full_mle",
        HacFitMethod::Smle => "smle",
        HacFitMethod::Dmle => "dmle",
    }
}

fn hac_fit_method_from_name(name: &str) -> PyResult<HacFitMethod> {
    match name.trim().to_ascii_lowercase().as_str() {
        "tau_init" | "tau" => Ok(HacFitMethod::TauInit),
        "recursive_mle" | "recursive" => Ok(HacFitMethod::RecursiveMle),
        "full_mle" | "full" => Ok(HacFitMethod::FullMle),
        "smle" => Ok(HacFitMethod::Smle),
        "dmle" => Ok(HacFitMethod::Dmle),
        other => Err(PyValueError::new_err(format!(
            "unsupported HAC fit method '{other}'"
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
        .ok_or_else(|| PyValueError::new_err("HAC node dictionaries require a 'family' key"))?
        .extract::<String>()?;
    let theta = dict
        .get_item("theta")?
        .ok_or_else(|| PyValueError::new_err("HAC node dictionaries require a 'theta' key"))?
        .extract::<f64>()?;
    let children_value = dict
        .get_item("children")?
        .ok_or_else(|| PyValueError::new_err("HAC node dictionaries require a 'children' key"))?;
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
    }
}

fn attach_pair_components<'py>(
    py: Python<'py>,
    dict: &Bound<'py, PyDict>,
    spec: &rscopulas::PairCopulaSpec,
) -> PyResult<()> {
    if let PairCopulaParams::Khoudraji(params) = &spec.params {
        dict.set_item("shape_1", params.shape_first)?;
        dict.set_item("shape_2", params.shape_second)?;
        dict.set_item("base_copula_1", pair_spec_to_py(py, &params.first)?)?;
        dict.set_item("base_copula_2", pair_spec_to_py(py, &params.second)?)?;
    }
    Ok(())
}

fn pair_spec_to_py<'py>(
    py: Python<'py>,
    spec: &rscopulas::PairCopulaSpec,
) -> PyResult<Bound<'py, PyDict>> {
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
        .ok_or_else(|| PyValueError::new_err("vine edge dictionaries require 'conditioned'"))?
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
        .ok_or_else(|| PyValueError::new_err("vine tree dictionaries require 'level'"))?
        .extract::<usize>()?;
    let edges = dict
        .get_item("edges")?
        .ok_or_else(|| PyValueError::new_err("vine tree dictionaries require 'edges'"))?
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
}

impl From<FitDiagnostics> for PyFitDiagnostics {
    fn from(value: FitDiagnostics) -> Self {
        Self {
            loglik: value.loglik,
            aic: value.aic,
            bic: value.bic,
            converged: value.converged,
            n_iter: value.n_iter,
        }
    }
}

#[pymethods]
impl PyFitDiagnostics {
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
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_GaussianCopula"
)]
#[derive(Clone)]
struct PyGaussianCopula {
    inner: GaussianCopula,
}

#[pymethods]
impl PyGaussianCopula {
    #[staticmethod]
    fn from_params(correlation: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        GaussianCopula::new(matrix_from_py(correlation))
            .map(|inner| Self { inner })
            .map_err(to_pyerr)
    }

    #[staticmethod]
    #[pyo3(signature = (data, clip_eps=1e-12, max_iter=500))]
    fn fit(
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
        max_iter: usize,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let result =
                GaussianCopula::fit(&data, &fit_options(clip_eps, max_iter)).map_err(to_pyerr)?;
            Ok((
                Self {
                    inner: result.model,
                },
                result.diagnostics.into(),
            ))
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
        let data = pseudo_obs_from_py(data)?;
        let values = self
            .inner
            .log_pdf(&data, &eval_options(clip_eps))
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut rng = rng_from_seed(seed);
        let values = self
            .inner
            .sample(n, &mut rng, &sample_options())
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_StudentTCopula"
)]
#[derive(Clone)]
struct PyStudentTCopula {
    inner: StudentTCopula,
}

#[pymethods]
impl PyStudentTCopula {
    #[staticmethod]
    fn from_params(
        correlation: PyReadonlyArray2<'_, f64>,
        degrees_of_freedom: f64,
    ) -> PyResult<Self> {
        StudentTCopula::new(matrix_from_py(correlation), degrees_of_freedom)
            .map(|inner| Self { inner })
            .map_err(to_pyerr)
    }

    #[staticmethod]
    #[pyo3(signature = (data, clip_eps=1e-12, max_iter=500))]
    fn fit(
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
        max_iter: usize,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let result =
                StudentTCopula::fit(&data, &fit_options(clip_eps, max_iter)).map_err(to_pyerr)?;
            Ok((
                Self {
                    inner: result.model,
                },
                result.diagnostics.into(),
            ))
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
        let data = pseudo_obs_from_py(data)?;
        let values = self
            .inner
            .log_pdf(&data, &eval_options(clip_eps))
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut rng = rng_from_seed(seed);
        let values = self
            .inner
            .sample(n, &mut rng, &sample_options())
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_ClaytonCopula"
)]
#[derive(Clone)]
struct PyClaytonCopula {
    inner: ClaytonCopula,
}

#[pymethods]
impl PyClaytonCopula {
    #[staticmethod]
    fn from_params(dim: usize, theta: f64) -> PyResult<Self> {
        ClaytonCopula::new(dim, theta)
            .map(|inner| Self { inner })
            .map_err(to_pyerr)
    }

    #[staticmethod]
    #[pyo3(signature = (data, clip_eps=1e-12, max_iter=500))]
    fn fit(
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
        max_iter: usize,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let result =
                ClaytonCopula::fit(&data, &fit_options(clip_eps, max_iter)).map_err(to_pyerr)?;
            Ok((
                Self {
                    inner: result.model,
                },
                result.diagnostics.into(),
            ))
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
        let data = pseudo_obs_from_py(data)?;
        let values = self
            .inner
            .log_pdf(&data, &eval_options(clip_eps))
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut rng = rng_from_seed(seed);
        let values = self
            .inner
            .sample(n, &mut rng, &sample_options())
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_FrankCopula"
)]
#[derive(Clone)]
struct PyFrankCopula {
    inner: FrankCopula,
}

#[pymethods]
impl PyFrankCopula {
    #[staticmethod]
    fn from_params(dim: usize, theta: f64) -> PyResult<Self> {
        FrankCopula::new(dim, theta)
            .map(|inner| Self { inner })
            .map_err(to_pyerr)
    }

    #[staticmethod]
    #[pyo3(signature = (data, clip_eps=1e-12, max_iter=500))]
    fn fit(
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
        max_iter: usize,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let result =
                FrankCopula::fit(&data, &fit_options(clip_eps, max_iter)).map_err(to_pyerr)?;
            Ok((
                Self {
                    inner: result.model,
                },
                result.diagnostics.into(),
            ))
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
        let data = pseudo_obs_from_py(data)?;
        let values = self
            .inner
            .log_pdf(&data, &eval_options(clip_eps))
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut rng = rng_from_seed(seed);
        let values = self
            .inner
            .sample(n, &mut rng, &sample_options())
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_GumbelCopula"
)]
#[derive(Clone)]
struct PyGumbelCopula {
    inner: GumbelHougaardCopula,
}

#[pymethods]
impl PyGumbelCopula {
    #[staticmethod]
    fn from_params(dim: usize, theta: f64) -> PyResult<Self> {
        GumbelHougaardCopula::new(dim, theta)
            .map(|inner| Self { inner })
            .map_err(to_pyerr)
    }

    #[staticmethod]
    #[pyo3(signature = (data, clip_eps=1e-12, max_iter=500))]
    fn fit(
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
        max_iter: usize,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let result = GumbelHougaardCopula::fit(&data, &fit_options(clip_eps, max_iter))
                .map_err(to_pyerr)?;
            Ok((
                Self {
                    inner: result.model,
                },
                result.diagnostics.into(),
            ))
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
        let data = pseudo_obs_from_py(data)?;
        let values = self
            .inner
            .log_pdf(&data, &eval_options(clip_eps))
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut rng = rng_from_seed(seed);
        let values = self
            .inner
            .sample(n, &mut rng, &sample_options())
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_PairCopula"
)]
#[derive(Clone)]
struct PyPairCopula {
    inner: rscopulas::PairCopulaSpec,
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
        mut callback: F,
    ) -> PyResult<Bound<'py, PyArray1<f64>>>
    where
        F: FnMut(&rscopulas::PairCopulaSpec, f64, f64, f64) -> Result<f64, CopulaError>,
    {
        let (left_values, right_values) =
            paired_vectors_from_py(left, right, left_name, right_name)?;
        let values = left_values
            .into_iter()
            .zip(right_values)
            .map(|(first, second)| callback(&self.inner, first, second, clip_eps).map_err(to_pyerr))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(values.into_pyarray(py))
    }
}

#[pymethods]
impl PyPairCopula {
    #[staticmethod]
    #[pyo3(signature = (family, parameters=None, rotation="R0"))]
    fn from_spec(family: &str, parameters: Option<Vec<f64>>, rotation: &str) -> PyResult<Self> {
        if pair_family_from_name(family)? == PairCopulaFamily::Khoudraji {
            return Err(PyValueError::new_err(
                "use PairCopula.from_khoudraji(...) for khoudraji specifications",
            ));
        }
        Ok(Self {
            inner: pair_spec_from_values(family, rotation, parameters.unwrap_or_default())?,
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
        Ok(Self {
            inner: rscopulas::PairCopulaSpec {
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
        pair_spec_to_py(py, &self.inner)
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
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_VineCopula"
)]
#[derive(Clone)]
struct PyVineCopula {
    inner: VineCopula,
}

#[pymethods]
impl PyVineCopula {
    #[staticmethod]
    fn from_trees(
        kind: &str,
        trees: &Bound<'_, PyAny>,
        truncation_level: Option<usize>,
    ) -> PyResult<Self> {
        let trees = trees
            .cast::<PyList>()?
            .iter()
            .map(|tree| vine_tree_from_py(&tree))
            .collect::<PyResult<Vec<_>>>()?;
        VineCopula::from_trees(vine_kind_from_name(kind)?, trees, truncation_level)
            .map(|inner| Self { inner })
            .map_err(to_pyerr)
    }

    #[staticmethod]
    fn gaussian_c_vine(
        order: Vec<usize>,
        correlation: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        VineCopula::gaussian_c_vine(order, matrix_from_py(correlation))
            .map(|inner| Self { inner })
            .map_err(to_pyerr)
    }

    #[staticmethod]
    fn gaussian_d_vine(
        order: Vec<usize>,
        correlation: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        VineCopula::gaussian_d_vine(order, matrix_from_py(correlation))
            .map(|inner| Self { inner })
            .map_err(to_pyerr)
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (data, family_set=None, include_rotations=true, criterion="aic", truncation_level=None, independence_threshold=None, clip_eps=1e-12, max_iter=500, order=None))]
    fn fit_c(
        data: PyReadonlyArray2<'_, f64>,
        family_set: Option<Vec<String>>,
        include_rotations: bool,
        criterion: &str,
        truncation_level: Option<usize>,
        independence_threshold: Option<f64>,
        clip_eps: f64,
        max_iter: usize,
        order: Option<Vec<usize>>,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let options = vine_fit_options(
                family_set,
                include_rotations,
                criterion,
                truncation_level,
                independence_threshold,
                clip_eps,
                max_iter,
            )?;
            let result = match order {
                Some(order) => VineCopula::fit_c_vine_with_order(&data, &order, &options),
                None => VineCopula::fit_c_vine(&data, &options),
            }
            .map_err(to_pyerr)?;
            Ok((
                Self {
                    inner: result.model,
                },
                result.diagnostics.into(),
            ))
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (data, family_set=None, include_rotations=true, criterion="aic", truncation_level=None, independence_threshold=None, clip_eps=1e-12, max_iter=500, order=None))]
    fn fit_d(
        data: PyReadonlyArray2<'_, f64>,
        family_set: Option<Vec<String>>,
        include_rotations: bool,
        criterion: &str,
        truncation_level: Option<usize>,
        independence_threshold: Option<f64>,
        clip_eps: f64,
        max_iter: usize,
        order: Option<Vec<usize>>,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let options = vine_fit_options(
                family_set,
                include_rotations,
                criterion,
                truncation_level,
                independence_threshold,
                clip_eps,
                max_iter,
            )?;
            let result = match order {
                Some(order) => VineCopula::fit_d_vine_with_order(&data, &order, &options),
                None => VineCopula::fit_d_vine(&data, &options),
            }
            .map_err(to_pyerr)?;
            Ok((
                Self {
                    inner: result.model,
                },
                result.diagnostics.into(),
            ))
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (data, family_set=None, include_rotations=true, criterion="aic", truncation_level=None, independence_threshold=None, clip_eps=1e-12, max_iter=500))]
    fn fit_r(
        data: PyReadonlyArray2<'_, f64>,
        family_set: Option<Vec<String>>,
        include_rotations: bool,
        criterion: &str,
        truncation_level: Option<usize>,
        independence_threshold: Option<f64>,
        clip_eps: f64,
        max_iter: usize,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
        catch_internal_panic(|| {
            let data = pseudo_obs_from_py(data)?;
            let options = vine_fit_options(
                family_set,
                include_rotations,
                criterion,
                truncation_level,
                independence_threshold,
                clip_eps,
                max_iter,
            )?;
            let result = VineCopula::fit_r_vine(&data, &options).map_err(to_pyerr)?;
            Ok((
                Self {
                    inner: result.model,
                },
                result.diagnostics.into(),
            ))
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
    fn structure_kind(&self) -> &'static str {
        vine_kind_name(self.inner.structure())
    }

    #[getter]
    fn truncation_level(&self) -> Option<usize> {
        self.inner.truncation_level()
    }

    fn order(&self) -> Vec<usize> {
        self.inner.order()
    }

    fn pair_parameters(&self) -> Vec<f64> {
        self.inner.pair_parameters()
    }

    fn structure_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = self.inner.structure_info();
        let dict = PyDict::new(py);
        dict.set_item("kind", vine_kind_name(info.kind))?;
        dict.set_item("matrix", info.matrix.clone().into_pyarray(py))?;
        dict.set_item("truncation_level", info.truncation_level)?;
        Ok(dict)
    }

    fn trees<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
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
    }

    #[pyo3(signature = (data, clip_eps=1e-12))]
    fn log_pdf<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data = pseudo_obs_from_py(data)?;
        let values = self
            .inner
            .log_pdf(&data, &eval_options(clip_eps))
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut rng = rng_from_seed(seed);
        let values = self
            .inner
            .sample(n, &mut rng, &sample_options())
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
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
        let view = data.as_array();
        let values = self
            .inner
            .rosenblatt(view, &sample_options())
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }

    /// Inverse Rosenblatt transform `V = F^{-1}(U)` indexed by original
    /// variable label. This is the primitive behind `sample` and
    /// `sample_conditional`.
    fn inverse_rosenblatt<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let view = data.as_array();
        let values = self
            .inner
            .inverse_rosenblatt(view, &sample_options())
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
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
        let view = data.as_array();
        let values = self
            .inner
            .rosenblatt_prefix(view, col_limit, &sample_options())
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }
}

#[pyclass(
    skip_from_py_object,
    module = "rscopulas._rscopulas",
    name = "_HierarchicalArchimedeanCopula"
)]
#[derive(Clone)]
struct PyHierarchicalArchimedeanCopula {
    inner: HierarchicalArchimedeanCopula,
}

#[pymethods]
impl PyHierarchicalArchimedeanCopula {
    #[staticmethod]
    fn from_tree(tree: &Bound<'_, PyAny>) -> PyResult<Self> {
        let tree = hac_tree_from_py(tree)?;
        HierarchicalArchimedeanCopula::new(tree)
            .map(|inner| Self { inner })
            .map_err(to_pyerr)
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (data, tree=None, family_set=None, structure_method="agglomerative_tau_then_collapse", fit_method="recursive_mle", collapse_eps=0.05, mc_samples=256, allow_experimental=true, clip_eps=1e-12, max_iter=500))]
    fn fit(
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
            let result = match tree {
                Some(tree) => {
                    let parsed_tree = hac_tree_from_py(tree)?;
                    HierarchicalArchimedeanCopula::fit_with_tree(&data, parsed_tree, &options)
                }
                None => HierarchicalArchimedeanCopula::fit(&data, &options),
            }
            .map_err(to_pyerr)?;
            Ok((
                Self {
                    inner: result.model,
                },
                result.diagnostics.into(),
            ))
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
        hac_tree_to_py(py, self.inner.tree())
    }

    fn leaf_order(&self) -> Vec<usize> {
        self.inner.leaf_order()
    }

    fn parameters(&self) -> Vec<f64> {
        self.inner.parameters()
    }

    fn families(&self) -> Vec<String> {
        self.inner
            .families()
            .into_iter()
            .map(|family| hac_family_name(family).to_string())
            .collect()
    }

    #[pyo3(signature = (data, clip_eps=1e-12))]
    fn log_pdf<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'_, f64>,
        clip_eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data = pseudo_obs_from_py(data)?;
        let values = self
            .inner
            .log_pdf(&data, &eval_options(clip_eps))
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }

    #[pyo3(signature = (n, seed=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut rng = rng_from_seed(seed);
        let values = self
            .inner
            .sample(n, &mut rng, &sample_options())
            .map_err(to_pyerr)?;
        Ok(values.into_pyarray(py))
    }
}

#[pymodule]
fn _rscopulas(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
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
    Ok(())
}
