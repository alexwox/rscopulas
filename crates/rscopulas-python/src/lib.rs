use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::{
    Bound, create_exception,
    exceptions::{PyException, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyModule},
};
use rand::{SeedableRng, random, rngs::StdRng};
use rscopulas_core::{
    ClaytonCopula, CopulaError, CopulaFamily, CopulaModel, EvalOptions, ExecPolicy, FitDiagnostics,
    FitOptions, FrankCopula, GaussianCopula, GumbelHougaardCopula, HacFamily, HacFitMethod,
    HacFitOptions, HacNode, HacStructureMethod, HacTree, HierarchicalArchimedeanCopula,
    PairCopulaFamily, PairCopulaParams, Rotation, SampleOptions, SelectionCriterion,
    StudentTCopula, VineCopula, VineFitOptions, VineStructureKind,
};

create_exception!(rscopulas, RscopulasError, PyException);
create_exception!(rscopulas, InvalidInputError, RscopulasError);
create_exception!(rscopulas, ModelFitError, RscopulasError);
create_exception!(rscopulas, NumericalError, RscopulasError);
create_exception!(rscopulas, BackendError, RscopulasError);

fn to_pyerr(error: CopulaError) -> PyErr {
    match error {
        CopulaError::InvalidInput(inner) => InvalidInputError::new_err(inner.to_string()),
        CopulaError::FitFailed(inner) => ModelFitError::new_err(inner.to_string()),
        CopulaError::Numerical(inner) => NumericalError::new_err(inner.to_string()),
        CopulaError::Backend(inner) => BackendError::new_err(inner.to_string()),
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

fn pseudo_obs_from_py(data: PyReadonlyArray2<'_, f64>) -> PyResult<rscopulas_core::PseudoObs> {
    rscopulas_core::PseudoObs::from_view(data.as_array())
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
        other => Err(PyValueError::new_err(format!(
            "unsupported pair family '{other}'; expected one of independence, gaussian, student_t, clayton, frank, gumbel"
        ))),
    }
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

fn params_to_vec(params: PairCopulaParams) -> Vec<f64> {
    match params {
        PairCopulaParams::None => Vec::new(),
        PairCopulaParams::One(value) => vec![value],
        PairCopulaParams::Two(first, second) => vec![first, second],
    }
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
        let data = pseudo_obs_from_py(data)?;
        let result =
            GaussianCopula::fit(&data, &fit_options(clip_eps, max_iter)).map_err(to_pyerr)?;
        Ok((
            Self {
                inner: result.model,
            },
            result.diagnostics.into(),
        ))
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
        let data = pseudo_obs_from_py(data)?;
        let result =
            StudentTCopula::fit(&data, &fit_options(clip_eps, max_iter)).map_err(to_pyerr)?;
        Ok((
            Self {
                inner: result.model,
            },
            result.diagnostics.into(),
        ))
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
        let data = pseudo_obs_from_py(data)?;
        let result =
            ClaytonCopula::fit(&data, &fit_options(clip_eps, max_iter)).map_err(to_pyerr)?;
        Ok((
            Self {
                inner: result.model,
            },
            result.diagnostics.into(),
        ))
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
        let data = pseudo_obs_from_py(data)?;
        let result = FrankCopula::fit(&data, &fit_options(clip_eps, max_iter)).map_err(to_pyerr)?;
        Ok((
            Self {
                inner: result.model,
            },
            result.diagnostics.into(),
        ))
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
        let data = pseudo_obs_from_py(data)?;
        let result =
            GumbelHougaardCopula::fit(&data, &fit_options(clip_eps, max_iter)).map_err(to_pyerr)?;
        Ok((
            Self {
                inner: result.model,
            },
            result.diagnostics.into(),
        ))
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
    name = "_VineCopula"
)]
#[derive(Clone)]
struct PyVineCopula {
    inner: VineCopula,
}

#[pymethods]
impl PyVineCopula {
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
    #[pyo3(signature = (data, family_set=None, include_rotations=true, criterion="aic", truncation_level=None, independence_threshold=None, clip_eps=1e-12, max_iter=500))]
    fn fit_c(
        data: PyReadonlyArray2<'_, f64>,
        family_set: Option<Vec<String>>,
        include_rotations: bool,
        criterion: &str,
        truncation_level: Option<usize>,
        independence_threshold: Option<f64>,
        clip_eps: f64,
        max_iter: usize,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
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
        let result = VineCopula::fit_c_vine(&data, &options).map_err(to_pyerr)?;
        Ok((
            Self {
                inner: result.model,
            },
            result.diagnostics.into(),
        ))
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (data, family_set=None, include_rotations=true, criterion="aic", truncation_level=None, independence_threshold=None, clip_eps=1e-12, max_iter=500))]
    fn fit_d(
        data: PyReadonlyArray2<'_, f64>,
        family_set: Option<Vec<String>>,
        include_rotations: bool,
        criterion: &str,
        truncation_level: Option<usize>,
        independence_threshold: Option<f64>,
        clip_eps: f64,
        max_iter: usize,
    ) -> PyResult<(Self, PyFitDiagnostics)> {
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
        let result = VineCopula::fit_d_vine(&data, &options).map_err(to_pyerr)?;
        Ok((
            Self {
                inner: result.model,
            },
            result.diagnostics.into(),
        ))
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
                let edge_dict = PyDict::new(py);
                edge_dict.set_item("tree", edge.tree)?;
                edge_dict.set_item("conditioned", (edge.conditioned.0, edge.conditioned.1))?;
                edge_dict.set_item("conditioning", edge.conditioning.clone())?;
                edge_dict.set_item("family", pair_family_name(edge.copula.family))?;
                edge_dict.set_item("rotation", rotation_name(edge.copula.rotation))?;
                edge_dict.set_item("parameters", params_to_vec(edge.copula.params))?;
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

    module.add_class::<PyFitDiagnostics>()?;
    module.add_class::<PyGaussianCopula>()?;
    module.add_class::<PyStudentTCopula>()?;
    module.add_class::<PyClaytonCopula>()?;
    module.add_class::<PyFrankCopula>()?;
    module.add_class::<PyGumbelCopula>()?;
    module.add_class::<PyVineCopula>()?;
    module.add_class::<PyHierarchicalArchimedeanCopula>()?;
    Ok(())
}
