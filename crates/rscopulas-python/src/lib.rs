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
    FitOptions, FrankCopula, GaussianCopula, GumbelHougaardCopula, PairCopulaFamily,
    PairCopulaParams, Rotation, SampleOptions, SelectionCriterion, StudentTCopula, VineCopula,
    VineFitOptions, VineStructureKind,
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
        CopulaFamily::Vine => "vine",
    }
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
    Ok(())
}
