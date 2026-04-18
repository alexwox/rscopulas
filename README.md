# rscopulas

`rscopulas` is a copula workspace centered on one core idea: operate on
validated pseudo-observations and make fitting, density evaluation, and
sampling predictable across single-family copulas, pair copulas, and vine
copulas.

The current source of truth in this repository is the Rust crate
`rscopulas-core`, with a NumPy-first Python package layered on top of it.

## What the package gives you

- Single-family copulas for Gaussian, Student t, Clayton, Frank, and
  Gumbel-Hougaard.
- Bivariate pair-copula kernels with h-functions and inverse h-functions.
- Asymmetric Khoudraji pair-copulas that can participate in vine edge fitting.
- C-vine, D-vine, and R-vine construction, fitting, scoring, and sampling.
- Explicit data validation through `PseudoObs`, so the library only accepts
  inputs in the open unit hypercube `(0, 1)^d`.
- Likelihood diagnostics on fitted models: log-likelihood, AIC, BIC,
  convergence flag, and iteration count.
- An execution policy layer that keeps CPU, CUDA, and Metal behavior honest
  instead of silently pretending every path is accelerated.

## Workspace layout

- `crates/rscopulas-core`: the main Rust API and the package most users should
  start with.
- `crates/rscopulas-accel`: backend helpers for CPU parallelism plus narrow
  CUDA and Metal acceleration.
- `crates/rscopulas-python`: local Python bindings built on top of
  `rscopulas-core`.

## Core mental model

Everything starts with pseudo-observations:

- Each value must be finite.
- Each value must satisfy `0 < u < 1`.
- The library does not estimate marginal distributions for you.
- You fit a model from `PseudoObs`, then evaluate `log_pdf(...)` or
  `sample(...)` from the fitted model.

If you already have ranked and scaled data, you are in the right place. If you
have raw observations, transform them to pseudo-observations before calling the
API.

## Rust variant

The primary Rust crate is `rscopulas-core`.

### Add the crate

This repository is currently organized as a workspace, so local development
typically depends on the path crate directly:

```toml
[dependencies]
ndarray = "0.17"
rand = "0.9"
rscopulas-core = { path = "crates/rscopulas-core" }
```

### Fit a Gaussian copula

```rust
use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};
use rscopulas_core::{CopulaModel, FitOptions, GaussianCopula, PseudoObs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = PseudoObs::new(array![
        [0.12, 0.18],
        [0.21, 0.25],
        [0.27, 0.22],
        [0.35, 0.42],
        [0.48, 0.51],
        [0.56, 0.49],
        [0.68, 0.73],
        [0.82, 0.79],
    ])?;

    let fit = GaussianCopula::fit(&data, &FitOptions::default())?;
    println!("AIC: {}", fit.diagnostics.aic);
    println!("Correlation:\n{:?}", fit.model.correlation());

    let log_pdf = fit.model.log_pdf(&data, &Default::default())?;
    println!("First log density: {}", log_pdf[0]);

    let mut rng = StdRng::seed_from_u64(7);
    let sample = fit.model.sample(4, &mut rng, &Default::default())?;
    println!("Sample:\n{:?}", sample);
    Ok(())
}
```

### Fit a mixed-family R-vine

```rust
use ndarray::array;
use rscopulas_core::{
    PairCopulaFamily, PseudoObs, SelectionCriterion, VineCopula, VineFitOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = PseudoObs::new(array![
        [0.12, 0.18, 0.21],
        [0.21, 0.25, 0.29],
        [0.27, 0.22, 0.31],
        [0.35, 0.42, 0.39],
        [0.48, 0.51, 0.46],
        [0.56, 0.49, 0.58],
        [0.68, 0.73, 0.69],
        [0.82, 0.79, 0.76],
    ])?;

    let options = VineFitOptions {
        family_set: vec![
            PairCopulaFamily::Independence,
            PairCopulaFamily::Gaussian,
            PairCopulaFamily::Clayton,
            PairCopulaFamily::Frank,
            PairCopulaFamily::Gumbel,
        PairCopulaFamily::Khoudraji,
        ],
        include_rotations: true,
        criterion: SelectionCriterion::Aic,
        truncation_level: Some(1),
        ..VineFitOptions::default()
    };

    let fit = VineCopula::fit_r_vine(&data, &options)?;
    println!("structure = {:?}", fit.model.structure());
    println!("order = {:?}", fit.model.order());
    println!("pair parameters = {:?}", fit.model.pair_parameters());
    Ok(())
}
```

### Build a vine from explicit Gaussian parameters

Use this path when you already know the structure and want a model directly,
without fitting pair families from data.

```rust
use ndarray::array;
use rscopulas_core::VineCopula;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let correlation = array![
        [1.0, 0.60, 0.35],
        [0.60, 1.0, 0.25],
        [0.35, 0.25, 1.0],
    ];

    let model = VineCopula::gaussian_c_vine(vec![0, 1, 2], correlation)?;
    println!("order = {:?}", model.order());
    Ok(())
}
```

### Work directly with pair copulas

The pair-copula layer is useful when you want low-level control over a vine
edge, need h-functions explicitly, or want to inspect the selected family on one
pair before fitting a full vine.

```rust
use rscopulas_core::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spec = PairCopulaSpec {
        family: PairCopulaFamily::Clayton,
        rotation: Rotation::R90,
        params: PairCopulaParams::One(1.4),
    };

    let log_pdf = spec.log_pdf(0.32, 0.77, 1e-12)?;
    let h = spec.cond_first_given_second(0.32, 0.77, 1e-12)?;

    println!("log_pdf = {log_pdf}");
    println!("h_1|2 = {h}");
    Ok(())
}
```

You can also build an asymmetric Khoudraji edge from two existing pair kernels:

```rust
use rscopulas_core::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let first = PairCopulaSpec {
        family: PairCopulaFamily::Gaussian,
        rotation: Rotation::R0,
        params: PairCopulaParams::One(0.45),
    };
    let second = PairCopulaSpec {
        family: PairCopulaFamily::Clayton,
        rotation: Rotation::R0,
        params: PairCopulaParams::One(2.0),
    };
    let khoudraji = PairCopulaSpec::khoudraji(first, second, 0.35, 0.80)?;
    println!("log_pdf = {}", khoudraji.log_pdf(0.32, 0.77, 1e-12)?);
    Ok(())
}
```

### Important API notes

- Import the `CopulaModel` trait when calling trait methods like `log_pdf(...)`
  and `sample(...)` on concrete models.
- `PseudoObs::new(...)` validates that your matrix has at least two columns and
  all values lie strictly inside `(0, 1)`.
- `GaussianCopula::fit(...)` inverts the Kendall tau matrix.
- `StudentTCopula::fit(...)` combines Kendall tau inversion with a grid search
  over degrees of freedom.
- `ClaytonCopula`, `FrankCopula`, and `GumbelHougaardCopula` fit a single
  dependence parameter from the data.
- `PairCopulaSpec::khoudraji(...)` builds a generic asymmetric pair copula from
  two existing pair kernels and two shape parameters.
- `VineFitOptions` controls candidate pair families, rotations, criterion,
  truncation, and optional independence selection thresholds. Khoudraji
  candidates currently run on CPU only and use a bounded internal search over
  supported base families.

### Execution backends

Execution is explicit by design:

- `ExecPolicy::Auto` is conservative and currently means CPU reference or
  CPU-parallel behavior only.
- `ExecPolicy::Force(Device::Cuda(_))` and `ExecPolicy::Force(Device::Metal)`
  only work on the narrow GPU-aware paths that exist today.
- Single-family `log_pdf(...)`, Kendall tau estimation, and sampling remain CPU
  paths.
- Gaussian pair-batch evaluation and Gaussian vine density evaluation are the
  main accelerated paths today.

If you need deterministic backend selection, prefer explicit `Force(...)`
policies instead of relying on `Auto`.

## Python variant

The Python package is named `rscopulas` and is built with `maturin` on top of
the Rust core. The Python layer is intentionally thin: it keeps the same model
families and fitting behavior, but exposes them in a NumPy-first shape with
Pythonic result objects.

### Install for local development

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,viz]"
maturin develop --manifest-path crates/rscopulas-python/Cargo.toml
pytest
```

The project metadata lives in the root `pyproject.toml`, and the extension
module is exported as `rscopulas._rscopulas`.

If you only need the numeric API, `python -m pip install -e ".[dev]"` is still
enough. The optional plotting helpers live in `rscopulas.plotting` and require
the `viz` extra.

### Example scripts (visuals)

After installing with the `viz` extra and running `maturin develop`, you can
generate figures from the repository root:

```bash
PYTHONPATH=python python python/examples/copula_visualisation.py
PYTHONPATH=python python python/examples/copula_gallery.py
```

- `python/examples/copula_visualisation.py` writes a single multi-panel overview
  (`python/examples/output/copula_visualisation.png`).
- `python/examples/copula_gallery.py` writes one PNG per model kind (Gaussian,
  Student t, Clayton, Frank, Gumbel, vine, nested Gumbel hierarchical Archimedean, and pair-copula kernel) under `python/examples/output/gallery_*.png`.

### Fit a Gaussian copula

```python
import numpy as np

from rscopulas import GaussianCopula

data = np.array(
    [
        [0.12, 0.18],
        [0.21, 0.25],
        [0.27, 0.22],
        [0.35, 0.42],
        [0.48, 0.51],
        [0.56, 0.49],
        [0.68, 0.73],
        [0.82, 0.79],
    ],
    dtype=np.float64,
)

fit = GaussianCopula.fit(data)
print("family:", fit.model.family)
print("dim:", fit.model.dim)
print("AIC:", fit.diagnostics.aic)
print("correlation:\n", fit.model.correlation)
print("sample:\n", fit.model.sample(4, seed=7))
```

### Build a Khoudraji pair kernel in Python

```python
import numpy as np

from rscopulas import PairCopula

model = PairCopula.from_khoudraji(
    "gaussian",
    "clayton",
    shape_1=0.35,
    shape_2=0.8,
    first_parameters=[0.45],
    second_parameters=[2.0],
)
u1 = np.array([0.17, 0.31, 0.62, 0.88], dtype=np.float64)
u2 = np.array([0.23, 0.54, 0.41, 0.79], dtype=np.float64)
print("family:", model.family)
print("spec:", model.spec)
print("log_pdf:", model.log_pdf(u1, u2))
```

### Quick visualization

The plotting helpers are intentionally lightweight and sit in
`rscopulas.plotting` so the base package stays NumPy-first.

```python
import matplotlib.pyplot as plt
import numpy as np

from rscopulas import GaussianCopula, VineCopula
from rscopulas.plotting import plot_density, plot_scatter, plot_vine_structure

data = np.array(
    [
        [0.12, 0.18, 0.21],
        [0.21, 0.25, 0.29],
        [0.27, 0.22, 0.31],
        [0.35, 0.42, 0.39],
        [0.48, 0.51, 0.46],
        [0.56, 0.49, 0.58],
        [0.68, 0.73, 0.69],
        [0.82, 0.79, 0.76],
    ],
    dtype=np.float64,
)

gaussian_fit = GaussianCopula.fit(data[:, :2])
plot_scatter(sample=data[:, :2])
plot_density(gaussian_fit.model)

vine_fit = VineCopula.fit_r(
    data,
    family_set=["independence", "gaussian", "clayton", "frank", "gumbel", "khoudraji"],
    truncation_level=1,
)
plot_vine_structure(vine_fit.model)

plt.show()
```

### Fit and inspect an R-vine

```python
import numpy as np

from rscopulas import VineCopula

data = np.array(
    [
        [0.12, 0.18, 0.21],
        [0.21, 0.25, 0.29],
        [0.27, 0.22, 0.31],
        [0.35, 0.42, 0.39],
        [0.48, 0.51, 0.46],
        [0.56, 0.49, 0.58],
        [0.68, 0.73, 0.69],
        [0.82, 0.79, 0.76],
    ],
    dtype=np.float64,
)

fit = VineCopula.fit_r(
    data,
    family_set=["independence", "gaussian", "clayton", "frank", "gumbel", "khoudraji"],
    include_rotations=True,
    criterion="aic",
    truncation_level=1,
)

print("kind:", fit.model.structure_kind)
print("order:", fit.model.order)
print("pair parameters:", fit.model.pair_parameters)
print("structure info matrix:\n", fit.model.structure_info.matrix)
print("tree count:", len(fit.model.trees))
```

### Build models from known parameters

```python
import numpy as np

from rscopulas import GaussianCopula, StudentTCopula, VineCopula

gaussian = GaussianCopula.from_params(
    np.array([[1.0, 0.6], [0.6, 1.0]], dtype=np.float64)
)

student_t = StudentTCopula.from_params(
    np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64),
    degrees_of_freedom=6.0,
)

vine = VineCopula.gaussian_c_vine(
    [0, 1, 2],
    np.array(
        [
            [1.0, 0.60, 0.35],
            [0.60, 1.0, 0.25],
            [0.35, 0.25, 1.0],
        ],
        dtype=np.float64,
    ),
)

print(gaussian.correlation)
print(student_t.degrees_of_freedom)
print(vine.order)
```

### Python API shape

- `GaussianCopula`, `StudentTCopula`, `ClaytonCopula`, `FrankCopula`,
  `GumbelCopula`, `HierarchicalArchimedeanCopula`, and `VineCopula` are the
  main model classes. `PairCopula` exposes low-level bivariate kernels.
- `fit(...)` returns a `FitResult` with `model` and `diagnostics`.
- `diagnostics` exposes `loglik`, `aic`, `bic`, `converged`, and `n_iter`.
- `log_pdf(...)` accepts array-like input and returns a NumPy array.
- `sample(...)` returns a NumPy array and accepts an optional integer seed.
- Vine models additionally expose `structure_kind`, `truncation_level`, `order`,
  `pair_parameters`, `structure_info`, and `trees`.
- Optional plotting helpers are available from `rscopulas.plotting`.

### Python caveats

- Inputs still need to be pseudo-observations in the open interval `(0, 1)`.
- The Python layer uses the Rust core with `ExecPolicy::Auto`; backend selection
  is not exposed as a Python API yet.
- The current workflow is local-development-first rather than published-wheel
  distribution.

## Current status

### Implemented now

- `PseudoObs` validation for `(0, 1)^d` inputs.
- Single-family copulas with `new`, `fit`, `log_pdf`, and `sample`.
- Pair-copula kernels for independence, Gaussian, Student t, Clayton, Frank,
  and Gumbel, including rotations where supported.
- `fit_pair_copula(...)` with AIC/BIC-driven family selection.
- `VineCopula::gaussian_c_vine(...)` and `VineCopula::gaussian_d_vine(...)`.
- `VineCopula::fit_c_vine(...)`, `fit_d_vine(...)`, and `fit_r_vine(...)`.
- `VineCopula::from_trees(...)` for explicit vine construction.
- R-based reference fixtures for single-family, pair-copula, and vine coverage.

### Deliberately not polished yet

- There is no top-level one-call facade that fits an unknown family and returns
  a single ergonomic model wrapper.
- The exported `Copula` enum exists, but the crate does not yet center its UX
  around it.
- The Python package is local-development-first and is not set up for published
  wheel distribution.

## Reference fixtures

Fixture generation scripts live under `scripts/reference/` and regenerate JSON
fixtures in place for:

- `fixtures/reference/r-copula/v1_1_3/` from R package `copula` 1.1-3,
- `fixtures/reference/vinecopula/v2/` from R package `VineCopula` 2.6.1.

This fixture suite is what keeps the numerical claims grounded rather than
hand-wavy.

## Benchmarks

Cross-language speed comparisons between R and `rscopulas` live under
`benchmarks/`.

The benchmark harness reuses the same JSON fixtures as the reference tests and
covers:

- single-family `log_pdf`, `fit`, and `sample`
- pair-copula density plus h/hinv kernels
- mixed R-vine `log_pdf`, `sample`, and `fit`

Run the orchestrator from the workspace root:

```bash
python benchmarks/run.py
```

Outputs land in `benchmarks/output/` as both JSON and Markdown summaries.

See `benchmarks/README.md` for prerequisites, case filters, and iteration
scaling.

## Development

Run the main quality gates from the workspace root:

```bash
cargo fmt --check
cargo test
cargo bench --no-run
cargo clippy --all-targets --all-features -- -D warnings
```
