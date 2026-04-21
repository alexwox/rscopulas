<div align="center">
  <img alt="rscopulas" src="https://pub-ca706ca75c8a4972b721945607f0ff01.r2.dev/rscopulas-logo.png" width="64" height="64" />
</div>

# rscopulas

**Copula modeling for Python and Rust** on the same core: fit, score, and sample dependence structures from **pseudo-observations** in the open unit hypercube
`(0, 1)^d`, with explicit validation, diagnostics, and reference-backed tests.

**Up to ~100× faster** than R’s [`copula`](https://cran.r-project.org/package=copula) / [`VineCopula`](https://cran.r-project.org/package=VineCopula) stack on the mixed R-vine `log_pdf` workload in our cross-language harness (see [Performance](#performance-and-benchmarking)).

| Surface                     | Role                                                                                                                             |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Python** (`rscopulas`)    | NumPy-first API built with PyO3/maturin; optional plotting via `rscopulas.plotting` (`viz` extra).                               |
| **Rust** (`rscopulas`) | Primary library crate: traits, `PseudoObs`, execution policy (`ExecPolicy` / `Device`), and accelerated paths where implemented. |


**Documentation:** start with [docs/README.md](docs/README.md) (guides, specs, benchmarks). This README stays the install-and-quickstart landing page.

## Why rscopulas

- **Validated inputs** — Only finite values strictly inside `(0, 1)`; no silent coercion at the boundary.
- **Single-family, pair, and vine copulas** — Gaussian, Student t, Archimedean families, **Khoudraji** asymmetric pair copulas, hierarchical Archimedean (HAC), and C-/D-/R-vines with mixed pair families.
- **Fast Rust core for Python and Rust users** — Heavy vine work runs in `rscopulas`, so Python users keep a NumPy-first API without falling back to pure-Python orchestration on the hot path.
- **Large speedups vs R on the benchmarked vine workloads** — On the shared mixed R-vine harness (`python benchmarks/run.py`), mean iteration time vs `VineCopula`/R is about **3–4× faster** for `fit` and `sample`, and about **100× faster** for `log_pdf` (Rust and Python track each other; see Performance below).
- **Diagnostics** — Log-likelihood, AIC, BIC, convergence, iteration count on fitted models.
- **Reference fixtures** — JSON fixtures checked against R packages `copula` and `VineCopula`; see [docs/specs.md](docs/specs.md).
- **Benchmarks** — Cross-language harness (Rust, Python, R) plus Criterion benches; see [docs/benchmarks.md](docs/benchmarks.md).

## Pseudo-observations (read this once)

- You supply **pseudo-observations**; the library does **not** fit marginals for you.
- Each entry must satisfy `0 < u < 1` (strict).
- Workflow: transform raw data to uniforms → build `PseudoObs` / NumPy `float64` arrays → `fit` → `log_pdf` / `sample`.

## Quick start — Python

Install with [uv](https://docs.astral.sh/uv/) using the [project workflow](https://docs.astral.sh/uv/guides/projects/) (lockfile + `uv sync` / `uv run`):

- **In a uv project** (recommended): `uv add rscopulas` (adds a dependency and updates `uv.lock`).
- **Without a project** (pip-style): `uv pip install rscopulas`.

```bash
uv add rscopulas
```

Minimal fit:

```python
import numpy as np
from rscopulas import GaussianCopula

data = np.array([[0.12, 0.18], [0.21, 0.25], [0.82, 0.79]], dtype=np.float64)
fit = GaussianCopula.fit(data)
print("AIC:", fit.diagnostics.aic)
print("sample:\n", fit.model.sample(4, seed=7))
```

For local development from this repository, use [uv sync](https://docs.astral.sh/uv/concepts/projects/sync/) so the lockfile and `.venv` stay aligned (see `uv.lock` in the repo root):

```bash
uv sync --all-extras
uv run maturin develop --manifest-path crates/rscopulas-python/Cargo.toml
uv run pytest
```

`uv sync` installs this package in editable mode and pulls the **dev** dependency group (pytest, maturin); `--all-extras` adds the optional **`viz`** extra (matplotlib) for plotting examples.

**Example scripts** (after `uv sync --all-extras` and `uv run maturin develop`; or install `rscopulas[viz]` from PyPI when you only need plotting):

```bash
uv run python python/examples/quickstart.py
uv run python python/examples/copula_visualisation.py
uv run python python/examples/copula_gallery.py
```

Gallery figures are written under `python/examples/output/`; see [docs/examples.md](docs/examples.md).

## Quick start — Rust

Add the crate (path dependency while developing in this workspace):

```toml
[dependencies]
ndarray = "0.17"
rand = "0.9"
rscopulas = { path = "crates/rscopulas-core" }
```

```rust
use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};
use rscopulas::{CopulaModel, FitOptions, GaussianCopula, PseudoObs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = PseudoObs::new(array![
        [0.12_f64, 0.18],
        [0.21, 0.25],
        [0.82, 0.79],
    ])?;
    let fit = GaussianCopula::fit(&data, &FitOptions::default())?;
    println!("AIC: {}", fit.diagnostics.aic);
    let mut rng = StdRng::seed_from_u64(7);
    println!("sample:\n{:?}", fit.model.sample(4, &mut rng, &Default::default())?);
    Ok(())
}
```

Runnable examples live under `crates/rscopulas-core/examples/` (e.g. `cargo run -p rscopulas --example quickstart_gaussian` from the workspace root). See [docs/rust.md](docs/rust.md).

## Performance and benchmarking

- **Cross-language wall times** (Rust vs Python vs R on shared fixtures): `python benchmarks/run.py` — details in [benchmarks/README.md](benchmarks/README.md) and [docs/benchmarks.md](docs/benchmarks.md).
- **Criterion microbenchmarks** (Rust, manifest-driven where applicable): `cargo bench -p rscopulas` (see crate benches).

Representative mixed R-vine harness results (same fixture and iteration counts for each language; **speedup** = R mean time ÷ ours, so higher is better):

| Workload | Rust vs R | Python vs R | Mean times (Rust / Python / R) |
| --- | ---: | ---: | --- |
| **Fit** | **3.1×** | **3.4×** | 7.1 ms / 6.5 ms / 22.1 ms |
| **Sample** | **4.0×** | **4.0×** | 28.4 ms / 27.9 ms / 112.9 ms |
| **`log_pdf`** | **114×** | **111×** | 22.9 µs / 23.5 µs / 2.60 ms |

Those multiples are from one development machine run (2026-04-18). Reproduce with `python benchmarks/run.py` and read `benchmarks/output/latest.md` locally—numbers depend on CPU and R package versions. They show you can keep the **Python API** and still sit on the same fast Rust core instead of trading ergonomics for performance.

Do not compare harness wall times to Criterion reports directly; methodology differs. [docs/benchmarks.md](docs/benchmarks.md) explains both.

## Feature overview

- Single-family: Gaussian, Student t, Clayton, Frank, Gumbel–Hougaard.
- Pair copulas: density, h-functions, inverses; rotations where supported; **Khoudraji** composition of two base pair kernels ([docs/khoudraji.md](docs/khoudraji.md)).
- Vines: C-vine, D-vine, R-vine — construct, fit, `log_pdf`, sample, and **exact conditional sampling** via the Rosenblatt transform ([docs/vines.md](docs/vines.md)).
- HAC: hierarchical Archimedean construction and fitting — behavior and sampling caveats: [docs/hac.md](docs/hac.md) (Python/Rust API also in [docs/python.md](docs/python.md)).

## Workspace layout (for contributors)

- `crates/rscopulas-core` — Rust library package `rscopulas`, including internal CPU/GPU acceleration support.
- `crates/rscopulas-python` — PyO3 bindings.
- `python/` — Python package source.
- `fixtures/reference/` — R-generated JSON for tests and benchmarks.
- `benchmarks/` — cross-language benchmark orchestration.
- `scripts/reference/` — R scripts to regenerate fixtures.

## Longer examples in this README

The sections below mirror the quick starts with fuller snippets (R-vine, pair copulas, Khoudraji, plotting). For narrative guides, prefer [docs/getting-started.md](docs/getting-started.md).

### Rust — Fit a mixed-family R-vine

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

### Rust — Gaussian C-vine from correlation

```rust
use ndarray::array;
use rscopulas_core::VineCopula;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let correlation = array![
        [1.0_f64, 0.60, 0.35],
        [0.60, 1.0, 0.25],
        [0.35, 0.25, 1.0],
    ];
    let model = VineCopula::gaussian_c_vine(vec![0, 1, 2], correlation)?;
    println!("order = {:?}", model.order());
    Ok(())
}
```

### Rust — Pair copula and Khoudraji

```rust
use rscopulas_core::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spec = PairCopulaSpec {
        family: PairCopulaFamily::Clayton,
        rotation: Rotation::R90,
        params: PairCopulaParams::One(1.4),
    };
    println!("log_pdf = {}", spec.log_pdf(0.32, 0.77, 1e-12)?);

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
    println!("Khoudraji log_pdf = {}", khoudraji.log_pdf(0.32, 0.77, 1e-12)?);
    Ok(())
}
```

### Rust API notes

- Import `CopulaModel` for `log_pdf` / `sample` on concrete copulas.
- `VineFitOptions` controls families (including Khoudraji), rotations, criterion, truncation, independence thresholds. Khoudraji fitting uses a bounded internal search over supported base families (CPU).
- `ExecPolicy::Auto` is conservative; use `ExecPolicy::Force(Device::Cpu)` (or CUDA/Metal where implemented) for deterministic backend choice. See [docs/rust.md](docs/rust.md).

### Python — Visualization

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

### Python — Khoudraji `PairCopula`

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
print("log_pdf:", model.log_pdf(u1, u2))
```

### Python API summary

- Models: `GaussianCopula`, `StudentTCopula`, Archimedean families, `HierarchicalArchimedeanCopula`, `VineCopula`, `PairCopula`.
- `fit(...)` returns `FitResult` with `model` and `diagnostics` (`loglik`, `aic`, `bic`, `converged`, `n_iter`).
- Python uses `ExecPolicy::Auto` internally; GPU/device selection is not exposed yet.

## Current status

**Implemented:** pseudo-obs validation; single-family fit/score/sample; pair kernels and `fit_pair_copula`; vines (Gaussian construction + mixed-family fitting); HAC; Khoudraji in pair and vine fitting; reference tests and benchmarks.

**Still evolving:** no single “auto-discover family” facade; `Copula` enum not the main UX; Rust crate publishing and multi-platform Python wheel automation are still separate release tasks.

## Reference fixtures

Regeneration scripts live in `scripts/reference/` for:

- `fixtures/reference/r-copula/v1_1_3/` — R package `copula` 1.1-3 (includes Khoudraji fixtures).
- `fixtures/reference/vinecopula/v2/` — R package `VineCopula` 2.6.1.

## Development

```bash
cargo fmt --check
cargo test
cargo bench --no-run
cargo clippy --all-targets --all-features -- -D warnings
```

