# rscopulas

`rscopulas` is a Rust workspace for fitting, evaluating, and sampling copulas
from pseudo-observations.

## Workspace

- `crates/rscopulas-core`: core data validation, copula models, pair-copula
  primitives, vine structure / fitting / evaluation / sampling, math, stats,
  reference tests, and benchmarks
- `crates/rscopulas-accel`: CPU-parallel helpers plus the first narrow CUDA /
  Metal batch-kernel facade
- `crates/rscopulas-python`: PyO3 extension crate plus the local NumPy-first
  Python package surface

## Current Status

### Working now

- pseudo-observation validation through `PseudoObs`
- single-family copulas (`new`, `fit`, `log_pdf`, `sample`):
  - `GaussianCopula` (Kendall tau inversion for the correlation matrix)
  - `StudentTCopula` (tau inversion plus a degrees-of-freedom grid search)
  - `ClaytonCopula`, `FrankCopula`, `GumbelHougaardCopula` (mean Kendall tau
    inversion, Gaussian-style AIC/BIC diagnostics)
- pair-copula primitives in `rscopulas_core::paircopula` exposing
  `log_pdf`, `cond_first_given_second`, `cond_second_given_first`,
  `inv_first_given_second`, and `inv_second_given_first` for:
  - Independence, Gaussian, Student t, Clayton, Frank, Gumbel-Hougaard
  - rotations `R0`, `R90`, `R180`, `R270` (Gaussian and Student t only use `R0`
    by construction; Clayton / Frank / Gumbel exercise all four rotations)
- `fit_pair_copula` selects a family + rotation by AIC or BIC, with an optional
  Kendall tau independence threshold
- vine copulas (`CopulaModel::log_pdf`, `CopulaModel::sample`) over C, D, and
  R structures:
  - Gaussian-parameterised constructors: `VineCopula::gaussian_c_vine`,
    `VineCopula::gaussian_d_vine`
  - mixed-family fitters: `VineCopula::fit_c_vine`, `VineCopula::fit_d_vine`,
    `VineCopula::fit_r_vine` with configurable `family_set`, rotations,
    AIC/BIC selection, optional truncation level, and optional independence
    threshold (`VineFitOptions`)
  - `VineCopula::from_trees` for building a vine from pre-specified trees
- reference tests against fixtures generated from R:
  - `copula` 1.1-3 at `d = 2` for Gaussian, Student t, Clayton, Frank, Gumbel
    (log pdf, fit, sample summary)
  - `VineCopula` 2.6.1 pair-copula fixtures (log pdf, both h-functions, both
    inverse h-functions) for Gaussian, Student t, Clayton, Frank, Gumbel, plus
    Clayton and Gumbel at `R90`, `R180`, `R270`
  - `VineCopula` 2.6.1 vine fixtures: Gaussian C-vine and D-vine at `d = 4`;
    a mixed-family R-vine at `d = 5` for both the full and truncation-level-2
    cases; and an R-vine fit non-triviality check against a Dissmann-style
    reference matrix
- Criterion benchmarks for Gaussian `fit`, `log_pdf`, and `sample`; Student t,
  Clayton, Frank, and Gumbel `log_pdf`; and Gaussian C-vine and D-vine
  `log_pdf`
- execution controls through `ExecPolicy` / `Device`
- quality gates enforced in CI:
  - `cargo fmt --check`
  - `cargo test`
  - `cargo bench --no-run`
  - `cargo clippy --all-targets --all-features -- -D warnings`

### Not working yet / scaffolding

- there is no top-level ergonomic facade yet for selecting a family and
  returning a boxed or enum-backed fitted model from one entry point; the
  `Copula` enum is exported but unused by the library
- reference coverage is strongest at `d = 2` for single families, `d = 4` for
  Gaussian vines, and `d = 5` for mixed R-vines; broader dimensional
  regression coverage still needs to be added
- Python packaging currently targets local development through `maturin develop`;
  wheel automation and PyPI publishing are not set up yet

### Known caveats

- `ExecPolicy::Auto` is intentionally conservative: it can pick CPU-parallel
  execution, but it does not automatically jump to CUDA or Metal yet
- forced GPU execution is narrow and explicit:
  - CUDA currently accelerates the Gaussian pair batch kernel used by
    pair-fit scoring and Gaussian vine `log_pdf`
  - Metal currently accelerates the same Gaussian pair batch surface through an
    `f32` mixed-precision kernel; broader `f64`-sensitive work remains on CPU
    or unsupported, depending on the operation
- single-family copula `log_pdf`, Kendall tau, and sampling remain CPU paths;
  forcing CUDA/Metal for those top-level operations still surfaces a backend
  error
- mixed-family pair and vine work can still contain deliberate CPU fallback
  within a forced GPU request, because only the Gaussian pair kernel is
  accelerated today

## Acceleration Contract

The backend story is intentionally staged and honest:

- CPU: the reference implementation is complete, and `Auto` can choose Rayon
  parallelism for batch-heavy paths that already support it
- CUDA: first true GPU backend for this library's `f64`-heavy numerics; the
  current kernel set is the Gaussian pair batch surface reused by Gaussian vine
  evaluation
- Metal: bounded mixed-precision backend; it shares the same Gaussian pair
  batch contract as CUDA, but with parity checked against tolerances instead of
  exact equality

If you need predictable device behavior today, prefer
`Force(Device::Cpu)`, `Force(Device::Cuda(_))`, or `Force(Device::Metal)`
instead of relying on `Auto`.

## Reference Fixtures

Fixture generation scripts live in `scripts/reference/` and write JSON
fixtures under:

- `fixtures/reference/r-copula/v1_1_3/` — single-family fixtures from R
  package `copula` 1.1-3
- `fixtures/reference/vinecopula/v2/` — pair-copula and vine fixtures from R
  package `VineCopula` 2.6.1

The R scripts (`generate_{gaussian,student_t,clayton,frank,gumbel}_fixtures.R`,
`generate_paircopula_fixtures.R`, `generate_vine_fixtures.R`) are runnable from
the workspace root with `Rscript` and regenerate the JSON fixtures in-place.

## Development

Run the main quality gates from the workspace root:

```bash
cargo fmt --check
cargo test
cargo bench --no-run
cargo clippy --all-targets --all-features -- -D warnings
```

## Python Bindings

The Python package is intentionally scoped for local development first. It wraps
the stable `rscopulas-core` surface with a NumPy-first API and avoids exposing
backend-selection or persistence features that are still evolving.

Current Python surface:

- `GaussianCopula`, `StudentTCopula`, `ClaytonCopula`, `FrankCopula`,
  `GumbelCopula`, and `VineCopula`
- `fit(...)`, `from_params(...)`, `log_pdf(...)`, and `sample(...)`
- fit diagnostics and vine structure inspection helpers

Local install workflow:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip maturin ".[dev]"
maturin develop --manifest-path crates/rscopulas-python/Cargo.toml
pytest python/tests
```

Minimal example:

```python
import numpy as np

from rscopulas import GaussianCopula, VineCopula

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
print(gaussian_fit.diagnostics.aic)
print(gaussian_fit.model.sample(4, seed=7))

vine_fit = VineCopula.fit_r(
    data,
    family_set=["independence", "gaussian", "clayton", "frank", "gumbel"],
    truncation_level=1,
)
print(vine_fit.model.structure_kind)
print(vine_fit.model.order)
```

Deliberately deferred in Python for now:

- model save/load helpers
- backend selection (`ExecPolicy` / `Device`)
- packaging for published wheels and PyPI
