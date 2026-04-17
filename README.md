# rscopulas

`rscopulas` is a Rust workspace for fitting, evaluating, and sampling copulas
from pseudo-observations.

## Workspace

- `crates/rscopulas-core`: core data validation, copula models, fitting, math,
  tests, and benchmarks
- `crates/rscopulas-accel`: backend capability detection scaffolding for future
  CPU/GPU dispatch
- `crates/rscopulas-python`: placeholder crate for future Python bindings

## Current Status

### Working now

- pseudo-observation validation through `PseudoObs`
- Gaussian copula:
  - `new`
  - `fit` via Kendall's tau inversion
  - `log_pdf`
  - `sample`
- Student t copula:
  - `new`
  - `fit` via Kendall's tau inversion plus a degrees-of-freedom grid search
  - `log_pdf`
  - `sample`
- Archimedean copulas:
  - Clayton `new`, `fit`, `log_pdf`, `sample`
  - Frank `new`, `fit`, `log_pdf`, `sample`
  - Gumbel-Hougaard `new`, `fit`, `log_pdf`, `sample`
- Gaussian vine copulas:
  - `gaussian_c_vine`
  - `gaussian_d_vine`
  - `fit_c_vine`
  - `fit_d_vine`
  - `fit_r_vine`
  - `log_pdf`
  - `sample`
- reference tests against fixtures generated from R package `copula` `1.1-3`
  for:
  - Gaussian
  - Student t
  - Clayton
  - Frank
  - Gumbel
- reference tests against R package `VineCopula` `2.6.1` for Gaussian C-vine
  and D-vine fixtures
- Criterion benchmarks for Gaussian fit/log-density/sample and the additional
  family log-density paths, including Gaussian vine log-density
- `cargo test`
- `cargo bench --no-run`
- `cargo clippy --all-targets --all-features -- -D warnings`

### Not working yet

- full mixed-family pair-copula vines are not implemented yet:
  - the current `VineCopula` path is a Gaussian vine layer backed by the
    equivalent Gaussian copula
  - there are no non-Gaussian pair-copula h-functions or inverse h-functions
    in the core crate yet
- `rscopulas-accel` only reports detected capabilities; it does not execute any
  accelerated kernels yet
- `rscopulas-python` does not expose a Python API yet
- there is no top-level ergonomic facade yet for selecting a family and
  returning a boxed or enum-backed fitted model from one entry point
- coverage is currently strongest for the 2D reference fixtures checked against
  R; broader dimensional regression coverage still needs to be added

## Reference Fixtures

Fixture generation scripts live in `scripts/reference/` and write JSON fixtures
under `fixtures/reference/r-copula/v1_1_3/`.

## Development

Run the main quality gates from the workspace root:

```bash
cargo test
cargo bench --no-run
cargo clippy --all-targets --all-features -- -D warnings
```
