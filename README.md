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
- pair copulas for vines:
  - Independence, Gaussian, Student t, Clayton, Frank, Gumbel
  - rotated Clayton and Gumbel through `R90`, `R180`, and `R270`
  - `log_pdf`
  - h-functions and inverse h-functions
- simplified vine copulas:
  - `gaussian_c_vine`
  - `gaussian_d_vine`
  - `fit_c_vine`
  - `fit_d_vine`
  - `fit_r_vine`
  - `from_trees`
  - `log_pdf`
  - `sample`
  - truncation via `VineFitOptions::truncation_level`
- reference tests against fixtures generated from R package `copula` `1.1-3`
  for:
  - Gaussian
  - Student t
  - Clayton
  - Frank
  - Gumbel
- reference tests against R package `VineCopula` `2.6.1` for:
  - pair-copula densities, h-functions, inverse h-functions, and rotations
  - Gaussian C-vine and D-vine fixtures
  - mixed-family R-vine fixtures
  - truncated R-vine fixtures
- Criterion benchmarks for Gaussian fit/log-density/sample and the additional
  family log-density paths, including Gaussian vine log-density
- `cargo test`
- `cargo bench --no-run`
- `cargo clippy --all-targets --all-features -- -D warnings`

### Not working yet

- non-simplified vines are not implemented
- there is no full external fit-parity suite against `VineCopula` for
  `fit_r_vine`; the current reference coverage is strongest for pair kernels,
  vine density evaluation, and sampling
- `rscopulas-accel` only reports detected capabilities; it does not execute any
  accelerated kernels yet
- `rscopulas-python` does not expose a Python API yet
- there is no top-level ergonomic facade yet for selecting a family and
  returning a boxed or enum-backed fitted model from one entry point
- broader dimensional regression coverage still needs to be added for larger
  vine models and more fit scenarios

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
