# rscopulas

`rscopulas` is a Rust workspace for fitting, evaluating, and sampling copulas
from pseudo-observations.

## Workspace

- `crates/rscopulas-core`: core data validation, copula models, pair-copula
  primitives, vine structure / fitting / evaluation / sampling, math, stats,
  reference tests, and benchmarks
- `crates/rscopulas-accel`: CPU-parallel helpers plus the first narrow CUDA /
  Metal batch-kernel facade
- `crates/rscopulas-python`: placeholder crate for future Python bindings

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

- `rscopulas-python` exposes no Python API yet; the crate is a `cdylib`
  placeholder
- there is no top-level ergonomic facade yet for selecting a family and
  returning a boxed or enum-backed fitted model from one entry point; the
  `Copula` enum is exported but unused by the library
- reference coverage is strongest at `d = 2` for single families, `d = 4` for
  Gaussian vines, and `d = 5` for mixed R-vines; broader dimensional
  regression coverage still needs to be added

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
