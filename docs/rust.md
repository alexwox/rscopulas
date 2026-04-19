# Rust API (`rscopulas`)

`rscopulas` is the main library crate. Import the prelude-style re-exports from the crate root.

## Key concepts

- **`PseudoObs`** — validated `ndarray` wrapper for `(0,1)^d` data.
- **`CopulaModel`** — trait for `log_pdf`, `sample`, etc. on fitted / constructed models.
- **`FitOptions` / `EvalOptions` / `SampleOptions`** — carry `ExecPolicy` and related knobs.

## Execution policy

- **`ExecPolicy::Auto`** — conservative; does not guarantee GPU on all heavy paths.
- **`ExecPolicy::Force(Device::Cpu)`** — deterministic CPU (also used in manifest-driven Criterion benches).
- **`ExecPolicy::Force(Device::Cuda(_))` / `Metal`** — only where implemented; see crate docs and accelerated paths in the codebase.

Python bindings currently fix policy to `Auto`; see [python.md](python.md).

## Vines

- Construct: `VineCopula::gaussian_c_vine`, `gaussian_d_vine`, `from_trees`, etc.
- Fit: `VineCopula::fit_c_vine`, `fit_d_vine`, `fit_r_vine` with `VineFitOptions` (family set, rotations, criterion, truncation, independence thresholds).

Khoudraji is available as `PairCopulaFamily::Khoudraji` in the family set; fitting uses a bounded internal search (CPU).

## Examples

Runnable examples ship in `crates/rscopulas-core/examples/`:

| Example | Command |
|---------|---------|
| Gaussian quickstart | `cargo run -p rscopulas --example quickstart_gaussian` |
| R-vine fit | `cargo run -p rscopulas --example vine_r_vine_fit` |
| Pair copula | `cargo run -p rscopulas --example pair_copula_clayton` |
| Khoudraji pair | `cargo run -p rscopulas --example khoudraji_pair` |

Benchmark runners: `benchmark_runner` (cross-language harness binary), Criterion targets in `benches/`.

## Workspace dependency

Inside this repo, depend on the path crate with `rscopulas = { path = "crates/rscopulas-core" }`. For a released crate later, use the crates.io version once published.
