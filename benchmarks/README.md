# Benchmarks

This folder contains the cross-language benchmark harness for comparing R and
`rscopulas` on equivalent copula workloads.

## What it covers

The default manifest in `cases.json` measures:

- single-family `log_pdf`, `fit`, and `sample`
- pair-copula density plus h/hinv kernels (including **Khoudraji** via `pair_khoudraji_kernels`)
- mixed R-vine `log_pdf`, `sample`, and `fit`

Each case runs against the same JSON fixtures already used elsewhere in the
repository for reference testing.

## Layout

- `cases.json`: benchmark manifest and iteration counts
- `common.py`: shared Python-side path, timing, and schema helpers
- `run.py`: top-level orchestrator that runs Rust, Python, and R
- `python_runner.py`: Python package benchmarks
- `r_runner.R`: R package benchmarks
- `output/`: generated JSON and Markdown summaries

The Rust runner lives at `crates/rscopulas-core/examples/benchmark_runner.rs`
so it can use the crate's native API directly without inventing a second Cargo
package under this directory.

## Prerequisites

Rust:

```sh
cargo build --release --manifest-path crates/rscopulas-core/Cargo.toml --example benchmark_runner
```

Python bindings (from the local checkout):

```sh
maturin develop --release
```

R packages:

```r
install.packages(c("copula", "VineCopula", "jsonlite"))
```

## Running the suite

Run every implementation:

```sh
python benchmarks/run.py
```

Run only one implementation or a subset of cases:

```sh
python benchmarks/run.py --implementation rust --case vine
python benchmarks/run.py --implementation r --implementation python --case pair
```

Reduce or increase runtime by scaling iterations:

```sh
python benchmarks/run.py --iteration-scale 0.25
python benchmarks/run.py --iteration-scale 2.0
```

## Outputs

The orchestrator writes:

- `benchmarks/output/latest.json`: machine-readable benchmark results
- `benchmarks/output/latest.md`: human-readable summary with relative speed vs R

Each result row includes the benchmark id, implementation, fixture path,
iteration count, total wall time, mean time per iteration, and basic
environment metadata.
