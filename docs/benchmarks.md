# Benchmarks

There are **two** complementary benchmarking surfaces. They share the same **fixture JSON** as tests where noted, but **methodology differs** — do not compare numbers across them naively.

## 1. Cross-language harness (`benchmarks/`)

**What it measures:** Wall-clock time for end-to-end workloads in **Rust**, **Python**, and **R** on the same inputs.

**Driver:** `python benchmarks/run.py` (see [benchmarks/README.md](../benchmarks/README.md)).

**Manifest:** [benchmarks/cases.json](../benchmarks/cases.json) lists cases, fixture paths, iteration counts, and which implementations run.

**Coverage (default manifest) includes:**

- Single-family: `log_pdf`, `fit`, `sample`
- Pair copula: density and **h / hinv** kernels (including **`pair_khoudraji_kernels`** → `fixtures/reference/r-copula/v1_1_3/khoudraji_pair_case01.json`)
- Mixed R-vine: `log_pdf`, `sample`, `fit`

**Outputs:** Written under `benchmarks/output/` (e.g. `latest.json`, `latest.md`). That directory is gitignored except `.gitignore`; results are **local**.

**Rust binary:** Built as `rscopulas-core` example `benchmark_runner` (invoked by the orchestrator).

## 2. Criterion benches (`crates/rscopulas-core/benches/`)

**What it measures:** Microbenchmark statistics (typically per-iteration) for Rust-only code paths.

**Notable targets:**

- `cross_language.rs` — reads `benchmarks/cases.json` and benches Rust operations for cases that request `"rust"` (uses `ExecPolicy::Force(Device::Cpu)` for stable CPU timing).
- `gaussian.rs` — additional Gaussian-focused scenarios.

**Output:** Printed to the terminal / Criterion reports, not the harness `latest.json`.

## Reproducing results

**Harness (all implementations):**

```bash
python benchmarks/run.py
```

Filter:

```bash
python benchmarks/run.py --implementation rust --case vine
```

**Criterion (Rust only):**

```bash
cargo bench -p rscopulas-core
```

## What not to infer

- Harness times include interpreter / FFI / JSON load overhead for Python and R; Criterion benches do not.
- Relative speed vs R in `latest.md` is **machine-dependent**; use for regression tracking on a fixed runner, not as absolute marketing numbers without context.
