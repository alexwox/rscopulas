# Examples index

## Python (`python/examples/`)

| Script | Purpose |
|--------|---------|
| [quickstart.py](../python/examples/quickstart.py) | Gaussian + R-vine fit demo (CLI) |
| [copula_visualisation.py](../python/examples/copula_visualisation.py) | Multi-panel figure → `python/examples/output/copula_visualisation.png` |
| [copula_gallery.py](../python/examples/copula_gallery.py) | One PNG per model kind → `python/examples/output/gallery_*.png` |
| [portfolio_tail_risk.py](../python/examples/portfolio_tail_risk.py) | Worked risk example: simulated heavy-tailed returns, R-vine vs Gaussian-copula VaR/ES, conditional stress scenario via `sample_conditional` → `python/examples/output/portfolio_tail_risk.png` |

Run from repo root after installing `rscopulas` (for example `uv add "rscopulas[viz]"` in a uv project, or `uv sync --all-extras` in this repo) and `uv run maturin develop` when working on the Rust extension:

```bash
uv run python python/examples/quickstart.py
uv run python python/examples/copula_visualisation.py
uv run python python/examples/copula_gallery.py
uv run python python/examples/portfolio_tail_risk.py
```

**Gallery outputs:** PNGs under `python/examples/output/` are **checked in** as documentation assets. Regenerate when plotting or model behavior changes.

### Portfolio tail risk (`portfolio_tail_risk.py`)

Runs offline in well under a minute and needs only NumPy (the figure is written
only when matplotlib is importable). The script:

1. simulates 5 asset return series from a known mixed D-vine (Clayton, survival
   Gumbel, Student-t, Frank and rotated edges) built with `VineCopula.from_trees`,
   with Student-t(4) marginals whose quantile has a closed form in NumPy;
2. rank-transforms the returns to pseudo-observations and fits an R-vine with
   the default family set plus a Gaussian copula;
3. estimates 1-day 99 % portfolio VaR and ES by simulation from each fitted
   copula (empirical marginals), next to the true copula;
4. pins one asset at its 1st percentile with `sample_conditional` and compares
   the conditional loss distribution of the vine against the Gaussian copula's
   analytic conditional and the true model;
5. prints a small table and saves `python/examples/output/portfolio_tail_risk.png`.

`sample_conditional` requires the pinned column to be the fitted vine's
Rosenblatt anchor (`variable_order[0]`); the script stresses that asset and
explains how to pin a chosen asset instead with `fit_c(order=[..., asset])`.

## Rust (`rscopulas`, examples under `crates/rscopulas-core/examples/`)

| Example | Command |
|---------|---------|
| `quickstart_gaussian` | `cargo run -p rscopulas --example quickstart_gaussian` |
| `vine_r_vine_fit` | `cargo run -p rscopulas --example vine_r_vine_fit` |
| `pair_copula_clayton` | `cargo run -p rscopulas --example pair_copula_clayton` |
| `khoudraji_pair` | `cargo run -p rscopulas --example khoudraji_pair` |
| `benchmark_runner` | Built for the cross-language harness (see [benchmarks.md](benchmarks.md)) |

## Benchmarks

See [benchmarks.md](benchmarks.md) and [benchmarks/README.md](../benchmarks/README.md).
