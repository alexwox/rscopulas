# Python API

The package name is `rscopulas`. The native extension is `rscopulas._rscopulas`; public types are re-exported from `rscopulas`.

## Core types

- **Models:** `GaussianCopula`, `StudentTCopula`, `ClaytonCopula`, `FrankCopula`, `GumbelCopula`, `VineCopula`, `HierarchicalArchimedeanCopula`, `PairCopula`
- **Fitting:** `Model.fit(data)` → `FitResult` with `.model` and `.diagnostics`
- **Diagnostics:** `loglik`, `aic`, `bic`, `converged`, `n_iter`

## Arrays

- Use `numpy.ndarray` with `dtype=float64` for numeric inputs.
- Data must lie strictly in `(0, 1)`.

## Vine fitting

`VineCopula.fit_c`, `fit_d`, and `fit_r` accept string lists for `family_set`, e.g.:

```python
family_set=["independence", "gaussian", "clayton", "frank", "gumbel", "khoudraji"]
```

Rotations, criterion (`"aic"` / `"bic"`), and `truncation_level` map to the Rust `VineFitOptions`.

## Hierarchical Archimedean copulas (HAC)

`HierarchicalArchimedeanCopula.from_tree(...)` builds a nested Archimedean model from a nested dict (see tests for shape). **Density** uses an exact exchangeable path when the tree is a single Archimedean fan; nested trees use a composite pair-copula evaluation.

**Sampling:** nested **same-family Gumbel** clusters and **fully exchangeable** Archimedean trees are the scenarios validated for Monte Carlo use. **Mixed-family** nesting (different Archimedean family on a child node than on its parent) uses a numerical frailty sampler that can **degenerate** (e.g. coordinates near 1); do not rely on `sample()` for those trees until the implementation is improved.

Full detail: [hac.md](hac.md).

## Plotting (optional)

Install with the `viz` extra (`uv add "rscopulas[viz]"` in a uv project, or `uv sync --extra viz` in this repo). For local repo development, run `uv sync` then `uv run maturin develop`. Import from `rscopulas.plotting`:

- `plot_density`, `plot_scatter`, `plot_vine_structure`

The base package does not depend on Matplotlib.

## Gallery figures

`python/examples/copula_gallery.py` generates one PNG per model kind under `python/examples/output/`. These files are **versioned as documentation assets** so the README and gallery stay in sync without running the script. Regenerate after behavior or plotting changes:

```bash
uv run python python/examples/copula_gallery.py
```

See [examples.md](examples.md).

## Rust-only behavior today

The Python bindings run evaluation with `ExecPolicy::Auto`. Explicit `Device` / CUDA / Metal selection is **not** exposed on the Python surface. For backend control, use Rust ([rust.md](rust.md)).

## Packaging note

`rscopulas` is published on PyPI for normal installation. Development in this repository still uses editable install + `maturin develop`.
