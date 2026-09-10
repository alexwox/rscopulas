# Python API

The package name is `rscopulas`. The native extension is `rscopulas._rscopulas`; public types are re-exported from `rscopulas`.

## Core types

- **Models:** `GaussianCopula`, `StudentTCopula`, `ClaytonCopula`, `FrankCopula`, `GumbelCopula`, `VineCopula`, `HierarchicalArchimedeanCopula`, `FactorCopula`, `PairCopula`
- **Fitting:** `Model.fit(data)` → `FitResult` with `.model` and `.diagnostics`
- **Diagnostics:** `loglik`, `aic`, `bic`, `converged`, `n_iter`, `likelihood_kind`
- **Helpers:** `to_pseudo_obs(x)` builds pseudo-observations from raw data; `rscopulas.__version__` reports the installed version

## Arrays

- Use `numpy.ndarray` with `dtype=float64` for numeric inputs.
- Data must lie strictly in `(0, 1)`.
- `rscopulas.to_pseudo_obs(x, ties="average", scaling="n+1")` turns a raw 1-D vector or `(n, d)` matrix into pseudo-observations by ranking each column. `ties` is one of `average`, `min`, `max`, `ordinal` (matching `scipy.stats.rankdata`); `scaling="n+1"` maps rank `r` to `r / (n + 1)` and `"n"` to `(r - 0.5) / n`. NaN input raises `InvalidInputError` naming the offending position; ±inf rank at the ends. NumPy only, no SciPy dependency.

```python
import numpy as np
from rscopulas import VineCopula, to_pseudo_obs

returns = np.loadtxt("returns.csv", delimiter=",")  # raw (n, d) observations
u = to_pseudo_obs(returns)
fit = VineCopula.fit_r(u)
```

## Vine fitting

`VineCopula.fit_c`, `fit_d`, and `fit_r` accept string lists for `family_set`, e.g.:

```python
family_set=["independence", "gaussian", "clayton", "frank", "gumbel", "khoudraji"]
```

Supported names are `independence`, `gaussian`, `student_t`, `clayton`, `frank`, `gumbel`, `joe`, `bb1`, `bb6`, `bb7`, `bb8`, `tawn1`, `tawn2`, `tll`, and `khoudraji`. `family_set=None` (the default) delegates to the core default set — `independence`, `gaussian`, `student_t`, `clayton`, `frank`, `gumbel`, `joe`, `bb1`, `bb7`; the Python layer keeps no list of its own. `khoudraji` and `tll` are opt-in because they dominate fit time. Unknown names raise `InvalidInputError`.

Rotations, criterion (`"aic"` / `"bic"` / `"mbicv"`), `truncation_level`, `independence_threshold`, and `independence_test_level` (a significance level in `(0, 1)` for the per-edge Kendall-τ independence test; `None` disables it) map to the Rust `VineFitOptions`.

`fit_c` and `fit_d` additionally accept `order=[...]` (integer column indices) to pin the canonical variable order explicitly. This is how you set up exact conditional sampling — see [vines.md](vines.md#conditional-sampling).

## Conditional sampling from a vine

Fitted `VineCopula` models expose the Rosenblatt transform and a
`sample_conditional(known, n, seed=None)` convenience:

```python
vine = VineCopula.fit_c(u, order=[*others, US10Y_IDX]).model
assert vine.variable_order[0] == US10Y_IDX
scenarios = vine.sample_conditional({US10Y_IDX: yield_uniforms}, n=10_000, seed=0)
```

Inputs outside a diagonal prefix of `variable_order` raise
`rscopulas.NonPrefixConditioningError`. The free columns are drawn from the
seeded Rust generator that `sample` uses, so `seed` alone determines the
result; NumPy's global random state is not involved. Full semantics, the
`variable_order == order[-1]` convention, and the forward/inverse Rosenblatt
primitives are documented in [vines.md](vines.md#conditional-sampling).

## Hierarchical Archimedean copulas (HAC)

`HierarchicalArchimedeanCopula.from_tree(...)` builds a nested Archimedean model from a nested dict (see tests for shape). **Density** uses an exact exchangeable path when the tree is a single Archimedean fan; nested trees require explicit `composite_log_pdf(data)` scoring. Their `log_pdf(data)` raises an unsupported-density error.

**Sampling:** nested **same-family Gumbel** clusters and **fully exchangeable** Archimedean trees are the scenarios validated for Monte Carlo use. **Mixed-family** nesting (different Archimedean family on a child node than on its parent) uses a numerical frailty sampler that can **degenerate** (e.g. coordinates near 1); do not rely on `sample()` for those trees until the implementation is improved.

Full detail: [hac.md](hac.md).

## Serialization, pickling, and copying

Every model class (`GaussianCopula`, `StudentTCopula`, `ClaytonCopula`, `FrankCopula`, `GumbelCopula`, `VineCopula`, `HierarchicalArchimedeanCopula`, `FactorCopula`, `PairCopula`) supports:

- `model.to_json()` / `Model.from_json(payload)` — JSON mirroring the Rust `serde` representation. Payloads are validated on load (`InvalidInputError` for malformed or out-of-range state); vine payloads carry a `format_version` and unversioned payloads are rejected.
- `pickle.dumps` / `pickle.loads` — round trips preserve `log_pdf` and seeded `sample` output exactly. Fitted TLL pair-copula grids are re-normalized on load and match to floating-point rounding.
- `copy.copy` / `copy.deepcopy` — independent copies.
- `==` — structural equality on the serialized state (models are deliberately unhashable). `repr` shows the class, family, dimension, and key parameters, truncated for large models.

```python
import pickle

restored = pickle.loads(pickle.dumps(fit.model))
assert restored == fit.model
payload = fit.model.to_json()          # store alongside your data
same = type(fit.model).from_json(payload)
```

## Exceptions

```
RscopulasError (Exception)
├── InvalidInputError (also a ValueError)
│   └── NonPrefixConditioningError
├── ModelFitError
├── NumericalError
├── BackendError
└── InternalError
```

- `InvalidInputError` covers everything wrong with what you passed in: values outside `(0, 1)`, wrong array rank, evaluation-time dimension mismatches (`log_pdf`, `rosenblatt`, ...), unknown family/rotation/criterion/method strings, invalid seeds and sample counts, and malformed JSON. Because it is also a `ValueError`, existing `except ValueError` handlers keep working.
- `ModelFitError` is reserved for estimation failures and unsupported fitting configurations reported by the core; `NumericalError` for numerical breakdowns; `BackendError` for execution-backend problems.
- `InternalError` wraps a Rust panic caught at the binding boundary (every binding is wrapped, so `pyo3_runtime.PanicException` never escapes); please report it.

## Threads and the GIL

All compute-bound bindings — every fitter, `log_pdf`/`composite_log_pdf`, `sample`, the Rosenblatt transforms, pair-copula batch kernels, and TLL fits — copy their inputs and release the GIL while Rust works. Other Python threads keep running during a long fit, and model objects can be shared across threads: they are immutable, and concurrent `log_pdf`/`sample` calls return the same results as serial ones. A `KeyboardInterrupt` is still delivered only when the call returns.

## Seeds and sample counts

- `seed` must be `None` or an integer in `[0, 2**64)` (`int` or NumPy integer). Negative or larger values raise `InvalidInputError`; non-integers raise `TypeError`.
- `n` must be a positive `int`: `n=5.0` raises `TypeError`, `n=0` raises `InvalidInputError`, checked up front in every sampling method.
- `VineCopula.sample_conditional` draws its free columns from the seeded Rust generator, so a seed reproduces the draw regardless of NumPy's global random state.

## Typing and version

The package ships `py.typed`, a hand-written stub for the compiled extension (`rscopulas/_rscopulas.pyi`), and full annotations in the wrapper layer, so `mypy`/`pyright` see precise types. `rscopulas.__version__` reports the installed distribution version.

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
