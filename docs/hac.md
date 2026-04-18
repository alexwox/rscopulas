# Hierarchical Archimedean copulas (HAC)

This note describes how **hierarchical Archimedean copulas** are implemented in `rscopulas`, what is **well tested**, and where **sampling** can be unreliable so you can choose trees and APIs deliberately.

## Model shape

- A **tree** whose internal nodes carry an Archimedean **family** (`clayton`, `frank`, or `gumbel`) and a **parameter** `theta`, and whose **leaves** are column indices `0 … d-1` (pseudo-observation dimensions).
- Rust: `HierarchicalArchimedeanCopula` and `HacTree` / `HacNode` in `rscopulas-core`.
- Python: `HierarchicalArchimedeanCopula.from_tree({...})` with the same JSON-like nesting as in tests (`family`, `theta`, `children`).

Input data are always **pseudo-observations** in `(0, 1)^d`, consistent with the rest of the library.

## Density evaluation (`log_pdf`)

Two paths exist:

1. **Exchangeable Archimedean subtree (exact special case)**  
   If the entire model is a **single** Archimedean node whose children are **only leaves** (a classical exchangeable Archimedean copula in dimension `d`), density evaluation uses the **same** implementation as the corresponding `ClaytonCopula` / `FrankCopula` / `GumbelHougaardCopula`. This is checked in tests (e.g. agreement with a direct Clayton copula on the same data).

2. **General nested tree (composite evaluation)**  
   For deeper nesting, the implementation evaluates a **composite** expression built from **pair-copula** terms associated with the tree (see `composite_log_pdf_rows` in `crates/rscopulas-core/src/hac/mod.rs`). This matches the intended construction for many HAC specifications used in estimation workflows; it is **not** the same code path as the exchangeable closed form.

Use `exact_loglik` / `is_exact` on the Rust model (and the corresponding Python properties) to see whether the **exchangeable** fast path applies to your fitted or constructed model.

## Sampling (`sample`)

Sampling walks the tree, draws **frailties** at each node, and fills leaf uniforms. The difficulty is **nested nodes whose Archimedean family differs from the parent** (e.g. Gumbel above Clayton, or Clayton above Gumbel). For those edges the implementation uses **numerical Laplace inversion** (Stehfest weights + bisection) to sample conditional child frailties.

### Validated / reliable scenarios

These are the cases you can treat as **supported for Monte Carlo use** today:

- **Exchangeable:** one Archimedean node over all leaves (same as a standard multivariate Archimedean copula).
- **Nested Archimedean with compatible structure:** in particular **nested Gumbel** with an inner Gumbel cluster uses a **closed-form** frailty update for the inner group. Core tests check that simulated data stay in `(0,1)` and that **Kendall’s tau** blocks look consistent with the tree (see `nested_gumbel_sampling_recovers_hierarchical_tau_blocks` in `crates/rscopulas-core/tests/reference_hac.rs`).

The Python gallery figure for HAC (`python/examples/copula_gallery.py`) uses **nested Gumbel** for this reason.

### Mixed-family nesting (important caveat)

If **parent and child nodes use different Archimedean families**, sampling goes through the **numerical frailty** path described above. That path is **not** robust today: in practice it can produce **degenerate** draws (e.g. some coordinates **stuck near 1**), even when `log_pdf` still returns finite values. So:

- **Do not rely on `sample()` for arbitrary mixed-family HAC trees** for inference, calibration, or publication-quality plots until this sampler is improved or replaced.
- For **demos and dependence structure**, prefer **nested same-family** specifications (especially **Gumbel–Gumbel** as in the tests) or **fully exchangeable** Archimedean trees.

This limitation is **implementation / numerics**, not a claim that mixed HACs are undefined mathematically.

## Fitting

HAC fitting (structure methods, recursive MLE, optional Monte Carlo likelihood) is exposed in Rust and Python. Fitting quality and identifiability depend on the tree and `family_set`, as usual. Sampling caveats above apply **after** fit: if the **fitted** tree contains mixed families, treat `sample()` with the same caution.

## Further reading in-repo

- Tests: `crates/rscopulas-core/tests/reference_hac.rs`
- Implementation: `crates/rscopulas-core/src/hac/mod.rs`
- Python bindings: `crates/rscopulas-python` (`_HierarchicalArchimedeanCopula`)
