# Changelog

## 0.4.0 — unreleased

- Fit Gaussian pair copulas by maximum likelihood (Brent's method on
  `atanh ρ`, warm-started at the Kendall-τ inversion) instead of τ inversion
  alone, so AIC/BIC family selection no longer favours the MLE-fitted
  Archimedean candidates by construction.
- Polish every two-parameter pair family (Student-t, BB1, BB6, BB7, BB8,
  Tawn1, Tawn2) with a bounded Nelder–Mead joint maximisation after the grid
  warm start. Student-t degrees of freedom are now continuous on a log scale
  over `[2, 200]` rather than a 24-point grid capped at 50.
- Allow negative Frank parameters. `PairCopulaSpec::validate` accepts any
  finite non-zero `θ`, the kernels, h-inverses, and CDF use the reflection
  `c_{-θ}(u, v) = c_θ(1 − u, v)`, and the pair fitter can now select Frank
  for negatively dependent pairs (previously such edges could fail with
  "pair-copula selection produced no candidate" when Frank was the only
  candidate).
- Add `VineFitOptions::independence_test_level`: an optional significance
  level for the asymptotic Kendall-τ independence test run before family
  selection on every edge. Disabled by default; `independence_threshold`
  keeps its raw cut-off semantics. Exposed in Python as the
  `independence_test_level` keyword of `VineCopula.fit_c`/`fit_d`/`fit_r`.
- **Behaviour change:** `VineFitOptions::default()` no longer includes
  `Khoudraji` in `family_set`. It dominated the default fit time and rarely
  won selection; list it explicitly to keep the previous candidate set.
- Honour `FitOptions::max_iter` in pair fitting. The previous silent clamps
  (16–64 iterations for parametric families, 8–16 / 8–12 for Khoudraji) are
  gone; all scalar searches now use Brent's method with tolerance-based early
  stopping capped at `max_iter`, so the default of 500 does not add cost.
- Add `math::maximize_scalar_brent`, `math::nelder_mead_maximize`,
  `stats::kendall_tau_test_statistic`, and
  `stats::kendall_tau_rejects_independence`.
- Fix a false convergence plateau in the factor-copula adaptive quadrature.
  Strong links at extreme observations concentrate the latent posterior in a
  spike narrower than the node spacing of both comparison rules; because the
  spike sits on a mesh break, the interval on its long side was reported as
  empty and half of its mass was dropped while still reporting convergence
  (three Joe(40) links at `u = 1 - 1e-10` returned 49.3268 instead of 50.0112;
  three Clayton(30) links at `u = 1e-12` returned 58.0018 instead of 58.6835).
  Each row's mesh is now seeded from the local scale of every link's
  conditional density (conditional quantiles for Khoudraji/Tawn links), and
  every break is evaluated so that an interval whose end point hides mass from
  both rules is refined instead of accepted. `quadrature_max_nodes` now also
  counts these break-point evaluations. Benign rows cost about 5–10 % more.
- Add `benchmarks/compare_pyvinecopulib.py`, a comparison against pyvinecopulib
  on vines simulated by pyvinecopulib (fit time, selected families, in- and
  out-of-sample log-likelihood, Rosenblatt calibration), with the report
  committed as `docs/pyvinecopulib-comparison.md` and a methodology page under
  `docs/mdx/performance/comparison.mdx`. rscopulas fits comparably well but is
  currently 7–15× slower at select-and-fit with a nine-family candidate set.
- Add `python/examples/portfolio_tail_risk.py`: simulated heavy-tailed asset
  returns from a known mixed vine, R-vine versus Gaussian-copula VaR/ES, and a
  conditional stress scenario through `sample_conditional`.
- Release the GIL in every compute-bound Python binding (all fitters,
  `log_pdf`/`composite_log_pdf`, `sample`, the Rosenblatt transforms, pair
  batch kernels, and TLL fits). Other Python threads keep running during
  long fits; every model type is `Send + Sync`.
- Add `to_json`/`from_json`, `pickle`, `copy`/`deepcopy`, `==`, and `repr` to
  every Python model class (`GaussianCopula`, `StudentTCopula`,
  `ClaytonCopula`, `FrankCopula`, `GumbelCopula`, `VineCopula`,
  `HierarchicalArchimedeanCopula`, `FactorCopula`, `PairCopula`). Vine
  payloads carry `format_version`; all payloads are validated on load.
- `InvalidInputError` now inherits from `ValueError` as well as
  `RscopulasError`. Unsupported family/rotation/criterion/method strings and
  evaluation-time dimension mismatches raise `InvalidInputError` (previously
  bare `ValueError` or `ModelFitError`); the core reports mismatches as
  `InputError::DimensionMismatch`. Panics in any binding surface as
  `InternalError` instead of `pyo3_runtime.PanicException`.
- `VineCopula.sample_conditional` draws its free columns from the seeded Rust
  generator (`rscopulas._rscopulas.uniform_matrix`), so a seed reproduces
  the same draw regardless of NumPy's global state; draws for a given seed
  differ from 0.3. Negative or too-large seeds raise `InvalidInputError`
  everywhere, non-integer `n` raises `TypeError`, and `n < 1` raises
  `InvalidInputError` in every sampling method.
- Add `rscopulas.to_pseudo_obs(x, ties=..., scaling=...)`, a NumPy rank
  transform that builds valid pseudo-observations from raw data.
- Ship `py.typed`, a `rscopulas/_rscopulas.pyi` stub for the extension, full
  annotations in the wrapper layer, and `rscopulas.__version__`.
- Run the Python test suite on Python 3.10 (NumPy 1.26), 3.12, and 3.13 in
  CI, with SciPy available for the optional statistical checks.
- `VineCopula.fit_c`/`fit_d`/`fit_r` and `FactorCopula.fit` with
  `family_set=None` delegate to the core default candidate sets (vines:
  `independence`, `gaussian`, `student_t`, `clayton`, `frank`, `gumbel`,
  `joe`, `bb1`, `bb7`; factor: `independence`, `gaussian`, `clayton`,
  `frank`, `gumbel`) rather than carrying their own list, so a default
  five-column vine fit takes about a second. `khoudraji` and `tll` stay
  opt-in; the accepted family strings are documented on `fit_r`.

## 0.3.0 — unreleased

This minor release contains breaking corrections to the pre-1.0 numerical and
serialization contracts. See [migration notes](docs/mdx/guides/migrating-to-0-3.mdx).

- Align pair, vine, conditional, and fixture orientation; enforce truncation in
  every vine fitter and optimize mBICV over all prefix lengths.
- Repair nested Clayton/Frank HAC sampling and Frank's large-parameter sampler.
- Expose nested HAC scores as `composite_log_pdf`. Nested `log_pdf` and
  unimplemented full/simulated likelihood optimizers now error. The default
  method is `composite_mle`; `recursive_mle` remains an alias. Nonzero
  `mc_samples` is rejected because no supported optimizer uses simulation.
- Add `likelihood_kind` to fit diagnostics. Composite scores have NaN AIC/BIC.
- Replace inconsistent TLL density/conditional calculations with a normalized
  distribution. Correct the bandwidth square-root scaling, make Rust grid
  state immutable, and preserve fitted state in Python specifications.
- Stabilize Clayton, BB1, BB6/7/8, Gumbel, and Joe pair kernels in the tails.
  Valid zero-density rows return `-inf`; NaN and positive infinity remain errors.
- Use symmetric, convergence-checked Gaussian/Student-t CDF integration in
  Khoudraji models. Remove the 0.98 Gaussian pair-fitting clamp.
- Use exact Gaussian factor densities and adaptive interval integration for
  other links. Add work/tolerance controls and explicit fixed-node mode.
- Validate model state and inputs; honor forced backend policies and retain
  Gaussian tail precision in Metal evaluation.
- Version serialized vines; reject old unversioned payloads instead of silently
  changing their orientation. Correct pair plotting, examples, and documentation.
- Gate Python releases on library CI and installed-wheel smoke tests. Run demo
  builds and dependency audits in a separate workflow.
