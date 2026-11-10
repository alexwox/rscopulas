# Changelog

## 0.5.0 — 2026-11-10

Performance work on the numerical core. Measured on an Apple M3 Pro against
0.4.0 with `examples/perf_suite.rs`, both builds run back to back (medians;
all 110 cases in `benchmarks/perf/`; figures move by ~10 % between runs):
default R-vine fit about 10×, C/D-vine fit about 30×, default pair fit
20–30×, factor-copula fit 16–26×, nested Archimedean sampling 22–35×,
inverse h-functions 7–14×, mixed-vine sampling 7–8×, mixed-vine density 4×,
Gaussian-vine density 46×. Scalar densities and h-functions of the
Archimedean pair families are unchanged; no case is slower.

- Add a golden-snapshot regression suite
  (`tests/golden_snapshots.rs`, fixtures under `fixtures/golden/v1/`): every
  pair kernel, fitter, sampler, transform and statistic on fixed inputs,
  recorded from 0.4.0 and compared at 1e-9 relative (closed forms), by
  residual (inverse problems) or one-sidedly (fitted log-likelihoods may only
  improve). `tests/pair_fit_sweep.rs` holds 2250 pair fits to the
  log-likelihoods 0.4.0 reached, and `tests/numerical_invariants.rs` checks
  Gaussian vines against closed forms. Re-recordings and their justification
  are logged in `fixtures/golden/v1/CHANGES.md`.
- Inverse h-functions without a closed form (Gumbel, Joe, BB1/6/7/8, Tawn,
  Khoudraji) use a bracketed root search on the logit scale of both axes
  instead of a 90-step bisection: 5–8 evaluations of the h-function per
  inverse, and full relative accuracy for roots near 0 or 1.
- Pair fitting: candidates (family × rotation) and the grid points of
  two-parameter warm starts are fitted concurrently; the joint polish is a
  bounded quasi-Newton search (`math::quasi_newton_maximize`) with coordinate
  probes and Nelder–Mead only as a fallback; the Student t fit maximises the
  profile likelihood in `ν` over cached quantiles; likelihood passes cache the
  logs of the sample and hoist parameter-only terms.
- C- and D-vine fits now fit the edges of a tree concurrently, as the R-vine
  fit already did.
- Khoudraji fits warm-start their base pairs concurrently and polish with the
  quasi-Newton search. On one of the snapshot datasets this finds a better
  optimum than 0.4.0 did (log-likelihood 61.72 vs 58.77 for the vine).
- Vines whose edges are all Gaussian or independence copulas are evaluated on
  the normal-score scale, where h-functions are linear. Besides the speed-up
  this removes a loss of accuracy in the upper tail: density and Rosenblatt
  transform now match their closed forms to 1e-15 (0.4.0: 2e-8 on upper-tail
  rows).
- Replace the `statrs` normal and Student t quantile/CDF calls on hot paths
  with dedicated implementations (`special.rs`): Wichura's AS 241 normal
  quantile, Cody's normal CDF, and a Student t distribution with precomputed
  constants, continued-fraction CDF and Hill-plus-Halley quantile, checked
  against SciPy and exact closed forms. **Accuracy change:** the old normal
  CDF was only accurate to ~1e-10 relative in the tails, so results that
  depend on it move by up to that much.
- `StudentTCopula::fit` inverts each distinct pseudo-observation once per
  candidate `ν` instead of once per cell, and evaluates candidates
  concurrently.
- Factor copula: density rows are integrated concurrently, the joint polish
  uses Brent's method along each coordinate, and sampling applies the inverse
  h-functions in parallel (the random stream is unchanged).
- Nested Archimedean sampling draws its random numbers sequentially and
  solves for the nested frailties in parallel with a superlinear root search;
  seeded output is unchanged.
- TLL fits tabulate the separable kernel per axis (bit-identical grids,
  12× faster). Hoeffding's D uses an O(n log n) sweep instead of the O(n²)
  pairwise count (identical values).
- Add `math::quasi_newton_maximize` with `QuasiNewtonOptions` /
  `QuasiNewtonResult`.
- **Behaviour change:** Khoudraji h-functions no longer evaluate the base
  densities, so they succeed at points where a base density is not finite.

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
  keeps its raw cut-off semantics.
- Fit Khoudraji pair copulas by joint maximum likelihood. Base parameters
  and both shapes are optimised together (block coordinate ascent from three
  shape seeds, then a bounded Nelder–Mead polish of the leading base pairs),
  only unordered base pairs are enumerated because `(C₁, C₂, a, b)` and
  `(C₂, C₁, 1 − a, 1 − b)` are the same copula, and the winning base pair is
  chosen by the vine's selection criterion instead of raw log-likelihood.
  Previously the shapes were fitted after the bases had been fitted to the
  raw sample; on the copula-package reference fixture that stopped at
  loglik 9.10 while R's joint `Independence ⊗ Clayton` optimum scores 11.53,
  which the same base pair now reproduces (θ 4.926 vs 4.927, shapes
  (0.7615, 0.4147) vs (0.7615, 0.4147)).
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
  `docs/mdx/performance/comparison.mdx`. with the maximum-likelihood
  fitters rscopulas selects the same model under BIC (and a better one out of
  sample under AIC) and is 1.6–2.7× slower at select-and-fit with a nine-family
  candidate set.
- Add `python/examples/portfolio_tail_risk.py`: simulated heavy-tailed asset
  returns from a known mixed vine, R-vine versus Gaussian-copula VaR/ES, and a
  conditional stress scenario through `sample_conditional`.
- Release the GIL in every compute-bound Python binding (all fitters,
  `log_pdf`/`composite_log_pdf`, `sample`, the Rosenblatt transforms, pair
  batch kernels, and TLL fits). Other Python threads keep running during
  long fits; every model type is `Send + Sync`.
- Make JSON serialization bit-exact. serde_json's default float parser is
  best-effort and can be one ulp off, so `from_json`, pickle and `copy` could
  return a model that compared unequal to the original (observed on Windows
  CI for a fitted vine); the `float_roundtrip` feature is now enabled and a
  regression test round-trips fitted models and a sweep of doubles exactly.
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

## 0.3.0 — 2026-09-10

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
