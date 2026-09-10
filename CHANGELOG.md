# Changelog

## 0.4.0 — unreleased

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
