# Changelog

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
