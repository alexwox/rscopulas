# Cross-checks against outside references

Scripts used for the correctness pass before release 0.5.0. They compare the
installed `rscopulas` Python package with references that share no code with
it. They are not part of CI: they need `pyvinecopulib`, `scipy`, and for one
of them R with the `copula` package, and several run for minutes.

```sh
maturin build --release --out dist && pip install --force-reinstall dist/rscopulas-*.whl
pip install pyvinecopulib scipy
python scripts/crosschecks/<script>.py
```

| Script | Reference | Result on 0.5.0 |
| --- | --- | --- |
| `pair_kernels_vs_vinecopulib.py` | vinecopulib (C++) on a 15 × 15 grid reaching 1e-6 from the edges | Gaussian, Student t, Clayton, Gumbel, Frank, Joe densities agree to 5e-13. Where the two disagree (BB families and inverses in the tails) the 400-digit fixture in `fixtures/reference/mpmath/` sides with rscopulas. |
| `multivariate_density_vs_r.py` (after `Rscript multivariate_density_reference.R`) | R `copula::dCopula`, d = 2, 3, 5, 8 | Gaussian, Student t, Clayton, Frank, Gumbel agree to 2e-13 relative. |
| `factor_density_vs_quadrature.py` | `scipy.integrate.quad` over the latent factor, split at both ends | One-factor log-densities agree to 1.5e-9. |
| `vine_vs_naive_recursion.py` | A set-based textbook recursion built only from pair kernels | Log-density agrees to 1e-11 away from the upper corner; Rosenblatt agrees; samples pass a uniformity and independence test after the naive transform. |
| `nested_sampler_vs_cdf.py` | Closed-form nested distribution function | Root-mean-square z-score 1.00 over 9600 points. This is the check that found the nested Clayton and Frank sampler bug fixed in 0.5.0. |
| `pair_fits_global_optimum.py` | Differential evolution over every rotation, and vinecopulib's fit | 336 fits. One-parameter families, Student t and BB8 reach the global optimum. BB1/6/7 stop 5e-5 inside the parameter bound (log-likelihood within 6e-3). Tawn fits on near-independent data return a smooth local optimum; the global one is a spike at the edge of the parameter box worth 1 to 3.5 log-likelihood units. |
