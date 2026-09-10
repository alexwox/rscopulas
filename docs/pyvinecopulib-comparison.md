# rscopulas vs pyvinecopulib

Generated 2026-09-10 19:24 UTC by `benchmarks/compare_pyvinecopulib.py`.

| Environment | |
| --- | --- |
| platform | macOS-15.5-arm64-arm-64bit |
| cpu | Apple M3 Pro |
| python | 3.12.11 |
| numpy | 2.5.3 |
| rscopulas | 0.3.0 |
| pyvinecopulib | 0.7.6 |

## Method

Each dataset is simulated with `pyvinecopulib.Vinecop.simulate` from a known vine, so the truth is independent of rscopulas. Training and test sets are rank-transformed separately (`rank / (n + 1)`). Both libraries select structure and pair-copula families on the training set from the same candidate set — independence, Gaussian, Student-t, Clayton, Frank, Gumbel, Joe, BB1, BB7, rotations allowed — with Kendall's tau spanning trees. Each library uses its own default selection criterion: **rscopulas `fit_r` defaults to AIC, pyvinecopulib `FitControlsVinecop` defaults to BIC**; a third row refits rscopulas with BIC. pyvinecopulib keeps its default `preselect_families=True` (families are pruned by symmetry of the data before fitting), rscopulas fits every candidate. Both run single-threaded. Fit time is the median of 3 wall-clock runs of the complete select-and-fit call in one Python process. Log-likelihoods are the sums of each fitted model's own log-density over the training and test pseudo-observations; the truth row evaluates the simulating vine. Rosenblatt diagnostics transform the **test** set with each fitted model: `KS max` is the largest per-column Kolmogorov-Smirnov distance from U(0, 1) with its asymptotic p-value, `Spearman max` the largest absolute Spearman correlation between transformed columns (both should be small for a well-specified model).

## gaussian_rvine_5d (d = 5, n_train = 2000, n_test = 2000)

| Model | Fit time (s) | Log-lik train | Log-lik test | Params | KS max (p) | Spearman max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rscopulas (AIC) | 44.825 | 2608.4 | 2607.9 | 10 | 0.0219 (0.29) | 0.0519 |
| pyvinecopulib (BIC) | 3.097 | 2608.9 | 2607.6 | 10 | 0.0232 (0.23) | 0.0493 |
| rscopulas (BIC) | 27.447 | 2608.4 | 2607.9 | 10 | 0.0219 (0.29) | 0.0519 |
| truth (simulating vine) | – | 2606.3 | 2609.0 | – | 0.0208 (0.35) | 0.0355 |

Fit-time ratio rscopulas (AIC) / pyvinecopulib (BIC): **14.5x**. Out-of-sample log-likelihood gap to the truth: rscopulas (AIC) -1.0, pyvinecopulib (BIC) -1.4, rscopulas (BIC) -1.0.

Selected first-tree edges (variables are 0-based; `r` = rotation in degrees):

| Pair | Truth | rscopulas (AIC) | pyvinecopulib (BIC) | rscopulas (BIC) |
| --- | --- | --- | --- | --- |
| (0, 2) | gaussian [0.65] | gaussian [0.645] | gaussian [0.65] | gaussian [0.645] |
| (1, 2) | gaussian [0.5] | – | – | – |
| (1, 3) | gaussian [-0.6] | gaussian [-0.6] | gaussian [-0.601] | gaussian [-0.6] |
| (1, 4) | – | gaussian [0.599] | gaussian [0.604] | gaussian [0.599] |
| (2, 4) | gaussian [0.75] | gaussian [0.762] | gaussian [0.765] | gaussian [0.762] |

## mixed_rvine_5d (d = 5, n_train = 2000, n_test = 2000)

| Model | Fit time (s) | Log-lik train | Log-lik test | Params | KS max (p) | Spearman max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rscopulas (AIC) | 17.969 | 4378.7 | 4444.2 | 13 | 0.0288 (0.07) | 0.1028 |
| pyvinecopulib (BIC) | 1.289 | 4378.7 | 4443.8 | 13 | 0.0212 (0.33) | 0.0835 |
| rscopulas (BIC) | 17.129 | 4378.7 | 4444.2 | 13 | 0.0288 (0.07) | 0.1028 |
| truth (simulating vine) | – | 4595.8 | 4644.5 | – | 0.0192 (0.45) | 0.0375 |

Fit-time ratio rscopulas (AIC) / pyvinecopulib (BIC): **13.9x**. Out-of-sample log-likelihood gap to the truth: rscopulas (AIC) -200.4, pyvinecopulib (BIC) -200.7, rscopulas (BIC) -200.4.

Selected first-tree edges (variables are 0-based; `r` = rotation in degrees):

| Pair | Truth | rscopulas (AIC) | pyvinecopulib (BIC) | rscopulas (BIC) |
| --- | --- | --- | --- | --- |
| (0, 1) | – | student_t [0.63, 10.9] | student_t [0.63, 11.5] | student_t [0.63, 10.9] |
| (0, 2) | student_t [0.6, 4] | – | – | – |
| (0, 4) | gumbel r180 [2.5] | gumbel r180 [2.55] | gumbel r180 [2.55] | gumbel r180 [2.55] |
| (1, 2) | frank [6] | frank [6.01] | frank [6.01] | frank [6.01] |
| (1, 3) | clayton [2] | clayton [1.93] | clayton [1.93] | clayton [1.93] |

## mixed_rvine_8d (d = 8, n_train = 2000, n_test = 2000)

| Model | Fit time (s) | Log-lik train | Log-lik test | Params | KS max (p) | Spearman max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rscopulas (AIC) | 28.684 | 7701.5 | 7386.2 | 33 | 0.0379 (0.01) | 0.0848 |
| pyvinecopulib (BIC) | 4.222 | 7708.7 | 7391.2 | 30 | 0.0292 (0.06) | 0.0940 |
| rscopulas (BIC) | 28.492 | 7684.3 | 7379.6 | 29 | 0.0334 (0.02) | 0.1006 |
| truth (simulating vine) | – | 7941.2 | 7690.2 | – | 0.0284 (0.08) | 0.0612 |

Fit-time ratio rscopulas (AIC) / pyvinecopulib (BIC): **6.8x**. Out-of-sample log-likelihood gap to the truth: rscopulas (AIC) -304.0, pyvinecopulib (BIC) -299.0, rscopulas (BIC) -310.6.

Selected first-tree edges (variables are 0-based; `r` = rotation in degrees):

| Pair | Truth | rscopulas (AIC) | pyvinecopulib (BIC) | rscopulas (BIC) |
| --- | --- | --- | --- | --- |
| (0, 2) | frank [7] | frank [7.3] | frank [7.3] | frank [7.3] |
| (1, 2) | clayton r180 [1.8] | clayton r180 [1.93] | clayton r180 [1.93] | clayton r180 [1.93] |
| (2, 4) | clayton [2.5] | clayton [2.58] | clayton [2.58] | clayton [2.58] |
| (2, 5) | – | student_t [0.647, 8.22] | bb1 r180 [0.155, 1.66] | student_t [0.647, 8.22] |
| (2, 6) | gumbel r180 [2.2] | gumbel r180 [2.14] | gumbel r180 [2.14] | gumbel r180 [2.14] |
| (2, 7) | student_t [-0.55, 6] | – | – | – |
| (3, 7) | student_t [0.7, 4] | student_t [0.684, 3.57] | student_t [0.687, 3.79] | student_t [0.684, 3.57] |
| (5, 7) | gumbel r90 [2] | gumbel r90 [2.02] | gumbel r90 [2.02] | gumbel r90 [2.02] |

