# Golden snapshot change log

The snapshots in this directory were recorded from release 0.4.0 (commit
`bd07898`) before the performance work on the numerical core. They define "no
regression": every later change must reproduce them within the tolerances in
`crates/rscopulas-core/tests/support/golden.rs`.

An entry is re-recorded only when the *new* value is demonstrably the more
accurate one. Each such case is listed here with the evidence.

## Re-recorded entries

### `vine_evaluation.json`: `gaussian_d7/log_pdf`, `gaussian_c4/log_pdf`

- **Change:** 5 of 96 values moved by up to 4.4e-10 absolute (2.4e-9 relative).
- **Cause:** the standard normal CDF switched from `statrs`' `erfc` (accurate
  to ~1e-10 relative in the tails: `Φ(Φ⁻¹(1e-4))` returned
  `1.00000000008e-4`) to Cody's rational approximation (~1e-16 relative,
  checked against SciPy on 277 points in
  `special::tests::matches_scipy_reference_values`).
- **Evidence that the new values are the correct ones:** a Gaussian vine built
  from a correlation matrix is exactly the Gaussian copula with that matrix,
  whose density needs no normal CDF at all. The test
  `numerical_invariants::gaussian_vine_density_equals_gaussian_copula_density`
  compares the two: 0.4.0 deviated by 5.8e-11, the current code by 5e-15.

### `tll.json`: `quadratic/log_pdf`; `vine_evaluation.json`: `fitted_tll_trunc2/log_pdf`; `factor.json`: `mixed/log_pdf`

- **Change:** 1 of 80 values moved by 7.8e-11, 3 of 48 by up to 1.7e-10, and
  1 of 48 by 2.3e-11 (absolute; 1.5e-9 to 9e-9 relative, the last on a
  log-density of -0.0024).
- **Cause:** the same normal-CDF change as above. The TLL grid places its
  knots at `Φ(z)` and the factor copula maps latent normal scores through
  `Φ`, so both inherit the 1e-10 correction.
- **Evidence:** substituting the old `erfc`-based `Φ` back into the current
  code reproduces the 0.4.0 values of all three entries exactly (checked while
  making the change); nothing else in these code paths moved.

### `hac.json`: `clayton/sample`, `frank/sample`

- **Change:** every value in the nested (non-leaf) columns; these are new
  random draws, not a small numerical shift.
- **Cause:** the frailty of a Clayton node nested in a Clayton node, or a
  Frank node in a Frank node, was drawn by inverting a 12-term Stehfest
  approximation of its Laplace transform. It is now drawn exactly: a tilted
  stable variate by rejection for Clayton, a sum of tilted Sibuya variates
  for Frank (Hofert 2011), with the stable limit above a parent frailty of
  1e4.
- **Evidence that the old values were wrong:** the distribution function of
  a nested Archimedean copula is closed-form and involves no frailty. With
  2e6 draws the old sampler missed it by up to 0.21 in absolute terms
  (Frank 0.2 > 0.3 > 30) and 0.09 (Clayton 0.05 > 0.1 > 12); its marginals
  failed a Kolmogorov–Smirnov test with `√n·D` between 59 and 580 (the 1 %
  point is 1.63). The new sampler stays within binomial noise on the same
  points (root-mean-square z-score 1.00 over 9600 comparisons, largest 3.8)
  with `√n·D` at most 1.7. `reference_hac::
  nested_samplers_follow_the_nested_distribution_function` pins this at
  n = 2e5. Nested Gumbel and exchangeable trees were already exact by the
  same test and are bit-for-bit unchanged.

## Comparison rules added after recording

- **Conditioning slack** (`conditioning_slack` in `golden_snapshots.rs`): a
  vine or model log-density is also accepted when it differs from the stored
  value by less than four times the change caused by nudging the inputs by 45
  ulps. This affects 4 of 4368 values — rows where an intermediate conditional
  lands within 1e-11 of 1 and its complement has only five significant digits,
  so the stored value is itself rounding noise at the 1e-5 level.
- **TLL upper-tail allowance** (`snapshot_model_with_noise`): rows of the TLL
  vine with a coordinate above `1 − 1e-3` are compared at 1e-6. Past its last
  grid knot the TLL kernel locates a point through `1 − (accumulated CDF)`,
  which carries ~1e-7 relative rounding noise at `u = 1 − 1e-9`.

### `vine_fits.json`: `options/khoudraji/{structure,params,loglik}`

- **Change:** the R-vine fitted with `family_set = [Independence, Gaussian,
  Khoudraji]` on the 120 × 3 upper-tail sample selects different Khoudraji
  base pairs on two of its three edges; log-likelihood 58.775 → 61.716, AIC
  -101.55 → -103.43.
- **Cause:** the Khoudraji joint polish moved from Nelder–Mead to the
  quasi-Newton search with coordinate probes (`math::polish_maximize`). On
  edge (0, 1) it reaches log-likelihood 28.708 for the 180°-rotated candidate,
  where the simplex search had stalled below the unrotated candidate's 28.069.
- **Evidence:** selection minimises AIC, and the new model is better on that
  criterion and on raw log-likelihood. Fitted in isolation, the other two
  edges agree with 0.4.0 to 4e-7 relative (28.457390 vs 28.457395 and
  21.593444 vs 21.593452), inside the polish's own 1e-6 stopping tolerance.

### `vine_evaluation.json`: `gaussian_d7/{log_pdf,rosenblatt,rosenblatt_prefix}`, `gaussian_c4/{log_pdf,rosenblatt,rosenblatt_prefix}` (second re-recording)

- **Change:** on the two rows of the evaluation set that sit in the upper
  tail (all coordinates 0.9999 or `1 − 1e-9`), log-densities moved by up to
  1.3e-7 and Rosenblatt uniforms by up to 5e-9.
- **Cause:** vines whose edges are all Gaussian (or independence) are now
  evaluated on the normal-score scale (`vine/gaussian_chain.rs`), where both
  h-functions are linear. The 0.4.0 recursion stored every conditional as a
  probability; one within 1e-8 of 1 keeps only eight digits of its
  complement, and the next tree's `Φ⁻¹` amplifies that loss.
- **Evidence:** two closed forms that involve no vine recursion at all, in
  `tests/numerical_invariants.rs`. The density must equal the Gaussian
  copula's quadratic-form density: the current code agrees to 2e-15 on every
  row including the upper tail, 0.4.0 was off by 2e-8 there. The Rosenblatt
  transform must equal Cholesky whitening, `Φ⁻¹(U) = L⁻¹ Φ⁻¹(v)`: the current
  code agrees to 5e-16 on the score scale in both directions, 0.4.0 fails the
  same test at 1e-11.

## `pair_fit_sweep.txt`

Not a snapshot of the current code but of release 0.4.0: the log-likelihood
and AIC its pair fitters reached on 2250 fits (see
`crates/rscopulas-core/tests/pair_fit_sweep.rs`). The rewritten optimisers
must reach those values; they may exceed them.

Four fits are exempt, all the Tawn1/Tawn2 fit with rotations on one sample of
300 independent observations. There 0.4.0's simplex search ended on an
isolated likelihood spike at the edge of the parameter box (θ = 1 + e⁴ with an
asymmetry weight of 0.004; the likelihood is `-inf` one step further and lower
than at the starting point on the way there). The current search returns the
smooth local optimum 1–2.3 log-likelihood units lower. Over the wider
4500-fit sweep run during development the new optimisers were below 0.4.0 in
7 fits (all of this kind) and above it in 5; default-set selection was never
worse on AIC.
