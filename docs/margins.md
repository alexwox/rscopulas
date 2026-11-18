# Margins and extreme-value tails

A copula describes dependence between uniforms. To go from raw data to a joint
model and back to the raw scale, each column also needs:

- a distribution function `F`: raw value to uniform, and
- its inverse: uniform to raw value.

`to_pseudo_obs` gives the first direction only. Ranks also say nothing beyond
the largest observation: a simulated uniform of `0.99999` cannot be mapped to a
level the data never reached. The margins layer supplies both directions and
extrapolates.

## The model

Each column gets a `SemiParametricMargin`:

```text
       lower tail            body                upper tail
  generalized Pareto | interpolated ECDF | generalized Pareto
                    u_L                 u_U
```

- **Body.** The empirical distribution function `#{x_i <= x} / (n + 1)`,
  linear between observations. At an observation it equals `to_pseudo_obs`.
- **Tails.** Beyond each threshold, the distance past the threshold follows a
  generalized Pareto distribution (GPD) fitted by maximum likelihood. This is
  what the Pickands–Balkema–de Haan theorem says excesses over a high
  threshold approach.
- The pieces join continuously and the function is strictly increasing, so
  the quantile function inverts it exactly.

By default the lowest and highest tenth of the sample are tails.

## Raw data to joint tail estimates

```python
import numpy as np
from rscopulas import Margins, VineCopula

margins = Margins.fit(x)                     # x: raw (n, d) observations
u = margins.cdf(x)                           # pseudo-observations
model = VineCopula.fit_r(u).model            # any rscopulas model

simulated = margins.quantile(model.sample(1_000_000, seed=1))   # raw scale
p = np.mean((simulated[:, 0] > a) & (simulated[:, 1] > b))      # joint tail
```

```rust
use rscopulas::{FitOptions, GumbelHougaardCopula, MarginFitOptions, Margins};

let margins = Margins::fit(raw.view(), &MarginFitOptions::default())?;
let data = margins.to_pseudo_obs(raw.view())?;
let fit = GumbelHougaardCopula::fit(&data, &FitOptions::default())?;
let simulated = margins.quantile(sample.view())?;
```

`Margins` serializes like the copula models (`to_json` / `from_json`, pickle,
`serde`), so the margins can be stored next to the copula they were used with.

### Settings

| Python | Rust | Meaning |
| --- | --- | --- |
| `upper_tail=0.1` | `Threshold::Fraction(0.1)` | Fraction of the sample in the tail, in `(0, 0.5)` |
| `upper_threshold=2.5` | `Threshold::Value(2.5)` | Start the tail at this raw value; moved to the nearest observation on the body side |
| `upper_tail=None` | `None` | Keep that tail empirical: flat beyond the sample |
| `min_exceedances=10` | `min_exceedances` | Fewest observations beyond a threshold for a tail fit |

The lower tail has the same settings. For different settings per column, fit
`SemiParametricMargin` one column at a time and combine with
`Margins.from_margins([...])`.

### Return levels

`margin.return_level(m)` is the level exceeded once every `m` observations on
average, the `1 - 1/m` quantile. With daily data, `return_level(365.25 * 100)`
is the 100-year level, provided the observations are independent.

## Choosing the threshold

The threshold is a bias–variance trade-off. Too low and the GPD is fitted to
data that are not in the tail yet (bias). Too high and few points are left
(variance). Two standard plots help.

```python
from rscopulas.plotting import plot_mean_excess, plot_threshold_stability

plot_mean_excess(x[:, 0])
plot_threshold_stability(x[:, 0])
```

- **Mean residual life** (`mean_excess`). If excesses over `u0` are GPD, the
  mean excess over any higher threshold is linear in the threshold. Take the
  lowest threshold from which the curve is straight within its band. Rising
  means a heavy tail, flat an exponential tail, falling a bounded tail.
- **Parameter stability** (`threshold_stability`). Above a valid threshold the
  fitted shape, and the modified scale `scale - shape * threshold`, stay
  constant up to noise. Take the lowest threshold from which both are flat
  within their bands.

## Block maxima

`GeneralizedExtremeValue` models the maximum of a block (a year of daily
values, say):

```python
from rscopulas import GeneralizedExtremeValue, block_maxima

fit = GeneralizedExtremeValue.fit(block_maxima(daily, 365))
level = fit.model.return_level(100)              # 100-block return level
error = fit.return_level_std_error(100)          # delta method
```

The interval `level ± 1.96 · error` is symmetric. For long periods and heavy
tails the true uncertainty is skewed upward, so read it as a lower bound on
the upper side.

`GeneralizedPareto.fit(excesses)` is the tail fit on its own. Both fits return
the log-likelihood and the asymptotic covariance of the parameters.

## Conventions and limits

- **Shape sign.** Positive shape is a heavy (power-law) tail, zero an
  exponential-type tail, negative a tail with a finite end point.
  `scipy.stats.genpareto` uses `c = shape`; `scipy.stats.genextreme` uses
  `c = -shape`.
- **Shape range.** GPD fits are restricted to `shape >= -1`, where the
  likelihood is bounded; GEV fits to `[-1, 5]`. Standard errors are not
  reported on a boundary, and the usual theory needs `shape > -1/2`.
- **Independence.** The fits treat observations as independent and
  identically distributed. For time series, filter first (for example fit the
  margins to GARCH residuals) or decluster.
- **How far to extrapolate.** With 2000 tail points and true shape 1/3, the
  estimated survival probability at a level 50 times rarer than the rarest
  observation has a standard deviation of 0.7 on the log scale: right to
  within a factor of two about two times in three. The error averages out
  over repeated samples, but one sample is one sample.
- A fitted tail with negative shape assigns probability exactly 0 or 1 beyond
  its end point. `Margins.cdf` keeps its output `1e-12` away from 0 and 1 so
  the result is always valid copula input.

## How this is tested

`fixtures/reference/evt/fits.json` is written by
`scripts/reference/generate_evt_fixtures.py`:

- Distribution functions, densities and quantiles agree with
  `scipy.stats.genpareto` / `genextreme` to `1e-12` or better, including a
  shape of `1e-9` and survival probabilities down to `2.6e-11`.
- Six GPD and five GEV samples are fitted three ways. This crate reaches a
  log-likelihood at least as high as a tightly converged Nelder–Mead search on
  SciPy's log-likelihood and as R's `evd::fpot` / `evd::fgev`; parameters agree
  with SciPy to `5e-6` (GPD) and `2e-5` (GEV), standard errors to `2e-4`.
- GEV return levels and their delta-method standard errors agree with
  `extRemes` to `1e-3` and `5e-3`.
- The margin is checked on a distribution with a closed-form distribution
  function and exactly generalized Pareto tails: continuity at both joins,
  monotonicity, exact inversion, and no bias in the extrapolated survival
  probability over 40 samples.
- An end-to-end test (raw data, margins, Gumbel copula, one million simulated
  points, back to the raw scale) recovers a joint exceedance probability of
  `2.9e-4` known in closed form.
