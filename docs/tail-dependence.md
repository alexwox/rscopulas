# Tail dependence

Tail dependence asks whether the extremes of two variables arrive together.
For a pair of uniforms `(U1, U2)`:

```text
lower = lim  P(U2 <= q | U1 <= q)   as q -> 0
upper = lim  P(U2 >  q | U1 >  q)   as q -> 1
```

A value of 0 means joint extremes are no more likely than under independence
in the limit; 1 means they always coincide. Kendall's tau and the fitted
log-likelihood are dominated by the body of the distribution, so two families
with the same tau can disagree completely here.

`rscopulas` gives both sides of the comparison: what a fitted model implies,
and what the data show.

## What a model implies

| Family | Lower | Upper |
| --- | --- | --- |
| Independence, Gaussian, Frank | 0 | 0 |
| Student t (`rho`, `nu`) | `2 T_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))` | same |
| Clayton (`theta`) | `2^(-1/theta)` | 0 |
| Gumbel, Joe (`theta`) | 0 | `2 - 2^(1/theta)` |
| BB1 (`theta`, `delta`) | `2^(-1/(theta delta))` | `2 - 2^(1/delta)` |
| BB6 (`theta`, `delta`) | 0 | `2 - 2^(1/(theta delta))` |
| BB7 (`theta`, `delta`) | `2^(-1/delta)` | `2 - 2^(1/theta)` |
| BB8 (`theta`, `delta`) | 0 | 0 (Joe's value when `delta = 1`) |
| Tawn 1, Tawn 2 (`theta`, `s`) | 0 | `1 + s - (1 + s^theta)^(1/theta)` |

A 180 degree rotation swaps the two columns. A 90 or 270 degree rotation
moves the dependence to the off-diagonal corners, so both coefficients are 0
(for Student t it is the same as negating `rho`). Khoudraji and TLL pair
copulas have no closed form and report `None`.

```python
from rscopulas import ClaytonCopula, PairCopula

PairCopula.from_spec("clayton", [2.0]).tail_dependence()
# TailDependence(lower=0.7071067811865476, upper=0.0)

ClaytonCopula.from_params(3, 2.0).tail_dependence().lower   # (3, 3) matrix
```

```rust
use rscopulas::{PairCopulaFamily, PairCopulaParams, PairCopulaSpec, Rotation};

let spec = PairCopulaSpec {
    family: PairCopulaFamily::Clayton,
    rotation: Rotation::R0,
    params: PairCopulaParams::One(2.0),
};
let pair = spec.tail_dependence()?.expect("Clayton has a closed form");
assert_eq!(pair.upper, 0.0);
```

`GaussianCopula`, `StudentTCopula`, `ClaytonCopula`, `FrankCopula` and
`GumbelCopula` return a `TailDependenceMatrix` with one entry per pair. Vine,
factor and hierarchical models do not report coefficients.

## What the data show

All estimators use only the ranks of their input, so raw observations and
pseudo-observations give the same result.

```python
import numpy as np
from rscopulas import empirical_tail_dependence, empirical_tail_dependence_matrix

estimate = empirical_tail_dependence(x[:, 0], x[:, 1])        # k = floor(sqrt(n))
matrix = empirical_tail_dependence_matrix(x)                  # every pair
```

The estimator counts how often both variables are among their `k` smallest
(or `k` largest) values:

```text
lower = #{i : R_i <= k     and S_i <= k    } / k
upper = #{i : R_i >  n - k and S_i >  n - k} / k
```

### Choosing `k`

`k` is a trade-off. A small `k` looks only at genuine extremes but averages
over few points, so the estimate is noisy. A large `k` is stable but reaches
into the body of the distribution, where the conditional probability is not
the tail limit yet. The default is `floor(sqrt(n))`.

Do not trust one `k`. Plot the estimate over a range and read the flat part:

```python
from rscopulas.plotting import plot_tail_dependence

plot_tail_dependence(x, dims=(0, 1), model=fitted_pair)
```

`empirical_tail_dependence_curve(u1, u2)` returns the same numbers.

### A zero coefficient can still look large

The limit can be reached slowly. A Gumbel copula with `theta = 2.5` has lower
coefficient 0, but `P(U2 <= q | U1 <= q)` only falls like `q^0.32`. With
200 000 observations and `k = 447` the estimator looks at `q = 0.002`, where
that probability is still 0.14. A Gaussian copula is slower again. Compare
the estimate with the model at the same level, not with the limit.

That is what the tail concentration function is for:

```python
from rscopulas import PairCopula, empirical_tail_concentration
from rscopulas.plotting import plot_tail_concentration

observed = empirical_tail_concentration(u[:, 0], u[:, 1])      # q = 0.01 .. 0.99
implied = fitted_pair.tail_concentration(observed.q)

plot_tail_concentration(u, model=fitted_pair)
```

For `q <= 0.5` it is `C(q, q) / q`; above, `(1 - 2q + C(q, q)) / (1 - q)`. Its
two ends approach the coefficients. A model curve that leaves the data at one
end is the wrong family for that tail.

## Rust

```rust
use rscopulas::tail::{
    default_tail_k, empirical_tail_concentration, empirical_tail_dependence,
    empirical_tail_dependence_curve, empirical_tail_dependence_matrix,
};

let k = default_tail_k(first.len());
let estimate = empirical_tail_dependence(&first, &second, k)?;
let matrix = empirical_tail_dependence_matrix(data.as_view(), k)?;
```

## How this is tested

- `fixtures/reference/mpmath/tail_dependence.json` holds the coefficients of
  43 parameter sets, computed by
  `scripts/reference/generate_tail_dependence_truth.py` as numerical limits of
  each family's textbook distribution function at 1300 digits. The script
  never uses a closed-form coefficient. The crate agrees to `1e-10`.
- Student t values also match R `copula::lambda` to `1e-13`.
- On pseudo-observations the matrix estimator equals R
  `copula::fitLambda(u, method = "Schmidt.Stadtmueller", p = k / n)` to
  2e-16 for both tails (checked by hand on a 20 000 × 10 sample; R thresholds
  the values it is given instead of ranking them, so it needs
  pseudo-observations to agree).
- The estimators are compared with a literal transcription of their
  definitions, and on 200 000-point samples they recover the known
  coefficients of Clayton, Gumbel and Student t copulas.

## Speed

Measured on an Apple M3 Pro, best of several runs, identical results:

| Task | rscopulas | NumPy / SciPy | R `copula` |
| --- | --- | --- | --- |
| Both tail matrices, 20 000 × 10 | 1.1 ms | 17 ms | 14 ms (`fitLambda`, two calls) |
| Curve over 4 000 values of `k`, one pair | 1.5 ms | 3.7 ms vectorised, 66 ms as a loop | |

The matrix estimator ranks each column once, in parallel, and then looks
only at the roughly `k` observations in each tail.
