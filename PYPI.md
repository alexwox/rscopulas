![rscopulas logo](https://pub-ca706ca75c8a4972b721945607f0ff01.r2.dev/rscopulas-logo.png)

# rscopulas

`rscopulas` is a Python package for fitting, evaluating, and sampling copula models on validated pseudo-observations, backed by a Rust core built with PyO3 and maturin.

## Install

```bash
pip install rscopulas
```

Optional plotting helpers are available with:

```bash
pip install "rscopulas[viz]"
```

## Included models

- Gaussian and Student t copulas
- Archimedean families including Clayton, Frank, and Gumbel
- Pair copulas, including Khoudraji constructions
- C-vine, D-vine, and R-vine copulas
- Hierarchical Archimedean copulas

## Python example

```python
import numpy as np
from rscopulas import GaussianCopula

data = np.array(
    [
        [0.12, 0.18],
        [0.21, 0.25],
        [0.82, 0.79],
    ],
    dtype=np.float64,
)

fit = GaussianCopula.fit(data)
print("AIC:", fit.diagnostics.aic)
print("sample:\n", fit.model.sample(4, seed=7))
```

## Sampling example

```python
from rscopulas import GumbelCopula

model = GumbelCopula.from_params(2, 2.1)
samples = model.sample(5, seed=11)

print("family:", model.family)
print("samples:\n", samples)
print("log_pdf:\n", model.log_pdf(samples))
```

## Advanced example: fit an R-vine

```python
import numpy as np
from rscopulas import VineCopula

data = np.array(
    [
        [0.10, 0.14, 0.18, 0.22],
        [0.18, 0.21, 0.24, 0.27],
        [0.24, 0.29, 0.33, 0.31],
        [0.33, 0.30, 0.38, 0.41],
        [0.47, 0.45, 0.43, 0.49],
        [0.52, 0.58, 0.55, 0.53],
        [0.69, 0.63, 0.67, 0.71],
        [0.81, 0.78, 0.74, 0.76],
    ],
    dtype=np.float64,
)

fit = VineCopula.fit_r(
    data,
    family_set=["independence", "gaussian", "clayton", "frank", "gumbel"],
    truncation_level=2,
    max_iter=200,
)

print("structure kind:", fit.model.structure_kind)
print("order:", fit.model.order)
print("first-tree families:", [edge.family for edge in fit.model.trees[0].edges])
print("sample:\n", fit.model.sample(4, seed=13))
```

## Project links

- Source: https://github.com/alexwox/rscopulas
- Documentation: https://github.com/alexwox/rscopulas/tree/master/docs
- Issue tracker: https://github.com/alexwox/rscopulas/issues
