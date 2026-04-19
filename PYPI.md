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

## Project links

- Source: https://github.com/alexwox/rscopulas
- Documentation: https://github.com/alexwox/rscopulas/tree/master/docs
- Issue tracker: https://github.com/alexwox/rscopulas/issues
