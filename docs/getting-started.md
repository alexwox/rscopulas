# Getting started

## Pseudo-observations

`rscopulas` expects data in **pseudo-observation** form: a matrix `U` with entries strictly in `(0, 1)` and finite.

- The library does **not** estimate marginal distributions.
- If you have raw `X`, transform each margin to uniforms (parametric, empirical, or rank-based) first, then call the API.

Invalid values are rejected (e.g. `PseudoObs::new` in Rust; Python raises mapped errors).

## Python

Install from PyPI:

```bash
python -m pip install -U rscopulas
```

For local development from this repository:

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,viz]"
maturin develop --manifest-path crates/rscopulas-python/Cargo.toml
pytest
```

Minimal fit:

```python
import numpy as np
from rscopulas import GaussianCopula

data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64)
result = GaussianCopula.fit(data)
print(result.diagnostics.aic)
print(result.model.sample(5, seed=1))
```

See [python.md](python.md) and [examples.md](examples.md).

## Rust

Add `rscopulas` to your crate (path while working inside this workspace):

```toml
rscopulas = { path = "crates/rscopulas-core" }
ndarray = "0.17"
rand = "0.9"
```

Minimal fit:

```rust
use ndarray::array;
use rscopulas::{CopulaModel, FitOptions, GaussianCopula, PseudoObs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = PseudoObs::new(array![[0.1_f64, 0.2], [0.3, 0.4], [0.5, 0.6]])?;
    let fit = GaussianCopula::fit(&data, &FitOptions::default())?;
    println!("AIC: {}", fit.diagnostics.aic);
    Ok(())
}
```

Run the shipped examples from the `rscopulas` workspace crate:

```bash
cargo run -p rscopulas --example quickstart_gaussian
```

See [rust.md](rust.md).

## Next steps

- Vines: [vines.md](vines.md)
- Pair copulas and Khoudraji: [pair-copulas.md](pair-copulas.md), [khoudraji.md](khoudraji.md)
- Trust / coverage: [specs.md](specs.md)
- Performance: [benchmarks.md](benchmarks.md)
