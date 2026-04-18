# Examples index

## Python (`python/examples/`)

| Script | Purpose |
|--------|---------|
| [quickstart.py](../python/examples/quickstart.py) | Gaussian + R-vine fit demo (CLI) |
| [copula_visualisation.py](../python/examples/copula_visualisation.py) | Multi-panel figure → `python/examples/output/copula_visualisation.png` |
| [copula_gallery.py](../python/examples/copula_gallery.py) | One PNG per model kind → `python/examples/output/gallery_*.png` |

Run from repo root after `pip install -e ".[viz]"` and `maturin develop`:

```bash
PYTHONPATH=python python python/examples/quickstart.py
PYTHONPATH=python python python/examples/copula_visualisation.py
PYTHONPATH=python python python/examples/copula_gallery.py
```

**Gallery outputs:** PNGs under `python/examples/output/` are **checked in** as documentation assets. Regenerate when plotting or model behavior changes.

## Rust (`crates/rscopulas-core/examples/`)

| Example | Command |
|---------|---------|
| `quickstart_gaussian` | `cargo run -p rscopulas-core --example quickstart_gaussian` |
| `vine_r_vine_fit` | `cargo run -p rscopulas-core --example vine_r_vine_fit` |
| `pair_copula_clayton` | `cargo run -p rscopulas-core --example pair_copula_clayton` |
| `khoudraji_pair` | `cargo run -p rscopulas-core --example khoudraji_pair` |
| `benchmark_runner` | Built for the cross-language harness (see [benchmarks.md](benchmarks.md)) |

## Benchmarks

See [benchmarks.md](benchmarks.md) and [benchmarks/README.md](../benchmarks/README.md).
