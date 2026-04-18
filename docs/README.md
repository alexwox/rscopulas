# rscopulas documentation

Start here for guides beyond the root [README.md](../README.md).

## Guides

| Document | Contents |
|----------|----------|
| [getting-started.md](getting-started.md) | Pseudo-observations, install, first fit in Python and Rust |
| [python.md](python.md) | Python API, plotting, caveats |
| [rust.md](rust.md) | Rust crate, traits, `ExecPolicy`, development tips |
| [vines.md](vines.md) | C-vine, D-vine, R-vine usage |
| [pair-copulas.md](pair-copulas.md) | Pair kernels, rotations, selection |
| [khoudraji.md](khoudraji.md) | Asymmetric Khoudraji pair copulas |
| [benchmarks.md](benchmarks.md) | Cross-language harness vs Criterion |
| [specs.md](specs.md) | What is validated, reference sources, scope |
| [examples.md](examples.md) | Index of runnable example scripts |

## Repository map

- Rust library: `crates/rscopulas-core`
- Python package: `python/rscopulas`
- Reference JSON: `fixtures/reference/`
- Benchmark orchestration: `benchmarks/`
- R fixture generators: `scripts/reference/`
