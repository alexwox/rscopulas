# rscopulas-core

`rscopulas-core` is the primary Rust crate in the `rscopulas` workspace. It provides validated pseudo-observation handling, single-family and pair-copula models, and C-vine, D-vine, and R-vine fitting, evaluation, and sampling.

## Add the crate

```toml
[dependencies]
rscopulas-core = "0.1.1"
```

## What it includes

- `PseudoObs` for validated data in `(0, 1)^d`
- Gaussian, Student t, Clayton, Frank, and Gumbel copulas
- Pair copulas, including Khoudraji constructions
- Vine copulas with fitting and diagnostics
- Explicit execution-policy controls via `ExecPolicy` and `Device`

## Links

- Repository: https://github.com/alexwox/rscopulas
- Docs: https://docs.rs/rscopulas-core
