# Specifications and validation scope

This page summarizes **what the codebase is designed to enforce** and **how it is tested**. It is not a legal warranty; it is a map from claims to evidence in the repo.

## Input contract

- **Domain:** Pseudo-observations only — finite values strictly in `(0, 1)` per dimension.
- **Rust:** `PseudoObs::new` validates shape and bounds.
- **Python:** Equivalent checks in bindings; errors map to Python exceptions.

## Model surface (high level)

| Area | Rust (`rscopulas`) | Python (`rscopulas`) |
|------|-------------------------|----------------------|
| Single-family copulas | Yes | Yes |
| Pair copulas + rotations | Yes | Yes |
| Khoudraji pair copulas | Yes | Yes |
| Vine C/D/R | Yes | Yes |
| HAC | Yes | Yes |
| Explicit `ExecPolicy` / device | Yes | Auto only (today) |

**HAC:** density and sampling are not validated to the same standard for every tree shape; see [hac.md](hac.md) (exchangeable vs composite `log_pdf`, **mixed-family sampling** limitations).

## Fit / diagnostics contract

Fitting returns:

- A **model** handle with family-specific parameters.
- **Diagnostics:** log-likelihood, AIC, BIC, convergence flag, iteration count where applicable.

Exact fields are stable within a release series; see type definitions in Rust and Python wrappers.

## Reference validation

### R package `copula` (1.1-3)

Fixtures under `fixtures/reference/r-copula/v1_1_3/` including single-family, pair, **Khoudraji**, and related cases. Regenerated via scripts in `scripts/reference/` (e.g. `generate_khoudraji_fixtures.R`).

### R package `VineCopula` (2.6.1)

Fixtures under `fixtures/reference/vinecopula/v2/` for pair copulas and vine scenarios.

### Tests

Rust integration tests in `crates/rscopulas-core/tests/reference_*.rs` compare against these JSON fixtures.

Python tests in `python/tests/test_bindings.py` exercise bindings and behavior consistent with the core.

## Benchmark validation

[benchmarks/cases.json](../benchmarks/cases.json) reuses the same fixture paths for cross-language timing. Passing tests do **not** guarantee identical timings across languages; they ground **correctness** on shared inputs.

## Out of scope (today)

- Automatic marginal estimation / pipeline from raw data to pseudo-observations.
- A single “auto pick copula family” high-level facade.
- Published PyPI/crates.io release automation (see project README for dev workflow).

## Versioning note

When bumping versions, regenerate or re-verify reference JSON if numerical algorithms change; CI should run `cargo test` and Python `pytest`.
