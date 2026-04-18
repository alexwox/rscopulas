# Khoudraji pair copulas

**Khoudraji** copulas are **asymmetric** bivariate copulas formed from two existing pair-copula kernels and two **shape** parameters. They can appear as **vine edges** and are exposed as a dedicated pair family.

## Motivation

Classical symmetric Archimedean or elliptical pairs cannot capture all dependence asymmetries. Khoudraji’s construction mixes two base copulas with power transforms of the uniforms, producing flexible skewed dependence.

## Rust

- Build a spec: `PairCopulaSpec::khoudraji(first, second, shape_first, shape_second)?` where `first` and `second` are `PairCopulaSpec` instances for supported base families.
- Include in vine fitting: add `PairCopulaFamily::Khoudraji` to `VineFitOptions::family_set`.

Khoudraji edge fitting uses a **bounded internal search** over supported base families and is **CPU-only** in current implementations.

## Python

- `PairCopula.from_khoudraji(base_family_1, base_family_2, shape_1, shape_2, first_parameters, second_parameters, ...)`
- Vine `family_set` may include `"khoudraji"`.

## Validation and references

- JSON fixtures: `fixtures/reference/r-copula/v1_1_3/khoudraji_*.json`
- Regenerator: `scripts/reference/generate_khoudraji_fixtures.R`
- Rust tests: `crates/rscopulas-core/tests/reference_khoudraji.rs`

## Benchmarks

The cross-language manifest includes `pair_khoudraji_kernels` in `benchmarks/cases.json` (pair density and h/hinv-style kernel workload vs R/Python). See [benchmarks.md](benchmarks.md).
