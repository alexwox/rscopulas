# Pair copulas

The pair-copula layer provides **bivariate** density evaluation, **conditional** distributions (h-functions), and **inverses** where implemented — the building blocks for vines and for direct two-dimensional modeling.

## Families

Supported kernels include independence, Gaussian, Student t, Clayton, Frank, and Gumbel (with **rotations** where applicable). See `PairCopulaFamily` and `PairCopulaSpec` in Rust.

## Construction (Rust)

`PairCopulaSpec` holds `family`, `rotation`, and `PairCopulaParams` (one- or two-parameter forms depending on family).

Typical calls:

- `log_pdf(u1, u2, clip_eps)`
- `cond_first_given_second`, `cond_second_given_first`
- `inv_first_given_second`, `inv_second_given_first` (where defined)

## Construction (Python)

`PairCopula` wraps the native spec. Use `from_family` / family-specific constructors or `from_khoudraji` for asymmetric Khoudraji copulas ([khoudraji.md](khoudraji.md)).

## Pair selection

`fit_pair_copula` (Rust) performs AIC/BIC-driven selection over a candidate set. Vines use analogous logic per edge subject to `VineFitOptions`.

## Reference tests

Pair fixtures under `fixtures/reference/vinecopula/v2/` and R `copula` JSON under `fixtures/reference/r-copula/v1_1_3/` are exercised in `crates/rscopulas-core/tests/reference_paircopula*.rs` and related modules.
