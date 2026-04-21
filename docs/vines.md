# Vine copulas

`rscopulas` supports **C-vine**, **D-vine**, and **R-vine** structures: construction from known parameters (e.g. Gaussian vines), and **mixed pair-copula fitting** from pseudo-observations.

## When to use which

- **C-vine / D-vine** — Tractable when a natural ordering exists (e.g. root variable for C-vine).
- **R-vine** — General regular vine; structure is selected during fitting (along with pair families on edges).

## API entry points (Rust)

- Fit: `VineCopula::fit_c_vine`, `fit_d_vine`, `fit_r_vine` with `VineFitOptions`
- Fit with an explicit variable order: `VineCopula::fit_c_vine_with_order`, `fit_d_vine_with_order`
- Gaussian shortcuts: `VineCopula::gaussian_c_vine`, `gaussian_d_vine`
- Explicit trees: `VineCopula::from_trees`
- Rosenblatt transforms: `VineCopula::rosenblatt`, `inverse_rosenblatt`, `rosenblatt_prefix`, `variable_order`

## API entry points (Python)

- `VineCopula.fit_c`, `fit_d`, `fit_r` with `family_set`, `criterion`, `truncation_level`, etc.
- `VineCopula.fit_c` / `fit_d` also accept `order=[...]` to pin the variable order (used to set up conditional sampling).
- `VineCopula.gaussian_c_vine`, `gaussian_d_vine`
- `VineCopula.rosenblatt`, `inverse_rosenblatt`, `variable_order`, `sample_conditional`

## Pair families on edges

Candidate families include independence, Gaussian, Student t, Clayton, Frank, Gumbel, and **Khoudraji** (asymmetric). Rotations apply where supported for classical families.

Asymmetric families (e.g. Khoudraji) affect edge likelihood and structure selection; treat fitted structures as tied to the chosen `family_set` and options.

## Inspecting a fitted vine

Python exposes `structure_kind`, `order`, `variable_order`, `pair_parameters`, `structure_info`, `trees`. Rust exposes analogous accessors on `VineCopula`.

For visualization of structure in Python, use `rscopulas.plotting.plot_vine_structure`.

## Conditional sampling

A fitted vine is built from pair-copula h-functions and their inverses, so it
natively supports **conditional simulation**: given observed values on a subset
of columns, draw the remaining columns from the vine's conditional law without
importance reweighting, Metropolis acceptance, or any other sampler tax. The
tail dependence encoded in each pair copula propagates through exactly.

The primitives are the two Rosenblatt transforms — the forward map
`U = F(V)` that turns vine-distributed data into independent uniforms, and the
inverse map `V = F^{-1}(U)` that does the opposite. Unconditional sampling is
just `inverse_rosenblatt` of a uniform matrix, and conditional sampling pins
a prefix of the uniforms to match known values.

### The `variable_order` convention

Every fitted vine has a diagonal order `variable_order` giving the sequence in
which the Rosenblatt chain emits variables. `variable_order[0]` is the
**Rosenblatt anchor**: its initial uniform passes through unchanged, so
`V[:, variable_order[0]]` equals the anchor uniform bit-for-bit.

For canonical C- and D-vines the relationship to the user-supplied `order` is
the same:

```text
variable_order[0] == order[-1]
```

So to pin column `X` at the anchor, place it at the **end** of `order`:

```python
from rscopulas import VineCopula

fit = VineCopula.fit_c(u, order=[*other_cols, X], family_set=["gaussian", "frank"])
vine = fit.model
assert vine.variable_order[0] == X
```

For a D-vine `variable_order` is exactly `list(reversed(order))`; for a C-vine
the remaining positions are permuted by the structure-matrix construction —
inspect `vine.variable_order` after fitting if you care about positions beyond
the anchor. R-vines (`fit_r`) pick their own structure and do not accept an
`order=` argument; if you need conditional sampling, fit a C- or D-vine with
an explicit `order`.

### `sample_conditional`

```python
vine.sample_conditional(
    known={X: u_series_on_X},   # 1D array of length n, values in (0, 1)
    n=10_000,
    seed=2026,
)
```

Returns an `(n, dim)` matrix in original variable-label order. The column
for `X` equals `np.clip(u_series_on_X, eps, 1 - eps)` exactly when `X` is the
anchor (the single-variable fast path is bit-exact). For `k >= 2` known
columns, the returned values for the known columns match the input up to
~`1e-8` drift.

Requirements:

- `known` must supply a *diagonal prefix* of `variable_order`: its keys must
  equal `set(variable_order[:k])` for some `k`. Otherwise the method raises
  `rscopulas.NonPrefixConditioningError`.
- Set up the prefix at fit time with `fit_c(order=[...])` /
  `fit_d(order=[...])`: to condition on `[Y, X]` (in that Rosenblatt order)
  place them at the end of `order` so that
  `variable_order[:2] == [Y, X]`.

### Rosenblatt / inverse Rosenblatt directly

Both transforms accept and return `(n, dim)` NumPy arrays indexed by the
original variable label. They round-trip up to `1e-8`:

```python
V = vine.sample(1_000, seed=42)
U = vine.rosenblatt(V)              # independent uniforms, same column layout
V_back = vine.inverse_rosenblatt(U) # exact inverse
```

Rust users get the same methods on `VineCopula` with `ArrayView2<f64>` inputs.

### What about R-vines with an arbitrary conditioning subset?

The Rosenblatt chain only yields the right conditional law when the known
columns sit at the diagonal prefix. General-subset conditioning on an
arbitrary fitted R-vine requires either MCMC/rejection (approximate, with
finite ESS) or re-fitting with an appropriate structure. This library only
supports the exact path: re-fit with `fit_c` / `fit_d` and an `order` that
places the conditioning variables at the end.

## Further reading

- Reference tests: `crates/rscopulas-core/tests/reference_vine*.rs`,
  `crates/rscopulas-core/tests/rosenblatt_transforms.rs`
- Python tests: `python/tests/test_conditional_sampling.py`
- Fixtures: `fixtures/reference/vinecopula/v2/`
