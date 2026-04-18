# Vine copulas

`rscopulas` supports **C-vine**, **D-vine**, and **R-vine** structures: construction from known parameters (e.g. Gaussian vines), and **mixed pair-copula fitting** from pseudo-observations.

## When to use which

- **C-vine / D-vine** — Tractable when a natural ordering exists (e.g. root variable for C-vine).
- **R-vine** — General regular vine; structure is selected during fitting (along with pair families on edges).

## API entry points (Rust)

- Fit: `VineCopula::fit_c_vine`, `fit_d_vine`, `fit_r_vine` with `VineFitOptions`
- Gaussian shortcuts: `VineCopula::gaussian_c_vine`, `gaussian_d_vine`
- Explicit trees: `VineCopula::from_trees`

## API entry points (Python)

- `VineCopula.fit_c`, `fit_d`, `fit_r` with `family_set`, `criterion`, `truncation_level`, etc.
- `VineCopula.gaussian_c_vine`, `gaussian_d_vine`

## Pair families on edges

Candidate families include independence, Gaussian, Student t, Clayton, Frank, Gumbel, and **Khoudraji** (asymmetric). Rotations apply where supported for classical families.

Asymmetric families (e.g. Khoudraji) affect edge likelihood and structure selection; treat fitted structures as tied to the chosen `family_set` and options.

## Inspecting a fitted vine

Python exposes `structure_kind`, `order`, `pair_parameters`, `structure_info`, `trees`. Rust exposes analogous accessors on `VineCopula`.

For visualization of structure in Python, use `rscopulas.plotting.plot_vine_structure`.

## Further reading

- Reference tests: `crates/rscopulas-core/tests/reference_vine*.rs`
- Fixtures: `fixtures/reference/vinecopula/v2/`
