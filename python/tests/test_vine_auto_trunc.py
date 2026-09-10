"""Python-side tests for the new pyvinecopulib-parity kwargs on
``VineCopula.fit_r`` — the mBICV auto-truncation, Spearman / Hoeffding tree
criteria, and Prim / Wilson tree algorithms. Mirrors Rust coverage in
``reference_tree_select.rs``; this file is a smoke test that every kwarg
reaches the Rust side correctly and returns a valid model.
"""

from __future__ import annotations

import numpy as np
import pytest

from rscopulas import GaussianCopula, InvalidInputError, ModelFitError, VineCopula

# Family kernels and default family selection are tested separately.
FAMILIES = ["independence", "gaussian"]


def _weak_chain_sample(dim: int, n: int, seed: int) -> np.ndarray:
    """6-dim DGP with real dependence only in the first three columns; the
    trailing columns are (near-)independent uniforms. Used to exercise
    mBICV auto-truncation."""
    rng = np.random.default_rng(seed)
    anchor = rng.uniform(size=n)
    sample = np.empty((n, dim), dtype=np.float64)
    for col in range(dim):
        noise = rng.uniform(size=n)
        coupling = 0.9 if col < 3 else 0.02
        sample[:, col] = np.clip(coupling * anchor + (1 - coupling) * noise, 1e-6, 1 - 1e-6)
    return sample


def _correlated_sample(dim: int, n: int, seed: int) -> np.ndarray:
    """Full-dependence Gaussian copula sample (all pairs positive τ).
    Useful for checking that the fitter runs end-to-end on every
    tree_algorithm / tree_criterion combination."""
    corr = np.array(
        [[0.6 ** abs(i - j) for j in range(dim)] for i in range(dim)], dtype=np.float64
    )
    return GaussianCopula.from_params(corr).sample(n, seed=seed)


def test_fit_r_mbicv_auto_truncation_drops_weak_trees() -> None:
    data = _weak_chain_sample(6, 1500, seed=7)
    fit = VineCopula.fit_r(data, family_set=FAMILIES, criterion="mbicv", select_trunc_lvl=True)
    assert fit.model.truncation_level is not None
    # Should truncate below the full depth (d-1 = 5).
    assert fit.model.truncation_level < 5
    # Should keep at least the first real tree.
    assert fit.model.truncation_level >= 1


def test_fit_r_mbicv_accepts_custom_psi0() -> None:
    data = _weak_chain_sample(6, 1500, seed=9)
    fit = VineCopula.fit_r(data, family_set=FAMILIES, criterion="mbicv:0.95", select_trunc_lvl=True)
    assert fit.model.truncation_level is not None


def test_fit_r_rejects_invalid_mbicv_psi0() -> None:
    data = _weak_chain_sample(6, 500, seed=11)
    with pytest.raises(InvalidInputError, match="psi0"):
        VineCopula.fit_r(data, family_set=FAMILIES, criterion="mbicv:1.5", select_trunc_lvl=True)


def test_fit_r_with_spearman_rho_criterion() -> None:
    data = _correlated_sample(5, 400, seed=13)
    fit = VineCopula.fit_r(data, family_set=FAMILIES, tree_criterion="rho")
    assert fit.model.dim == 5


def test_fit_r_with_hoeffding_criterion() -> None:
    data = _correlated_sample(5, 400, seed=17)
    fit = VineCopula.fit_r(data, family_set=FAMILIES, tree_criterion="hoeffding")
    assert fit.model.dim == 5


def test_fit_r_prim_algorithm_matches_kruskal() -> None:
    data = _correlated_sample(5, 400, seed=19)
    k = VineCopula.fit_r(data, family_set=FAMILIES, tree_algorithm="kruskal")
    p = VineCopula.fit_r(data, family_set=FAMILIES, tree_algorithm="prim")
    # Same MST objective → loglik within 1%.
    rel_gap = abs(k.diagnostics.loglik - p.diagnostics.loglik) / max(abs(k.diagnostics.loglik), 1.0)
    assert rel_gap < 0.01


def test_fit_r_random_weighted_is_reproducible() -> None:
    data = _correlated_sample(5, 400, seed=23)
    a = VineCopula.fit_r(data, family_set=FAMILIES, tree_algorithm="random_weighted", rng_seed=42)
    b = VineCopula.fit_r(data, family_set=FAMILIES, tree_algorithm="random_weighted", rng_seed=42)
    assert a.diagnostics.loglik == b.diagnostics.loglik


def test_fit_c_rejects_non_kruskal() -> None:
    data = _correlated_sample(4, 200, seed=29)
    with pytest.raises(ModelFitError, match="tree_algorithm"):
        VineCopula.fit_c(data, tree_algorithm="prim")


def test_fit_d_rejects_non_kruskal() -> None:
    data = _correlated_sample(4, 200, seed=31)
    with pytest.raises(ModelFitError, match="tree_algorithm"):
        VineCopula.fit_d(data, tree_algorithm="prim")


def _n_independence_edges(model: VineCopula) -> int:
    return sum(edge.family == "independence" for tree in model.trees for edge in tree.edges)


def test_independence_test_level_reaches_the_fitter() -> None:
    # Fix the C-vine order (order[0] is the first-tree hub) and stop after the
    # first tree so the structure is identical across fits and only the
    # per-edge independence decision moves. Hub 0 is strongly tied to columns
    # 1-2 and nearly independent of columns 3-5.
    data = _weak_chain_sample(6, 800, seed=37)
    order = [0, 1, 2, 3, 4, 5]
    kwargs = dict(family_set=["gaussian", "clayton"], order=order, truncation_level=1)
    plain = VineCopula.fit_c(data, **kwargs).model
    loose = VineCopula.fit_c(data, independence_test_level=0.05, **kwargs).model
    strict = VineCopula.fit_c(data, independence_test_level=1e-6, **kwargs).model
    assert _n_independence_edges(strict) >= _n_independence_edges(loose) >= _n_independence_edges(plain)
    assert _n_independence_edges(strict) >= 1
    # The strongly dependent edges around the anchor must survive the test.
    assert any(edge.family != "independence" for edge in strict.trees[0].edges)

    assert VineCopula.fit_d(data, independence_test_level=0.05, family_set=FAMILIES).model.dim == 6
    assert VineCopula.fit_r(data, independence_test_level=0.05, family_set=FAMILIES).model.dim == 6
    for fitter in (VineCopula.fit_c, VineCopula.fit_d, VineCopula.fit_r):
        with pytest.raises(InvalidInputError, match="independence_test_level"):
            fitter(data, family_set=FAMILIES, independence_test_level=1.0)


CORE_DEFAULT_FAMILIES = {
    "independence", "gaussian", "student_t", "clayton", "frank", "gumbel", "joe", "bb1", "bb7",
}


def test_default_family_set_follows_the_core_default() -> None:
    # `family_set=None` must delegate to `VineFitOptions::default()`; the
    # binding keeps no list of its own, so khoudraji/tll never sneak in.
    data = _correlated_sample(4, 300, seed=41)
    families = set()
    for fitter in (VineCopula.fit_c, VineCopula.fit_d, VineCopula.fit_r):
        model = fitter(data).model
        families |= {edge.family for tree in model.trees for edge in tree.edges}
    assert families
    assert families <= CORE_DEFAULT_FAMILIES
