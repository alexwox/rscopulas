"""Python-side tests for the new pyvinecopulib-parity kwargs on
``VineCopula.fit_r`` — the mBICV auto-truncation, Spearman / Hoeffding tree
criteria, and Prim / Wilson tree algorithms. Mirrors Rust coverage in
``reference_tree_select.rs``; this file is a smoke test that every kwarg
reaches the Rust side correctly and returns a valid model.
"""

from __future__ import annotations

import numpy as np
import pytest

from rscopulas import VineCopula


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
    rng = np.random.default_rng(seed)
    # Toeplitz-like correlation.
    corr = np.array(
        [[0.6 ** abs(i - j) for j in range(dim)] for i in range(dim)], dtype=np.float64
    )
    z = rng.standard_normal(size=(n, dim))
    # Ballpark Cholesky: multiply by corr triangular factor via np.linalg.cholesky.
    l = np.linalg.cholesky(corr)
    y = z @ l.T
    # Φ CDF → uniform.
    from scipy.special import ndtr  # type: ignore

    return np.clip(ndtr(y), 1e-6, 1 - 1e-6)


def test_fit_r_mbicv_auto_truncation_drops_weak_trees() -> None:
    data = _weak_chain_sample(6, 1500, seed=7)
    fit = VineCopula.fit_r(data, criterion="mbicv", select_trunc_lvl=True)
    assert fit.model.truncation_level is not None
    # Should truncate below the full depth (d-1 = 5).
    assert fit.model.truncation_level < 5
    # Should keep at least the first real tree.
    assert fit.model.truncation_level >= 1


def test_fit_r_mbicv_accepts_custom_psi0() -> None:
    data = _weak_chain_sample(6, 1500, seed=9)
    fit = VineCopula.fit_r(data, criterion="mbicv:0.95", select_trunc_lvl=True)
    assert fit.model.truncation_level is not None


def test_fit_r_rejects_invalid_mbicv_psi0() -> None:
    data = _weak_chain_sample(6, 500, seed=11)
    with pytest.raises(Exception):
        VineCopula.fit_r(data, criterion="mbicv:1.5", select_trunc_lvl=True)


def test_fit_r_with_spearman_rho_criterion() -> None:
    try:
        data = _correlated_sample(5, 400, seed=13)
    except ImportError:
        # SciPy optional; fall back to naive uniform shuffle.
        rng = np.random.default_rng(13)
        data = np.clip(rng.uniform(size=(400, 5)), 1e-6, 1 - 1e-6)
    fit = VineCopula.fit_r(data, tree_criterion="rho")
    assert fit.model.dim == 5


def test_fit_r_with_hoeffding_criterion() -> None:
    try:
        data = _correlated_sample(5, 400, seed=17)
    except ImportError:
        rng = np.random.default_rng(17)
        data = np.clip(rng.uniform(size=(400, 5)), 1e-6, 1 - 1e-6)
    fit = VineCopula.fit_r(data, tree_criterion="hoeffding")
    assert fit.model.dim == 5


def test_fit_r_prim_algorithm_matches_kruskal() -> None:
    try:
        data = _correlated_sample(5, 400, seed=19)
    except ImportError:
        rng = np.random.default_rng(19)
        data = np.clip(rng.uniform(size=(400, 5)), 1e-6, 1 - 1e-6)
    k = VineCopula.fit_r(data, tree_algorithm="kruskal")
    p = VineCopula.fit_r(data, tree_algorithm="prim")
    # Same MST objective → loglik within 1%.
    rel_gap = abs(k.diagnostics.loglik - p.diagnostics.loglik) / max(abs(k.diagnostics.loglik), 1.0)
    assert rel_gap < 0.01


def test_fit_r_random_weighted_is_reproducible() -> None:
    try:
        data = _correlated_sample(5, 400, seed=23)
    except ImportError:
        rng = np.random.default_rng(23)
        data = np.clip(rng.uniform(size=(400, 5)), 1e-6, 1 - 1e-6)
    a = VineCopula.fit_r(data, tree_algorithm="random_weighted", rng_seed=42)
    b = VineCopula.fit_r(data, tree_algorithm="random_weighted", rng_seed=42)
    assert a.diagnostics.loglik == b.diagnostics.loglik


def test_fit_c_rejects_non_kruskal() -> None:
    try:
        data = _correlated_sample(4, 200, seed=29)
    except ImportError:
        rng = np.random.default_rng(29)
        data = np.clip(rng.uniform(size=(200, 4)), 1e-6, 1 - 1e-6)
    with pytest.raises(Exception):
        VineCopula.fit_c(data, tree_algorithm="prim")


def test_fit_d_rejects_non_kruskal() -> None:
    try:
        data = _correlated_sample(4, 200, seed=31)
    except ImportError:
        rng = np.random.default_rng(31)
        data = np.clip(rng.uniform(size=(200, 4)), 1e-6, 1 - 1e-6)
    with pytest.raises(Exception):
        VineCopula.fit_d(data, tree_algorithm="prim")
