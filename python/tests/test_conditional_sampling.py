"""Tests for the Rosenblatt transforms and `sample_conditional` on VineCopula."""

from __future__ import annotations

import numpy as np
import pytest

from rscopulas import NonPrefixConditioningError, VineCopula


def _pseudo_obs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Build a synthetic pseudo-obs matrix with real dependence."""
    z = rng.standard_normal((n, d))
    # Chain z[0] -> z[1] -> z[2] -> ... so every pair has a nonzero tau.
    for j in range(1, d):
        z[:, j] += 0.55 * z[:, j - 1]
    ranks = np.argsort(np.argsort(z, axis=0), axis=0) + 1
    return ranks / (n + 1)


@pytest.fixture(scope="module")
def mixed_vine() -> VineCopula:
    rng = np.random.default_rng(2026)
    u = _pseudo_obs(400, 5, rng)
    # Fit a full R-vine over the mixed family set (default).
    return VineCopula.fit_r(
        u,
        family_set=["gaussian", "clayton", "frank", "gumbel"],
        include_rotations=False,
    ).model


@pytest.fixture(scope="module")
def anchored_c_vine() -> tuple[VineCopula, int]:
    """C-vine whose Rosenblatt anchor is a known column.

    For fit_c, ``variable_order[0] == order[-1]``, so placing the target
    column at the end of ``order`` pins it to the anchor position.
    """
    rng = np.random.default_rng(7)
    d = 5
    u = _pseudo_obs(500, d, rng)
    anchor = 2  # arbitrary user-chosen column
    remainder = [c for c in range(d) if c != anchor]
    order = [*remainder, anchor]
    model = VineCopula.fit_c(
        u,
        order=order,
        family_set=["gaussian", "frank"],
        include_rotations=False,
    ).model
    return model, anchor


@pytest.fixture(scope="module")
def anchored_d_vine() -> tuple[VineCopula, list[int]]:
    """D-vine whose first two Rosenblatt positions are pinned.

    For fit_d, ``variable_order == list(reversed(order))``, so the last two
    elements of ``order`` become the first two Rosenblatt positions (in
    reversed order).
    """
    rng = np.random.default_rng(99)
    d = 5
    u = _pseudo_obs(500, d, rng)
    # Pin variable_order[0] = 3, variable_order[1] = 1 by using order[-1]=3, order[-2]=1.
    order = [0, 2, 4, 1, 3]
    model = VineCopula.fit_d(
        u,
        order=order,
        family_set=["gaussian", "frank"],
        include_rotations=False,
    ).model
    return model, [3, 1]


def test_variable_order_is_list_of_ints(mixed_vine: VineCopula) -> None:
    vo = mixed_vine.variable_order
    assert isinstance(vo, list)
    assert len(vo) == mixed_vine.dim
    assert set(vo) == set(range(mixed_vine.dim))
    assert all(isinstance(idx, int) for idx in vo)


def test_round_trip_v_to_u_to_v(mixed_vine: VineCopula) -> None:
    v = mixed_vine.sample(200, seed=42)
    u = mixed_vine.rosenblatt(v)
    v_back = mixed_vine.inverse_rosenblatt(u)
    np.testing.assert_allclose(v, v_back, atol=1e-8)


def test_round_trip_u_to_v_to_u(mixed_vine: VineCopula) -> None:
    rng = np.random.default_rng(77)
    u = rng.uniform(1e-6, 1 - 1e-6, size=(200, mixed_vine.dim))
    v = mixed_vine.inverse_rosenblatt(u)
    u_back = mixed_vine.rosenblatt(v)
    np.testing.assert_allclose(u, u_back, atol=1e-8)


def test_rosenblatt_shapes(mixed_vine: VineCopula) -> None:
    v = mixed_vine.sample(50, seed=1)
    u = mixed_vine.rosenblatt(v)
    assert u.shape == v.shape

    u2 = mixed_vine.inverse_rosenblatt(u)
    assert u2.shape == v.shape


def test_rosenblatt_rejects_wrong_dim(mixed_vine: VineCopula) -> None:
    bad = np.zeros((10, mixed_vine.dim + 1))
    with pytest.raises(Exception):
        mixed_vine.rosenblatt(bad)
    with pytest.raises(Exception):
        mixed_vine.inverse_rosenblatt(bad)


# --- sample_conditional ------------------------------------------------------


def test_sample_conditional_k1_exact_match(anchored_c_vine) -> None:
    vine, anchor = anchored_c_vine
    assert vine.variable_order[0] == anchor

    rng = np.random.default_rng(0)
    x = rng.uniform(0.05, 0.95, size=300)
    v = vine.sample_conditional({anchor: x}, n=300, seed=123)

    assert v.shape == (300, vine.dim)
    # The k == 1 fast path writes `np.clip(x, eps, 1-eps)` directly into U[:, anchor]
    # and the Rosenblatt anchor's initial uniform propagates through untouched,
    # so the V column for the anchor equals the clipped input exactly.
    expected = np.clip(x, 1e-12, 1.0 - 1e-12)
    np.testing.assert_array_equal(v[:, anchor], expected)


def test_sample_conditional_k2_round_trip_exact(anchored_d_vine) -> None:
    vine, prefix = anchored_d_vine
    assert vine.variable_order[: len(prefix)] == prefix

    rng = np.random.default_rng(1)
    x0 = rng.uniform(0.05, 0.95, size=200)
    x1 = rng.uniform(0.05, 0.95, size=200)
    v = vine.sample_conditional({prefix[0]: x0, prefix[1]: x1}, n=200, seed=456)

    assert v.shape == (200, vine.dim)
    # The k >= 2 path uses rosenblatt_prefix + inverse_rosenblatt, so the
    # round-trip is tight but not bit-exact. 1e-6 is well within the
    # Gaussian/Frank precision bound.
    np.testing.assert_allclose(
        v[:, prefix[0]], np.clip(x0, 1e-12, 1 - 1e-12), atol=1e-8
    )
    np.testing.assert_allclose(
        v[:, prefix[1]], np.clip(x1, 1e-12, 1 - 1e-12), atol=1e-8
    )


def test_sample_conditional_non_prefix_raises(anchored_c_vine) -> None:
    vine, anchor = anchored_c_vine
    # Pick a column that is NOT variable_order[0].
    non_anchor = next(c for c in range(vine.dim) if c != anchor)

    values = np.full(50, 0.5)
    with pytest.raises(NonPrefixConditioningError) as excinfo:
        vine.sample_conditional({non_anchor: values}, n=50, seed=0)

    message = str(excinfo.value)
    assert "variable_order" in message
    assert "fit_c" in message or "fit_d" in message


def test_sample_conditional_seed_determinism(anchored_c_vine) -> None:
    vine, anchor = anchored_c_vine
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 0.9, size=100)
    first = vine.sample_conditional({anchor: x}, n=100, seed=2026)
    second = vine.sample_conditional({anchor: x}, n=100, seed=2026)
    np.testing.assert_array_equal(first, second)

    # Different seed must produce a different trajectory on at least the
    # non-conditioned columns.
    third = vine.sample_conditional({anchor: x}, n=100, seed=2027)
    free_cols = [c for c in range(vine.dim) if c != anchor]
    assert not np.array_equal(first[:, free_cols], third[:, free_cols])


def test_sample_conditional_rejects_wrong_length(anchored_c_vine) -> None:
    vine, anchor = anchored_c_vine
    values = np.full(7, 0.5)
    with pytest.raises(ValueError):
        vine.sample_conditional({anchor: values}, n=8, seed=0)


def test_sample_conditional_rejects_empty_known(anchored_c_vine) -> None:
    vine, _ = anchored_c_vine
    with pytest.raises(ValueError):
        vine.sample_conditional({}, n=8, seed=0)


# --- marginal sanity check ---------------------------------------------------


def test_sample_conditional_narrow_band_preserves_conditional_distribution(
    anchored_c_vine,
) -> None:
    """Narrow-band sanity: conditional samples drawn with the anchor pinned
    inside a tight interval should, restricted to the non-conditioned
    columns, be close to the unconditional sample restricted to observations
    where the anchor happens to fall in the same band.

    This is a weak property but catches gross directional bugs (e.g. wiring
    the anchor uniform into the wrong column of the workspace).
    """
    stats = pytest.importorskip("scipy.stats")

    vine, anchor = anchored_c_vine
    free_cols = [c for c in range(vine.dim) if c != anchor]

    # Unconditional reference: keep only the samples whose anchor falls in a
    # narrow interval.
    big = vine.sample(20_000, seed=314)
    lo, hi = 0.55, 0.65
    mask = (big[:, anchor] > lo) & (big[:, anchor] < hi)
    reference = big[mask][:, free_cols]
    if reference.shape[0] < 500:
        pytest.skip(
            f"unconditional reference has only {reference.shape[0]} obs in the band"
        )

    # Conditional: pin the anchor to a grid of values in the same band.
    rng = np.random.default_rng(271828)
    x = rng.uniform(lo, hi, size=reference.shape[0])
    conditional = vine.sample_conditional(
        {anchor: x}, n=reference.shape[0], seed=16180
    )[:, free_cols]

    # Compare column-wise distributions via two-sample KS. We require each
    # column's p-value above a permissive threshold; a genuine bug in the
    # wiring would drive p below 1e-6 reliably.
    for idx in range(len(free_cols)):
        _, pvalue = stats.ks_2samp(reference[:, idx], conditional[:, idx])
        assert pvalue > 1e-3, (
            f"free column {free_cols[idx]} conditional distribution drifts "
            f"from unconditional-in-band (p={pvalue:e})"
        )
