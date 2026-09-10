"""Tests for the rank-based pseudo-observation helper."""

from __future__ import annotations

import numpy as np
import pytest

from rscopulas import GaussianCopula, InvalidInputError, to_pseudo_obs


def test_known_ranks_without_ties() -> None:
    np.testing.assert_allclose(to_pseudo_obs([3.0, 1.0, 2.0]), [0.75, 0.25, 0.5])


@pytest.mark.parametrize(
    ("ties", "expected_ranks"),
    [
        ("average", [1.0, 2.5, 2.5, 4.0]),
        ("min", [1.0, 2.0, 2.0, 4.0]),
        ("max", [1.0, 3.0, 3.0, 4.0]),
        ("ordinal", [1.0, 2.0, 3.0, 4.0]),
    ],
)
def test_tie_methods_match_known_ranks(ties: str, expected_ranks: list[float]) -> None:
    x = np.array([1.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(to_pseudo_obs(x, ties=ties), np.array(expected_ranks) / 5.0)


def test_ordinal_breaks_ties_by_order_of_appearance() -> None:
    x = np.array([2.0, 1.0, 2.0, 2.0])
    np.testing.assert_allclose(to_pseudo_obs(x, ties="ordinal") * 5.0, [2.0, 1.0, 3.0, 4.0])


def test_all_equal_column() -> None:
    np.testing.assert_allclose(to_pseudo_obs([7.0, 7.0, 7.0]), [0.5, 0.5, 0.5])
    np.testing.assert_allclose(to_pseudo_obs([7.0, 7.0, 7.0], ties="ordinal"), [0.25, 0.5, 0.75])


def test_scaling_n_uses_hazen_positions() -> None:
    x = np.array([10.0, 30.0, 20.0])
    np.testing.assert_allclose(to_pseudo_obs(x, scaling="n"), [0.5 / 3, 2.5 / 3, 1.5 / 3])


def test_matrix_input_ranks_each_column_independently() -> None:
    x = np.array([[3.0, 10.0], [1.0, 30.0], [2.0, 20.0]])
    np.testing.assert_allclose(to_pseudo_obs(x), [[0.75, 0.25], [0.25, 0.75], [0.5, 0.5]])


def test_output_shape_dtype_and_open_interval() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 3))
    x[:10, 0] = 1.0  # ties
    u = to_pseudo_obs(x)
    assert u.shape == x.shape
    assert u.dtype == np.float64
    assert np.all(u > 0.0) and np.all(u < 1.0)
    assert to_pseudo_obs(x[:, 0]).shape == (50,)
    assert to_pseudo_obs(x[:1]).tolist() == [[0.5, 0.5, 0.5]]


def test_accepts_lists_and_integers() -> None:
    np.testing.assert_allclose(to_pseudo_obs([[3, 1], [1, 3]]), [[2 / 3, 1 / 3], [1 / 3, 2 / 3]])


def test_infinite_values_rank_at_the_ends() -> None:
    np.testing.assert_allclose(to_pseudo_obs([np.inf, 0.0, -np.inf]), [0.75, 0.5, 0.25])


def test_nan_is_rejected_with_its_location() -> None:
    x = np.array([[1.0, 2.0], [np.nan, 3.0]])
    with pytest.raises(InvalidInputError, match="NaN") as excinfo:
        to_pseudo_obs(x)
    assert "row 1, column 0" in str(excinfo.value)
    with pytest.raises(ValueError):
        to_pseudo_obs([1.0, np.nan])


@pytest.mark.parametrize(
    "bad",
    [np.zeros((0, 2)), np.zeros(0), np.zeros((2, 2, 2)), 3.0],
    ids=["empty_matrix", "empty_vector", "3d", "scalar"],
)
def test_bad_shapes_are_rejected(bad: object) -> None:
    with pytest.raises(InvalidInputError):
        to_pseudo_obs(bad)


def test_non_numeric_input_is_rejected() -> None:
    with pytest.raises(InvalidInputError, match="float64"):
        to_pseudo_obs(["a", "b"])


def test_unknown_options_are_rejected() -> None:
    with pytest.raises(InvalidInputError, match="ties"):
        to_pseudo_obs([1.0, 2.0], ties="dense")
    with pytest.raises(InvalidInputError, match="scaling"):
        to_pseudo_obs([1.0, 2.0], scaling="n+2")


def test_matches_scipy_rankdata() -> None:
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(42)
    x = rng.integers(0, 5, size=(200, 4)).astype(np.float64)  # heavy ties
    n = x.shape[0]
    for method in ("average", "min", "max", "ordinal"):
        expected = np.apply_along_axis(
            lambda col: stats.rankdata(col, method=method), 0, x
        ) / (n + 1)
        np.testing.assert_allclose(to_pseudo_obs(x, ties=method), expected)


def test_output_is_valid_model_input() -> None:
    rng = np.random.default_rng(7)
    z = rng.standard_normal((300, 2))
    z[:, 1] += 0.8 * z[:, 0]
    fit = GaussianCopula.fit(to_pseudo_obs(z))
    assert fit.model.correlation[0, 1] > 0.4
