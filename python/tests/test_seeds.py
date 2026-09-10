"""Seed and sample-count validation, plus NumPy-independent reproducibility."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rscopulas import (
    ClaytonCopula,
    FactorCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    HierarchicalArchimedeanCopula,
    InvalidInputError,
    StudentTCopula,
    VineCopula,
)
from rscopulas._rscopulas import uniform_matrix

CORR = [[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]]

MODELS = [
    GaussianCopula.from_params(CORR),
    StudentTCopula.from_params(CORR, 4.0),
    ClaytonCopula.from_params(3, 1.5),
    FrankCopula.from_params(3, 2.0),
    GumbelCopula.from_params(3, 1.4),
    HierarchicalArchimedeanCopula.from_tree(
        {"family": "gumbel", "theta": 1.4, "children": [0, 1, 2]}
    ),
    FactorCopula.from_links([{"family": "gaussian", "parameters": [0.6]}] * 3),
]
MODEL_IDS = [type(model).__name__ for model in MODELS]


def _pseudo_obs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal((n, d))
    for j in range(1, d):
        z[:, j] += 0.55 * z[:, j - 1]
    ranks = np.argsort(np.argsort(z, axis=0), axis=0) + 1
    return ranks / (n + 1)


@pytest.fixture(scope="module")
def anchored_vine() -> tuple[VineCopula, int]:
    rng = np.random.default_rng(11)
    u = _pseudo_obs(300, 4, rng)
    anchor = 1
    model = VineCopula.fit_c(
        u, order=[0, 2, 3, anchor], family_set=["gaussian", "frank"], include_rotations=False
    ).model
    assert model.variable_order[0] == anchor
    return model, anchor


@pytest.fixture(scope="module")
def two_pinned_vine() -> tuple[VineCopula, list[int]]:
    rng = np.random.default_rng(12)
    u = _pseudo_obs(300, 4, rng)
    model = VineCopula.fit_d(
        u, order=[0, 2, 3, 1], family_set=["gaussian", "frank"], include_rotations=False
    ).model
    return model, model.variable_order[:2]


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_sample_seed_is_reproducible_and_independent_of_numpy(model: Any) -> None:
    first = model.sample(16, seed=2024)
    np.random.seed(1)
    np.random.default_rng(2).uniform(size=8)
    second = model.sample(16, seed=2024)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, model.sample(16, seed=2025))


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
@pytest.mark.parametrize("seed", [-1, 2**64, np.int64(-5)], ids=["negative", "too_large", "numpy_negative"])
def test_out_of_range_seed_raises_invalid_input(model: Any, seed: Any) -> None:
    with pytest.raises(InvalidInputError, match="seed"):
        model.sample(4, seed=seed)


def test_non_integer_seed_raises_type_error() -> None:
    model = MODELS[0]
    with pytest.raises(TypeError, match="seed"):
        model.sample(4, seed=1.5)
    with pytest.raises(TypeError, match="seed"):
        model.sample(4, seed="7")


def test_numpy_integer_seeds_are_accepted() -> None:
    model = MODELS[0]
    np.testing.assert_array_equal(model.sample(4, seed=np.uint64(7)), model.sample(4, seed=7))
    np.testing.assert_array_equal(model.sample(4, seed=np.int32(7)), model.sample(4, seed=7))


@pytest.mark.parametrize("bad_n", [5.0, "5", True, None], ids=["float", "str", "bool", "none"])
def test_non_integer_n_raises_type_error(bad_n: Any) -> None:
    with pytest.raises(TypeError, match="n must be an int"):
        MODELS[0].sample(bad_n)


@pytest.mark.parametrize("bad_n", [0, -3])
def test_non_positive_n_raises_invalid_input(bad_n: int) -> None:
    with pytest.raises(InvalidInputError, match="positive"):
        MODELS[0].sample(bad_n)


def test_numpy_integer_n_is_accepted() -> None:
    assert MODELS[0].sample(np.int64(4), seed=1).shape == (4, 3)


def test_sample_conditional_validates_n_and_seed(anchored_vine: tuple[VineCopula, int]) -> None:
    vine, anchor = anchored_vine
    x = np.full(5, 0.5)
    with pytest.raises(TypeError, match="n must be an int"):
        vine.sample_conditional({anchor: x}, n=5.0, seed=0)
    with pytest.raises(InvalidInputError, match="positive"):
        vine.sample_conditional({anchor: np.zeros(0)}, n=0, seed=0)
    with pytest.raises(InvalidInputError, match="seed"):
        vine.sample_conditional({anchor: x}, n=5, seed=-1)
    with pytest.raises(InvalidInputError, match="seed"):
        vine.sample_conditional({anchor: x}, n=5, seed=2**64)
    with pytest.raises(TypeError, match="seed"):
        vine.sample_conditional({anchor: x}, n=5, seed=0.5)


def test_vine_fit_rng_seed_is_validated() -> None:
    data = GaussianCopula.from_params(CORR).sample(120, seed=3)
    with pytest.raises(InvalidInputError, match="seed"):
        VineCopula.fit_r(data, family_set=["gaussian"], rng_seed=-1)
    with pytest.raises(InvalidInputError, match="seed"):
        VineCopula.fit_c(data, family_set=["gaussian"], rng_seed=2**64)
    with pytest.raises(InvalidInputError, match="seed"):
        VineCopula.fit_d(data, family_set=["gaussian"], rng_seed=-7)
    assert VineCopula.fit_r(data, family_set=["gaussian"], rng_seed=np.uint64(3)).model.dim == 3


def test_sample_conditional_k1_is_independent_of_numpy_state(
    anchored_vine: tuple[VineCopula, int],
) -> None:
    vine, anchor = anchored_vine
    x = np.linspace(0.05, 0.95, 40)
    first = vine.sample_conditional({anchor: x}, n=40, seed=99)
    np.random.seed(123)
    np.random.default_rng(5).uniform(size=(40, 3))
    second = vine.sample_conditional({anchor: x}, n=40, seed=99)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, anchor], x)
    assert not np.array_equal(first, vine.sample_conditional({anchor: x}, n=40, seed=100))


def test_sample_conditional_k2_is_independent_of_numpy_state(
    two_pinned_vine: tuple[VineCopula, list[int]],
) -> None:
    vine, prefix = two_pinned_vine
    x0 = np.linspace(0.1, 0.9, 30)
    x1 = np.linspace(0.9, 0.1, 30)
    known = {prefix[0]: x0, prefix[1]: x1}
    first = vine.sample_conditional(known, n=30, seed=7)
    np.random.seed(0)
    second = vine.sample_conditional(known, n=30, seed=7)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[:, prefix[0]], x0, atol=1e-8)
    np.testing.assert_allclose(first[:, prefix[1]], x1, atol=1e-8)


def test_sample_conditional_free_columns_come_from_rust_uniforms(
    anchored_vine: tuple[VineCopula, int],
) -> None:
    """The free columns are exactly inverse_rosenblatt(U) with U drawn by uniform_matrix."""
    vine, anchor = anchored_vine
    n = 25
    x = np.linspace(0.2, 0.8, n)
    u = np.empty((n, vine.dim))
    u[:, anchor] = x
    u[:, vine.variable_order[1:]] = uniform_matrix(n, vine.dim - 1, 31)
    expected = vine.inverse_rosenblatt(u)
    np.testing.assert_array_equal(vine.sample_conditional({anchor: x}, n=n, seed=31), expected)


def test_uniform_matrix_contract() -> None:
    first = uniform_matrix(6, 3, 5)
    np.testing.assert_array_equal(first, uniform_matrix(6, 3, seed=5))
    assert first.shape == (6, 3)
    assert first.dtype == np.float64
    assert np.all(first > 0.0) and np.all(first < 1.0)
    assert not np.array_equal(first, uniform_matrix(6, 3, 6))
    assert uniform_matrix(4, 0, 1).shape == (4, 0)
    assert uniform_matrix(4, 2).shape == (4, 2)
    with pytest.raises(InvalidInputError, match="positive"):
        uniform_matrix(0, 3, 1)
    with pytest.raises(InvalidInputError, match="seed"):
        uniform_matrix(2, 2, -1)
