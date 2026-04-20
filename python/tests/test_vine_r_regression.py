from __future__ import annotations

import numpy as np
import pytest

from rscopulas import GaussianCopula, InternalError, ModelFitError, VineCopula

FAMILY_SET = ["independence", "gaussian", "clayton", "frank", "gumbel"]


def _star_correlation(dim: int, rho: float = 0.7) -> np.ndarray:
    correlation = np.full((dim, dim), rho * rho, dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)
    for leaf in range(1, dim):
        correlation[0, leaf] = rho
        correlation[leaf, 0] = rho
    return correlation


def _path_correlation(dim: int, rho: float = 0.8) -> np.ndarray:
    correlation = np.zeros((dim, dim), dtype=np.float64)
    for row in range(dim):
        for col in range(dim):
            correlation[row, col] = rho ** abs(row - col)
    return correlation


def _gaussian_data(correlation: np.ndarray, n_obs: int, seed: int) -> np.ndarray:
    model = GaussianCopula.from_params(correlation)
    return model.sample(n_obs, seed=seed)


def _assert_vine_invariants(model: VineCopula, dim: int) -> None:
    assert model.dim == dim
    assert len(model.order) == dim
    assert sorted(model.order) == list(range(dim))
    assert model.structure_info.matrix.shape == (dim, dim)
    assert np.all(model.structure_info.matrix >= 0)
    assert np.all(model.structure_info.matrix < dim)


@pytest.mark.parametrize("dim", [4, 5, 6, 7])
@pytest.mark.parametrize(
    ("topology", "builder"),
    [("path", _path_correlation), ("star", _star_correlation)],
)
def test_vine_fitters_preserve_dimension_and_order(dim: int, topology: str, builder) -> None:
    data = _gaussian_data(builder(dim), 384, seed=dim)

    fit_c = VineCopula.fit_c(data, family_set=FAMILY_SET, max_iter=100)
    fit_d = VineCopula.fit_d(data, family_set=FAMILY_SET, max_iter=100)
    fit_r = VineCopula.fit_r(data, family_set=FAMILY_SET, max_iter=100)

    assert fit_c.model.structure_kind == "c"
    assert fit_d.model.structure_kind == "d"
    assert fit_r.model.structure_kind == "r"

    _assert_vine_invariants(fit_c.model, dim)
    _assert_vine_invariants(fit_d.model, dim)
    _assert_vine_invariants(fit_r.model, dim)


@pytest.mark.parametrize("truncation_level", [1, 2, 3, 4, 5, None])
def test_d6_star_truncation_levels_do_not_raise_internal_errors(
    truncation_level: int | None,
) -> None:
    data = _gaussian_data(_star_correlation(6), 512, seed=19)

    try:
        fit = VineCopula.fit_r(
            data,
            family_set=FAMILY_SET,
            truncation_level=truncation_level,
            max_iter=100,
        )
    except InternalError as exc:  # pragma: no cover - explicit contract check
        pytest.fail(f"fit_r leaked an internal panic for trunc={truncation_level}: {exc}")
    except ModelFitError as exc:  # pragma: no cover - should no longer happen here
        pytest.fail(f"fit_r should succeed on d=6 star data for trunc={truncation_level}: {exc}")

    _assert_vine_invariants(fit.model, 6)


@pytest.mark.parametrize("dim", [5, 6, 7, 8])
@pytest.mark.parametrize("truncation_level", [1, 2])
def test_fit_r_model_dim_equals_input_dim_on_truncated_path_data(
    dim: int, truncation_level: int
) -> None:
    data = _gaussian_data(_path_correlation(dim), 384, seed=5 + dim + truncation_level)
    fit = VineCopula.fit_r(
        data,
        family_set=FAMILY_SET,
        truncation_level=truncation_level,
        max_iter=100,
    )

    _assert_vine_invariants(fit.model, dim)
    assert fit.model.truncation_level == truncation_level


def test_vine_structure_kind_and_rotation_contracts() -> None:
    data = _gaussian_data(_path_correlation(4), 256, seed=13)
    fits = [
        VineCopula.fit_c(data, family_set=FAMILY_SET, max_iter=100),
        VineCopula.fit_d(data, family_set=FAMILY_SET, max_iter=100),
        VineCopula.fit_r(data, family_set=FAMILY_SET, max_iter=100),
    ]

    allowed_kinds = {"c", "d", "r"}
    allowed_rotations = {"R0", "R90", "R180", "R270"}

    for fit in fits:
        assert fit.model.structure_kind in allowed_kinds
        assert fit.model.structure_info.kind == fit.model.structure_kind
        assert len(fit.model.order) == fit.model.dim
        for tree in fit.model.trees:
            for edge in tree.edges:
                assert edge.rotation in allowed_rotations
