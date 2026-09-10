"""Pickle, copy, JSON, equality, and repr contracts shared by every model."""

from __future__ import annotations

import copy
import json
import pickle
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
    PairCopula,
    StudentTCopula,
    VineCopula,
)

CORR3 = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
DATA3 = GaussianCopula.from_params(CORR3).sample(64, seed=3)


def _vine() -> VineCopula:
    return VineCopula.fit_r(
        DATA3, family_set=["gaussian", "clayton", "frank"], max_iter=100
    ).model


def _hac() -> HierarchicalArchimedeanCopula:
    # A flat (exchangeable) tree keeps the exact `log_pdf` available.
    return HierarchicalArchimedeanCopula.from_tree(
        {"family": "gumbel", "theta": 1.4, "children": [0, 1, 2]}
    )


def _factor() -> FactorCopula:
    return FactorCopula.from_links(
        [
            {"family": "gaussian", "parameters": [0.7]},
            {"family": "clayton", "parameters": [1.5]},
            {"family": "frank", "parameters": [3.0]},
        ]
    )


MODEL_FACTORIES = {
    "gaussian": lambda: GaussianCopula.from_params(CORR3),
    "student_t": lambda: StudentTCopula.from_params(CORR3, 5.0),
    "clayton": lambda: ClaytonCopula.from_params(3, 1.5),
    "frank": lambda: FrankCopula.from_params(3, 4.0),
    "gumbel": lambda: GumbelCopula.from_params(3, 1.3),
    "vine": _vine,
    "hac": _hac,
    "factor": _factor,
}


def _first_json_difference(left: Any, right: Any, path: str = "$") -> str | None:
    """Return the first differing path between two parsed JSON payloads."""
    if type(left) is not type(right):
        return f"{path}: {type(left).__name__} != {type(right).__name__} ({left!r} vs {right!r})"
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return f"{path}.{key}: present on one side only"
            found = _first_json_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for idx, (a, b) in enumerate(zip(left, right)):
            found = _first_json_difference(a, b, f"{path}[{idx}]")
            if found:
                return found
        return None
    if left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def assert_models_equal(restored: Any, model: Any) -> None:
    """`restored == model`, with the first differing serialized field on failure."""
    if restored == model:
        return
    diff = _first_json_difference(json.loads(restored.to_json()), json.loads(model.to_json()))
    raise AssertionError(f"{restored!r} != {model!r}; first serialized difference: {diff}")


@pytest.fixture(scope="module", params=sorted(MODEL_FACTORIES))
def model(request: pytest.FixtureRequest) -> Any:
    return MODEL_FACTORIES[request.param]()


def test_pickle_round_trip_preserves_log_pdf_exactly(model: Any) -> None:
    for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(model, protocol=protocol))
        assert type(restored) is type(model)
        assert restored._core is not model._core
        assert_models_equal(restored, model)
        np.testing.assert_array_equal(restored.log_pdf(DATA3), model.log_pdf(DATA3))
        np.testing.assert_array_equal(restored.sample(5, seed=9), model.sample(5, seed=9))


def test_deepcopy_and_copy_are_independent(model: Any) -> None:
    for clone in (copy.deepcopy(model), copy.copy(model)):
        assert clone is not model
        assert clone._core is not model._core
        assert_models_equal(clone, model)
        np.testing.assert_array_equal(clone.log_pdf(DATA3), model.log_pdf(DATA3))


def test_json_round_trip(model: Any) -> None:
    payload = model.to_json()
    assert isinstance(json.loads(payload), dict)
    restored = type(model).from_json(payload)
    assert_models_equal(restored, model)
    assert restored.dim == model.dim
    assert restored.family == model.family
    np.testing.assert_array_equal(restored.log_pdf(DATA3), model.log_pdf(DATA3))


def test_repr_mentions_family_and_dim(model: Any) -> None:
    text = repr(model)
    assert text.startswith(type(model).__name__ + "(")
    assert f"family={model.family!r}" in text
    assert f"dim={model.dim}" in text


def test_core_objects_pickle_copy_and_compare(model: Any) -> None:
    core = model._core
    restored = pickle.loads(pickle.dumps(core))
    assert restored is not core
    assert restored == core
    assert copy.deepcopy(core) == core
    assert "dim=" in repr(core)
    assert (core == object()) is False
    with pytest.raises(TypeError):
        hash(core)


def _replace_floats(obj: Any, value: float) -> Any:
    if isinstance(obj, float):
        return value
    if isinstance(obj, list):
        return [_replace_floats(item, value) for item in obj]
    if isinstance(obj, dict):
        return {key: _replace_floats(item, value) for key, item in obj.items()}
    return obj


def test_from_json_validates_state(model: Any) -> None:
    if isinstance(model, FrankCopula):
        pytest.skip("negative Frank parameters are valid")
    corrupted = json.dumps(_replace_floats(json.loads(model.to_json()), -3.0))
    with pytest.raises(InvalidInputError):
        type(model).from_json(corrupted)
    with pytest.raises(ValueError):
        type(model).from_json("not json at all")


def test_setstate_rejects_foreign_state(model: Any) -> None:
    blank = object.__new__(type(model))
    with pytest.raises(InvalidInputError):
        blank.__setstate__({"bogus": 1})


def test_equality_semantics() -> None:
    a = ClaytonCopula.from_params(3, 1.5)
    assert a == ClaytonCopula.from_params(3, 1.5)
    assert a != ClaytonCopula.from_params(3, 1.6)
    assert a != ClaytonCopula.from_params(2, 1.5)
    assert a != FrankCopula.from_params(3, 1.5)
    assert a != 1.5
    assert (a == "clayton") is False
    with pytest.raises(TypeError):
        hash(a)
    with pytest.raises(TypeError):
        {a}


def test_repr_truncates_large_parameters() -> None:
    dim = 9
    corr = np.full((dim, dim), 0.3)
    np.fill_diagonal(corr, 1.0)
    text = repr(GaussianCopula.from_params(corr))
    assert "..." in text
    assert len(text) < 400
    assert "dim=9" in text

    hac = HierarchicalArchimedeanCopula.from_tree(
        {"family": "clayton", "theta": 1.2, "children": list(range(dim))}
    )
    assert "families=['clayton']" in repr(hac)
    assert "parameters=[1.2]" in repr(hac)


def test_vine_json_carries_format_version_and_rejects_unversioned() -> None:
    vine = _vine()
    payload = json.loads(vine.to_json())
    assert payload["format_version"] == 1
    payload.pop("format_version")
    with pytest.raises(InvalidInputError):
        VineCopula.from_json(json.dumps(payload))


def _tll_pair() -> PairCopula:
    return PairCopula.fit_tll(DATA3[:, 0], DATA3[:, 1])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PairCopula.from_spec("gaussian", [0.6]),
        lambda: PairCopula.from_spec("clayton", [1.5], rotation="R90"),
        lambda: PairCopula.from_spec("bb1", [1.2, 1.5]),
        lambda: PairCopula.from_khoudraji(
            "gaussian", "clayton", shape_1=0.35, shape_2=0.8,
            first_parameters=[0.45], second_parameters=[2.0],
        ),
        _tll_pair,
    ],
    ids=["gaussian", "clayton_r90", "bb1", "khoudraji", "tll"],
)
def test_pair_copula_protocols(factory: Any) -> None:
    pair = factory()
    u1, u2 = DATA3[:, 0], DATA3[:, 1]
    expected = pair.log_pdf(u1, u2)
    clones = [
        pickle.loads(pickle.dumps(pair)),
        copy.deepcopy(pair),
        copy.copy(pair),
        PairCopula.from_json(pair.to_json()),
    ]
    for clone in clones:
        assert clone._core is not pair._core
        assert clone.family == pair.family
        assert clone.rotation == pair.rotation
        if pair.family == "tll":
            # Loading TLL state re-normalizes the density grid, so a restored
            # TLL spec (and therefore `==`) matches the fitted one only up to
            # floating-point rounding.
            np.testing.assert_allclose(clone.log_pdf(u1, u2), expected, rtol=1e-10)
            assert clone.spec["state"]["bandwidth"] == pair.spec["state"]["bandwidth"]
        else:
            assert clone == pair
            assert clone.spec == pair.spec
            np.testing.assert_array_equal(clone.log_pdf(u1, u2), expected)
    text = repr(pair)
    assert f"family={pair.family!r}" in text
    assert f"rotation={pair.rotation!r}" in text
    assert "dim=2" in text
    assert pair != PairCopula.from_spec("frank", [2.0])
    with pytest.raises(TypeError):
        hash(pair)
