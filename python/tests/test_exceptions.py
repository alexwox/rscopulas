"""Exception hierarchy and error-classification contracts."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

from rscopulas import (
    BackendError,
    ClaytonCopula,
    FactorCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    HierarchicalArchimedeanCopula,
    InternalError,
    InvalidInputError,
    ModelFitError,
    NonPrefixConditioningError,
    NumericalError,
    PairCopula,
    RscopulasError,
    StudentTCopula,
    VineCopula,
)

CORR3 = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
DATA3 = GaussianCopula.from_params(CORR3).sample(120, seed=5)
WRONG_WIDTH = np.full((4, 4), 0.4)


def test_hierarchy() -> None:
    assert issubclass(RscopulasError, Exception)
    assert not issubclass(RscopulasError, ValueError)
    assert issubclass(InvalidInputError, RscopulasError)
    assert issubclass(InvalidInputError, ValueError)
    assert issubclass(NonPrefixConditioningError, InvalidInputError)
    for exc in (ModelFitError, NumericalError, BackendError, InternalError):
        assert issubclass(exc, RscopulasError)
        assert not issubclass(exc, ValueError)
    assert InvalidInputError.__module__ == "rscopulas"
    assert InvalidInputError.__name__ == "InvalidInputError"


def test_invalid_input_can_be_caught_as_value_error() -> None:
    with pytest.raises(ValueError) as excinfo:
        GaussianCopula.fit(np.array([[0.0, 0.5], [0.4, 0.6]]))
    assert isinstance(excinfo.value, InvalidInputError)
    assert isinstance(excinfo.value, RscopulasError)


@pytest.mark.parametrize(
    "call",
    [
        lambda: PairCopula.from_spec("nope", [1.0]),
        lambda: PairCopula.from_spec("clayton", [1.0], rotation="R45"),
        lambda: PairCopula.from_spec("gaussian", [1.0, 2.0]),
        lambda: PairCopula.from_spec("independence", [1.0]),
        lambda: PairCopula.fit_tll(DATA3[:, 0], DATA3[:, 1], method="cubic"),
        lambda: VineCopula.fit_r(DATA3, criterion="nope"),
        lambda: VineCopula.fit_r(DATA3, criterion="mbicv:1.5"),
        lambda: VineCopula.fit_r(DATA3, tree_algorithm="nope"),
        lambda: VineCopula.fit_r(DATA3, tree_criterion="nope"),
        lambda: VineCopula.fit_r(DATA3, family_set=["gaussian", "nope"]),
        lambda: VineCopula.fit_r(DATA3, family_set=["gaussian"], independence_test_level=1.5),
        lambda: VineCopula.fit_c(DATA3, family_set=["gaussian"], independence_test_level=0.0),
        lambda: VineCopula.from_trees("q", []),
        lambda: HierarchicalArchimedeanCopula.fit(DATA3, structure_method="nope"),
        lambda: HierarchicalArchimedeanCopula.fit(DATA3, fit_method="nope"),
        lambda: HierarchicalArchimedeanCopula.fit(DATA3, family_set=["nope"]),
        lambda: HierarchicalArchimedeanCopula.from_tree(
            {"family": "nope", "theta": 1.0, "children": [0, 1]}
        ),
        lambda: FactorCopula.fit(DATA3, layout="nope"),
        lambda: FactorCopula.fit(DATA3, criterion="nope"),
        lambda: FactorCopula.from_links([{"family": "nope"}] * 2),
    ],
    ids=[
        "pair_family",
        "pair_rotation",
        "pair_param_count",
        "independence_params",
        "tll_method",
        "vine_criterion",
        "vine_mbicv_psi0",
        "vine_tree_algorithm",
        "vine_tree_criterion",
        "vine_family_set",
        "vine_independence_test_level_high",
        "vine_independence_test_level_zero",
        "vine_kind",
        "hac_structure_method",
        "hac_fit_method",
        "hac_family_set",
        "hac_tree_family",
        "factor_layout",
        "factor_criterion",
        "factor_link_family",
    ],
)
def test_unsupported_names_raise_invalid_input(call: Callable[[], Any]) -> None:
    with pytest.raises(InvalidInputError):
        call()


def _flat_hac() -> HierarchicalArchimedeanCopula:
    return HierarchicalArchimedeanCopula.from_tree(
        {"family": "gumbel", "theta": 1.4, "children": [0, 1, 2]}
    )


def _vine() -> VineCopula:
    return VineCopula.fit_r(DATA3, family_set=["gaussian"]).model


@pytest.mark.parametrize(
    "call",
    [
        lambda: GaussianCopula.from_params(CORR3).log_pdf(WRONG_WIDTH),
        lambda: StudentTCopula.from_params(CORR3, 4.0).log_pdf(WRONG_WIDTH),
        lambda: ClaytonCopula.from_params(3, 1.5).log_pdf(WRONG_WIDTH),
        lambda: FrankCopula.from_params(3, 2.0).log_pdf(WRONG_WIDTH),
        lambda: GumbelCopula.from_params(3, 1.4).log_pdf(WRONG_WIDTH),
        lambda: _vine().log_pdf(WRONG_WIDTH),
        lambda: _vine().rosenblatt(WRONG_WIDTH),
        lambda: _vine().inverse_rosenblatt(WRONG_WIDTH),
        lambda: _flat_hac().log_pdf(WRONG_WIDTH),
        lambda: _flat_hac().composite_log_pdf(WRONG_WIDTH),
        lambda: FactorCopula.from_links(
            [{"family": "gaussian", "parameters": [0.6]}] * 3
        ).log_pdf(WRONG_WIDTH),
    ],
    ids=[
        "gaussian",
        "student_t",
        "clayton",
        "frank",
        "gumbel",
        "vine_log_pdf",
        "vine_rosenblatt",
        "vine_inverse_rosenblatt",
        "hac_log_pdf",
        "hac_composite_log_pdf",
        "factor",
    ],
)
def test_dimension_mismatch_is_an_input_error(call: Callable[[], Any]) -> None:
    with pytest.raises(InvalidInputError, match="dimension") as excinfo:
        call()
    assert not isinstance(excinfo.value, ModelFitError)


def test_dimension_mismatch_message_reports_both_sizes() -> None:
    with pytest.raises(InvalidInputError, match=r"input dimension 4 does not match model dimension 3"):
        GaussianCopula.from_params(CORR3).log_pdf(WRONG_WIDTH)


def test_non_matrix_input_is_an_input_error() -> None:
    model = GaussianCopula.from_params(CORR3)
    with pytest.raises(InvalidInputError, match="2D"):
        model.log_pdf([0.1, 0.2, 0.3])
    with pytest.raises(InvalidInputError, match="1D"):
        PairCopula.from_spec("gaussian", [0.5]).log_pdf(DATA3[:, :2], DATA3[:, 0])


def test_pair_length_mismatch_is_an_input_error() -> None:
    pair = PairCopula.from_spec("gaussian", [0.5])
    with pytest.raises(InvalidInputError, match="same length"):
        pair.log_pdf(np.array([0.1, 0.2]), np.array([0.3]))


def test_model_fit_error_still_reports_genuine_fit_failures() -> None:
    with pytest.raises(ModelFitError, match="dimension 1"):
        ClaytonCopula.from_params(1, 1.5)
    with pytest.raises(ModelFitError, match="tree_algorithm"):
        VineCopula.fit_c(DATA3, family_set=["gaussian"], tree_algorithm="prim")
