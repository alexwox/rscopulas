import numpy as np
import pytest

from rscopulas import (
    ClaytonCopula,
    FitDiagnostics,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    HierarchicalArchimedeanCopula,
    InvalidInputError,
    StudentTCopula,
    VineCopula,
)


DATA_2D = np.array(
    [
        [0.12, 0.18],
        [0.21, 0.25],
        [0.27, 0.22],
        [0.35, 0.42],
        [0.48, 0.51],
        [0.56, 0.49],
        [0.68, 0.73],
        [0.82, 0.79],
    ],
    dtype=np.float64,
)

DATA_4D = np.array(
    [
        [0.10, 0.14, 0.18, 0.22],
        [0.18, 0.21, 0.24, 0.27],
        [0.24, 0.29, 0.33, 0.31],
        [0.33, 0.30, 0.38, 0.41],
        [0.47, 0.45, 0.43, 0.49],
        [0.52, 0.58, 0.55, 0.53],
        [0.69, 0.63, 0.67, 0.71],
        [0.81, 0.78, 0.74, 0.76],
    ],
    dtype=np.float64,
)


def test_gaussian_fit_returns_pythonic_result() -> None:
    fit = GaussianCopula.fit(DATA_2D)

    assert isinstance(fit.diagnostics, FitDiagnostics)
    assert fit.model.family == "gaussian"
    assert fit.model.dim == 2
    assert fit.model.correlation.shape == (2, 2)
    assert fit.model.log_pdf(DATA_2D).shape == (DATA_2D.shape[0],)

    samples = fit.model.sample(6, seed=7)
    assert samples.shape == (6, 2)
    assert np.all(samples > 0.0)
    assert np.all(samples < 1.0)


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (StudentTCopula.from_params([[1.0, 0.5], [0.5, 1.0]], 4.0), DATA_2D),
        (ClaytonCopula.from_params(2, 1.5), DATA_2D),
        (FrankCopula.from_params(2, 4.0), DATA_2D),
        (GumbelCopula.from_params(2, 1.3), DATA_2D),
    ],
)
def test_family_wrappers_support_log_pdf_and_sample(model, data: np.ndarray) -> None:
    values = model.log_pdf(data)
    assert values.shape == (data.shape[0],)

    samples = model.sample(5, seed=11)
    assert samples.shape == (5, model.dim)
    assert np.all(samples > 0.0)
    assert np.all(samples < 1.0)


def test_vine_fit_r_exposes_structure_and_parameters() -> None:
    fit = VineCopula.fit_r(
        DATA_4D,
        family_set=["independence", "gaussian", "clayton", "frank", "gumbel"],
        truncation_level=2,
        max_iter=200,
    )

    assert fit.model.family == "vine"
    assert fit.model.structure_kind == "r"
    assert fit.model.dim == 4
    assert fit.model.order

    structure = fit.model.structure_info
    assert structure.kind == "r"
    assert structure.matrix.shape == (4, 4)
    assert structure.truncation_level == 2

    trees = fit.model.trees
    assert len(trees) >= 2
    assert trees[0].edges

    pair_parameters = fit.model.pair_parameters
    assert pair_parameters.ndim == 1

    samples = fit.model.sample(4, seed=13)
    assert samples.shape == (4, 4)
    assert np.all(samples > 0.0)
    assert np.all(samples < 1.0)


def test_invalid_input_error_surfaces_from_core_validation() -> None:
    with pytest.raises(InvalidInputError):
        GaussianCopula.fit(np.array([[0.0, 0.5], [0.4, 0.6]], dtype=np.float64))


def test_hac_wrapper_exposes_tree_and_sampling() -> None:
    tree = {
        "family": "gumbel",
        "theta": 1.2,
        "children": [
            0,
            1,
            {"family": "gumbel", "theta": 2.0, "children": [2, 3]},
        ],
    }

    model = HierarchicalArchimedeanCopula.from_tree(tree)
    assert model.family == "hierarchical_archimedean"
    assert model.dim == 4
    assert model.is_exact is True
    assert model.tree["family"] == "gumbel"

    samples = model.sample(16, seed=19)
    assert samples.shape == (16, 4)
    assert np.all(samples > 0.0)
    assert np.all(samples < 1.0)

    fit = HierarchicalArchimedeanCopula.fit(
        samples,
        family_set=["gumbel"],
        structure_method="agglomerative_tau_then_collapse",
        fit_method="recursive_mle",
    )
    assert fit.model.dim == 4
    assert fit.model.structure_method == "agglomerative_tau_then_collapse"
    assert fit.model.fit_method == "recursive_mle"
