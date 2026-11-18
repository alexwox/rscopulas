import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rscopulas import ClaytonCopula, GaussianCopula, PairCopula, VineCopula
from rscopulas.plotting import (
    plot_density,
    plot_scatter,
    plot_tail_concentration,
    plot_tail_dependence,
    plot_vine_structure,
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


@pytest.fixture(autouse=True)
def close_figures() -> None:
    yield
    plt.close("all")


def test_plot_density_returns_axes_for_bivariate_model() -> None:
    model = GaussianCopula.fit(DATA_2D).model

    ax = plot_density(model, grid_size=24, levels=8)

    assert ax.get_title() == "Gaussian density"
    assert ax.get_xlabel() == "u1"
    assert ax.get_ylabel() == "u2"
    assert len(ax.collections) >= 1


def test_plot_density_rejects_non_bivariate_models() -> None:
    model = ClaytonCopula.from_params(3, 1.5)

    with pytest.raises(ValueError, match="bivariate"):
        plot_density(model)


def test_plot_density_accepts_pair_copulas() -> None:
    ax = plot_density(PairCopula.from_spec("gaussian", [.6]), grid_size=12)
    assert ax.get_title() == "Gaussian density"
    assert len(ax.collections) >= 1


def test_plot_scatter_supports_samples_and_model_sampling() -> None:
    sample_ax = plot_scatter(sample=DATA_2D, alpha=0.6, s=16)
    assert sample_ax.get_title() == "Observed pseudo-observations"
    assert sample_ax.get_xlabel() == "u1"
    assert sample_ax.get_ylabel() == "u2"

    model_ax = plot_scatter(model=ClaytonCopula.from_params(3, 1.2), n=64, seed=5, dims=(0, 2))
    assert model_ax.get_title() == "Clayton samples"
    assert model_ax.get_xlabel() == "u1"
    assert model_ax.get_ylabel() == "u3"


def test_plot_scatter_validates_arguments() -> None:
    with pytest.raises(ValueError, match="either `sample` or `model`"):
        plot_scatter()

    with pytest.raises(ValueError, match="not both"):
        plot_scatter(sample=DATA_2D, model=GaussianCopula.fit(DATA_2D).model)

    with pytest.raises(ValueError, match="out of bounds"):
        plot_scatter(sample=np.ones((4, 3), dtype=np.float64) * 0.5, dims=(0, 3))


def test_plot_vine_structure_renders_matrix_and_annotations() -> None:
    vine = VineCopula.gaussian_c_vine(
        [0, 1, 2],
        np.array(
            [
                [1.0, 0.60, 0.35],
                [0.60, 1.0, 0.25],
                [0.35, 0.25, 1.0],
            ],
            dtype=np.float64,
        ),
    )

    ax = plot_vine_structure(vine)

    assert ax.get_title() == "C-vine structure matrix"
    assert ax.get_xlabel() == "column"
    assert ax.get_ylabel() == "row"
    assert len(ax.images) == 1
    assert len(ax.texts) >= 1


def test_plot_vine_structure_rejects_non_vine_models() -> None:
    with pytest.raises(TypeError, match="vine model"):
        plot_vine_structure(GaussianCopula.fit(DATA_2D).model)


def test_plot_tail_concentration_overlays_a_model() -> None:
    sample = ClaytonCopula.from_params(3, 2.0).sample(500, seed=1)
    model = PairCopula.from_spec("clayton", [2.0])

    ax = plot_tail_concentration(sample, model=model, dims=(0, 2))

    assert ax.get_title() == "Tail concentration of u1, u3"
    labels = [line.get_label() for line in ax.get_lines()]
    assert labels[:2] == ["empirical", "Clayton"]
    empirical = ax.get_lines()[0].get_ydata()
    assert len(empirical) == 99 and 0.0 <= empirical.min() and empirical.max() <= 1.0

    with pytest.raises(TypeError, match="PairCopula"):
        plot_tail_concentration(sample, model=object())
    with pytest.raises(ValueError, match="distinct"):
        plot_tail_concentration(sample, dims=(1, 1))


def test_plot_tail_dependence_marks_default_k_and_model() -> None:
    sample = ClaytonCopula.from_params(2, 2.0).sample(400, seed=1)

    ax = plot_tail_dependence(sample, model=PairCopula.from_spec("clayton", [2.0]))

    labels = [line.get_label() for line in ax.get_lines()]
    assert labels == ["lower", "upper", "model lower", "model upper", "sqrt(n)"]
    assert ax.get_lines()[0].get_xdata().tolist() == list(range(1, 81))
    assert ax.get_lines()[2].get_ydata()[0] == pytest.approx(2.0**-0.5)
    assert ax.get_lines()[4].get_xdata()[0] == 20
