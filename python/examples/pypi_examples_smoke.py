import numpy as np

from rscopulas import GaussianCopula, GumbelCopula, VineCopula


def _assert_unit_cube(sample: np.ndarray) -> None:
    assert np.all(sample > 0.0)
    assert np.all(sample < 1.0)


def main() -> None:
    gaussian_data = np.array(
        [
            [0.12, 0.18],
            [0.21, 0.25],
            [0.82, 0.79],
        ],
        dtype=np.float64,
    )
    gaussian_fit = GaussianCopula.fit(gaussian_data)
    gaussian_sample = gaussian_fit.model.sample(4, seed=7)
    assert gaussian_fit.model.family == "gaussian"
    assert gaussian_sample.shape == (4, 2)
    _assert_unit_cube(gaussian_sample)
    print("Example 1 ok: Gaussian fit + sample")
    print("  AIC:", gaussian_fit.diagnostics.aic)

    gumbel = GumbelCopula.from_params(2, 2.1)
    gumbel_sample = gumbel.sample(5, seed=11)
    gumbel_log_pdf = gumbel.log_pdf(gumbel_sample)
    assert gumbel.family == "gumbel"
    assert gumbel_sample.shape == (5, 2)
    assert gumbel_log_pdf.shape == (5,)
    _assert_unit_cube(gumbel_sample)
    print("Example 2 ok: Gumbel sample + log_pdf")
    print("  First log density:", float(gumbel_log_pdf[0]))

    vine_data = np.array(
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
    vine_fit = VineCopula.fit_r(
        vine_data,
        family_set=["independence", "gaussian", "clayton", "frank", "gumbel"],
        truncation_level=2,
        max_iter=200,
    )
    vine_sample = vine_fit.model.sample(4, seed=13)
    first_tree_families = [edge.family for edge in vine_fit.model.trees[0].edges]
    assert vine_fit.model.structure_kind == "r"
    assert vine_fit.model.order
    assert first_tree_families
    assert vine_sample.shape == (4, 4)
    _assert_unit_cube(vine_sample)
    print("Example 3 ok: R-vine fit + sample")
    print("  Order:", vine_fit.model.order)
    print("  First-tree families:", first_tree_families)

    print("All three PYPI.md examples ran successfully.")


if __name__ == "__main__":
    main()
