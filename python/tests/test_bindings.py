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
    PairCopula,
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


def test_vine_from_trees_rebuilds_model() -> None:
    fit = VineCopula.fit_r(
        DATA_4D,
        family_set=["independence", "gaussian", "clayton", "frank", "gumbel"],
        truncation_level=2,
        max_iter=200,
    )
    rebuilt = VineCopula.from_trees(
        fit.model.structure_kind,
        fit.model.trees,
        truncation_level=fit.model.truncation_level,
    )

    assert rebuilt.structure_kind == fit.model.structure_kind
    assert rebuilt.dim == fit.model.dim
    np.testing.assert_allclose(rebuilt.log_pdf(DATA_4D), fit.model.log_pdf(DATA_4D))


def test_pair_wrapper_exposes_density_and_h_kernels() -> None:
    model = PairCopula.from_spec("gaussian", [0.7])
    u1 = np.array([0.17, 0.31, 0.62, 0.88], dtype=np.float64)
    u2 = np.array([0.23, 0.54, 0.41, 0.79], dtype=np.float64)
    p = np.array([0.27, 0.45, 0.73, 0.91], dtype=np.float64)

    log_pdf = model.log_pdf(u1, u2)
    cond_first = model.cond_first_given_second(u1, u2)
    cond_second = model.cond_second_given_first(u1, u2)
    inv_first = model.inv_first_given_second(p, u2)
    inv_second = model.inv_second_given_first(u1, p)

    assert model.family == "gaussian"
    assert model.rotation == "R0"
    assert model.parameters == (0.7,)
    assert log_pdf.shape == (4,)
    assert cond_first.shape == (4,)
    assert cond_second.shape == (4,)
    assert inv_first.shape == (4,)
    assert inv_second.shape == (4,)
    np.testing.assert_allclose(log_pdf[0], 0.6046870303398274)
    np.testing.assert_allclose(cond_first[0], 0.27030730996634816)
    np.testing.assert_allclose(cond_second[0], 0.46044070722116331)
    np.testing.assert_allclose(inv_first[-1], 0.93599364105697047)
    np.testing.assert_allclose(inv_second[-1], 0.96246049482916851)


def test_joe_pair_copula() -> None:
    # Joe is a 1-parameter Archimedean family with upper-tail dependence;
    # the smoke test mirrors the Gaussian pair test and checks that the new
    # binding is reachable via the "joe" family string and evaluates the
    # density / h-function kernels without error.
    model = PairCopula.from_spec("joe", [2.5])
    u1 = np.array([0.17, 0.31, 0.62, 0.88], dtype=np.float64)
    u2 = np.array([0.23, 0.54, 0.41, 0.79], dtype=np.float64)
    p = np.array([0.27, 0.45, 0.73, 0.91], dtype=np.float64)

    log_pdf = model.log_pdf(u1, u2)
    cond_first = model.cond_first_given_second(u1, u2)
    cond_second = model.cond_second_given_first(u1, u2)
    inv_first = model.inv_first_given_second(p, u2)
    inv_second = model.inv_second_given_first(u1, p)

    assert model.family == "joe"
    assert model.rotation == "R0"
    assert model.parameters == (2.5,)
    assert log_pdf.shape == (4,)
    assert np.all(np.isfinite(log_pdf))
    assert cond_first.shape == (4,)
    assert np.all((cond_first > 0.0) & (cond_first < 1.0))
    assert cond_second.shape == (4,)
    assert np.all((cond_second > 0.0) & (cond_second < 1.0))
    assert inv_first.shape == (4,)
    assert np.all((inv_first > 0.0) & (inv_first < 1.0))
    assert inv_second.shape == (4,)
    assert np.all((inv_second > 0.0) & (inv_second < 1.0))


def test_bb1_pair_copula() -> None:
    # BB1 is a 2-parameter Archimedean (Clayton-Gumbel blend) with both lower
    # and upper tail dependence; the smoke test checks that the binding
    # round-trips a (theta, delta) pair and that all h-functions produce
    # values strictly inside (0, 1).
    model = PairCopula.from_spec("bb1", [1.5, 1.5])
    u1 = np.array([0.17, 0.31, 0.62, 0.88], dtype=np.float64)
    u2 = np.array([0.23, 0.54, 0.41, 0.79], dtype=np.float64)
    p = np.array([0.27, 0.45, 0.73, 0.91], dtype=np.float64)

    log_pdf = model.log_pdf(u1, u2)
    cond_first = model.cond_first_given_second(u1, u2)
    cond_second = model.cond_second_given_first(u1, u2)
    inv_first = model.inv_first_given_second(p, u2)
    inv_second = model.inv_second_given_first(u1, p)

    assert model.family == "bb1"
    assert model.rotation == "R0"
    assert model.parameters == (1.5, 1.5)
    assert log_pdf.shape == (4,)
    assert np.all(np.isfinite(log_pdf))
    assert cond_first.shape == (4,)
    assert np.all((cond_first > 0.0) & (cond_first < 1.0))
    assert cond_second.shape == (4,)
    assert np.all((cond_second > 0.0) & (cond_second < 1.0))
    assert inv_first.shape == (4,)
    assert np.all((inv_first > 0.0) & (inv_first < 1.0))
    assert inv_second.shape == (4,)
    assert np.all((inv_second > 0.0) & (inv_second < 1.0))


@pytest.mark.parametrize(
    ("family", "params"),
    [
        ("bb6", (2.0, 1.5)),
        ("bb7", (1.5, 1.5)),
        ("bb8", (2.0, 0.8)),
        ("tawn1", (2.0, 0.6)),
        ("tawn2", (2.0, 0.6)),
    ],
)
def test_bb_pair_copulas_round_trip(family: str, params: tuple[float, float]) -> None:
    # BB6 (Joe-Gumbel), BB7 (Joe-Clayton), BB8 (Joe-Frank) — all 2-parameter
    # Archimedeans. Smoke-test that each binding round-trips parameters and
    # evaluates density / h-functions without non-finite values.
    model = PairCopula.from_spec(family, list(params))
    u1 = np.array([0.17, 0.31, 0.62, 0.88], dtype=np.float64)
    u2 = np.array([0.23, 0.54, 0.41, 0.79], dtype=np.float64)
    p = np.array([0.27, 0.45, 0.73, 0.91], dtype=np.float64)

    log_pdf = model.log_pdf(u1, u2)
    cond_first = model.cond_first_given_second(u1, u2)
    cond_second = model.cond_second_given_first(u1, u2)
    inv_first = model.inv_first_given_second(p, u2)
    inv_second = model.inv_second_given_first(u1, p)

    assert model.family == family
    assert model.rotation == "R0"
    assert model.parameters == params
    assert log_pdf.shape == (4,)
    assert np.all(np.isfinite(log_pdf))
    for arr in (cond_first, cond_second, inv_first, inv_second):
        assert arr.shape == (4,)
        assert np.all((arr > 0.0) & (arr < 1.0))


def test_tll_pair_copula_fits_from_gaussian_sample() -> None:
    # TLL is nonparametric — fit on a Gaussian copula sample with ρ=0.5 and
    # verify the estimated density at the centre is roughly 1/sqrt(1-ρ²) ≈
    # 1.1547. Also verify h-functions stay in (0, 1) and are monotone.
    rho = 0.5
    correlation = np.array([[1.0, rho], [rho, 1.0]])
    gauss_model = GaussianCopula.from_params(correlation)
    samples = gauss_model.sample(2000, seed=7)
    u1, u2 = samples[:, 0], samples[:, 1]

    model = PairCopula.fit_tll(u1, u2, method="constant")
    assert model.family == "tll"
    assert model.rotation == "R0"
    # `parameters` exposes the effective dof summary.
    assert len(model.parameters) == 1
    assert model.parameters[0] > 0.0

    # Density at the centre should be close to Gaussian copula density.
    log_pdf = model.log_pdf(np.array([0.5]), np.array([0.5]))
    density = float(np.exp(log_pdf[0]))
    target = 1.0 / np.sqrt(1.0 - rho ** 2)
    assert abs(density - target) / target < 0.2, (
        f"tll density {density} should be within 20% of target {target}"
    )

    # h-function sanity: h_{1|2} should stay in (0, 1) and be monotone.
    u_grid = np.array([0.2, 0.4, 0.6, 0.8])
    v_fixed = np.full_like(u_grid, 0.4)
    h_vals = model.cond_first_given_second(u_grid, v_fixed)
    assert np.all((h_vals > 0.0) & (h_vals < 1.0))
    assert np.all(np.diff(h_vals) >= -1e-9), f"h values should be non-decreasing: {h_vals}"

    # Direct-params spec should raise a clear error (TLL is fit-only).
    with pytest.raises(Exception, match="tll"):
        PairCopula.from_spec("tll", [2.0])


def test_khoudraji_pair_wrapper_round_trips_structured_spec() -> None:
    model = PairCopula.from_khoudraji(
        "gaussian",
        "clayton",
        shape_1=0.35,
        shape_2=0.8,
        first_parameters=[0.45],
        second_parameters=[2.0],
    )
    u1 = np.array([0.17, 0.31, 0.62, 0.88], dtype=np.float64)
    u2 = np.array([0.23, 0.54, 0.41, 0.79], dtype=np.float64)

    log_pdf = model.log_pdf(u1, u2)
    spec = model.spec

    assert model.family == "khoudraji"
    assert model.rotation == "R0"
    assert len(model.parameters) >= 4
    assert spec["base_copula_1"]["family"] == "gaussian"
    assert spec["base_copula_2"]["family"] == "clayton"
    assert spec["shape_1"] == pytest.approx(0.35)
    assert spec["shape_2"] == pytest.approx(0.8)
    assert log_pdf.shape == (4,)


def test_vine_from_trees_accepts_khoudraji_edge_payload() -> None:
    model = VineCopula.from_trees(
        "r",
        [
            {
                "level": 1,
                "edges": [
                    {
                        "conditioned": (0, 1),
                        "conditioning": [],
                        "family": "khoudraji",
                        "rotation": "R0",
                        "shape_1": 0.35,
                        "shape_2": 0.8,
                        "base_copula_1": {
                            "family": "gaussian",
                            "rotation": "R0",
                            "parameters": [0.45],
                        },
                        "base_copula_2": {
                            "family": "clayton",
                            "rotation": "R0",
                            "parameters": [2.0],
                        },
                    }
                ],
            }
        ],
    )

    trees = model.trees
    assert trees[0].edges[0].family == "khoudraji"
    assert trees[0].edges[0].rotation == "R0"
    assert len(trees[0].edges[0].parameters) >= 4


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
