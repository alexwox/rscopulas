"""Tail-dependence diagnostics: model-implied values and rank-based estimates.

Closed forms are checked against ``scipy`` (Student t) and textbook values
here, and against 1300-digit limits in the Rust suite
(``crates/rscopulas-core/tests/reference_tail_dependence.rs``).
"""

import numpy as np
import pytest

import rscopulas
from rscopulas import (
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    InvalidInputError,
    PairCopula,
    StudentTCopula,
    TailDependence,
    default_tail_k,
    empirical_tail_concentration,
    empirical_tail_dependence,
    empirical_tail_dependence_curve,
    empirical_tail_dependence_matrix,
    to_pseudo_obs,
)


def test_pair_copula_closed_forms() -> None:
    assert PairCopula.from_spec("clayton", [2.0]).tail_dependence() == TailDependence(
        2.0 ** -0.5, 0.0
    )
    gumbel = PairCopula.from_spec("gumbel", [2.0]).tail_dependence()
    assert gumbel == TailDependence(0.0, pytest.approx(2.0 - 2.0**0.5, abs=1e-15))
    assert PairCopula.from_spec("gaussian", [0.95]).tail_dependence() == TailDependence(0.0, 0.0)
    assert PairCopula.from_spec("frank", [9.0]).tail_dependence() == TailDependence(0.0, 0.0)

    # A 180 degree rotation moves the dependence to the other tail.
    survival = PairCopula.from_spec("clayton", [2.0], rotation="R180").tail_dependence()
    assert survival == TailDependence(0.0, 2.0 ** -0.5)
    # A 90 degree rotation points it at the off-diagonal corners.
    assert PairCopula.from_spec("clayton", [2.0], rotation="R90").tail_dependence() == (
        TailDependence(0.0, 0.0)
    )


def test_student_t_matches_scipy() -> None:
    stats = pytest.importorskip("scipy.stats")
    for rho, nu in [(0.5, 3.0), (-0.4, 2.0), (0.9, 6.0), (0.3, 25.0)]:
        expected = 2.0 * stats.t.cdf(-np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho)), nu + 1.0)
        got = PairCopula.from_spec("student_t", [rho, nu]).tail_dependence()
        assert got.lower == pytest.approx(expected, rel=1e-12)
        assert got.upper == got.lower

    correlation = np.array([[1.0, 0.5, -0.4], [0.5, 1.0, 0.0], [-0.4, 0.0, 1.0]])
    matrix = StudentTCopula.from_params(correlation, 3.0).tail_dependence()
    assert matrix.lower[0, 1] == pytest.approx(0.3125, abs=1e-13)
    np.testing.assert_array_equal(matrix.lower, matrix.upper)
    np.testing.assert_array_equal(matrix.lower, matrix.lower.T)
    np.testing.assert_array_equal(np.diag(matrix.lower), np.ones(3))


def test_multivariate_models_report_matrices() -> None:
    clayton = ClaytonCopula.from_params(3, 2.0).tail_dependence()
    assert clayton.lower.shape == (3, 3)
    assert clayton.lower[0, 2] == pytest.approx(2.0**-0.5)
    assert clayton.upper[0, 2] == 0.0
    gumbel = GumbelCopula.from_params(3, 2.0).tail_dependence()
    assert gumbel.upper[1, 2] == pytest.approx(2.0 - 2.0**0.5)
    for model in (
        FrankCopula.from_params(3, 4.0),
        GaussianCopula.from_params(np.array([[1.0, 0.8], [0.8, 1.0]])),
    ):
        matrix = model.tail_dependence()
        off_diagonal = ~np.eye(model.dim, dtype=bool)
        assert not matrix.lower[off_diagonal].any()
        assert not matrix.upper[off_diagonal].any()


def test_families_without_a_closed_form_return_none() -> None:
    khoudraji = PairCopula.from_khoudraji(
        "clayton",
        "gumbel",
        shape_1=0.4,
        shape_2=0.6,
        first_parameters=[2.0],
        second_parameters=[2.0],
    )
    assert khoudraji.tail_dependence() is None


def test_estimates_recover_the_model() -> None:
    model = ClaytonCopula.from_params(2, 2.5)
    sample = model.sample(100_000, seed=7)
    estimate = empirical_tail_dependence(sample[:, 0], sample[:, 1])
    assert estimate.lower == pytest.approx(2.0 ** (-1.0 / 2.5), abs=0.08)
    assert estimate.upper < 0.08

    pair = PairCopula.from_spec("clayton", [2.5])
    observed = empirical_tail_concentration(sample[:, 0], sample[:, 1])
    assert observed.q[0] == 0.01 and observed.q[-1] == 0.99
    np.testing.assert_allclose(observed.values, pair.tail_concentration(observed.q), atol=0.05)


def test_estimators_accept_raw_data() -> None:
    rng = np.random.default_rng(0)
    raw = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.7], [0.7, 1.0]], size=2_000)
    raw[:, 0] = np.exp(raw[:, 0])  # any increasing transform keeps the ranks
    uniforms = to_pseudo_obs(raw)
    assert empirical_tail_dependence(raw[:, 0], raw[:, 1]) == empirical_tail_dependence(
        uniforms[:, 0], uniforms[:, 1]
    )
    matrix = empirical_tail_dependence_matrix(raw)
    pair = empirical_tail_dependence(raw[:, 0], raw[:, 1], k=default_tail_k(2_000))
    assert matrix.lower[0, 1] == pair.lower and matrix.upper[1, 0] == pair.upper


def test_curve_matches_single_estimates_and_definition() -> None:
    sample = GumbelCopula.from_params(2, 2.0).sample(300, seed=3)
    u1, u2 = sample[:, 0], sample[:, 1]
    curve = empirical_tail_dependence_curve(u1, u2)
    np.testing.assert_array_equal(curve.k, np.arange(1, 61))
    # Written out with scipy's ranks.
    stats = pytest.importorskip("scipy.stats")
    r, s = stats.rankdata(u1), stats.rankdata(u2)
    for index, k in enumerate(curve.k):
        assert curve.lower[index] == np.sum((r <= k) & (s <= k)) / k
        assert curve.upper[index] == np.sum((r > 300 - k) & (s > 300 - k)) / k
        single = empirical_tail_dependence(u1, u2, k=int(k))
        assert (single.lower, single.upper) == (curve.lower[index], curve.upper[index])
    custom = empirical_tail_dependence_curve(u1, u2, k=[5, 50])
    assert custom.lower.tolist() == [curve.lower[4], curve.lower[49]]


def test_default_k() -> None:
    assert default_tail_k(100) == 10
    assert default_tail_k(99) == 9
    assert default_tail_k(1) == 1


def test_invalid_input_is_reported() -> None:
    u = np.linspace(0.1, 0.9, 9)
    with pytest.raises(InvalidInputError):
        empirical_tail_dependence(u, u[:-1])
    with pytest.raises(InvalidInputError):
        empirical_tail_dependence(u, u, k=0)
    with pytest.raises(InvalidInputError):
        empirical_tail_dependence(u, u, k=10)
    with pytest.raises(InvalidInputError):
        empirical_tail_dependence(u, u, k=2.5)
    with pytest.raises(InvalidInputError):
        empirical_tail_dependence(u, np.where(u > 0.5, np.nan, u))
    with pytest.raises(InvalidInputError):
        empirical_tail_dependence_curve(u, u, k=[0, 1])
    with pytest.raises(InvalidInputError):
        empirical_tail_concentration(u, u, q=[0.0, 0.5])
    with pytest.raises(InvalidInputError):
        empirical_tail_dependence_matrix(u)
    with pytest.raises(InvalidInputError):
        PairCopula.from_spec("clayton", [2.0]).tail_concentration([1.0])
    # InvalidInputError is also a ValueError.
    with pytest.raises(ValueError):
        empirical_tail_dependence(u, u, k=0)


def test_public_names_are_exported() -> None:
    for name in (
        "TailConcentration",
        "TailDependence",
        "TailDependenceCurve",
        "TailDependenceMatrix",
        "default_tail_k",
        "empirical_tail_concentration",
        "empirical_tail_dependence",
        "empirical_tail_dependence_curve",
        "empirical_tail_dependence_matrix",
    ):
        assert name in rscopulas.__all__
