"""Marginal models: generalized Pareto, generalized extreme value, and the
semi-parametric margin, checked against ``scipy.stats`` and closed forms.

The Rust suite (``crates/rscopulas-core/tests/reference_margins.rs``) holds
the fits against SciPy and R on recorded samples; these tests cover the
Python surface and repeat the key comparisons live.
"""

import copy
import json
import pickle

import numpy as np
import pytest

import rscopulas
from rscopulas import (
    GeneralizedExtremeValue,
    GeneralizedPareto,
    GumbelCopula,
    InvalidInputError,
    Margins,
    SemiParametricMargin,
    block_maxima,
    mean_excess,
    threshold_stability,
    to_pseudo_obs,
)

stats = pytest.importorskip("scipy.stats")
optimize = pytest.importorskip("scipy.optimize")


def test_generalized_pareto_matches_scipy() -> None:
    y = np.array([1e-8, 0.01, 0.5, 1.0, 3.0, 7.4, 50.0])
    p = np.array([0.0, 1e-9, 0.01, 0.5, 0.9, 0.999])
    for scale, shape in [(2.0, 0.3), (1.5, -0.2), (1.0, 0.0), (0.7, 0.8)]:
        ours, theirs = GeneralizedPareto(scale, shape), stats.genpareto(shape, scale=scale)
        np.testing.assert_allclose(ours.cdf(y), theirs.cdf(y), rtol=1e-13, atol=1e-300)
        np.testing.assert_allclose(ours.sf(y), theirs.sf(y), rtol=1e-12, atol=0)
        np.testing.assert_allclose(ours.quantile(p), theirs.ppf(p), rtol=1e-12)
        finite = np.isfinite(theirs.logpdf(y))
        np.testing.assert_allclose(ours.logpdf(y)[finite], theirs.logpdf(y)[finite], rtol=1e-12)
        assert np.all(ours.logpdf(y)[~finite] == -np.inf)
    # Scalars in, scalars out; arrays keep their shape.
    model = GeneralizedPareto(2.0, 0.3)
    assert isinstance(model.cdf(1.0), float)
    assert model.cdf(np.ones((2, 3))).shape == (2, 3)
    with pytest.raises(InvalidInputError):
        GeneralizedPareto(-1.0, 0.1)
    with pytest.raises(InvalidInputError):
        model.quantile(1.0)


def test_generalized_extreme_value_matches_scipy() -> None:
    p = np.array([1e-9, 0.01, 0.5, 0.9, 0.999])
    for location, scale, shape in [(10.0, 3.0, 0.2), (-5.0, 2.0, -0.3), (0.0, 1.0, 0.0)]:
        ours = GeneralizedExtremeValue(location, scale, shape)
        theirs = stats.genextreme(-shape, loc=location, scale=scale)  # c = -shape
        x = location + scale * np.array([-1.0, 0.0, 0.5, 3.0])
        np.testing.assert_allclose(ours.cdf(x), theirs.cdf(x), rtol=1e-12)
        np.testing.assert_allclose(ours.logpdf(x), theirs.logpdf(x), rtol=1e-12)
        np.testing.assert_allclose(ours.quantile(p), theirs.ppf(p), rtol=1e-10)
        np.testing.assert_allclose(ours.return_level(100.0), theirs.ppf(0.99), rtol=1e-12)
    with pytest.raises(InvalidInputError):
        GeneralizedExtremeValue(0.0, 1.0, 0.1).return_level(1.0)


def _tight_maximum(negative_loglik, start):
    result = optimize.minimize(negative_loglik, start, method="Nelder-Mead")
    for _ in range(5):
        result = optimize.minimize(
            negative_loglik,
            result.x,
            method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 100_000, "maxfev": 100_000},
        )
    return result


@pytest.mark.parametrize("shape", [0.4, 0.0, -0.25])
def test_generalized_pareto_fit_is_the_maximum_of_scipys_likelihood(shape: float) -> None:
    y = stats.genpareto.rvs(shape, scale=2.0, size=800, random_state=np.random.default_rng(5))
    fit = GeneralizedPareto.fit(y)

    def negative_loglik(params):
        value = -np.sum(stats.genpareto.logpdf(y, params[1], scale=params[0]))
        return value if np.isfinite(value) and params[0] > 0 else np.inf

    reference = _tight_maximum(negative_loglik, [float(np.mean(y)), 0.1])
    assert fit.loglik >= -reference.fun - 1e-8
    assert fit.loglik == pytest.approx(-negative_loglik([fit.model.scale, fit.model.shape]), abs=1e-8)
    assert fit.model.scale == pytest.approx(reference.x[0], rel=1e-5)
    assert fit.model.shape == pytest.approx(reference.x[1], abs=1e-5)
    assert fit.n_obs == 800
    # Truth within three standard errors.
    assert abs(fit.model.shape - shape) < 3.0 * fit.std_errors[1]
    assert abs(fit.model.scale - 2.0) < 3.0 * fit.std_errors[0]


def test_generalized_extreme_value_fit_is_the_maximum_of_scipys_likelihood() -> None:
    x = stats.genextreme.rvs(-0.2, loc=10.0, scale=3.0, size=400, random_state=np.random.default_rng(8))
    fit = GeneralizedExtremeValue.fit(x)

    def negative_loglik(params):
        value = -np.sum(stats.genextreme.logpdf(x, -params[2], loc=params[0], scale=params[1]))
        return value if np.isfinite(value) and params[1] > 0 else np.inf

    reference = _tight_maximum(negative_loglik, [float(np.mean(x)), float(np.std(x)), 0.1])
    assert fit.loglik >= -reference.fun - 1e-7
    assert fit.model.location == pytest.approx(reference.x[0], abs=1e-4)
    assert fit.model.scale == pytest.approx(reference.x[1], abs=1e-4)
    assert fit.model.shape == pytest.approx(reference.x[2], abs=1e-4)
    for value, truth, error in zip(
        (fit.model.location, fit.model.scale, fit.model.shape), (10.0, 3.0, 0.2), fit.std_errors
    ):
        assert abs(value - truth) < 3.0 * error
    level = fit.model.return_level(100.0)
    error = fit.return_level_std_error(100.0)
    truth = stats.genextreme.ppf(0.99, -0.2, loc=10.0, scale=3.0)
    assert abs(level - truth) < 3.0 * error
    assert fit.return_level_std_error([10.0, 100.0]).shape == (2,)


def test_block_maxima() -> None:
    np.testing.assert_array_equal(block_maxima([1.0, 5.0, 2.0, 4.0, 9.0], 2), [5.0, 4.0])
    with pytest.raises(InvalidInputError):
        block_maxima([1.0, 2.0], 0)
    with pytest.raises(InvalidInputError):
        block_maxima([1.0, 2.0], 1.5)


def test_margin_body_is_the_interpolated_ecdf() -> None:
    x = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.5, 6.0, 5.0, 3.5])
    margin = SemiParametricMargin.fit(x, lower_tail=None, upper_tail=None)
    # At the observations this is exactly to_pseudo_obs.
    np.testing.assert_allclose(margin.cdf(x), to_pseudo_obs(x), rtol=1e-15)
    assert margin.cdf(4.5) == pytest.approx(0.65)
    assert margin.quantile(0.65) == pytest.approx(4.5)
    # An empirical margin cannot extrapolate.
    assert margin.cdf(1e6) == 0.9 and margin.quantile(0.9999) == 9.0
    assert margin.lower_tail is None and margin.upper_tail is None and margin.n_obs == 9


def test_margin_recovers_student_t_and_extrapolates() -> None:
    rng = np.random.default_rng(3)
    x = stats.t.rvs(4, size=20_000, random_state=rng)
    margin = SemiParametricMargin.fit(x)
    upper, lower = margin.upper_tail, margin.lower_tail
    assert upper.n_exceedances == 2000 and lower.n_exceedances == 2000
    assert upper.probability == pytest.approx(2001 / 20001)
    # t(4) has tail shape 1/4; the fit above the 90 % quantile is still
    # short of the limit, so allow for that.
    assert 0.1 < upper.gpd.shape < 0.35 and 0.1 < lower.gpd.shape < 0.35

    grid = np.linspace(-3.0, 3.0, 25)
    np.testing.assert_allclose(margin.cdf(grid), stats.t.cdf(grid, 4), atol=0.012)
    # A level exceeded once in 40 000 draws: twice the sample size, so the
    # data alone say nothing about it. (Much further out the estimate of a
    # single sample is too noisy to assert on; the Rust suite checks there
    # that the error averages out over 40 samples.)
    level = stats.t.isf(2.5e-5, 4)
    assert abs(np.log((1.0 - margin.cdf(level)) / 2.5e-5)) < 0.7
    # Exact inverse everywhere.
    p = np.array([1e-9, 1e-4, 0.05, 0.1001, 0.5, 0.8999, 0.97, 1 - 1e-7])
    np.testing.assert_allclose(margin.cdf(margin.quantile(p)), p, rtol=1e-9)
    np.testing.assert_allclose(margin.quantile(margin.cdf(x[:100])), x[:100], rtol=1e-9, atol=1e-9)
    assert margin.return_level(1e6) == pytest.approx(margin.quantile(1 - 1e-6), rel=1e-8)
    assert "GPD(threshold=" in repr(margin)


def test_margin_threshold_by_value() -> None:
    x = stats.expon.rvs(size=3_000, random_state=np.random.default_rng(4))
    margin = SemiParametricMargin.fit(x, lower_tail=None, upper_threshold=2.0)
    upper = margin.upper_tail
    assert upper.threshold == x[x <= 2.0].max()
    assert upper.n_exceedances == int(np.sum(x > 2.0))
    # Exponential tail: shape 0, scale 1.
    assert abs(upper.gpd.shape) < 0.15 and upper.gpd.scale == pytest.approx(1.0, abs=0.15)
    assert margin.lower_tail is None


def test_full_pipeline_estimates_a_joint_tail_probability() -> None:
    theta, n = 2.0, 20_000
    u = GumbelCopula.from_params(2, theta).sample(n, seed=11)
    raw = np.column_stack([stats.lomax.ppf(u[:, 0], 3.0), stats.expon.ppf(u[:, 1], scale=2.0)])

    margins = Margins.fit(raw)
    assert margins.dim == 2 and len(margins.margins) == 2
    uniforms = margins.cdf(raw)
    assert 0.0 < uniforms.min() and uniforms.max() < 1.0
    np.testing.assert_allclose(margins.quantile(uniforms), raw, rtol=1e-8, atol=1e-8)

    model = GumbelCopula.fit(uniforms).model
    assert model.theta == pytest.approx(theta, abs=0.08)
    simulated = margins.quantile(model.sample(1_000_000, seed=12))

    p = 0.9995
    a, b = stats.lomax.ppf(p, 3.0), stats.expon.ppf(p, scale=2.0)
    joint = np.exp(-((-np.log(p)) ** theta * 2) ** (1 / theta))
    exact = 1.0 - 2.0 * p + joint
    estimate = np.mean((simulated[:, 0] > a) & (simulated[:, 1] > b))
    assert abs(np.log(estimate / exact)) < 0.35
    # Ranks alone cannot place a simulated point beyond the sample maximum.
    assert simulated[:, 0].max() > raw[:, 0].max()
    np.testing.assert_allclose(
        margins.return_level(1e5), [m.return_level(1e5) for m in margins.margins]
    )


def test_margins_serialize_copy_and_compare() -> None:
    raw = stats.norm.rvs(size=(400, 3), random_state=np.random.default_rng(6))
    margins = Margins.fit(raw, lower_tail=0.05, upper_tail=0.2)
    assert [m.upper_tail.n_exceedances for m in margins.margins] == [80, 80, 80]
    restored = Margins.from_json(margins.to_json())
    assert restored == margins
    assert pickle.loads(pickle.dumps(margins)) == margins
    assert copy.deepcopy(margins) == margins
    np.testing.assert_array_equal(restored.cdf(raw), margins.cdf(raw))
    assert margins != Margins.fit(raw)

    # Per-column settings.
    parts = [
        SemiParametricMargin.fit(raw[:, 0]),
        SemiParametricMargin.fit(raw[:, 1], lower_tail=None, upper_threshold=1.0),
    ]
    combined = Margins.from_margins(parts)
    assert combined.dim == 2 and combined.margins[1] == parts[1]
    np.testing.assert_array_equal(combined.cdf(raw[:, :2])[:, 1], parts[1].cdf(raw[:, 1]))

    payload = json.loads(margins.to_json())
    payload["margins"][0]["probabilities"][0] = 0.99
    with pytest.raises(InvalidInputError, match="inconsistent"):
        Margins.from_json(json.dumps(payload))


def test_threshold_diagnostics_follow_theory() -> None:
    alpha = 4.0
    x = stats.lomax.rvs(alpha, size=100_000, random_state=np.random.default_rng(9))
    thresholds = np.array([0.0, 0.2, 0.5, 1.0])

    excess = mean_excess(x, thresholds)
    # Lomax: mean excess over u is (1 + u) / (alpha - 1).
    assert np.all(np.abs(excess.mean_excess - (1 + thresholds) / (alpha - 1)) < 5 * excess.std_error)
    for index, u in enumerate(thresholds):
        above = x[x > u] - u
        assert excess.n_exceedances[index] == above.size
        assert excess.mean_excess[index] == pytest.approx(above.mean(), rel=1e-12)
        assert excess.std_error[index] == pytest.approx(stats.sem(above), rel=1e-10)

    stability = threshold_stability(x, thresholds)
    # Lomax: shape 1 / alpha and modified scale 1 / alpha at every threshold.
    assert np.all(np.abs(stability.shape - 1 / alpha) < 4 * stability.shape_std_error)
    assert np.all(
        np.abs(stability.modified_scale - 1 / alpha) < 4 * stability.modified_scale_std_error
    )
    direct = GeneralizedPareto.fit(x[x > 0.5] - 0.5)
    assert stability.shape[2] == direct.model.shape
    assert stability.shape_std_error[2] == pytest.approx(direct.std_errors[1])

    # Defaults: 40 quantiles from the median up, 30 points left above the last.
    default = mean_excess(x)
    assert default.threshold.size == 40 and default.n_exceedances[-1] == 30
    with pytest.raises(InvalidInputError):
        mean_excess(x[:50])
    with pytest.raises(InvalidInputError):
        threshold_stability(x, [1e9])


def test_invalid_input_is_reported() -> None:
    x = np.random.default_rng(0).normal(size=200)
    with pytest.raises(InvalidInputError):
        SemiParametricMargin.fit(x, upper_tail=0.5)
    with pytest.raises(InvalidInputError):
        SemiParametricMargin.fit(x, upper_tail=0.01)  # 2 exceedances
    with pytest.raises(InvalidInputError):
        SemiParametricMargin.fit(np.where(x > 2, np.nan, x))
    with pytest.raises(InvalidInputError):
        SemiParametricMargin.fit(x.reshape(-1, 2))
    with pytest.raises(InvalidInputError):
        Margins.fit(x)
    margins = Margins.fit(x.reshape(-1, 2), lower_tail=0.2, upper_tail=0.2)
    with pytest.raises(InvalidInputError):
        margins.cdf(x.reshape(-1, 4))
    with pytest.raises(InvalidInputError):
        margins.quantile(np.array([[0.5, 1.0]]))
    with pytest.raises(InvalidInputError):
        Margins.from_margins([])
    with pytest.raises(InvalidInputError):
        GeneralizedPareto.fit([1.0, -2.0, 3.0])
    with pytest.raises(ValueError):  # InvalidInputError is a ValueError
        GeneralizedExtremeValue.fit([1.0, 2.0])


def test_public_names_are_exported() -> None:
    for name in (
        "GeneralizedExtremeValue",
        "GeneralizedPareto",
        "GevFit",
        "GpdFit",
        "GpdTail",
        "Margins",
        "MeanExcess",
        "SemiParametricMargin",
        "ThresholdStability",
        "block_maxima",
        "mean_excess",
        "threshold_stability",
    ):
        assert name in rscopulas.__all__
