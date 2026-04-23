"""Python-level tests for the Krupskii–Joe factor copula binding.

Mirrors the structure of the Rust `reference_factor.rs` suite: construction,
log_pdf numerics, sampling validity, serialization round-trip, and parameter
recovery on a mono-family Gaussian DGP. The Clayton-DGP selection test lives
on the Rust side because it exercises internal fit decisions we don't surface
directly through the Python API.
"""

from __future__ import annotations

import numpy as np
import pytest

from rscopulas import FactorCopula, FactorFitDiagnostics


def _reference_links() -> list[dict[str, object]]:
    return [
        {"family": "gaussian", "rotation": "R0", "parameters": [0.7]},
        {"family": "gaussian", "rotation": "R0", "parameters": [0.6]},
        {"family": "clayton", "rotation": "R0", "parameters": [2.0]},
        {"family": "gumbel", "rotation": "R0", "parameters": [1.8]},
        {"family": "gumbel", "rotation": "R0", "parameters": [2.2]},
    ]


def test_factor_copula_from_links_reports_basic_fields() -> None:
    model = FactorCopula.from_links(_reference_links(), quadrature_nodes=25)

    assert model.family == "factor"
    assert model.dim == 5
    assert model.num_factors == 1
    assert model.layout == "basic_1f"
    assert model.quadrature_nodes == 25
    assert len(model.links) == 5
    # Each link is a dict with at least these keys — matches PairCopula layout.
    for link in model.links:
        assert "family" in link
        assert "rotation" in link
        assert "parameters" in link


def test_factor_copula_sample_and_log_pdf_are_consistent() -> None:
    model = FactorCopula.from_links(_reference_links())
    sample = model.sample(256, seed=7)
    assert sample.shape == (256, 5)
    assert np.all(sample > 0.0)
    assert np.all(sample < 1.0)

    log_pdf = model.log_pdf(sample)
    assert log_pdf.shape == (256,)
    assert np.all(np.isfinite(log_pdf))

    # log_pdf is deterministic.
    log_pdf_again = model.log_pdf(sample)
    np.testing.assert_array_equal(log_pdf, log_pdf_again)


def test_factor_copula_to_json_round_trip_preserves_log_pdf() -> None:
    model = FactorCopula.from_links(_reference_links())
    payload = model.to_json()
    assert isinstance(payload, str)
    assert len(payload) > 0

    restored = FactorCopula.from_json(payload)
    assert restored.dim == model.dim
    assert restored.num_factors == model.num_factors
    assert restored.layout == model.layout
    assert restored.quadrature_nodes == model.quadrature_nodes

    sample = model.sample(64, seed=11)
    a = model.log_pdf(sample)
    b = restored.log_pdf(sample)
    np.testing.assert_array_equal(a, b)


def test_factor_copula_fit_recovers_gaussian_structure() -> None:
    # Mono-family Gaussian DGP — sequential MLE should recover rhos cleanly.
    truth = FactorCopula.from_links(
        [
            {"family": "gaussian", "rotation": "R0", "parameters": [0.75]},
            {"family": "gaussian", "rotation": "R0", "parameters": [0.65]},
            {"family": "gaussian", "rotation": "R0", "parameters": [0.55]},
            {"family": "gaussian", "rotation": "R0", "parameters": [0.45]},
        ]
    )
    sample = truth.sample(2000, seed=42)

    fit = FactorCopula.fit(
        sample,
        family_set=["gaussian"],
        include_rotations=False,
    )

    assert fit.model.family == "factor"
    assert fit.model.dim == 4
    assert isinstance(fit.diagnostics, FactorFitDiagnostics)
    assert fit.diagnostics.loglik > 0
    assert fit.diagnostics.converged

    # Every link should come out Gaussian (only family in the set).
    recovered_rhos = []
    for link in fit.model.links:
        assert link["family"] == "gaussian"
        params = link["parameters"]
        assert len(params) == 1
        recovered_rhos.append(float(params[0]))

    truth_rhos = [0.75, 0.65, 0.55, 0.45]
    for got, expected in zip(recovered_rhos, truth_rhos):
        assert 0.0 < got < 1.0, f"recovered rho {got} escaped (0, 1)"
        # Polish removes the attenuation bias that forced the old 0.15 tol.
        assert abs(got - expected) < 0.08, (
            f"rho recovery: got {got}, expected near {expected}"
        )

    # Delta-method standard errors: one per polished ρ, all positive finite.
    assert len(fit.diagnostics.std_errors) == len(truth_rhos)
    for se in fit.diagnostics.std_errors:
        assert np.isfinite(se) and se > 0


def test_factor_copula_fit_uses_default_family_set_when_unspecified() -> None:
    truth = FactorCopula.from_links(_reference_links())
    sample = truth.sample(1000, seed=13)
    fit = FactorCopula.fit(sample)
    assert fit.diagnostics.loglik > 50.0
    assert fit.model.dim == 5


def test_factor_copula_fit_rejects_unknown_layout() -> None:
    sample = FactorCopula.from_links(_reference_links()).sample(128, seed=1)
    with pytest.raises(Exception):
        FactorCopula.fit(sample, layout="not_a_layout")


def test_factor_copula_log_pdf_rejects_wrong_dimension() -> None:
    model = FactorCopula.from_links(_reference_links())  # dim = 5
    wrong = np.array(
        [
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
        ],
        dtype=np.float64,
    )
    with pytest.raises(Exception):
        model.log_pdf(wrong)


def test_factor_copula_from_links_rejects_degenerate_dim() -> None:
    with pytest.raises(Exception):
        FactorCopula.from_links(
            [{"family": "gaussian", "rotation": "R0", "parameters": [0.5]}]
        )


def test_factor_copula_polish_cycles_zero_disables_polish() -> None:
    # With polish disabled, the fit reduces to sequential+EM — loglik must be
    # ≤ the polished fit's loglik (polish is guarded by a bail-out so it can
    # only match or improve, never degrade).
    truth = FactorCopula.from_links(_reference_links())
    sample = truth.sample(1500, seed=17)

    fit_no_polish = FactorCopula.fit(sample, joint_polish_cycles=0)
    fit_polished = FactorCopula.fit(sample, joint_polish_cycles=5)

    assert fit_polished.diagnostics.loglik + 1e-6 >= fit_no_polish.diagnostics.loglik

    # Both return SE vectors of the same length (polish doesn't change the
    # parameter layout, only the parameter values).
    assert len(fit_polished.diagnostics.std_errors) == len(fit_no_polish.diagnostics.std_errors)


def test_factor_copula_from_links_rejects_too_few_quadrature_nodes() -> None:
    with pytest.raises(Exception):
        FactorCopula.from_links(
            [
                {"family": "gaussian", "rotation": "R0", "parameters": [0.5]},
                {"family": "gaussian", "rotation": "R0", "parameters": [0.5]},
            ],
            quadrature_nodes=1,
        )
