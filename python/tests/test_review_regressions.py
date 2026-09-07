import json

import numpy as np
import pytest

from rscopulas import FactorCopula, HierarchicalArchimedeanCopula, NumericalError


def test_factor_integration_controls_apply_to_fitting_and_round_trip():
    links = [{"family": "clayton", "parameters": [3.0]}] * 3
    data = FactorCopula.from_links(links).sample(40, seed=71)
    with pytest.raises(NumericalError, match="node budget"):
        FactorCopula.from_links(links, quadrature_max_nodes=48).log_pdf(data)

    fit = FactorCopula.fit(
        data, family_set=["clayton"], include_rotations=False,
        adaptive_quadrature=False, quadrature_nodes=60,
        quadrature_max_nodes=48, quadrature_rel_tol=1e-6,
        refine_iterations=0, joint_polish_cycles=0,
    )
    assert fit.model.adaptive_quadrature is False
    assert fit.model.quadrature_nodes == 60
    assert fit.model.quadrature_max_nodes == 48
    assert fit.model.quadrature_rel_tol == 1e-6
    np.testing.assert_allclose(fit.diagnostics.loglik, fit.model.log_pdf(data).sum(), atol=1e-9)
    restored = FactorCopula.from_json(fit.model.to_json())
    assert restored.adaptive_quadrature is False
    assert restored.quadrature_max_nodes == 48
    np.testing.assert_array_equal(restored.log_pdf(data), fit.model.log_pdf(data))

    state = json.loads(fit.model.to_json())
    state["quadrature"]["rel_tol"] = -1.0
    with pytest.raises(ValueError):
        FactorCopula.from_json(json.dumps(state))


@pytest.mark.parametrize("family,theta", [("clayton", 30.0), ("gumbel", 10.0), ("joe", 40.0)])
def test_factor_tail_observations_do_not_abort_the_batch(family, theta):
    model = FactorCopula.from_links([{"family": family, "parameters": [theta]}] * 3)
    data = np.array([[0.2, 0.3, 0.4], [1e-12] * 3, [1 - 1e-10] * 3])
    assert np.isfinite(model.log_pdf(data)).all()


def test_hac_rejects_unused_monte_carlo_budget():
    tree = {"family": "clayton", "theta": 1.0, "children": [0, 1]}
    model = HierarchicalArchimedeanCopula.from_tree(tree)
    with pytest.raises(Exception, match="mc_samples must be zero"):
        HierarchicalArchimedeanCopula.fit(model.sample(20, seed=7), tree=tree, mc_samples=256)
