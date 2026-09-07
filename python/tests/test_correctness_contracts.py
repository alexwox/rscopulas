import json
import subprocess
import sys

import numpy as np
import pytest

from rscopulas import (
    FactorCopula, GaussianCopula, HierarchicalArchimedeanCopula,
    InvalidInputError, ModelFitError, PairCopula, VineCopula,
)


def test_frank_extreme_sampling_finishes_in_a_subprocess():
    # A regression must time out instead of hanging the entire test runner.
    subprocess.run([
        sys.executable, "-c",
        "from rscopulas import FrankCopula; "
        "[FrankCopula.from_params(3,t).sample(10,seed=7) for t in (40.,1e-20)]",
    ], check=True, timeout=15)


def test_tll_specs_preserve_fitted_state_across_pair_vine_and_factor():
    data = GaussianCopula.from_params([[1., .6], [.6, 1.]]).sample(300, seed=17)
    pair = PairCopula.fit_tll(data[:, 0], data[:, 1])
    spec = json.loads(json.dumps(pair.spec))
    restored = PairCopula.from_spec(**spec)
    expected = pair.log_pdf(data[:, 0], data[:, 1])
    np.testing.assert_allclose(restored.log_pdf(data[:, 0], data[:, 1]), expected)
    trees = [{"level": 1, "edges": [{"conditioned": [0, 1], "conditioning": [], **spec}]}]
    vine = VineCopula.from_trees("r", trees)
    np.testing.assert_allclose(vine.log_pdf(data), expected)
    rebuilt = VineCopula.from_trees("r", vine.trees)
    np.testing.assert_allclose(rebuilt.log_pdf(data), expected)
    factor = FactorCopula.from_links([spec, {"family": "independence"}])
    np.testing.assert_allclose(PairCopula.from_spec(**factor.links[0]).log_pdf(data[:, 0], data[:, 1]), expected)
    np.testing.assert_allclose(factor.log_pdf(data[:4]), 0., atol=1e-10)
    mixed = FactorCopula.from_links([spec, spec])
    assert np.isfinite(mixed.log_pdf(data[:4])).all()
    invalid = json.loads(json.dumps(spec))
    invalid["state"]["bandwidth"] = -1
    with pytest.raises(ValueError):
        PairCopula.from_spec(**invalid)


@pytest.mark.parametrize("eps", [float("nan"), .6, 0., -1., 1e-30])
def test_invalid_clipping_raises_an_input_error(eps):
    model = GaussianCopula.from_params([[1., .5], [.5, 1.]])
    with pytest.raises(InvalidInputError):
        model.log_pdf([[.2, .3]], clip_eps=eps)


def test_malformed_factor_json_is_rejected_on_loading():
    factor = FactorCopula.from_links([{"family": "independence"}] * 2)
    payload = json.loads(factor.to_json())
    payload["links"].append(payload["links"][0])
    with pytest.raises(ValueError):
        FactorCopula.from_json(json.dumps(payload))


def test_independence_threshold_uses_selected_dependence_metric():
    data = GaussianCopula.from_params([[1., .75], [.75, 1.]]).sample(800, seed=12)
    options = dict(family_set=["gaussian"], independence_threshold=.55)
    rho = VineCopula.fit_r(data, tree_criterion="rho", **options)
    tau = VineCopula.fit_r(data, tree_criterion="tau", **options)
    assert rho.model.trees[0].edges[0].family == "gaussian"
    assert tau.model.trees[0].edges[0].family == "independence"


def test_nested_hac_requires_explicit_composite_scoring():
    tree = {"family": "gumbel", "theta": 1.2, "children": [
        0, {"family": "gumbel", "theta": 2., "children": [1, 2]},
    ]}
    model = HierarchicalArchimedeanCopula.from_tree(tree)
    data = model.sample(40, seed=4)
    with pytest.raises(ModelFitError, match="composite"):
        model.log_pdf(data)
    assert np.isfinite(model.composite_log_pdf(data)).all()
    fit = HierarchicalArchimedeanCopula.fit(data, tree=tree, family_set=["gumbel"], structure_method="given_tree")
    assert fit.diagnostics.likelihood_kind == "composite"
    assert np.isnan(fit.diagnostics.aic)
    assert np.isnan(fit.diagnostics.bic)
    for method in ("full_mle", "smle", "dmle"):
        with pytest.raises(ModelFitError, match="not implemented"):
            HierarchicalArchimedeanCopula.fit(data, fit_method=method)


def test_factor_diagnostics_count_completed_stages():
    model = FactorCopula.from_links([{"family": "gaussian", "parameters": [.7]}] * 3)
    fit = FactorCopula.fit(model.sample(40, seed=71), family_set=["gaussian"],
                           refine_iterations=0, joint_polish_cycles=0)
    assert fit.diagnostics.n_iter == 1
    assert not fit.diagnostics.converged
    assert fit.diagnostics.likelihood_kind == "joint"
    np.testing.assert_allclose(fit.diagnostics.loglik, fit.model.log_pdf(model.sample(40, seed=71)).sum())
