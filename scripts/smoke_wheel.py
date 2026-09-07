"""Exercise the installed distribution (run with python -I)."""

import numpy as np

from rscopulas import FactorCopula, GaussianCopula, PairCopula, VineCopula

gaussian = GaussianCopula.from_params([[1.0, 0.6], [0.6, 1.0]])
data = gaussian.sample(20, seed=7)
assert data.shape == (20, 2)
assert np.isfinite(gaussian.log_pdf(data)).all()
pair = PairCopula.from_spec("clayton", [2.0], rotation="R90")
assert np.isfinite(pair.log_pdf(data[:, 0], data[:, 1])).all()
vine = VineCopula.fit_r(data, family_set=["gaussian"]).model
np.testing.assert_allclose(vine.inverse_rosenblatt(vine.rosenblatt(data)), data, atol=1e-8)
factor = FactorCopula.from_links([{"family": "gaussian", "parameters": [0.7]}] * 2)
assert np.isfinite(FactorCopula.from_json(factor.to_json()).log_pdf(data)).all()
print("Installed wheel smoke checks passed")
