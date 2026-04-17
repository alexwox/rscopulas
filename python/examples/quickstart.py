import numpy as np

from rscopulas import GaussianCopula, VineCopula


data = np.array(
    [
        [0.12, 0.18, 0.21],
        [0.21, 0.25, 0.29],
        [0.27, 0.22, 0.31],
        [0.35, 0.42, 0.39],
        [0.48, 0.51, 0.46],
        [0.56, 0.49, 0.58],
        [0.68, 0.73, 0.69],
        [0.82, 0.79, 0.76],
    ],
    dtype=np.float64,
)

gaussian_fit = GaussianCopula.fit(data[:, :2])
print("Gaussian family:", gaussian_fit.model.family)
print("Gaussian AIC:", gaussian_fit.diagnostics.aic)
print("Gaussian samples:\n", gaussian_fit.model.sample(3, seed=7))

vine_fit = VineCopula.fit_r(
    data,
    family_set=["independence", "gaussian", "clayton", "frank", "gumbel"],
    truncation_level=1,
)
print("Vine structure:", vine_fit.model.structure_kind)
print("Vine order:", vine_fit.model.order)
print("Vine pair parameters:", vine_fit.model.pair_parameters)
