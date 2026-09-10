"""Portfolio tail risk: R-vine versus Gaussian copula.

A worked example that runs offline in well under a minute with NumPy only
(matplotlib is optional and only used to save a figure). It answers the
question a risk desk actually asks: *does the copula matter for the tail?*

1. Simulate 1500 days of returns for five assets from a **known** mixed vine
   (Clayton and survival-Gumbel edges give lower-tail dependence, a Student-t
   edge symmetric tail dependence, Frank/Gaussian edges none) with Student-t(4)
   marginals, whose quantile function has a closed form in NumPy.
2. Rank-transform the returns to pseudo-observations and fit an R-vine with a
   parametric family set plus a Gaussian copula.
3. Estimate 1-day 99 % portfolio VaR and ES by simulation from each fitted
   copula, mapping simulated uniforms through the empirical marginals of the
   history, next to the same numbers from the true copula.
4. Stress scenario: pin one asset at its 1st percentile with
   ``VineCopula.sample_conditional`` and compare the conditional portfolio
   loss distribution of the vine with the Gaussian copula's analytic
   conditional and with the true model (a thin slice of a large sample).
5. Print a small table and, if matplotlib is importable, save
   ``python/examples/output/portfolio_tail_risk.png``.

``sample_conditional`` needs the pinned column to be the fitted vine's
Rosenblatt anchor, ``model.variable_order[0]``; the script stresses that asset.
To stress a chosen asset instead, fit with ``VineCopula.fit_c(u, order=[...,
asset])`` (or ``fit_d``), which places it at the anchor.

Run from the repository root::

    python python/examples/portfolio_tail_risk.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

import rscopulas as rc

ASSETS = ["Small caps", "Credit", "Commodities", "FX carry", "Equity index"]
WEIGHTS = np.array([0.15, 0.25, 0.15, 0.10, 0.35])
DAILY_VOL = np.array([0.015, 0.006, 0.013, 0.008, 0.011])
N_HISTORY = 1500
N_SCENARIOS = 100_000
N_CONDITIONAL = 40_000
N_TRUTH_SLICE_SAMPLE = 1_000_000
STRESS_LEVEL = 0.01
ALPHA = 0.99
SEED = 20260910
# Parametric families only: the fitter's default set also includes the
# nonparametric TLL and Khoudraji candidates, which take about a minute at
# this sample size and are not needed to make the point.
FAMILY_SET = ["independence", "gaussian", "student_t", "clayton", "frank", "gumbel", "joe"]
OUTPUT = Path(__file__).resolve().parent / "output" / "portfolio_tail_risk.png"


def true_copula() -> rc.VineCopula:
    """A 5-dimensional D-vine on the path 0-1-2-3-4 with mixed tail behaviour."""
    edge_specs = {
        (1, 0): ("clayton", "R0", [2.0]),  # small caps – credit: lower-tail dependence
        (1, 1): ("gumbel", "R180", [2.2]),  # credit – commodities: lower-tail dependence
        (1, 2): ("student_t", "R0", [0.5, 4.0]),  # commodities – FX carry: symmetric tails
        (1, 3): ("clayton", "R0", [2.6]),  # FX carry – equity index: lower-tail dependence
        (2, 0): ("gumbel", "R90", [1.3]),
        (2, 1): ("clayton", "R270", [0.7]),
        (2, 2): ("gaussian", "R0", [0.3]),
        (3, 0): ("frank", "R0", [1.5]),
        (3, 1): ("clayton", "R180", [0.5]),
        (4, 0): ("gaussian", "R0", [0.1]),
    }
    order = list(range(len(ASSETS)))
    trees = []
    for level in range(1, len(ASSETS)):
        edges = []
        for k in range(len(ASSETS) - level):
            family, rotation, parameters = edge_specs[(level, k)]
            edges.append(
                {
                    "tree": level,
                    "conditioned": (order[k], order[k + level]),
                    "conditioning": [order[m] for m in range(k + 1, k + level)],
                    "family": family,
                    "rotation": rotation,
                    "parameters": parameters,
                }
            )
        trees.append({"level": level, "edges": edges})
    return rc.VineCopula.from_trees("r", trees)


def t4_quantile(p: np.ndarray) -> np.ndarray:
    """Quantile function of Student-t with 4 degrees of freedom (closed form)."""
    p = np.asarray(p, dtype=np.float64)
    alpha = 4.0 * p * (1.0 - p)
    q = np.cos(np.arccos(np.sqrt(alpha)) / 3.0) / np.sqrt(alpha)
    return np.sign(p - 0.5) * 2.0 * np.sqrt(q - 1.0)


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(x, axis=0), axis=0) + 1
    return ranks / (x.shape[0] + 1)


_erf = np.vectorize(math.erf, otypes=[np.float64])


def normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=np.float64) / math.sqrt(2.0)))


def normal_quantile(p: float) -> float:
    lower, upper = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lower + upper)
        if float(normal_cdf(np.array(mid))) < p:
            lower = mid
        else:
            upper = mid
    return 0.5 * (lower + upper)


def portfolio_returns(u: np.ndarray, history: np.ndarray) -> np.ndarray:
    """Map copula scenarios through the empirical marginals of ``history``."""
    columns = [np.quantile(history[:, j], u[:, j]) for j in range(history.shape[1])]
    return np.column_stack(columns) @ WEIGHTS


def var_es(pnl: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    losses = -pnl
    var = float(np.quantile(losses, alpha))
    es = float(losses[losses >= var].mean())
    return var, es


def gaussian_conditional_sample(
    correlation: np.ndarray, anchor: int, level: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Exact conditional sample of a Gaussian copula given ``U[anchor] = level``."""
    d = correlation.shape[0]
    rest = [j for j in range(d) if j != anchor]
    z_anchor = normal_quantile(level)
    sigma_ra = correlation[np.ix_(rest, [anchor])]
    sigma_rr = correlation[np.ix_(rest, rest)]
    mean = (sigma_ra * z_anchor).ravel()
    cov = sigma_rr - sigma_ra @ sigma_ra.T
    z = rng.multivariate_normal(mean, cov, size=n, method="cholesky")
    u = np.empty((n, d))
    u[:, rest] = normal_cdf(z)
    u[:, anchor] = level
    return np.clip(u, 1e-12, 1.0 - 1e-12)


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    truth = true_copula()

    # 1. Simulated history with heavy-tailed marginals (t4 has variance 2).
    u_history = truth.sample(N_HISTORY, seed=SEED)
    history = DAILY_VOL * t4_quantile(u_history) / math.sqrt(2.0)
    u = pseudo_obs(history)

    # 2. Fit an R-vine and a Gaussian copula to the pseudo-observations.
    t0 = time.perf_counter()
    vine = rc.VineCopula.fit_r(u, family_set=FAMILY_SET).model
    fit_seconds = time.perf_counter() - t0
    gaussian = rc.GaussianCopula.fit(u).model
    print(f"Fitted R-vine in {fit_seconds:.1f} s; first-tree edges:")
    for edge in vine.trees[0].edges:
        a, b = edge.conditioned
        params = ", ".join(f"{p:.2f}" for p in edge.parameters)
        print(f"  {ASSETS[a]:<12} - {ASSETS[b]:<12} {edge.family} {edge.rotation} [{params}]")

    # 3. Unconditional 1-day VaR / ES by simulation.
    models = {"true copula": truth, "R-vine": vine, "Gaussian copula": gaussian}
    pnl = {name: portfolio_returns(model.sample(N_SCENARIOS, seed=SEED + 1), history) for name, model in models.items()}
    print(f"\n1-day {ALPHA:.0%} portfolio VaR / ES ({N_SCENARIOS:,} scenarios, empirical marginals):")
    print(f"  {'model':<16} {'VaR':>8} {'ES':>8}")
    risk = {}
    for name, series in pnl.items():
        var, es = var_es(series)
        risk[name] = (var, es)
        print(f"  {name:<16} {var:8.2%} {es:8.2%}")

    # 4. Conditional stress: pin the vine's anchor asset at its 1st percentile.
    anchor = vine.variable_order[0]
    print(f"\nStress scenario: {ASSETS[anchor]} pinned at its {STRESS_LEVEL:.0%} quantile "
          f"(a {np.quantile(history[:, anchor], STRESS_LEVEL):.1%} daily return in the simulated history)")
    known = {anchor: np.full(N_CONDITIONAL, STRESS_LEVEL)}
    u_vine = vine.sample_conditional(known, N_CONDITIONAL, seed=SEED + 2)
    u_gauss = gaussian_conditional_sample(gaussian.correlation, anchor, STRESS_LEVEL, N_CONDITIONAL, rng)
    big = truth.sample(N_TRUTH_SLICE_SAMPLE, seed=SEED + 3)
    u_true = big[np.abs(big[:, anchor] - STRESS_LEVEL) < STRESS_LEVEL / 2.0]
    conditional = {
        "true copula": portfolio_returns(u_true, history),
        "R-vine": portfolio_returns(u_vine, history),
        "Gaussian copula": portfolio_returns(u_gauss, history),
    }
    unconditional_var = risk["true copula"][0]
    print(f"  {'model':<16} {'mean':>8} {'median':>8} {'5% q':>8} {'VaR99':>8} {'ES99':>8} {'P(loss>VaR99)':>14}")
    for name, series in conditional.items():
        var, es = var_es(series)
        exceed = float(np.mean(-series > unconditional_var))
        print(
            f"  {name:<16} {series.mean():8.2%} {np.median(series):8.2%} {np.quantile(series, 0.05):8.2%} "
            f"{var:8.2%} {es:8.2%} {exceed:14.1%}"
        )
    print(f"  (P(loss>VaR99) uses the unconditional true-copula VaR of {unconditional_var:.2%}; "
          f"true-copula row uses {len(u_true):,} rows with U[anchor] within ±{STRESS_LEVEL / 2:.1%} of the level)")

    # 5. Figure (optional).
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed; skipping the figure.")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        colours = {"true copula": "#4a4a4a", "R-vine": "#1f77b4", "Gaussian copula": "#d62728"}
        grid = np.linspace(0.90, 0.9995, 200)
        for name, series in pnl.items():
            axes[0].plot(grid, np.quantile(-series, grid), label=name, color=colours[name])
        axes[0].set_xlabel("confidence level")
        axes[0].set_ylabel("portfolio loss")
        axes[0].set_title("Unconditional loss quantiles (VaR curve)")
        axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axes[0].legend(frameon=False)
        bins = np.linspace(min(s.min() for s in conditional.values()), 0.02, 80)
        for name, series in conditional.items():
            axes[1].hist(series, bins=bins, density=True, histtype="step", linewidth=1.5, label=name, color=colours[name])
        axes[1].axvline(-unconditional_var, color="black", linestyle=":", linewidth=1, label="unconditional VaR99")
        axes[1].set_xlabel("portfolio return")
        axes[1].set_title(f"Conditional on {ASSETS[anchor]} at its 1st percentile")
        axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axes[1].legend(frameon=False)
        fig.tight_layout()
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=130)
        print(f"\nSaved {OUTPUT}")

    print(f"\nTotal wall time {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
