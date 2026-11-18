"""Reference fits for the generalized Pareto and generalized extreme value models.

Two outside references, neither sharing code with rscopulas:

1. SciPy. The log-likelihood is `scipy.stats.genpareto.logpdf` /
   `genextreme.logpdf`, maximised here with Nelder-Mead at tight tolerances
   from several starts (SciPy's own `.fit` stops at about four digits).
   Standard errors come from a Richardson-extrapolated Hessian of that
   log-likelihood (`numdifftools` is not needed: the stencil is written out).
2. R. `evd::fpot` / `evd::fgev` estimates and standard errors, and
   `extRemes::fevd` + `ci` for the delta-method return-level interval. This
   script runs `generate_evt_fixtures.R` on the same data and stores the
   result as an `r` block in every case.

Usage:

    pip install scipy          # and R with the evd and extRemes packages
    python scripts/reference/generate_evt_fixtures.py

Writes `fixtures/reference/evt/fits.json`, read by
`crates/rscopulas-core/tests/reference_margins.rs`.

Shape convention in the fixture: xi > 0 is a heavy tail (SciPy's genextreme
uses c = -xi).
"""

import json
import pathlib
import subprocess
import tempfile

import numpy as np
from scipy import optimize, stats

OUT = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "reference" / "evt"

GPD_CASES = [
    ("heavy", 2.0, 0.3, 500),
    ("bounded", 1.5, -0.2, 300),
    ("exponential", 1.0, 0.0, 200),
    ("very_heavy", 0.7, 0.8, 1000),
    ("small_sample", 3.0, 0.1, 30),
    ("light_bounded", 4.0, -0.4, 400),
]
GEV_CASES = [
    ("frechet", 10.0, 3.0, 0.2, 200),
    ("weibull", -5.0, 2.0, -0.3, 150),
    ("gumbel", 0.0, 1.0, 0.0, 100),
    ("heavy", 100.0, 20.0, 0.5, 500),
    ("small_sample", 2.0, 0.5, 0.1, 40),
]


def maximise(nll, starts):
    best = None
    for start in starts:
        result = optimize.minimize(
            nll, start, method="Nelder-Mead",
            options={"xatol": 1e-13, "fatol": 1e-13, "maxiter": 200_000, "maxfev": 200_000},
        )
        # Restart from the optimum: Nelder-Mead can stall on a collapsed simplex.
        for _ in range(4):
            result = optimize.minimize(
                nll, result.x, method="Nelder-Mead",
                options={"xatol": 1e-13, "fatol": 1e-13, "maxiter": 200_000, "maxfev": 200_000},
            )
        if best is None or result.fun < best.fun:
            best = result
    return best


def hessian(f, x, rel=1e-3):
    """Central second differences with one Richardson step."""
    x = np.asarray(x, dtype=float)
    scale = np.maximum(np.abs(x), 1.0)

    def at(h):
        n = len(x)
        out = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                ei = np.zeros(n); ej = np.zeros(n)
                ei[i] = h * scale[i]; ej[j] = h * scale[j]
                value = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (
                    4 * ei[i] * ej[j]
                )
                out[i, j] = out[j, i] = value
        return out

    return (4 * at(rel / 2) - at(rel)) / 3


def gpd_case(name, scale, shape, n, rng):
    y = stats.genpareto.rvs(shape, scale=scale, size=n, random_state=rng)

    def nll(p):
        sigma, xi = p
        if sigma <= 0:
            return np.inf
        value = -np.sum(stats.genpareto.logpdf(y, xi, scale=sigma))
        return value if np.isfinite(value) else np.inf

    best = maximise(nll, [(np.mean(y), 0.1), (np.mean(y) * 0.5, 0.5), (np.mean(y) * 1.5, -0.1)])
    cov = np.linalg.inv(hessian(nll, best.x))
    return {
        "name": name, "true": {"scale": scale, "shape": shape}, "excesses": y.tolist(),
        "scipy": {
            "scale": best.x[0], "shape": best.x[1], "loglik": -best.fun,
            "std_errors": np.sqrt(np.diag(cov)).tolist(),
        },
    }


def gev_case(name, loc, scale, shape, n, rng):
    x = stats.genextreme.rvs(-shape, loc=loc, scale=scale, size=n, random_state=rng)

    def nll(p):
        mu, sigma, xi = p
        if sigma <= 0:
            return np.inf
        value = -np.sum(stats.genextreme.logpdf(x, -xi, loc=mu, scale=sigma))
        return value if np.isfinite(value) else np.inf

    sd = np.std(x)
    starts = [(np.mean(x) - 0.45 * sd, 0.78 * sd, 0.0), (np.median(x), sd, 0.2), (np.mean(x), 2 * sd, 0.1)]
    best = maximise(nll, starts)
    cov = np.linalg.inv(hessian(nll, best.x))
    mu, sigma, xi = best.x
    levels = {}
    for period in (10.0, 100.0, 1000.0):
        level = stats.genextreme.ppf(1 - 1 / period, -xi, loc=mu, scale=sigma)

        def at(p):
            return stats.genextreme.ppf(1 - 1 / period, -p[2], loc=p[0], scale=p[1])

        grad = np.zeros(3)
        for i, h in enumerate((1e-6 * sigma, 1e-6 * sigma, 1e-6)):
            e = np.zeros(3); e[i] = h
            grad[i] = (at(best.x + e) - at(best.x - e)) / (2 * h)
        levels[str(int(period))] = {"level": float(level), "std_error": float(np.sqrt(grad @ cov @ grad))}
    return {
        "name": name, "true": {"location": loc, "scale": scale, "shape": shape}, "maxima": x.tolist(),
        "scipy": {
            "location": mu, "scale": sigma, "shape": xi, "loglik": -best.fun,
            "std_errors": np.sqrt(np.diag(cov)).tolist(), "return_levels": levels,
        },
    }


def distribution_values():
    """cdf, survival, log-density and quantiles from SciPy on fixed grids."""
    probabilities = [1e-9, 1e-4, 0.01, 0.25, 0.5, 0.9, 0.999, 1 - 1e-9]
    gpd = []
    for scale, shape in [(2.0, 0.3), (1.5, -0.2), (1.0, 0.0), (0.7, 0.8), (3.0, 1e-9), (2.0, -1.0)]:
        y = [1e-8, 0.01, 0.5, 1.0, 3.0, 7.4, 50.0, 1e4]
        dist = stats.genpareto(shape, scale=scale)
        gpd.append({
            "scale": scale, "shape": shape, "y": y,
            "cdf": dist.cdf(y).tolist(), "survival": dist.sf(y).tolist(),
            "ln_pdf": [v if np.isfinite(v) else None for v in dist.logpdf(y).tolist()],
            "p": probabilities, "quantile": dist.ppf(probabilities).tolist(),
        })
    gev = []
    for loc, scale, shape in [(10.0, 3.0, 0.2), (-5.0, 2.0, -0.3), (0.0, 1.0, 0.0), (100.0, 20.0, 0.5), (1.0, 2.0, 1e-9)]:
        x = [loc - 3 * scale, loc - scale, loc, loc + 0.5 * scale, loc + 3 * scale, loc + 30 * scale]
        dist = stats.genextreme(-shape, loc=loc, scale=scale)
        gev.append({
            "location": loc, "scale": scale, "shape": shape, "x": x,
            "cdf": dist.cdf(x).tolist(),
            "ln_pdf": [v if np.isfinite(v) else None for v in dist.logpdf(x).tolist()],
            "p": probabilities, "quantile": dist.ppf(probabilities).tolist(),
        })
    return {"gpd": gpd, "gev": gev}


def add_r_fits(payload):
    """Fit the same data in R and attach the estimates to each case."""
    script = pathlib.Path(__file__).with_suffix(".R")
    with tempfile.TemporaryDirectory() as directory:
        for kind, key in (("gpd", "excesses"), ("gev", "maxima")):
            for case in payload[kind]:
                path = pathlib.Path(directory) / f"{kind}_{case['name']}.txt"
                path.write_text("\n".join(repr(float(v)) for v in case[key]) + "\n")
        output = subprocess.run(
            ["Rscript", str(script), directory], check=True, capture_output=True, text=True
        ).stdout
    flat = {}
    for line in output.splitlines():
        kind, name, key, value = line.split()
        flat.setdefault((kind, name), {})[key] = float(value)
    for kind in ("gpd", "gev"):
        for case in payload[kind]:
            values = flat[(kind, case["name"])]
            names = ["scale", "shape"] if kind == "gpd" else ["location", "scale", "shape"]
            block = {name: values[name] for name in names}
            block["loglik"] = values["loglik"]
            block["std_errors"] = [values[f"se_{name}"] for name in names]
            if kind == "gev":
                block["return_levels"] = {
                    period: {"level": values[f"level_{period}"], "std_error": values[f"level_se_{period}"]}
                    for period in ("10", "100", "1000")
                }
            case["r"] = block


def main():
    rng = np.random.default_rng(20261118)
    payload = {
        "gpd": [gpd_case(*case, rng) for case in GPD_CASES],
        "gev": [gev_case(*case, rng) for case in GEV_CASES],
    }
    add_r_fits(payload)
    payload["distributions"] = distribution_values()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fits.json").write_text(json.dumps(payload, indent=1) + "\n")
    for kind in ("gpd", "gev"):
        for case in payload[kind]:
            for source in ("scipy", "r"):
                shown = {k: v for k, v in case[source].items() if k != "return_levels"}
                print(kind, case["name"], source, shown)


if __name__ == "__main__":
    main()
