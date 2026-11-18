"""Reference tail-dependence coefficients, computed as limits.

rscopulas returns closed-form coefficients. This script never uses those
closed forms. It takes each family's textbook distribution function C and
evaluates the defining limits numerically with mpmath at 1300 digits:

    lower = C(q, q) / q                        at q = 1e-250
    upper = (1 - 2u + C(u, u)) / (1 - u)       at u = 1 - 1e-250

Each limit is evaluated a second time at 1e-125. The two values must agree
to 1e-10, which shows the limit has been reached rather than assumed. (The
slowest case is the lower tail of Gumbel(8), which decays like q^0.09. The
digits are needed by BB7 with theta = 4, whose upper tail forms
1 - (1 - u)^4 = 1 - 1e-1000.)

The Student t copula has no closed-form distribution function, so its
diagonal is integrated instead:

    C(q, q) = integral over x <= x_q of T_{nu+1}(w(x)) t_nu(x) dx,
    w(x) = (x_q - rho x) / sqrt((nu + x^2)(1 - rho^2) / (nu + 1))

at q = 1e-60 and 1e-45 (the error decays like q^(2/nu); agreement to 1e-9
is required, and nu <= 6 keeps that reachable).

Usage:

    pip install mpmath
    python scripts/reference/generate_tail_dependence_truth.py

Writes `fixtures/reference/mpmath/tail_dependence.json`, read by
`crates/rscopulas-core/tests/reference_tail_dependence.rs`.
"""

import json
import pathlib

from mpmath import betainc, exp, findroot, gamma, inf, log, mp, mpf, pi, quad, sqrt

mp.dps = 1300
ONE = mpf(1)
OUT = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "reference" / "mpmath"


def independence(u, v):
    return u * v


def frank(u, v, t):
    return -log(1 + (exp(-t * u) - 1) * (exp(-t * v) - 1) / (exp(-t) - 1)) / t


def gumbel(u, v, t):
    return exp(-(((-log(u)) ** t + (-log(v)) ** t) ** (ONE / t)))


def clayton(u, v, t):
    return (u**-t + v**-t - 1) ** (-ONE / t)


def joe(u, v, t):
    a, b = (1 - u) ** t, (1 - v) ** t
    return 1 - (a + b - a * b) ** (ONE / t)


def bb1(u, v, t, d):
    return (1 + ((u**-t - 1) ** d + (v**-t - 1) ** d) ** (ONE / d)) ** (-ONE / t)


def bb6(u, v, t, d):
    x, y = -log(1 - (1 - u) ** t), -log(1 - (1 - v) ** t)
    return 1 - (1 - exp(-((x**d + y**d) ** (ONE / d)))) ** (ONE / t)


def bb7(u, v, t, d):
    x, y = (1 - (1 - u) ** t) ** -d, (1 - (1 - v) ** t) ** -d
    return 1 - (1 - (x + y - 1) ** (-ONE / d)) ** (ONE / t)


def bb8(u, v, t, d):
    x, y = 1 - (1 - d * u) ** t, 1 - (1 - d * v) ** t
    return (1 - (1 - x * y / (1 - (1 - d) ** t)) ** (ONE / t)) / d


def tawn(u, v, t, alpha, beta):
    """Extreme-value copula exp(-z A(w)), z = -ln(uv), w = -ln(v) / z."""
    z = -(log(u) + log(v))
    w = -log(v) / z
    a = (1 - alpha) * (1 - w) + (1 - beta) * w + ((alpha * (1 - w)) ** t + (beta * w) ** t) ** (ONE / t)
    return exp(-z * a)


def tawn1(u, v, t, alpha):
    return tawn(u, v, t, alpha, ONE)


def tawn2(u, v, t, beta):
    return tawn(u, v, t, ONE, beta)


CLOSED_FORM = {
    "independence": (independence, [[]]),
    "clayton": (clayton, [[0.2], [1.0], [2.5], [9.0]]),
    "frank": (frank, [[-6.0], [0.5], [12.0]]),
    "gumbel": (gumbel, [[1.0], [1.3], [2.5], [8.0]]),
    "joe": (joe, [[1.0], [1.4], [3.0], [10.0]]),
    "bb1": (bb1, [[0.3, 1.0], [0.5, 1.6], [2.0, 3.0], [0.2, 6.0]]),
    "bb6": (bb6, [[1.0, 1.0], [1.5, 1.3], [3.0, 2.5]]),
    "bb7": (bb7, [[1.0, 0.4], [1.8, 1.2], [4.0, 3.0]]),
    "bb8": (bb8, [[1.5, 0.7], [4.0, 0.95], [6.0, 0.3], [3.0, 1.0]]),
    "tawn1": (tawn1, [[1.5, 0.3], [3.0, 0.8], [6.0, 1.0], [2.0, 0.0]]),
    "tawn2": (tawn2, [[1.5, 0.3], [3.0, 0.8], [6.0, 1.0], [2.0, 0.0]]),
}


def limits(cdf, params, eps):
    q = mpf(10) ** -eps
    lower = cdf(q, q, *params) / q
    u = 1 - q
    upper = (1 - 2 * u + cdf(u, u, *params)) / q
    return lower, upper


def closed_form_case(name, cdf, raw):
    params = [mpf(str(value)) for value in raw]
    coarse = limits(cdf, params, 125)
    fine = limits(cdf, params, 250)
    for a, b in zip(coarse, fine):
        assert abs(a - b) < mpf(10) ** -10, (name, raw, a, b)
    return {
        "family": name,
        "parameters": raw,
        "lower": float(fine[0]),
        "upper": float(fine[1]),
    }


def t_pdf(x, nu):
    return gamma((nu + 1) / 2) / (sqrt(nu * pi) * gamma(nu / 2)) * (1 + x * x / nu) ** (-(nu + 1) / 2)


def t_cdf(x, nu):
    """Student t distribution function through the regularised incomplete beta."""
    tail = betainc(nu / 2, mpf(1) / 2, 0, nu / (nu + x * x), regularized=True) / 2
    return tail if x <= 0 else 1 - tail


def t_lower(rho, nu, eps):
    q = mpf(10) ** -eps
    # Tail of t_nu: F(x) ~ c |x|^(-nu); start the root search from that.
    c = gamma((nu + 1) / 2) / (sqrt(nu * pi) * gamma(nu / 2)) * nu ** ((nu - 1) / 2)
    start = -((c / q) ** (1 / nu))
    x_q = findroot(lambda x: log(t_cdf(x, nu)) - log(q), start)
    assert abs(t_cdf(x_q, nu) / q - 1) < mpf(10) ** -30

    def integrand(s):
        # x = x_q * s with s >= 1 covers (-inf, x_q].
        x = x_q * s
        w = (x_q - rho * x) / sqrt((nu + x * x) * (1 - rho * rho) / (nu + 1))
        return t_cdf(w, nu + 1) * t_pdf(x, nu) * (-x_q)

    return quad(integrand, [1, 2, 10, 100, inf]) / q


def student_t_case(rho, nu):
    mp.dps = 80
    rho_mp, nu_mp = mpf(str(rho)), mpf(str(nu))
    coarse = t_lower(rho_mp, nu_mp, 45)
    fine = t_lower(rho_mp, nu_mp, 60)
    assert abs(coarse - fine) < mpf(10) ** -9, (rho, nu, coarse, fine)
    mp.dps = 1300
    # Radial symmetry: the upper coefficient equals the lower one.
    return {
        "family": "student_t",
        "parameters": [rho, nu],
        "lower": float(fine),
        "upper": float(fine),
    }


def main():
    cases = []
    for name, (cdf, grid) in CLOSED_FORM.items():
        for raw in grid:
            cases.append(closed_form_case(name, cdf, raw))
    for rho, nu in [(0.5, 3.0), (-0.4, 2.0), (0.9, 6.0), (0.0, 4.0), (0.7, 1.0)]:
        cases.append(student_t_case(rho, nu))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tail_dependence.json"
    path.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
