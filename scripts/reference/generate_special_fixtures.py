"""Reference values for `rscopulas::special` from SciPy (Boost.Math backend).

Run from the repository root:

    python scripts/reference/generate_special_fixtures.py

Writes fixtures/reference/special/student_t_scipy.json, consumed by the unit
tests in crates/rscopulas-core/src/special.rs.
"""

import json
from decimal import Decimal, getcontext
from pathlib import Path

import scipy
from scipy.special import ndtr, ndtri
from scipy.stats import t

NUS = [0.5, 1.0, 1.5, 2.0, 2.05, 2.5, 3.0, 4.0, 7.3, 15.0, 30.0, 200.0, 1000.0]
PROBABILITIES = (
    [1e-15, 1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49]
    + [0.5]
    + [0.51, 0.6, 0.75, 0.9, 0.95, 0.99, 0.999, 1 - 1e-6, 1 - 1e-9, 1 - 1e-12]
)
# |x| below ~1e-4 is left out on purpose: SciPy's t.cdf loses digits there
# (t.cdf(1e-6, 1) is off from the exact Cauchy value 0.5 + atan(1e-6)/pi by
# 1.4e-11), so it cannot serve as a reference in that range.
POINTS = [-1e6, -300.0, -40.0, -8.0, -3.0, -1.2, -0.4, -1e-3, 0.0, 2e-3, 0.05, 0.7, 2.0, 5.5, 25.0, 1e4]

# Dense in the centre, sparse far into the lower tail.
NORMAL_POINTS = [i / 16.0 + 0.013 for i in range(-136, 136)] + [-37.5, -30.0, -20.0, -12.0, -9.5]

rows = []
for nu in NUS:
    rows.append(
        {
            "nu": nu,
            "ppf": [float(t.ppf(p, nu)) for p in PROBABILITIES],
            "cdf": [float(t.cdf(x, nu)) for x in POINTS],
            "logpdf": [float(t.logpdf(x, nu)) for x in POINTS],
        }
    )

# Exact references for even nu, where the CDF is the finite sum
#   F(t) = 1/2 + (sin(theta)/2) * sum_{k=0}^{nu/2-1} c_k cos(theta)^(2k),
#   c_0 = 1, c_k = c_{k-1} (2k-1)/(2k), sin(theta) = t / sqrt(nu + t^2),
# evaluated in 60-digit decimal arithmetic; quantiles by bisection on it.
getcontext().prec = 60


def exact_cdf(x, nu):
    x = Decimal(x)
    sin = (x * x / (nu + x * x)).sqrt() * (1 if x > 0 else -1)
    cos2 = Decimal(nu) / (nu + x * x)
    term = Decimal(1)
    total = Decimal(1)
    for k in range(1, nu // 2):
        term *= Decimal(2 * k - 1) / Decimal(2 * k) * cos2
        total += term
    return Decimal(1) / 2 + sin / 2 * total


def exact_ppf(p, nu):
    p = Decimal(p)
    low, high = -(Decimal(10) ** 12), Decimal(10) ** 12
    for _ in range(260):
        mid = (low + high) / 2
        if exact_cdf(mid, nu) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2


EXACT_X = [-300.0, -8.0, -3.0, -1.2, -0.4, -1e-3, -1e-7, 1e-6, 0.05, 0.7, 2.0, 5.5, 25.0]
EXACT_P = [1e-12, 1e-6, 1e-3, 0.05, 0.3, 0.49, 0.5000001, 0.75, 0.99, 1 - 1e-9]
exact_rows = []
for nu in [2, 4, 10, 30, 200]:
    exact_rows.append(
        {
            "nu": float(nu),
            "x": EXACT_X,
            "cdf": [float(exact_cdf(x, nu)) for x in EXACT_X],
            "p": EXACT_P,
            "ppf": [float(exact_ppf(p, nu)) for p in EXACT_P],
        }
    )

payload = {
    "student_t_exact": exact_rows,
    "source": f"scipy {scipy.__version__}",
    "probabilities": PROBABILITIES,
    "points": POINTS,
    "student_t": rows,
    "normal": {
        "ppf": [float(ndtri(p)) for p in PROBABILITIES],
        "cdf": [float(ndtr(x)) for x in NORMAL_POINTS],
        "cdf_points": NORMAL_POINTS,
    },
}
out = Path("fixtures/reference/special/student_t_scipy.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=1) + "\n")
print(f"wrote {out}")
