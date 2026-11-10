"""High-precision reference values for the Archimedean and BB pair kernels.

Each family's copula distribution function is written out from its textbook
closed form and differentiated with mpmath at 400 significant digits:

    h(u1, u2) = dC/du1          (the law of U2 given U1 = u1)
    c(u1, u2) = d2C/du1 du2

Nothing here shares code, formulas or rearrangements with rscopulas, so the
values are an independent statement of what the kernels should return. The
400 digits are needed: at u = (1e-6, 1 - 1e-6) the BB1 distribution function
cancels more than 70 digits.

Usage (about ten minutes per family; run the families in parallel):

    pip install mpmath
    python scripts/reference/generate_pair_kernel_truth.py bb1   # one family
    python scripts/reference/generate_pair_kernel_truth.py       # merge parts

Each family writes `fixtures/reference/mpmath/parts/<family>.json`; the call
without arguments merges the parts into `pair_kernels.json`, which
`crates/rscopulas-core/tests/reference_pair_kernels.rs` reads.
"""

import json
import pathlib
import sys

from mpmath import diff, exp, log, mp, mpf

mp.dps = 400
ONE = mpf(1)
OUT = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "reference" / "mpmath"


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
    x, y = 1 - (1 - u) ** t, 1 - (1 - v) ** t
    return 1 - (1 - (x**-d + y**-d - 1) ** (-ONE / d)) ** (ONE / t)


def bb8(u, v, t, d):
    x, y = 1 - (1 - d * u) ** t, 1 - (1 - d * v) ** t
    return (1 - (1 - x * y / (1 - (1 - d) ** t)) ** (ONE / t)) / d


CDF = {
    "clayton": clayton,
    "gumbel": gumbel,
    "frank": frank,
    "joe": joe,
    "bb1": bb1,
    "bb6": bb6,
    "bb7": bb7,
    "bb8": bb8,
}
PARAMETERS = {
    "clayton": [[t] for t in (0.05, 2.0, 15.0, 27.0)],
    "gumbel": [[t] for t in (1.02, 2.0, 12.0, 25.0)],
    "frank": [[t] for t in (-30.0, -0.5, 3.0, 12.0, 34.0)],
    "joe": [[t] for t in (1.05, 3.0, 20.0)],
    "bb1": [[a, b] for a in (0.05, 2.5, 6.0) for b in (1.02, 3.5, 6.5)],
    "bb6": [[a, b] for a in (1.05, 3.5, 5.5) for b in (1.02, 1.6, 7.0)],
    "bb7": [[a, b] for a in (1.05, 3.5, 5.5) for b in (0.05, 3.0, 20.0)],
    "bb8": [[a, b] for a in (1.2, 5.0, 7.5) for b in (0.1, 0.9, 0.99)],
}
POINTS = [1e-6, 1e-3, 0.05, 0.35, 0.65, 0.95, 0.999, 1 - 1e-4, 1 - 1e-6]


def family_cases(family):
    cases = []
    for parameters in PARAMETERS[family]:
        exact = [mpf(p) for p in parameters]

        def cdf(u, v):
            return CDF[family](u, v, *exact)

        u1, u2, log_pdf, h = [], [], [], []
        for a in POINTS:
            for b in POINTS:
                ma, mb = mpf(a), mpf(b)
                density = diff(cdf, (ma, mb), (1, 1))
                conditional = diff(lambda x: cdf(x, mb), ma)
                assert density > 0 and 0 <= conditional <= 1, (family, parameters, a, b)
                u1.append(a)
                u2.append(b)
                log_pdf.append(float(log(density)))
                # Twenty digits, so that values next to 0 and 1 survive.
                h.append(mp.nstr(conditional, 20))
        cases.append(
            {
                "family": family,
                "parameters": parameters,
                "u1": u1,
                "u2": u2,
                "log_pdf": log_pdf,
                "h_second_given_first": h,
            }
        )
        print(family, parameters, "done", flush=True)
    return cases


def main():
    parts = OUT / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        for family in sys.argv[1:]:
            (parts / f"{family}.json").write_text(json.dumps(family_cases(family)))
        return
    merged = []
    for family in CDF:
        merged.extend(json.loads((parts / f"{family}.json").read_text()))
    payload = {
        "source": "mpmath, 400 digits, closed-form distribution functions",
        "generator": "scripts/reference/generate_pair_kernel_truth.py",
        "cases": merged,
    }
    (OUT / "pair_kernels.json").write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", OUT / "pair_kernels.json", len(merged), "cases")


if __name__ == "__main__":
    main()
