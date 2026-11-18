"""Speed of every feature rscopulas shares with pyvinecopulib, side by side.

``compare_pyvinecopulib.py`` checks that both libraries fit the same models.
This script only measures time: pair kernels, pair fits, and vine density,
transforms, sampling and fitting, each on identical inputs and with the same
number of threads on both sides (``--threads``, default 1).

    python benchmarks/compare_pyvinecopulib_features.py
    python benchmarks/compare_pyvinecopulib_features.py --threads 12

A ratio above 1 means rscopulas is faster. Times are the best of several runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _threads_from_argv() -> int:
    for index, argument in enumerate(sys.argv):
        if argument == "--threads" and index + 1 < len(sys.argv):
            return max(1, int(sys.argv[index + 1]))
        if argument.startswith("--threads="):
            return max(1, int(argument.split("=", 1)[1]))
    return 1


# rscopulas sizes its thread pool from this variable on first use.
THREADS = _threads_from_argv()
os.environ["RAYON_NUM_THREADS"] = str(THREADS)

import numpy as np  # noqa: E402
import pyvinecopulib as pv  # noqa: E402

import rscopulas as rc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
F = pv.BicopFamily
PAIRS = {
    "gaussian": (F.gaussian, [0.6]),
    "student_t": (F.student, [0.6, 5.0]),
    "clayton": (F.clayton, [2.0]),
    "gumbel": (F.gumbel, [2.0]),
    "frank": (F.frank, [5.0]),
    "joe": (F.joe, [2.5]),
    "bb1": (F.bb1, [0.6, 1.6]),
    "bb6": (F.bb6, [1.8, 1.6]),
    "bb7": (F.bb7, [1.8, 0.8]),
    "bb8": (F.bb8, [2.5, 0.9]),
}
VINE_FAMILIES = ["independence", "gaussian", "student_t", "clayton", "frank", "gumbel", "joe", "bb1", "bb7"]
PV_VINE_FAMILIES = [F.indep, F.gaussian, F.student, F.clayton, F.frank, F.gumbel, F.joe, F.bb1, F.bb7]


def best(fn, repeats: int, min_time: float = 0.05) -> float:
    """Best wall time of one call, in seconds."""
    fn()
    result = float("inf")
    for _ in range(repeats):
        calls = 0
        start = time.perf_counter()
        while True:
            fn()
            calls += 1
            elapsed = time.perf_counter() - start
            if elapsed >= min_time:
                break
        result = min(result, elapsed / calls)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--only", type=str, default="", help="substring filter on the case name")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "output" / f"pyvinecopulib-features-t{THREADS}.json")
    args = parser.parse_args()
    rows: list[dict] = []

    def record(name: str, ours, theirs, repeats: int = args.repeats) -> None:
        if args.only and args.only not in name:
            return
        a, b = best(ours, repeats), best(theirs, repeats)
        rows.append({"case": name, "rscopulas_s": a, "pyvinecopulib_s": b, "ratio": b / a})
        flag = "" if b / a >= 1.0 else "   <-- slower"
        print(f"{name:42s} rscopulas {a * 1e3:10.4f} ms   pyvinecopulib {b * 1e3:10.4f} ms   {b / a:7.2f}x{flag}", flush=True)

    rng = np.random.default_rng(7)
    n = 20_000
    for name, (family, params) in PAIRS.items():
        theirs = pv.Bicop(family=family, parameters=np.array(params, float).reshape(-1, 1))
        ours = rc.PairCopula.from_spec(name, params)
        u = np.ascontiguousarray(theirs.simulate(n, seeds=[1, 2, 3]))
        u1, u2 = np.ascontiguousarray(u[:, 0]), np.ascontiguousarray(u[:, 1])
        p = np.column_stack([u1, rng.uniform(0.001, 0.999, n)])
        p1, p2 = np.ascontiguousarray(p[:, 0]), np.ascontiguousarray(p[:, 1])
        record(f"pair/{name}/pdf n=20k", lambda: ours.log_pdf(u1, u2), lambda: theirs.pdf(u))
        record(f"pair/{name}/hfunc1 n=20k", lambda: ours.cond_second_given_first(u1, u2), lambda: theirs.hfunc1(u))
        record(f"pair/{name}/hfunc2 n=20k", lambda: ours.cond_first_given_second(u1, u2), lambda: theirs.hfunc2(u))
        record(f"pair/{name}/hinv1 n=20k", lambda: ours.inv_second_given_first(p1, p2), lambda: theirs.hinv1(p))
        record(f"pair/{name}/hinv2 n=20k", lambda: ours.inv_first_given_second(p1, p2), lambda: theirs.hinv2(p))
        small = np.ascontiguousarray(u[:2000])
        controls = pv.FitControlsBicop(family_set=[family], num_threads=THREADS)
        record(
            f"pair/{name}/fit n=2k",
            lambda: rc.VineCopula.fit_r(small, family_set=[name], include_rotations=False, criterion="bic"),
            lambda: pv.Bicop.from_data(small, controls=controls),
            repeats=3,
        )

    # Family selection on one pair.
    theirs = pv.Bicop(family=F.gumbel, rotation=180, parameters=np.array([[2.0]]))
    small = np.ascontiguousarray(theirs.simulate(2000, seeds=[4, 5, 6]))
    controls = pv.FitControlsBicop(family_set=PV_VINE_FAMILIES, num_threads=THREADS)
    record(
        "pair/select 9 families n=2k",
        lambda: rc.VineCopula.fit_r(small, family_set=VINE_FAMILIES, include_rotations=True, criterion="bic"),
        lambda: pv.Bicop.from_data(small, controls=controls),
        repeats=3,
    )
    tll_controls = pv.FitControlsBicop(family_set=[F.tll], num_threads=THREADS)
    record(
        "pair/tll fit n=2k",
        lambda: rc.PairCopula.fit_tll(small[:, 0], small[:, 1], method="quadratic"),
        lambda: pv.Bicop.from_data(small, controls=tll_controls),
        repeats=3,
    )

    # Vines: both libraries fit the same data and are timed on their own fit.
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import compare_pyvinecopulib as base  # noqa: E402

    for label, build in (("mixed d=5", base.mixed_rvine_5), ("mixed d=8", base.mixed_rvine_8)):
        truth = build()
        train = np.ascontiguousarray(base.pseudo_obs(truth.simulate(2000, seeds=[11, 12, 13])))
        points = np.ascontiguousarray(base.pseudo_obs(truth.simulate(10_000, seeds=[21, 22, 23])))
        controls = pv.FitControlsVinecop(family_set=PV_VINE_FAMILIES, num_threads=THREADS)

        def fit_ours():
            return rc.VineCopula.fit_r(train, family_set=VINE_FAMILIES, include_rotations=True, criterion="bic")

        def fit_theirs():
            return pv.Vinecop.from_data(train, controls=controls)

        record(f"vine/{label}/fit n=2k", fit_ours, fit_theirs, repeats=2)
        ours, theirs = fit_ours().model, fit_theirs()
        record(f"vine/{label}/pdf n=10k", lambda: ours.log_pdf(points), lambda: theirs.pdf(points, num_threads=THREADS))
        record(
            f"vine/{label}/rosenblatt n=10k",
            lambda: ours.rosenblatt(points),
            lambda: theirs.rosenblatt(points, num_threads=THREADS),
        )
        record(
            f"vine/{label}/inverse_rosenblatt n=10k",
            lambda: ours.inverse_rosenblatt(points),
            lambda: theirs.inverse_rosenblatt(points, num_threads=THREADS),
        )
        record(
            f"vine/{label}/sample n=10k",
            lambda: ours.sample(10_000, seed=3),
            lambda: theirs.simulate(10_000, num_threads=THREADS, seeds=[3]),
        )

    truth = base.mixed_rvine_5()
    train = np.ascontiguousarray(base.pseudo_obs(truth.simulate(1000, seeds=[31, 32, 33])))
    tll = pv.FitControlsVinecop(family_set=[F.tll], num_threads=THREADS)
    record(
        "vine/tll d=5/fit n=1k",
        lambda: rc.VineCopula.fit_r(train, family_set=["tll"], criterion="bic"),
        lambda: pv.Vinecop.from_data(train, controls=tll),
        repeats=2,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"threads": THREADS, "rows": rows}, indent=2))
    slower = [row for row in rows if row["ratio"] < 1.0]
    print(f"\n{len(rows)} cases, {len(slower)} slower than pyvinecopulib; wrote {args.output}")


if __name__ == "__main__":
    main()
