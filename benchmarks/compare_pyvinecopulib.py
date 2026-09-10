"""Compare rscopulas against pyvinecopulib on vine copulas simulated by pyvinecopulib.

The ground truth never touches rscopulas: three known vines are built with
pyvinecopulib, simulated with ``Vinecop.simulate``, and split into a training and
a test set. Both libraries then select and fit an R-vine on rank-transformed
training data with the same candidate family set (independence, Gaussian,
Student-t, Clayton, Frank, Gumbel, Joe, BB1, BB7, rotations allowed) and each
library's default selection criterion (rscopulas: AIC, pyvinecopulib: BIC). A
third configuration refits rscopulas with BIC so the criterion effect can be
separated from the library effect.

Per dataset the report contains fit wall time (median of ``--repeats`` runs),
the selected family of every first-tree edge next to the truth, in-sample and
out-of-sample log-likelihood, and uniformity diagnostics of the Rosenblatt
transform of the test set (per-column Kolmogorov-Smirnov distance and the
largest absolute Spearman correlation between transformed columns).

Usage (from the repository root, with both packages importable)::

    python benchmarks/compare_pyvinecopulib.py
    python benchmarks/compare_pyvinecopulib.py --repeats 1 --n-train 500 --n-test 500

Results go to ``benchmarks/output/pyvinecopulib-comparison.{md,json}`` (that
directory is gitignored) and a committed copy of the Markdown report to
``docs/pyvinecopulib-comparison.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyvinecopulib as pv

import rscopulas as rc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "output" / "pyvinecopulib-comparison.md"
DEFAULT_DOCS_COPY = ROOT / "docs" / "pyvinecopulib-comparison.md"

FAMILY_NAMES = [
    "independence",
    "gaussian",
    "student_t",
    "clayton",
    "frank",
    "gumbel",
    "joe",
    "bb1",
    "bb7",
]
PV_FAMILIES = [
    pv.BicopFamily.indep,
    pv.BicopFamily.gaussian,
    pv.BicopFamily.student,
    pv.BicopFamily.clayton,
    pv.BicopFamily.frank,
    pv.BicopFamily.gumbel,
    pv.BicopFamily.joe,
    pv.BicopFamily.bb1,
    pv.BicopFamily.bb7,
]
PV_LABELS = {family: name for family, name in zip(PV_FAMILIES, FAMILY_NAMES)}
SEED_SIMULATE = 20260910


# --------------------------------------------------------------------------- #
# Known vines (pyvinecopulib objects only)                                     #
# --------------------------------------------------------------------------- #


def bicop(family: pv.BicopFamily, rotation: int, *parameters: float) -> pv.Bicop:
    params = np.asarray(parameters, dtype=np.float64).reshape(-1, 1)
    return pv.Bicop(family=family, rotation=rotation, parameters=params)


def gaussian_rvine_5() -> pv.Vinecop:
    structure = pv.RVineStructure.simulate(5, seeds=[501])
    rhos = {
        1: [0.75, -0.6, 0.65, 0.5],
        2: [0.4, -0.35, 0.3],
        3: [0.2, -0.15],
        4: [0.1],
    }
    pair_copulas = [
        [bicop(pv.BicopFamily.gaussian, 0, rho) for rho in rhos[tree]] for tree in range(1, 5)
    ]
    return pv.Vinecop.from_structure(structure=structure, pair_copulas=pair_copulas)


def mixed_rvine_5() -> pv.Vinecop:
    structure = pv.RVineStructure.simulate(5, seeds=[502])
    f = pv.BicopFamily
    pair_copulas = [
        [bicop(f.clayton, 0, 2.0), bicop(f.gumbel, 180, 2.5), bicop(f.frank, 0, 6.0), bicop(f.student, 0, 0.6, 4.0)],
        [bicop(f.clayton, 90, 1.2), bicop(f.gumbel, 270, 1.6), bicop(f.frank, 0, 3.0)],
        [bicop(f.student, 0, 0.3, 5.0), bicop(f.clayton, 180, 0.7)],
        [bicop(f.frank, 0, 1.5)],
    ]
    return pv.Vinecop.from_structure(structure=structure, pair_copulas=pair_copulas)


def mixed_rvine_8() -> pv.Vinecop:
    structure = pv.RVineStructure.simulate(8, seeds=[508])
    f = pv.BicopFamily
    pair_copulas = [
        [
            bicop(f.clayton, 0, 2.5),
            bicop(f.gumbel, 180, 2.2),
            bicop(f.frank, 0, 7.0),
            bicop(f.student, 0, 0.7, 4.0),
            bicop(f.clayton, 180, 1.8),
            bicop(f.gumbel, 90, 2.0),
            bicop(f.student, 0, -0.55, 6.0),
        ],
        [
            bicop(f.gumbel, 0, 1.5),
            bicop(f.clayton, 270, 1.0),
            bicop(f.frank, 0, 3.5),
            bicop(f.student, 0, 0.35, 5.0),
            bicop(f.clayton, 90, 0.9),
            bicop(f.gumbel, 180, 1.4),
        ],
        [
            bicop(f.frank, 0, 2.0),
            bicop(f.student, 0, 0.25, 6.0),
            bicop(f.clayton, 0, 0.5),
            bicop(f.gumbel, 0, 1.2),
            bicop(f.frank, 0, -1.5),
        ],
        [
            bicop(f.gaussian, 0, 0.2),
            bicop(f.frank, 0, 1.0),
            bicop(f.clayton, 180, 0.3),
            bicop(f.student, 0, 0.15, 8.0),
        ],
        [bicop(f.frank, 0, 0.8), bicop(f.gaussian, 0, -0.1), bicop(f.gaussian, 0, 0.1)],
        [bicop(f.gaussian, 0, 0.05), bicop(f.frank, 0, 0.5)],
        [bicop(f.gaussian, 0, 0.05)],
    ]
    return pv.Vinecop.from_structure(structure=structure, pair_copulas=pair_copulas)


DATASETS: dict[str, Callable[[], pv.Vinecop]] = {
    "gaussian_rvine_5d": gaussian_rvine_5,
    "mixed_rvine_5d": mixed_rvine_5,
    "mixed_rvine_8d": mixed_rvine_8,
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    """Rank transform ``rank / (n + 1)`` per column (no ties in simulated data)."""
    ranks = np.argsort(np.argsort(x, axis=0), axis=0) + 1
    return np.asarray(ranks / (x.shape[0] + 1), dtype=np.float64)


def ks_uniform_distance(column: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance of a sample from U(0, 1)."""
    x = np.sort(column)
    n = x.shape[0]
    grid = np.arange(1, n + 1) / n
    return float(max(np.max(grid - x), np.max(x - (grid - 1.0 / n))))


def ks_p_value(distance: float, n: int) -> float:
    """Asymptotic Kolmogorov p-value ``P(D_n > distance)``."""
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * distance
    if lam < 1e-8:
        return 1.0
    k = np.arange(1, 101)
    return float(min(1.0, max(0.0, 2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam)))))


def spearman_matrix(u: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(u, axis=0), axis=0).astype(np.float64)
    return np.corrcoef(ranks, rowvar=False)


def rosenblatt_diagnostics(transformed: np.ndarray) -> dict[str, float]:
    n, d = transformed.shape
    distances = [ks_uniform_distance(transformed[:, j]) for j in range(d)]
    worst = max(distances)
    rho = spearman_matrix(transformed)
    off_diagonal = np.abs(rho[np.triu_indices(d, k=1)])
    return {
        "ks_max": worst,
        "ks_max_p_value": ks_p_value(worst, n),
        "ks_mean": float(np.mean(distances)),
        "spearman_max_abs": float(np.max(off_diagonal)),
    }


def time_fit(fit: Callable[[], Any], repeats: int) -> tuple[Any, float, list[float]]:
    timings: list[float] = []
    model = None
    for _ in range(repeats):
        start = time.perf_counter()
        model = fit()
        timings.append(time.perf_counter() - start)
    return model, statistics.median(timings), timings


def fmt_params(values: Any) -> str:
    return ", ".join(f"{float(v):.3g}" for v in np.asarray(values, dtype=np.float64).ravel())


# --------------------------------------------------------------------------- #
# Library adapters                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class Edge:
    pair: tuple[int, int]
    family: str
    rotation: int
    parameters: str

    def label(self) -> str:
        rot = f" r{self.rotation}" if self.rotation else ""
        params = f" [{self.parameters}]" if self.parameters else ""
        return f"{self.family}{rot}{params}"


@dataclass
class FitSummary:
    library: str
    criterion: str
    fit_seconds_median: float
    fit_seconds: list[float]
    loglik_train: float
    loglik_test: float
    n_parameters: float
    tree1: list[Edge]
    rosenblatt_test: dict[str, float]
    notes: str = ""


@dataclass
class DatasetReport:
    name: str
    dim: int
    n_train: int
    n_test: int
    truth_loglik_train: float
    truth_loglik_test: float
    truth_tree1: list[Edge]
    truth_rosenblatt_test: dict[str, float]
    fits: list[FitSummary] = field(default_factory=list)


def pv_tree1_edges(vine: pv.Vinecop) -> list[Edge]:
    matrix = np.asarray(vine.matrix)
    d = matrix.shape[0]
    edges = []
    for column, pc in enumerate(vine.pair_copulas[0]):
        diagonal = int(matrix[d - 1 - column, column]) - 1
        partner = int(matrix[0, column]) - 1
        edges.append(
            Edge(
                pair=tuple(sorted((diagonal, partner))),
                family=PV_LABELS.get(pc.family, str(pc.family)),
                rotation=int(pc.rotation),
                parameters=fmt_params(pc.parameters) if pc.family != pv.BicopFamily.indep else "",
            )
        )
    return sorted(edges, key=lambda e: e.pair)


def rc_tree1_edges(model: rc.VineCopula) -> list[Edge]:
    edges = []
    for edge in model.trees[0].edges:
        rotation = int(edge.rotation.lstrip("R"))
        edges.append(
            Edge(
                pair=tuple(sorted(edge.conditioned)),
                family=edge.family,
                rotation=rotation,
                parameters=fmt_params(edge.parameters),
            )
        )
    return sorted(edges, key=lambda e: e.pair)


def rc_n_parameters(model: rc.VineCopula) -> int:
    return sum(len(edge.parameters) for tree in model.trees for edge in tree.edges)


def fit_rscopulas(u_train: np.ndarray, u_test: np.ndarray, criterion: str, repeats: int) -> FitSummary:
    def run() -> rc.VineCopula:
        return rc.VineCopula.fit_r(
            u_train, family_set=FAMILY_NAMES, include_rotations=True, criterion=criterion
        ).model

    model, median, timings = time_fit(run, repeats)
    return FitSummary(
        library="rscopulas",
        criterion=criterion.upper(),
        fit_seconds_median=median,
        fit_seconds=timings,
        loglik_train=float(np.sum(model.log_pdf(u_train))),
        loglik_test=float(np.sum(model.log_pdf(u_test))),
        n_parameters=rc_n_parameters(model),
        tree1=rc_tree1_edges(model),
        rosenblatt_test=rosenblatt_diagnostics(model.rosenblatt(u_test)),
        notes="fit_r, tree_algorithm=kruskal, tree_criterion=tau, exhaustive family search",
    )


def fit_pyvinecopulib(u_train: np.ndarray, u_test: np.ndarray, repeats: int) -> FitSummary:
    controls = pv.FitControlsVinecop(family_set=PV_FAMILIES, num_threads=1)

    def run() -> pv.Vinecop:
        return pv.Vinecop.from_data(u_train, controls=controls)

    model, median, timings = time_fit(run, repeats)
    return FitSummary(
        library="pyvinecopulib",
        criterion="BIC",
        fit_seconds_median=median,
        fit_seconds=timings,
        loglik_train=float(model.loglik(u_train)),
        loglik_test=float(model.loglik(u_test)),
        n_parameters=float(model.npars),
        tree1=pv_tree1_edges(model),
        rosenblatt_test=rosenblatt_diagnostics(np.asarray(model.rosenblatt(u_test, seeds=[1]))),
        notes="Vinecop.from_data, tree_algorithm=mst_prim, tree_criterion=tau, preselect_families=True, num_threads=1",
    )


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #


def run_dataset(name: str, build: Callable[[], pv.Vinecop], n_train: int, n_test: int, repeats: int) -> DatasetReport:
    truth = build()
    sample = np.asarray(truth.simulate(n_train + n_test, seeds=[SEED_SIMULATE]))
    u_train = pseudo_obs(sample[:n_train])
    u_test = pseudo_obs(sample[n_train:])
    report = DatasetReport(
        name=name,
        dim=int(truth.dim),
        n_train=n_train,
        n_test=n_test,
        truth_loglik_train=float(truth.loglik(u_train)),
        truth_loglik_test=float(truth.loglik(u_test)),
        truth_tree1=pv_tree1_edges(truth),
        truth_rosenblatt_test=rosenblatt_diagnostics(np.asarray(truth.rosenblatt(u_test, seeds=[1]))),
    )
    print(f"[{name}] d={report.dim} n_train={n_train} n_test={n_test}", flush=True)
    for label, runner in [
        ("rscopulas (AIC)", lambda: fit_rscopulas(u_train, u_test, "aic", repeats)),
        ("pyvinecopulib (BIC)", lambda: fit_pyvinecopulib(u_train, u_test, repeats)),
        ("rscopulas (BIC)", lambda: fit_rscopulas(u_train, u_test, "bic", repeats)),
    ]:
        summary = runner()
        report.fits.append(summary)
        print(
            f"  {label:<20} fit {summary.fit_seconds_median:8.3f} s  "
            f"loglik train {summary.loglik_train:10.2f}  test {summary.loglik_test:10.2f}",
            flush=True,
        )
    return report


def environment() -> dict[str, str]:
    def version(package: str) -> str:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
            ).stdout.strip() or cpu
        except OSError:
            pass
    return {
        "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "platform": platform.platform(),
        "cpu": cpu,
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "rscopulas": version("rscopulas"),
        "pyvinecopulib": version("pyvinecopulib"),
    }


def render_markdown(reports: list[DatasetReport], env: dict[str, str], repeats: int) -> str:
    lines: list[str] = []
    lines.append("# rscopulas vs pyvinecopulib")
    lines.append("")
    lines.append(f"Generated {env['date']} by `benchmarks/compare_pyvinecopulib.py`.")
    lines.append("")
    lines.append("| Environment | |")
    lines.append("| --- | --- |")
    for key in ["platform", "cpu", "python", "numpy", "rscopulas", "pyvinecopulib"]:
        lines.append(f"| {key} | {env[key]} |")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Each dataset is simulated with `pyvinecopulib.Vinecop.simulate` from a known vine, so the "
        "truth is independent of rscopulas. Training and test sets are rank-transformed separately "
        "(`rank / (n + 1)`). Both libraries select structure and pair-copula families on the training "
        "set from the same candidate set — independence, Gaussian, Student-t, Clayton, Frank, Gumbel, "
        "Joe, BB1, BB7, rotations allowed — with Kendall's tau spanning trees. Each library uses its "
        "own default selection criterion: **rscopulas `fit_r` defaults to AIC, pyvinecopulib "
        "`FitControlsVinecop` defaults to BIC**; a third row refits rscopulas with BIC. pyvinecopulib "
        "keeps its default `preselect_families=True` (families are pruned by symmetry of the data "
        "before fitting), rscopulas fits every candidate. Both run single-threaded. Fit time is the "
        f"median of {repeats} wall-clock runs of the complete select-and-fit call in one Python "
        "process. Log-likelihoods are the sums of each fitted model's own log-density over the "
        "training and test pseudo-observations; the truth row evaluates the simulating vine. "
        "Rosenblatt diagnostics transform the **test** set with each fitted model: `KS max` is the "
        "largest per-column Kolmogorov-Smirnov distance from U(0, 1) with its asymptotic p-value, "
        "`Spearman max` the largest absolute Spearman correlation between transformed columns "
        "(both should be small for a well-specified model)."
    )
    lines.append("")
    for report in reports:
        lines.append(f"## {report.name} (d = {report.dim}, n_train = {report.n_train}, n_test = {report.n_test})")
        lines.append("")
        lines.append("| Model | Fit time (s) | Log-lik train | Log-lik test | Params | KS max (p) | Spearman max |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for fit in report.fits:
            ros = fit.rosenblatt_test
            lines.append(
                f"| {fit.library} ({fit.criterion}) | {fit.fit_seconds_median:.3f} | {fit.loglik_train:.1f} | "
                f"{fit.loglik_test:.1f} | {fit.n_parameters:g} | {ros['ks_max']:.4f} ({ros['ks_max_p_value']:.2f}) | "
                f"{ros['spearman_max_abs']:.4f} |"
            )
        ros = report.truth_rosenblatt_test
        lines.append(
            f"| truth (simulating vine) | – | {report.truth_loglik_train:.1f} | {report.truth_loglik_test:.1f} | – | "
            f"{ros['ks_max']:.4f} ({ros['ks_max_p_value']:.2f}) | {ros['spearman_max_abs']:.4f} |"
        )
        lines.append("")
        rs_aic = report.fits[0]
        pvc = report.fits[1]
        ratio = rs_aic.fit_seconds_median / pvc.fit_seconds_median if pvc.fit_seconds_median > 0 else float("nan")
        lines.append(
            f"Fit-time ratio rscopulas (AIC) / pyvinecopulib (BIC): **{ratio:.1f}x**. "
            f"Out-of-sample log-likelihood gap to the truth: rscopulas (AIC) {rs_aic.loglik_test - report.truth_loglik_test:+.1f}, "
            f"pyvinecopulib (BIC) {pvc.loglik_test - report.truth_loglik_test:+.1f}, "
            f"rscopulas (BIC) {report.fits[2].loglik_test - report.truth_loglik_test:+.1f}."
        )
        lines.append("")
        lines.append("Selected first-tree edges (variables are 0-based; `r` = rotation in degrees):")
        lines.append("")
        header = "| Pair | Truth |" + "".join(f" {fit.library} ({fit.criterion}) |" for fit in report.fits)
        lines.append(header)
        lines.append("| --- | --- |" + " --- |" * len(report.fits))
        pairs = {edge.pair for edge in report.truth_tree1}
        for fit in report.fits:
            pairs.update(edge.pair for edge in fit.tree1)
        truth_map = {edge.pair: edge.label() for edge in report.truth_tree1}
        fit_maps = [{edge.pair: edge.label() for edge in fit.tree1} for fit in report.fits]
        for pair in sorted(pairs):
            row = f"| ({pair[0]}, {pair[1]}) | {truth_map.get(pair, '–')} |"
            row += "".join(f" {fit_map.get(pair, '–')} |" for fit_map in fit_maps)
            lines.append(row)
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3, help="fit repetitions per configuration (median reported)")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS), help="restrict to a dataset (repeatable)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown report path (JSON written next to it)")
    parser.add_argument("--docs-copy", type=str, default=str(DEFAULT_DOCS_COPY), help="committed copy of the report; pass '' to skip")
    args = parser.parse_args()
    docs_copy = Path(args.docs_copy) if args.docs_copy else None

    names = args.dataset or list(DATASETS)
    reports = [run_dataset(name, DATASETS[name], args.n_train, args.n_test, args.repeats) for name in names]
    env = environment()
    markdown = render_markdown(reports, env, args.repeats)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    args.output.with_suffix(".json").write_text(
        json.dumps({"environment": env, "repeats": args.repeats, "datasets": [asdict(r) for r in reports]}, indent=2)
        + "\n"
    )
    print(f"wrote {args.output}")
    if docs_copy is not None:
        docs_copy.parent.mkdir(parents=True, exist_ok=True)
        docs_copy.write_text(markdown)
        print(f"wrote {docs_copy}")


if __name__ == "__main__":
    main()
