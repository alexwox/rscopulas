from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARKS_DIR = Path(__file__).resolve().parent
WORKSPACE_PYTHON_DIR = BENCHMARKS_DIR.parent / "python"
for candidate in (BENCHMARKS_DIR, WORKSPACE_PYTHON_DIR):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from common import benchmark_environment, load_json, make_result, measure_iterations, write_json

try:
    from rscopulas import (
        ClaytonCopula,
        FrankCopula,
        GaussianCopula,
        GumbelCopula,
        PairCopula,
        StudentTCopula,
        VineCopula,
    )
except Exception as exc:  # pragma: no cover - exercised by runtime invocation
    raise SystemExit(
        "Failed to import rscopulas. Build the extension first with "
        "`maturin develop --release` and retry."
    ) from exc

FIT_VINE_FAMILY_SET = [
    "independence",
    "gaussian",
    "clayton",
    "frank",
    "gumbel",
]


def _family_model_class(family: str):
    return {
        "gaussian": GaussianCopula,
        "student_t": StudentTCopula,
        "clayton": ClaytonCopula,
        "frank": FrankCopula,
        "gumbel": GumbelCopula,
    }[family]


def _fixture_dim(fixture: dict[str, Any]) -> int:
    rows = fixture.get("inputs") or fixture.get("input_pobs")
    if rows:
        return int(np.asarray(rows, dtype=np.float64).shape[1])
    correlation = fixture.get("correlation")
    if correlation is not None:
        return int(np.asarray(correlation, dtype=np.float64).shape[0])
    if fixture.get("theta") is not None:
        return 2
    raise ValueError("fixture does not provide enough information to infer model dimension")


def _single_family_model(case: dict[str, Any], fixture: dict[str, Any]):
    family = case["family"]
    if family == "gaussian":
        return GaussianCopula.from_params(np.asarray(fixture["correlation"], dtype=np.float64))
    if family == "student_t":
        return StudentTCopula.from_params(
            np.asarray(fixture["correlation"], dtype=np.float64),
            float(fixture["degrees_of_freedom"]),
        )
    dim = _fixture_dim(fixture)
    theta = float(fixture["theta"])
    if family == "clayton":
        return ClaytonCopula.from_params(dim, theta)
    if family == "frank":
        return FrankCopula.from_params(dim, theta)
    if family == "gumbel":
        return GumbelCopula.from_params(dim, theta)
    raise KeyError(f"unsupported family {family}")


def _pair_parameters(fixture: dict[str, Any]) -> list[float]:
    parameters = [float(fixture["par"])]
    if float(fixture.get("par2", 0.0)) != 0.0 or fixture["family"] == "student_t":
        parameters.append(float(fixture.get("par2", 0.0)))
    return parameters


def _pair_family_and_rotation(fixture: dict[str, Any]) -> tuple[str, str]:
    family_code = fixture.get("family_code")
    if family_code is not None:
        mapping = {
            0: ("independence", "R0"),
            1: ("gaussian", "R0"),
            2: ("student_t", "R0"),
            3: ("clayton", "R0"),
            13: ("clayton", "R180"),
            23: ("clayton", "R90"),
            33: ("clayton", "R270"),
            4: ("gumbel", "R0"),
            14: ("gumbel", "R180"),
            24: ("gumbel", "R90"),
            34: ("gumbel", "R270"),
            5: ("frank", "R0"),
        }
        try:
            return mapping[int(family_code)]
        except KeyError as exc:
            raise ValueError(f"unsupported pair family_code {family_code}") from exc

    family = str(fixture["family"]).lower()
    if "_rot" in family:
        base_family, rotation_suffix = family.split("_rot", maxsplit=1)
        return base_family, f"R{rotation_suffix}"
    return family, str(fixture.get("rotation", "R0"))


def _pair_model_from_fixture(fixture: dict[str, Any]) -> PairCopula:
    if fixture.get("family") == "khoudraji":
        return PairCopula.from_khoudraji(
            fixture["base_copula_1"]["family"],
            fixture["base_copula_2"]["family"],
            shape_1=float(fixture["shape_1"]),
            shape_2=float(fixture["shape_2"]),
            first_parameters=fixture["base_copula_1"].get("parameters", []),
            second_parameters=fixture["base_copula_2"].get("parameters", []),
            rotation=str(fixture.get("rotation", "R0")),
            first_rotation=str(fixture["base_copula_1"].get("rotation", "R0")),
            second_rotation=str(fixture["base_copula_2"].get("rotation", "R0")),
        )
    family, rotation = _pair_family_and_rotation(fixture)
    return PairCopula.from_spec(
        family,
        _pair_parameters(fixture),
        rotation=rotation,
    )


def _run_single_family(case: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    family = case["family"]
    operation = case["operation"]
    model_class = _family_model_class(family)
    if operation == "log_pdf":
        model = _single_family_model(case, fixture)
        data = np.asarray(fixture["inputs"], dtype=np.float64)
        measurement = measure_iterations(case["iterations"], lambda: model.log_pdf(data))
        return make_result(
            case,
            measurement,
            {"dim": int(data.shape[1]), "observations": int(data.shape[0])},
        )
    if operation == "fit":
        data = np.asarray(fixture["input_pobs"], dtype=np.float64)
        measurement = measure_iterations(case["iterations"], lambda: model_class.fit(data))
        return make_result(
            case,
            measurement,
            {"dim": int(data.shape[1]), "observations": int(data.shape[0])},
        )
    if operation == "sample":
        model = _single_family_model(case, fixture)
        measurement = measure_iterations(
            case["iterations"],
            lambda: model.sample(int(fixture["sample_size"]), seed=int(fixture["seed"])),
        )
        return make_result(
            case,
            measurement,
            {"dim": model.dim, "sample_size": int(fixture["sample_size"])},
        )
    raise KeyError(f"unsupported single-family operation {operation}")


def _run_pair_kernels(case: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    model = _pair_model_from_fixture(fixture)
    u1 = np.asarray(fixture["u1"], dtype=np.float64)
    u2 = np.asarray(fixture["u2"], dtype=np.float64)
    p = np.asarray(fixture["p"], dtype=np.float64)

    def run() -> None:
        model.log_pdf(u1, u2)
        model.cond_first_given_second(u1, u2)
        model.cond_second_given_first(u1, u2)
        model.inv_first_given_second(p, u2)
        model.inv_second_given_first(u1, p)

    measurement = measure_iterations(case["iterations"], run)
    return make_result(case, measurement, {"observations": int(u1.shape[0])})


def _vine_model_from_fixture(fixture: dict[str, Any]) -> VineCopula:
    return VineCopula.from_trees(
        str(fixture["structure"]).lower(),
        fixture["trees"],
        truncation_level=fixture.get("truncation_level"),
    )


def _run_vine(case: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    operation = case["operation"]
    if operation == "log_pdf":
        model = _vine_model_from_fixture(fixture)
        data = np.asarray(fixture["inputs"], dtype=np.float64)
        measurement = measure_iterations(case["iterations"], lambda: model.log_pdf(data))
        return make_result(
            case,
            measurement,
            {"dim": int(data.shape[1]), "observations": int(data.shape[0])},
        )
    if operation == "sample":
        model = _vine_model_from_fixture(fixture)
        measurement = measure_iterations(
            case["iterations"],
            lambda: model.sample(int(fixture["sample_size"]), seed=int(fixture["seed"])),
        )
        return make_result(
            case,
            measurement,
            {"dim": model.dim, "sample_size": int(fixture["sample_size"])},
        )
    if operation == "fit":
        data = np.asarray(fixture["data"], dtype=np.float64)
        measurement = measure_iterations(
            case["iterations"],
            lambda: VineCopula.fit_r(
                data,
                family_set=FIT_VINE_FAMILY_SET,
                include_rotations=True,
                criterion="aic",
                truncation_level=2,
            ),
        )
        return make_result(
            case,
            measurement,
            {"dim": int(data.shape[1]), "observations": int(data.shape[0])},
        )
    raise KeyError(f"unsupported vine operation {operation}")


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    fixture = load_json(case["fixture"])
    category = case["category"]
    if category == "single_family":
        return _run_single_family(case, fixture)
    if category == "pair_copula":
        return _run_pair_kernels(case, fixture)
    if category == "vine":
        return _run_vine(case, fixture)
    raise KeyError(f"unsupported benchmark category {category}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Python benchmark cases.")
    parser.add_argument("--cases", required=True, help="Path to the benchmark manifest JSON.")
    parser.add_argument("--output", required=True, help="Path to the runner output JSON.")
    args = parser.parse_args()

    manifest = load_json(args.cases)
    cases = [case for case in manifest["cases"] if "python" in case["implementations"]]
    results = [run_case(case) for case in cases]
    payload = {
        "implementation": "python",
        "runner": "benchmarks/python_runner.py",
        "environment": benchmark_environment({"numpy_version": np.__version__}),
        "results": results,
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
