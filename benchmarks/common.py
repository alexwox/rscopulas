from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
DEFAULT_CASES_PATH = BENCHMARKS_DIR / "cases.json"
DEFAULT_OUTPUT_DIR = BENCHMARKS_DIR / "output"


def workspace_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_json(path: str | Path) -> Any:
    return json.loads(workspace_path(path).read_text())


def write_json(path: str | Path, payload: Any) -> None:
    target = workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def benchmark_environment(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    environment = {
        "cwd": str(Path.cwd()),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }
    if extra:
        environment.update(extra)
    return environment


def measure_iterations(iterations: int, callback: Callable[[], Any]) -> dict[str, int]:
    callback()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        callback()
    total_ns = time.perf_counter_ns() - start
    return {
        "iterations": iterations,
        "mean_ns": total_ns // iterations,
        "total_ns": total_ns,
    }


def make_result(case: dict[str, Any], measurement: dict[str, int], extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "category": case["category"],
        "fixture": case["fixture"],
        "family": case.get("family"),
        "operation": case["operation"],
        "structure": case.get("structure"),
        **measurement,
        **extra,
    }


def filter_manifest(
    manifest: dict[str, Any],
    implementations: set[str],
    case_filters: list[str],
    iteration_scale: float,
) -> dict[str, Any]:
    requested = []
    normalized_filters = [value.lower() for value in case_filters]
    for case in manifest["cases"]:
        available = set(case["implementations"])
        if not available.intersection(implementations):
            continue
        if normalized_filters and not any(token in case["id"].lower() for token in normalized_filters):
            continue
        clone = dict(case)
        clone["implementations"] = sorted(available.intersection(implementations))
        clone["iterations"] = max(1, int(round(clone["iterations"] * iteration_scale)))
        requested.append(clone)
    return {
        "schema_version": manifest["schema_version"],
        "cases": requested,
    }
