from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from common import DEFAULT_CASES_PATH, DEFAULT_OUTPUT_DIR, filter_manifest, load_json, write_json

RUST_MANIFEST_PATH = BENCHMARKS_DIR.parent / "crates" / "rscopulas-core" / "Cargo.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-language R vs rscopulas benchmarks.")
    parser.add_argument(
        "--implementation",
        action="append",
        choices=["rust", "python", "r", "all"],
        default=[],
        help="Implementation(s) to run. May be passed multiple times.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Only run benchmark cases whose id contains the provided token.",
    )
    parser.add_argument(
        "--iteration-scale",
        type=float,
        default=1.0,
        help="Multiply per-case iteration counts before running.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "latest.json"),
        help="Combined JSON output path.",
    )
    return parser.parse_args()


def selected_implementations(values: list[str]) -> set[str]:
    if not values:
        return {"rust", "python", "r"}
    picked = set(values)
    if "all" in picked:
        return {"rust", "python", "r"}
    return picked


def run_subprocess(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=BENCHMARKS_DIR.parent)


def runner_command(implementation: str, cases_path: Path, output_path: Path) -> list[str]:
    if implementation == "python":
        return [
            sys.executable,
            str(BENCHMARKS_DIR / "python_runner.py"),
            "--cases",
            str(cases_path),
            "--output",
            str(output_path),
        ]
    if implementation == "r":
        return [
            "Rscript",
            str(BENCHMARKS_DIR / "r_runner.R"),
            "--cases",
            str(cases_path),
            "--output",
            str(output_path),
        ]
    if implementation == "rust":
        return [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(RUST_MANIFEST_PATH),
            "--example",
            "benchmark_runner",
            "--",
            "--cases",
            str(cases_path),
            "--output",
            str(output_path),
        ]
    raise KeyError(f"unsupported implementation {implementation}")


def combine_payloads(runners: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    results = []
    for payload in runners:
        for result in payload["results"]:
            merged = dict(result)
            merged["implementation"] = payload["implementation"]
            results.append(merged)
    return {
        "schema_version": manifest["schema_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": manifest["cases"],
        "runners": runners,
        "results": results,
    }


def format_ns(value: int | float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}s"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}ms"
    if value >= 1_000:
        return f"{value / 1_000:.3f}us"
    return f"{value:.0f}ns"


def render_markdown(payload: dict[str, Any]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in payload["results"]:
        grouped[result["case_id"]].append(result)

    lines = [
        "# Benchmark Summary",
        "",
        f"Generated at: {payload['generated_at']}",
        "",
    ]
    for case_id in sorted(grouped):
        case_results = sorted(grouped[case_id], key=lambda item: item["mean_ns"])
        baseline = next((item for item in case_results if item["implementation"] == "r"), None)
        lines.append(f"## {case_id}")
        lines.append("")
        lines.append("| Implementation | Mean | Total | Iterations | Relative to R |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for result in case_results:
            relative = "-"
            if baseline is not None and result["implementation"] != "r":
                relative = f"{baseline['mean_ns'] / result['mean_ns']:.2f}x"
            lines.append(
                "| "
                f"{result['implementation']} | "
                f"{format_ns(result['mean_ns'])} | "
                f"{format_ns(result['total_ns'])} | "
                f"{result['iterations']} | "
                f"{relative} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    implementations = selected_implementations(args.implementation)
    manifest = filter_manifest(
        load_json(DEFAULT_CASES_PATH),
        implementations=implementations,
        case_filters=args.case,
        iteration_scale=args.iteration_scale,
    )
    if not manifest["cases"]:
        raise SystemExit("No benchmark cases matched the requested filters.")

    output_path = Path(args.output)
    markdown_path = output_path.with_suffix(".md")

    with tempfile.TemporaryDirectory(prefix="rscopulas-bench-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        manifest_path = tmpdir_path / "cases.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        runners = []
        for implementation in sorted(implementations):
            runner_output = tmpdir_path / f"{implementation}.json"
            run_subprocess(runner_command(implementation, manifest_path, runner_output))
            runners.append(load_json(runner_output))

    payload = combine_payloads(runners, manifest)
    write_json(output_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload))


if __name__ == "__main__":
    main()
