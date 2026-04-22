#!/usr/bin/env python3
"""Check reviewer experiment evidence readiness from results JSON/CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "experiments" / "manifest.yaml"
DEFAULT_RESULTS = REPO_ROOT / "results" / "eval"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _result_files(results_dir: Path) -> list[Path]:
    return sorted(path for path in results_dir.glob("*.json") if path.name != "result.json")


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    return [_load_json(path) for path in _result_files(results_dir)]


def _executed_ok(result: dict[str, Any]) -> bool:
    graph = result.get("graph") or {}
    return (
        bool(result.get("executed"))
        and graph.get("exists") is True
        and int(graph.get("object_node_count") or 0) > 0
        and not result.get("failure_flags")
    )


def _dataset_summary_ok(path: Path, scenarios: list[str], required_topics: list[str]) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing {path.relative_to(REPO_ROOT)}"
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seen = {row.get("scenario") for row in rows}
    missing = [scenario for scenario in scenarios if scenario not in seen]
    if missing:
        return False, f"missing scenarios: {', '.join(missing)}"
    for row in rows:
        for topic in required_topics:
            if int(row.get(f"topic:{topic}") or 0) <= 0:
                return False, f"{row.get('scenario')} has no messages for {topic}"
    return True, f"{len(rows)} scenarios with required topics"


def _full_runs_ok(results: list[dict[str, Any]], scenarios: list[str]) -> tuple[bool, str]:
    missing = []
    for scenario in scenarios:
        matches = [
            result
            for result in results
            if result.get("scenario") == scenario
            and result.get("variant") == "full"
            and _executed_ok(result)
        ]
        if not matches:
            missing.append(scenario)
    if missing:
        return False, "run full variant: " + ", ".join(missing)
    return True, "full variant exists for all scenarios"


def _ablation_ok(results: list[dict[str, Any]], scenarios: list[str]) -> tuple[bool, str]:
    required = {"full", "semantic-only", "place-gng-only", "no-llm"}
    missing = []
    for scenario in scenarios:
        seen = {
            result.get("variant")
            for result in results
            if result.get("scenario") == scenario and _executed_ok(result)
        }
        gap = sorted(required - seen)
        if gap:
            missing.append(f"{scenario}: {', '.join(gap)}")
    if missing:
        return False, "missing ablations: " + "; ".join(missing)
    return True, "all ablation variants exist"


def _runtime_ok(results: list[dict[str, Any]]) -> tuple[bool, str]:
    modules = {"groundingdino_predict", "sam_predict", "pose_association", "graph_update"}
    for result in results:
        if not _executed_ok(result):
            continue
        runtime = ((result.get("graph") or {}).get("runtime") or {}).get("timings") or {}
        if modules.issubset(runtime):
            return True, f"runtime timings in {result.get('run_id')}"
    return False, "no executed run has the required runtime modules"


def _benchmark_ok(results: list[dict[str, Any]], manifest: dict[str, Any]) -> tuple[bool, str]:
    scenarios = [
        name
        for name, data in sorted((manifest.get("scenarios") or {}).items())
        if data.get("ground_truth_path")
    ]
    if not scenarios:
        return True, "no scenarios declare ground truth"

    missing = []
    for scenario in scenarios:
        matches = [
            result
            for result in results
            if result.get("scenario") == scenario
            and result.get("variant") == "full"
            and _executed_ok(result)
            and ((result.get("benchmark") or {}).get("available") is True)
            and (((result.get("benchmark") or {}).get("summary") or {}).get("f1") is not None)
        ]
        if not matches:
            missing.append(scenario)
    if missing:
        return False, "missing benchmark metrics: " + ", ".join(missing)
    return True, "object-map benchmark metrics exist for ground-truth scenarios"


def _sensitivity_ok(results: list[dict[str, Any]], manifest: dict[str, Any]) -> tuple[bool, str]:
    expected_keys = set((manifest.get("sensitivity") or {}).keys())
    seen_keys = set()
    for result in results:
        sensitivity = result.get("sensitivity")
        if not sensitivity or not _executed_ok(result):
            continue
        key = str(sensitivity).split("=", 1)[0]
        seen_keys.add(key)
    missing = sorted(expected_keys - seen_keys)
    if missing:
        return False, "missing sensitivity evidence: " + ", ".join(missing)
    return True, "sensitivity evidence exists for all sweep keys"


def _print_row(ok: bool, item: str, detail: str) -> None:
    marker = "PASS" if ok else "FAIL"
    print(f"[{marker}] {item} - {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.is_absolute():
        results_dir = REPO_ROOT / results_dir

    manifest = _load_yaml(manifest_path)
    scenarios = sorted((manifest.get("scenarios") or {}).keys())
    required_topics = list(manifest.get("required_topics") or [])
    results = _load_results(results_dir)

    checks = [
        ("Dataset summary / Table A", *_dataset_summary_ok(results_dir / "dataset_summary.csv", scenarios, required_topics)),
        ("AE-1 / R1-1 full runs", *_full_runs_ok(results, scenarios)),
        ("AE-2 / AE-3 ablations", *_ablation_ok(results, scenarios)),
        ("AE-4 runtime timings", *_runtime_ok(results)),
        ("Object-map benchmark", *_benchmark_ok(results, manifest)),
        ("AE-9 sensitivity", *_sensitivity_ok(results, manifest)),
    ]

    failed = 0
    for item, ok, detail in checks:
        _print_row(ok, item, detail)
        failed += 0 if ok else 1

    if failed:
        print("\nBLOCKED: experiment evidence is incomplete.")
        print("Next useful commands:")
        print("  python3 scripts/experiments/run_matrix.py --scenario all --variant full --skip-bag-hash")
        print("  python3 scripts/experiments/run_matrix.py --scenario all --variant all --skip-bag-hash")
        print("  python3 scripts/experiments/run_matrix.py --scenario meeting --variant full --sweep sensitivity --skip-bag-hash")
        print("  python3 scripts/experiments/aggregate_results.py")
        return 1

    print("\nREADY: experiment evidence gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
