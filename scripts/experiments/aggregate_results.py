#!/usr/bin/env python3
"""Aggregate STCM experiment result JSON into deterministic CSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_results(results_dir: Path):
    artifact_root = results_dir / "artifacts"
    for path in sorted(results_dir.glob("*.json")):
        yield path, _load_json(path)
    if artifact_root.exists():
        for path in sorted(artifact_root.glob("*/result.json")):
            top_level = results_dir / f"{path.parent.name}.json"
            if top_level.exists():
                continue
            yield path, _load_json(path)


def _timing(graph: dict[str, Any], name: str, key: str) -> Any:
    runtime = graph.get("runtime", {})
    timings = runtime.get("timings", {}) if isinstance(runtime, dict) else {}
    return (timings.get(name) or {}).get(key)


def _event(graph: dict[str, Any], name: str) -> int:
    runtime = graph.get("runtime", {})
    events = runtime.get("events", {}) if isinstance(runtime, dict) else {}
    return int(events.get(name) or 0)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate(results_dir: Path) -> dict[str, Path]:
    rows = []
    runtime_rows = []
    for path, result in _iter_results(results_dir):
        graph = result.get("graph", {})
        bag = result.get("bag", {})
        rows.append(
            {
                "run_id": result.get("run_id"),
                "scenario": result.get("scenario"),
                "variant": result.get("variant"),
                "sensitivity": result.get("sensitivity") or "",
                "executed": result.get("executed"),
                "git_sha": (result.get("git") or {}).get("sha"),
                "bag_path": bag.get("path"),
                "duration_ns": bag.get("duration_ns"),
                "object_nodes": graph.get("object_node_count"),
                "object_edges": graph.get("object_edge_count"),
                "place_nodes": graph.get("place_node_count"),
                "place_edges": graph.get("place_edge_count"),
                "zero_detection_frames": _event(graph, "zero_detection_frames"),
                "frames_seen": _event(graph, "frames_seen"),
                "pose_failures": _event(graph, "pose_failures"),
                "tf_lookup_failures": _event(graph, "tf_lookup_failures"),
                "failure_flags": ";".join(result.get("failure_flags", [])),
                "result_path": str(path.relative_to(REPO_ROOT)),
            }
        )
        for module in (
            "frame_total",
            "groundingdino_predict",
            "sam_predict",
            "pose_association",
            "instance_gng_update",
            "place_gng_update",
            "graph_update",
        ):
            runtime_rows.append(
                {
                    "run_id": result.get("run_id"),
                    "scenario": result.get("scenario"),
                    "variant": result.get("variant"),
                    "module": module,
                    "n": _timing(graph, module, "n"),
                    "mean_ms": _timing(graph, module, "mean_ms"),
                    "p50_ms": _timing(graph, module, "p50_ms"),
                    "p95_ms": _timing(graph, module, "p95_ms"),
                }
            )

    rows.sort(key=lambda row: (row["scenario"] or "", row["variant"] or "", row["run_id"] or ""))
    runtime_rows.sort(
        key=lambda row: (
            row["scenario"] or "",
            row["variant"] or "",
            row["run_id"] or "",
            row["module"] or "",
        )
    )
    summary_path = results_dir / "summary.csv"
    runtime_path = REPO_ROOT / "results" / "bench" / "runtime_summary.csv"
    _write_csv(
        summary_path,
        rows,
        [
            "run_id",
            "scenario",
            "variant",
            "sensitivity",
            "executed",
            "git_sha",
            "bag_path",
            "duration_ns",
            "object_nodes",
            "object_edges",
            "place_nodes",
            "place_edges",
            "zero_detection_frames",
            "frames_seen",
            "pose_failures",
            "tf_lookup_failures",
            "failure_flags",
            "result_path",
        ],
    )
    _write_csv(
        runtime_path,
        runtime_rows,
        ["run_id", "scenario", "variant", "module", "n", "mean_ms", "p50_ms", "p95_ms"],
    )
    return {"summary": summary_path, "runtime": runtime_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results" / "eval"))
    args = parser.parse_args()
    paths = aggregate(Path(args.results_dir).expanduser())
    for path in paths.values():
        print(path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
