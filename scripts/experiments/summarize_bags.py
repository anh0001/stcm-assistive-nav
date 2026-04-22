#!/usr/bin/env python3
"""Summarize rosbag2 metadata declared in the experiment manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "experiments" / "manifest.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _bag_info(path: Path) -> dict[str, Any]:
    metadata = _load_yaml(path / "metadata.yaml").get("rosbag2_bagfile_information", {})
    topics = {}
    for item in metadata.get("topics_with_message_count", []) or []:
        topic = item.get("topic_metadata", {}).get("name")
        if topic:
            topics[topic] = item.get("message_count", 0)
    return {
        "duration_ns": (metadata.get("duration") or {}).get("nanoseconds"),
        "message_count": metadata.get("message_count"),
        "storage_identifier": metadata.get("storage_identifier"),
        "topic_counts": topics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(REPO_ROOT / "results" / "eval" / "dataset_summary.csv"))
    args = parser.parse_args()

    manifest = _load_yaml(Path(args.manifest).expanduser())
    rows = []
    for scenario, data in sorted((manifest.get("scenarios") or {}).items()):
        bag_path = Path(data["bag_path"]).expanduser()
        info = _bag_info(bag_path)
        row = {
            "scenario": scenario,
            "bag_path": str(bag_path),
            "storage_identifier": info["storage_identifier"],
            "duration_ns": info["duration_ns"],
            "message_count": info["message_count"],
        }
        for topic in manifest.get("required_topics", []):
            row[f"topic:{topic}"] = info["topic_counts"].get(topic, 0)
        rows.append(row)

    fields = list(rows[0]) if rows else []
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(output.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
