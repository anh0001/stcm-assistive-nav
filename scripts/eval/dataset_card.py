#!/usr/bin/env python3
"""Build dataset card (Table A) from manifest + bag metadata + GT graph.

Reads configs/experiments/manifest.yaml + GT graphs and writes a JSON
record per scene. Trajectory length comes from rosbag metadata (duration only)
unless `--odom` is provided.

Output schema (per scene):
  {
    "scene": "meeting",
    "bag_path": "...",
    "storage_id": "sqlite3",
    "duration_s": 230.4,
    "frame_count_estimate": null,
    "label_vocabulary": ["chair", ...],
    "gt_instance_count": 20,
    "label_distribution": {"chair": 3, ...}
  }

Usage:
  python3 scripts/eval/dataset_card.py \
      --manifest configs/experiments/manifest.yaml \
      --output paper/tables/dataset_card.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _bag_metadata_path(bag_dir: Path) -> Path | None:
    if not bag_dir.exists():
        return None
    meta = bag_dir / "metadata.yaml"
    if meta.exists():
        return meta
    return None


def _bag_duration_s(meta_path: Path) -> float | None:
    try:
        meta = yaml.safe_load(meta_path.read_text())
    except Exception:
        return None
    info = (meta or {}).get("rosbag2_bagfile_information", meta) or {}
    dur = info.get("duration", {})
    if isinstance(dur, dict):
        nanos = dur.get("nanoseconds")
        if nanos is not None:
            try:
                return float(nanos) / 1e9
            except (TypeError, ValueError):
                return None
    if isinstance(dur, (int, float)):
        return float(dur) / 1e9
    return None


def _gt_record(gt_path: Path) -> dict[str, Any]:
    if not gt_path.exists():
        return {"gt_instance_count": None, "label_distribution": {}, "label_vocabulary": []}
    data = json.loads(gt_path.read_text())
    nodes = data.get("semantic_graph", {}).get("nodes", [])
    labels = [str(n.get("category") or n.get("label") or "") for n in nodes]
    dist = dict(Counter(labels))
    return {
        "gt_instance_count": len(nodes),
        "label_distribution": dist,
        "label_vocabulary": sorted(dist.keys()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="configs/experiments/manifest.yaml", type=Path)
    ap.add_argument("--scenes", nargs="*", default=["meeting", "livinglab"])
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    scenarios = manifest.get("scenarios", {})

    cards: list[dict[str, Any]] = []
    for scene in args.scenes:
        sc = scenarios.get(scene, {})
        bag_path = Path(sc.get("bag_path", ""))
        meta = _bag_metadata_path(bag_path)
        duration = _bag_duration_s(meta) if meta else None

        gt_path_str = sc.get("ground_truth_path") or f"configs/experiments/ground_truth/{scene}_stcm_gt.json"
        gt_record = _gt_record(Path(gt_path_str))

        cards.append({
            "scene": scene,
            "bag_path": str(bag_path),
            "storage_id": sc.get("storage_id"),
            "duration_s": duration,
            "metadata_path": str(meta) if meta else None,
            **gt_record,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cards": cards}, indent=2) + "\n")
    print(f"wrote {args.output}")
    for c in cards:
        print(f"  {c['scene']}: dur={c['duration_s']}s, gt={c['gt_instance_count']} "
              f"vocab={len(c['label_vocabulary'])}")


if __name__ == "__main__":
    main()
