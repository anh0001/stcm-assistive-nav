#!/usr/bin/env python3
"""Position stability across N runs of the same scenario.

For a set of stcm.json predictions on the same scene, match nodes across runs
by (label, nearest pose within --match-radius), then report per-class and
overall position standard deviation in metres.

Usage:
  python3 scripts/eval/stability.py \
      --predictions run0/stcm.json run1/stcm.json run2/stcm.json \
      --output output/stability.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _xy(pose: list[float]) -> tuple[float, float] | None:
    if not pose or len(pose) < 2:
        return None
    try:
        x, y = float(pose[0]), float(pose[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _by_label(pred: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    sg = pred.get("semantic_graph") or {}
    out: dict[str, list[tuple[float, float]]] = {}
    for n in sg.get("nodes", []):
        label = str(n.get("category") or n.get("label") or "")
        xy = _xy(n.get("pose", []))
        if xy is None:
            continue
        out.setdefault(label, []).append(xy)
    return out


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", nargs="+", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--match-radius", type=float, default=1.0)
    args = ap.parse_args()

    runs = [_by_label(json.loads(p.read_text())) for p in args.predictions]
    n_runs = len(runs)
    if n_runs < 2:
        raise SystemExit("Need at least 2 prediction files for stability.")

    labels = sorted({label for run in runs for label in run.keys()})
    per_label_sigma: dict[str, dict[str, Any]] = {}
    all_sigmas: list[float] = []

    for label in labels:
        # Anchor on run 0 nodes; for each, find nearest pose in each other run.
        anchors = list(runs[0].get(label, []))
        if not anchors:
            continue
        cluster_sigmas: list[float] = []
        for anchor in anchors:
            xs = [anchor[0]]
            ys = [anchor[1]]
            for ri in range(1, n_runs):
                cands = runs[ri].get(label, [])
                if not cands:
                    continue
                best = min(cands, key=lambda c: _euclid(c, anchor))
                if _euclid(best, anchor) <= args.match_radius:
                    xs.append(best[0])
                    ys.append(best[1])
            if len(xs) >= 2:
                # 2D position sigma = sqrt(var_x + var_y).
                sigma = math.sqrt(_std(xs) ** 2 + _std(ys) ** 2)
                cluster_sigmas.append(sigma)
        if cluster_sigmas:
            mean_sig = sum(cluster_sigmas) / len(cluster_sigmas)
            per_label_sigma[label] = {
                "n_clusters": len(cluster_sigmas),
                "mean_sigma_m": mean_sig,
                "max_sigma_m": max(cluster_sigmas),
            }
            all_sigmas.extend(cluster_sigmas)

    overall = {
        "n_runs": n_runs,
        "n_clusters": len(all_sigmas),
        "mean_sigma_m": (sum(all_sigmas) / len(all_sigmas)) if all_sigmas else None,
        "max_sigma_m": max(all_sigmas) if all_sigmas else None,
        "per_label": per_label_sigma,
        "predictions": [str(p) for p in args.predictions],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(overall, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"  mean_sigma_m={overall['mean_sigma_m']}")


if __name__ == "__main__":
    main()
