#!/usr/bin/env python3
"""Per-scene per-subset recoverability breakdown for AE-3 reporting.

Codex review P6: distinguish commands the predicted graph CAN answer
(eligible / recoverable_by_pred_graph) from commands that auto-fail in
all arms (perception_missing). Otherwise the functional subset can look
like padding.

Reads grounding-report JSONs (with per-trial gold_referent_present /
anchor_nodes_present / eligible_grounding_given_perception flags) and
emits a CSV summary. Same JSONs power the McNemar pipeline.

Usage:
  python3 scripts/eval/recoverability_breakdown.py \
      --reports results/grounding/meeting_grounding_spatial.json \
                results/grounding/meeting_grounding_intent.json \
                results/grounding_llm/meeting_grounding.json \
                results/grounding/livinglab_grounding_spatial.json \
                results/grounding/livinglab_grounding_intent.json \
                results/grounding_llm/livinglab_grounding.json \
      --output paper/tables/C_recoverability_breakdown.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _arm_id(report: dict[str, Any], path: Path) -> str:
    if report.get("backend"):
        return f"+LLM ({report['backend']})"
    arm = report.get("arm")
    if arm == "template-spatial":
        return "no-LLM (template-spatial)"
    if arm == "template+intent-lexicon":
        return "+intent-lexicon"
    return path.stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", nargs="+", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    rows = []
    for path in args.reports:
        rep = json.loads(path.read_text())
        scene = rep.get("scene") or path.stem
        arm = _arm_id(rep, path)
        # bucket by (subset, recoverable)
        bucket = defaultdict(lambda: {"n": 0, "top1": 0})
        all_subset = defaultdict(lambda: {"n": 0, "top1": 0,
                                          "n_recov": 0, "top1_recov": 0,
                                          "n_missing": 0, "top1_missing": 0})
        for tr in rep.get("trials", []):
            sub = tr.get("subset", "unknown")
            recov = bool(tr.get("eligible_grounding_given_perception", True))
            top1 = bool(tr.get("top1"))
            st = all_subset[sub]
            st["n"] += 1
            st["top1"] += int(top1)
            if recov:
                st["n_recov"] += 1
                st["top1_recov"] += int(top1)
            else:
                st["n_missing"] += 1
                st["top1_missing"] += int(top1)
        for sub, st in sorted(all_subset.items()):
            top1_acc = st["top1"] / st["n"] if st["n"] else 0.0
            top1_recov_acc = st["top1_recov"] / st["n_recov"] if st["n_recov"] else 0.0
            top1_missing_acc = st["top1_missing"] / st["n_missing"] if st["n_missing"] else 0.0
            rows.append({
                "scene": scene,
                "arm": arm,
                "subset": sub,
                "n": st["n"],
                "top1": st["top1"],
                "top1_acc": round(top1_acc, 3),
                "n_recoverable": st["n_recov"],
                "top1_recoverable": st["top1_recov"],
                "top1_acc_recoverable": round(top1_recov_acc, 3),
                "n_perception_missing": st["n_missing"],
                "top1_perception_missing": st["top1_missing"],
                "top1_acc_perception_missing": round(top1_missing_acc, 3),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.output.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.output}  ({len(rows)} rows)")
    # console preview
    for r in rows:
        print(f"  {r['scene']:10s} {r['arm']:30s} {r['subset']:14s} "
              f"n={r['n']:2d} top1={r['top1_acc']:.3f}  "
              f"recov n={r['n_recoverable']:2d} top1={r['top1_acc_recoverable']:.3f}  "
              f"missing n={r['n_perception_missing']:2d}")


if __name__ == "__main__":
    main()
