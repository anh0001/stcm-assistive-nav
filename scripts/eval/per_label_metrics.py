#!/usr/bin/env python3
"""Per-class IoU / macro-F1 / micro-F1 from benchmark.json.

Reads benchmark output produced by scripts/experiments/benchmark_stcm_graph.py
and emits per-label IoU plus macro/micro aggregates.

Usage:
  python3 scripts/eval/per_label_metrics.py \
      --benchmark output/stcm_benchmark.json \
      --output output/per_label.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _iou(tp: int, fp: int, fn: int) -> float | None:
    denom = tp + fp + fn
    if denom == 0:
        return None
    return tp / denom


def _f1(tp: int, fp: int, fn: int) -> float | None:
    denom = 2 * tp + fp + fn
    if denom == 0:
        return None
    return (2 * tp) / denom


def compute(benchmark: dict[str, Any]) -> dict[str, Any]:
    per_label_in = benchmark.get("per_label", {}) or {}

    rows: list[dict[str, Any]] = []
    sum_tp = sum_fp = sum_fn = 0
    iou_vals: list[float] = []
    f1_vals: list[float] = []

    for label, stats in per_label_in.items():
        tp = int(stats.get("tp", 0))
        fp = int(stats.get("fp", 0))
        fn = int(stats.get("fn", 0))
        iou = _iou(tp, fp, fn)
        f1 = _f1(tp, fp, fn)
        rows.append({
            "label": label,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "iou": iou,
            "f1": f1,
            "precision": stats.get("precision"),
            "recall": stats.get("recall"),
        })
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
        if iou is not None:
            iou_vals.append(iou)
        if f1 is not None:
            f1_vals.append(f1)

    macro_iou = sum(iou_vals) / len(iou_vals) if iou_vals else None
    macro_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else None
    micro_iou = _iou(sum_tp, sum_fp, sum_fn)
    micro_f1 = _f1(sum_tp, sum_fp, sum_fn)

    return {
        "per_label": rows,
        "macro_iou": macro_iou,
        "macro_f1": macro_f1,
        "micro_iou": micro_iou,
        "micro_f1": micro_f1,
        "total_tp": sum_tp,
        "total_fp": sum_fp,
        "total_fn": sum_fn,
        "n_labels": len(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    benchmark = json.loads(args.benchmark.read_text())
    result = compute(benchmark)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"  macro_iou={result['macro_iou']}  macro_f1={result['macro_f1']}")
    print(f"  micro_iou={result['micro_iou']}  micro_f1={result['micro_f1']}")


if __name__ == "__main__":
    main()
