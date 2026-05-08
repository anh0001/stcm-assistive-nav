#!/usr/bin/env python3
"""Run a single DSE design point on outdoor_livinglab × full-nyu-proposals.

Reads a JSON of overrides for the scenario `config_overrides`, materializes
a temp manifest with those overrides applied, runs the experiment, parses
F1 from result.json, and appends a row to dse_results/dse_log.csv.

Usage:
    python3 dse_results/run_dse_point.py --iter 1 --params '{"max_observation_range_m": 7}' --note "phase1 R=7"

Designed to be invoked sequentially by the DSE orchestrator. One run = one row.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs" / "experiments" / "manifest.yaml"
DSE_DIR = REPO / "dse_results"
LOG_CSV = DSE_DIR / "dse_log.csv"
STATE_JSON = DSE_DIR / "DSE_STATE.json"
SCENARIO = "outdoor_livinglab"
VARIANT = "full-nyu-proposals"

CSV_FIELDS = [
    "iteration",
    "timestamp",
    "max_observation_range_m",
    "gng_min_observations_to_commit",
    "box_threshold",
    "text_threshold",
    "target_label_thresholds_trailer",
    "label_margin_min",
    "cross_label_merge_distance_m",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "object_nodes",
    "range_gate_rejections",
    "constraint_met",
    "result_json",
    "capture_method",
    "rc",
    "notes",
]


def load_manifest() -> dict:
    with MANIFEST.open() as f:
        return yaml.safe_load(f)


def save_manifest(data: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def apply_overrides(manifest: dict, overrides: dict) -> dict:
    """Patch outdoor_livinglab.config_overrides with design point values.
    Trailer threshold special-cased: must update target_label_thresholds[0]."""
    sc = manifest["scenarios"][SCENARIO]["config_overrides"]
    for k, v in overrides.items():
        if k == "target_label_thresholds_trailer":
            # First entry of target_label_thresholds is trailer (matches manifest order).
            sc["target_label_thresholds"][0] = float(v)
        elif k in ("box_threshold", "text_threshold", "label_margin_min",
                   "cross_label_merge_distance_m", "max_observation_range_m",
                   "gng_cluster_merge_distance", "gng_outlier_gate_meters",
                   "cross_label_merge_min_cosine"):
            sc[k] = float(v)
        elif k == "gng_min_observations_to_commit":
            sc[k] = int(v)
        else:
            sc[k] = v
    return manifest


def parse_result(result_json_path: Path) -> dict:
    with result_json_path.open() as f:
        r = json.load(f)
    g = r.get("graph", {})
    b = r.get("benchmark", {})
    s = b.get("summary", {})
    rt_events = g.get("runtime", {}).get("events", {}) if isinstance(g.get("runtime"), dict) else {}
    return {
        "tp": s.get("tp"),
        "fp": s.get("fp"),
        "fn": s.get("fn"),
        "precision": s.get("precision"),
        "recall": s.get("recall"),
        "f1": s.get("f1"),
        "object_nodes": g.get("object_node_count"),
        "range_gate_rejections": rt_events.get("range_gate_rejections"),
    }


def append_log_row(row: dict) -> None:
    new = not LOG_CSV.exists()
    with LOG_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--params", required=True, help="JSON string of overrides")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-run", action="store_true")
    args = ap.parse_args()

    overrides = json.loads(args.params)
    manifest = load_manifest()
    manifest = apply_overrides(manifest, overrides)

    cfg_dir = DSE_DIR / "configs" / f"iter_{args.iter:03d}"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg_dir / "manifest.yaml"
    save_manifest(manifest, manifest_path)
    (cfg_dir / "params.json").write_text(json.dumps(overrides, indent=2))

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "experiments" / "run_experiment.py"),
        "--manifest", str(manifest_path),
        "--scenario", SCENARIO,
        "--variant", VARIANT,
        "--skip-bag-hash",
    ]
    if args.no_run:
        cmd.append("--no-run")

    out_dir = DSE_DIR / "outputs" / f"iter_{args.iter:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "stdout.log"
    err_path = out_dir / "stderr.log"

    print(f"[iter {args.iter}] params={overrides}")
    print(f"[iter {args.iter}] cmd={' '.join(cmd)}")
    # Snapshot existing result files BEFORE the run so we can detect a
    # missing-output failure even if a stale file exists from a prior run.
    eval_dir = REPO / "results" / "eval"
    pre_existing = {p.resolve() for p in eval_dir.glob(f"*_{SCENARIO}_{VARIANT}.json")}
    with log_path.open("w") as out_f, err_path.open("w") as err_f:
        proc = subprocess.run(cmd, stdout=out_f, stderr=err_f, cwd=REPO)

    # Capture exact result path printed by run_experiment.py: line of form
    # "Wrote results/eval/<run_id>.json" (relative to REPO).
    result_json = None
    capture_method = None
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        for line in log_text.splitlines():
            if line.startswith("Wrote ") and line.endswith(".json"):
                rel = line[len("Wrote "):].strip()
                cand = (REPO / rel).resolve()
                if cand.exists():
                    result_json = cand
                    capture_method = "stdout_marker"
                    break
    except OSError:
        pass

    if result_json is None:
        # Fallback: look for ANY *new* result file produced after pre-snapshot.
        post = {p.resolve() for p in eval_dir.glob(f"*_{SCENARIO}_{VARIANT}.json")}
        new_files = sorted(post - pre_existing, key=lambda p: p.stat().st_mtime)
        if new_files:
            result_json = new_files[-1]
            capture_method = "post_snapshot_diff"

    if result_json is None:
        # Hard fail: do not silently log a stale result.
        print(
            f"[iter {args.iter}] ERROR: no new result JSON produced (rc={proc.returncode}). "
            f"Logged row will have constraint_met=False."
        )
        capture_method = "missing"

    metrics = parse_result(result_json) if result_json else {}
    constraint_met = (
        result_json is not None
        and metrics.get("tp") is not None
        and metrics.get("f1") is not None
    )

    sc = manifest["scenarios"][SCENARIO]["config_overrides"]
    row = {
        "iteration": args.iter,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "max_observation_range_m": sc.get("max_observation_range_m"),
        "gng_min_observations_to_commit": sc.get("gng_min_observations_to_commit"),
        "box_threshold": sc.get("box_threshold"),
        "text_threshold": sc.get("text_threshold"),
        "target_label_thresholds_trailer": sc.get("target_label_thresholds", [None])[0],
        "label_margin_min": sc.get("label_margin_min"),
        "cross_label_merge_distance_m": sc.get("cross_label_merge_distance_m"),
        "constraint_met": constraint_met,
        "result_json": str(result_json) if result_json else "",
        "capture_method": capture_method or "",
        "rc": proc.returncode,
        "notes": args.note,
        **metrics,
    }
    append_log_row(row)

    print(
        f"[iter {args.iter}] F1={metrics.get('f1')} TP={metrics.get('tp')} "
        f"FP={metrics.get('fp')} R={metrics.get('recall')} (rc={proc.returncode}, "
        f"capture={capture_method})"
    )
    if not constraint_met:
        # Loud fail signal so DSE orchestrator can decide to skip vs retry.
        return 2 if proc.returncode == 0 else proc.returncode
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
