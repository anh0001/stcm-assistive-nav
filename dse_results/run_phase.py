#!/usr/bin/env python3
"""Process a phase JSON file sequentially via run_dse_point.py.

Updates DSE_STATE.json after each iteration.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DSE = REPO / "dse_results"
RUNNER = DSE / "run_dse_point.py"
STATE = DSE / "DSE_STATE.json"
LOG = DSE / "dse_log.csv"


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {
        "iteration": 0,
        "status": "initializing",
        "best_metric": None,
        "best_params": None,
        "best_iter": None,
        "total_iterations": 0,
        "start_time": datetime.datetime.utcnow().isoformat(),
        "patience_counter": 0,
        "phase": None,
    }


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=2))


def best_so_far_from_log() -> tuple[float | None, int | None, dict | None]:
    if not LOG.exists():
        return None, None, None
    import csv
    best = None
    best_iter = None
    best_row = None
    with LOG.open() as f:
        for r in csv.DictReader(f):
            try:
                f1 = float(r.get("f1") or "nan")
            except ValueError:
                continue
            if f1 != f1:
                continue
            if r.get("constraint_met") != "True":
                continue
            if best is None or f1 > best:
                best = f1
                best_iter = int(r["iteration"])
                best_row = r
    return best, best_iter, best_row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-file", required=True)
    ap.add_argument("--phase-name", required=True)
    args = ap.parse_args()

    points = json.loads(Path(args.phase_file).read_text())
    s = load_state()
    s["phase"] = args.phase_name
    s["status"] = "in_progress"
    save_state(s)

    for p in points:
        rc = subprocess.call([
            sys.executable, str(RUNNER),
            "--iter", str(p["iter"]),
            "--params", json.dumps(p["params"]),
            "--note", p.get("note", ""),
        ], cwd=REPO)
        s["iteration"] = p["iter"]
        s["total_iterations"] += 1
        best, best_iter, best_row = best_so_far_from_log()
        s["best_metric"] = best
        s["best_iter"] = best_iter
        s["best_params"] = best_row
        save_state(s)
        if rc != 0:
            print(f"[phase {args.phase_name}] iter {p['iter']} returned rc={rc}, continuing")

    s["status"] = "phase_complete"
    s["end_time"] = datetime.datetime.utcnow().isoformat()
    save_state(s)
    print(f"[phase {args.phase_name}] complete. Best F1={s.get('best_metric')} at iter {s.get('best_iter')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
