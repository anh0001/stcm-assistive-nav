#!/usr/bin/env python3
"""Aggregate per-module runtime across STCM result.json files into Table D.

For AE-4 / R1-4 (computational performance, real-time feasibility).

Reads results/eval/artifacts/*<scenario>_<variant>/result.json on the current
git SHA, extracts runtime.timings + bag metadata, emits:

  results/bench/runtime_<sha>.json   # machine-readable evidence
  paper/tables/D.tex                 # LaTeX table

Per-module mean / p50 / p95 latency in ms, effective Hz from sample count
divided by bag duration. FAST-LIO2 rate computed from /fastlio2/lio_odom
message_count / duration_ns.

Usage:
  python3 scripts/render_table_d.py
  python3 scripts/render_table_d.py --sha d5f5887 --runs meeting:full,livinglab:full
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "results" / "eval" / "artifacts"
BENCH_DIR = REPO / "results" / "bench"
TABLES_DIR = REPO / "paper" / "tables"

MODULE_LABEL = {
    "groundingdino_predict": "GroundingDINO detect",
    "sam_predict": "MobileSAM segment",
    "detection_filter": "Detection filter",
    "pose_association": "LiDAR-image assoc",
    "instance_gng_update": "Instance-GNG update",
    "place_gng_update": "Place-GNG update",
    "graph_update": "Temporal fusion",
    "frame_total": "Per-frame pipeline",
}
MODULE_ORDER = list(MODULE_LABEL.keys())


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()


def collect_hw() -> Dict[str, str]:
    def sh(cmd: str) -> str:
        try:
            return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
        except subprocess.CalledProcessError:
            return ""

    cpu = sh("lscpu | awk -F: '/Model name/ {print $2}'").strip()
    cores = sh("lscpu | awk -F: '/^Core\\(s\\) per socket/ {print $2}'").strip()
    sockets = sh("lscpu | awk -F: '/^Socket\\(s\\)/ {print $2}'").strip()
    threads = sh("lscpu | awk -F: '/^CPU\\(s\\):/ {print $2}'").strip().split()[0]
    ram_gb = sh("free -g | awk '/Mem:/ {print $2}'")
    gpu = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    driver = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
    cuda = sh("nvcc --version | tail -2 | head -1 | awk -F, '{print $2}'").strip()
    osver = sh("lsb_release -ds")
    kernel = sh("uname -r")
    py = sh("/usr/bin/python3 -c 'import sys; print(sys.version.split()[0])'")
    torchver = sh("PYTHONUSERBASE=$HOME/.local/stcm_sys_py310 /usr/bin/python3 -c 'import torch; print(torch.__version__)'")
    return {
        "cpu_model": cpu,
        "cpu_sockets": sockets,
        "cpu_cores_per_socket": cores,
        "cpu_threads_total": threads,
        "ram_gb": ram_gb,
        "gpu": gpu,
        "nvidia_driver": driver,
        "cuda_toolkit": cuda,
        "os": osver,
        "kernel": kernel,
        "ros_distro": "humble",
        "python": py,
        "pytorch": torchver,
    }


def find_runs(sha_short: str, scenarios: List[str], variant: str = "full") -> Dict[str, Path]:
    """Pick most recent artifact per scenario matching variant on given SHA."""
    out: Dict[str, Path] = {}
    for sc in scenarios:
        # match e.g. *meeting_full*, *livinglab_full*, *livinglab_tuned_full*
        candidates = sorted(
            [p for p in ARTIFACTS.glob(f"*_{sc}_{variant}*") if (p / "result.json").exists()],
            reverse=True,
        )
        for c in candidates:
            try:
                meta = json.loads((c / "result.json").read_text())
                if meta.get("git", {}).get("sha", "").startswith(sha_short):
                    out[sc] = c
                    break
            except Exception:
                continue
    return out


def extract_runtime(result_path: Path) -> Dict:
    d = json.loads(result_path.read_text())
    timings = d.get("graph", {}).get("runtime", {}).get("timings", {})
    events = d.get("graph", {}).get("runtime", {}).get("events", {})
    bag = d.get("bag", {})
    duration_s = bag.get("duration_ns", 0) / 1e9 if bag.get("duration_ns") else None

    # FAST-LIO2 effective rate
    fastlio = bag.get("topics", {}).get("/fastlio2/lio_odom", {})
    fastlio_hz = (fastlio.get("message_count", 0) / duration_s) if (fastlio and duration_s) else None
    # camera publish rate (pipeline upper bound)
    cam = bag.get("topics", {}).get("/camera/image_raw", {})
    cam_hz = (cam.get("message_count", 0) / duration_s) if (cam and duration_s) else None

    modules = {}
    for k in MODULE_ORDER:
        t = timings.get(k)
        if not t or t.get("n", 0) < 10:
            continue
        n = t["n"]
        modules[k] = {
            "n": n,
            "mean_ms": t["mean_ms"],
            "p50_ms": t["p50_ms"],
            "p95_ms": t["p95_ms"],
            "max_ms": t.get("max_ms"),
            "effective_hz": (n / duration_s) if duration_s else None,
        }
    return {
        "scenario": d.get("scenario"),
        "variant": d.get("variant"),
        "duration_s": duration_s,
        "bag_path": bag.get("path"),
        "frames_seen": events.get("frames_seen"),
        "fastlio2_hz": fastlio_hz,
        "camera_publish_hz": cam_hz,
        "modules": modules,
        "events": events,
        "bias_notes": {
            "frame_total_includes_skip_paths": (
                events.get("zero_detection_frames", 0)
                + events.get("target_label_empty_frames", 0)
            ),
            "pose_failures": events.get("pose_failures", 0),
            "tf_lookup_latest_fallbacks": events.get("tf_lookup_latest_fallbacks", 0),
            "raw_detections": events.get("raw_detections", 0),
            "target_detections": events.get("target_detections", 0),
        },
        "failure_flags": d.get("failure_flags", []),
        "result_path": str(result_path.relative_to(REPO)),
    }


def render_latex(hw: Dict, runs: Dict[str, Dict], sha: str) -> str:
    """Emit Table D: per-module latency averaged across scenarios.

    Critical-vs-async split per .claude/rules/benchmark-protocol.md.
    """
    ASYNC = {"groundingdino_predict", "sam_predict", "graph_update", "frame_total"}

    # average mean/p50/p95 across runs (per module)
    agg: Dict[str, Dict[str, List[float]]] = {}
    for run in runs.values():
        for mod, m in run["modules"].items():
            agg.setdefault(mod, {"mean_ms": [], "p50_ms": [], "p95_ms": [], "hz": []})
            agg[mod]["mean_ms"].append(m["mean_ms"])
            agg[mod]["p50_ms"].append(m["p50_ms"])
            agg[mod]["p95_ms"].append(m["p95_ms"])
            if m["effective_hz"]:
                agg[mod]["hz"].append(m["effective_hz"])

    fastlio_hz = [r["fastlio2_hz"] for r in runs.values() if r["fastlio2_hz"]]
    cam_hz = [r["camera_publish_hz"] for r in runs.values() if r["camera_publish_hz"]]

    rows = []
    for mod in MODULE_ORDER:
        if mod not in agg:
            continue
        a = agg[mod]
        kind = "async" if mod in ASYNC else "critical"
        rows.append({
            "label": MODULE_LABEL[mod],
            "kind": kind,
            "mean_ms": statistics.mean(a["mean_ms"]),
            "p50_ms": statistics.mean(a["p50_ms"]),
            "p95_ms": statistics.mean(a["p95_ms"]),
            "hz": statistics.mean(a["hz"]) if a["hz"] else None,
        })

    # Also inject FAST-LIO2 row (no per-call latency, just effective rate from bag).
    if fastlio_hz:
        rows.insert(0, {
            "label": "FAST-LIO2 odom (publish)",
            "kind": "critical",
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "hz": statistics.mean(fastlio_hz),
        })

    def fmt(x: Optional[float], width: int = 6) -> str:
        if x is None:
            return "--"
        if x >= 100:
            return f"{x:.0f}"
        return f"{x:.1f}"

    scenarios_str = ", ".join(sorted({r["scenario"] for r in runs.values()}))

    lines = []
    lines.append("% Generated by scripts/render_table_d.py — do not hand-edit")
    lines.append("% AE-4 / R1-4: computational performance and real-time feasibility")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Per-module runtime of the STCM pipeline (AE-4 / R1-4). "
                 "Mean / p50 / p95 latencies averaged across "
                 f"{len(runs)} offline rosbag replays ({scenarios_str}) on git SHA \\texttt{{{sha[:7]}}}; "
                 "\\emph{Eff.\\,Hz} = sample count / bag duration. "
                 "\\emph{Critical} modules must close the navigation loop in real time; "
                 "\\emph{async} modules run on independent threads and do not block "
                 "Nav2 / FAST-LIO2 (semantic graph updates lag detections by one cycle). "
                 f"Hardware: {hw['cpu_model']} ({hw['cpu_cores_per_socket']}c/{hw['cpu_threads_total']}t) "
                 f"+ {hw['gpu']}, {hw['ram_gb']} GiB RAM, {hw['os']} kernel "
                 f"{hw['kernel']}, ROS 2 {hw['ros_distro'].title()}, CUDA toolkit {hw['cuda_toolkit'].replace('release ', '')}, "
                 f"PyTorch {hw['pytorch']}, NVIDIA driver {hw['nvidia_driver']}.}}")
    lines.append("\\label{tab:runtime}")
    lines.append("\\begin{tabular}{l l r r r r}")
    lines.append("\\toprule")
    lines.append("Module & Class & Mean (ms) & p50 (ms) & p95 (ms) & Eff.\\,Hz \\\\")
    lines.append("\\midrule")
    for r in rows:
        marker = ""
        if r["label"] in ("Per-frame pipeline", "LiDAR-image assoc"):
            marker = "$^{\\dagger}$"
        lines.append(
            f"{r['label']}{marker} & {r['kind']} & "
            f"{fmt(r['mean_ms'])} & {fmt(r['p50_ms'])} & {fmt(r['p95_ms'])} & "
            f"{fmt(r['hz'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\\\[2pt]")
    lines.append("\\footnotesize $^{\\dagger}$Sample includes cheap-path early "
                 "returns (zero-detection / target-label-empty frames for "
                 "\\emph{Per-frame pipeline}; TF-lookup fast-fail for "
                 "\\emph{LiDAR-image assoc}); reader should consult "
                 "\\texttt{events.zero\\_detection\\_frames}, "
                 "\\texttt{target\\_label\\_empty\\_frames}, and "
                 "\\texttt{pose\\_failures} in "
                 "\\texttt{results/bench/runtime\\_<sha>.json} for the denominator.")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", default=None, help="Short git SHA (default: HEAD)")
    ap.add_argument("--scenarios", default="meeting,livinglab,livinglab_tuned",
                    help="comma-separated scenarios")
    ap.add_argument("--variant", default="full")
    args = ap.parse_args()

    sha = args.sha or git_sha()
    sha_short = sha[:7]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    runs = find_runs(sha_short, scenarios, variant=args.variant)
    if not runs:
        print(f"ERR: no runs found on SHA {sha_short} for scenarios={scenarios}, variant={args.variant}")
        print(f"  searched: {ARTIFACTS}")
        return 2

    print(f"Found {len(runs)} runs on SHA {sha_short}:")
    for sc, p in runs.items():
        print(f"  {sc:20s} -> {p.relative_to(REPO)}")

    hw = collect_hw()
    extracted = {sc: extract_runtime(p / "result.json") for sc, p in runs.items()}

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out_json = BENCH_DIR / f"runtime_{sha_short}.json"
    payload = {
        "sha": sha,
        "hw": hw,
        "runs": extracted,
        "note": "AE-4 runtime evidence; aggregated by scripts/render_table_d.py",
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_json.relative_to(REPO)}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tex = render_latex(hw, extracted, sha)
    out_tex = TABLES_DIR / "D.tex"
    out_tex.write_text(tex)
    print(f"Wrote {out_tex.relative_to(REPO)}")

    print("\n=== Table D preview ===")
    print(tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
