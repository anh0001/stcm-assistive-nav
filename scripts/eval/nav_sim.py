#!/usr/bin/env python3
"""Graph-feasibility navigation success simulator.

No closed-loop control. Operates on the saved STCM place_graph + semantic_graph
and a ground-truth semantic graph. For each trial:
  1. Sample a goal label from `--goal-labels`.
  2. Pick a GT instance with that label (round-robin per seed).
  3. Find the predicted place node closest to the GT instance pose.
  4. Try networkx.shortest_path from a fixed start place node to that goal node.
  5. Success requires:
       - path exists in place_graph
       - some predicted object of `goal_label` lies within `--label-radius`
         metres of the GT instance pose
       - path terminal pose lies within `--terminal-radius` metres of GT pose

Reports success_rate with Wilson 95% CI and mean / p95 path-length ratio.

Honest framing for paper: "graph-feasibility navigation success — control loop
not closed; tests whether the built map supports planning to labelled goals."
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import networkx as nx


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


_LABEL_ALIASES = {
    "trash bins": "trash bin",
    "trash_bin": "trash bin",
    "table": "meeting table set",
    "tables": "meeting table set",
    "desk": "meeting table set",
    "meeting_table_set": "meeting table set",
    "elevator sliding door": "door",
    "elevator_sliding_door": "door",
    "vacuum_cleaner": "vacuum cleaner",
    "plant_pot": "plant pot",
    "cardboard_box": "cardboard box",
    "water_fountain": "water fountain",
    "emergency_exit_sign": "emergency exit sign",
    "electric_vehicle": "electric vehicle",
    "large_power_bank": "large power bank",
}


def _norm_label(label: str) -> str:
    key = str(label).strip().lower()
    return _LABEL_ALIASES.get(key, key)


def _xy(pose: list[float]) -> tuple[float, float] | None:
    if not pose or len(pose) < 2:
        return None
    try:
        x = float(pose[0])
        y = float(pose[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if abs(x) > 1e6 or abs(y) > 1e6:
        return None
    return x, y


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_place_graph(place_graph: dict[str, Any]) -> tuple[nx.Graph, dict[str, tuple[float, float]]]:
    g = nx.Graph()
    pose_map: dict[str, tuple[float, float]] = {}
    for node in place_graph.get("nodes", []):
        nid = str(node.get("id"))
        xy = _xy(node.get("pose", []))
        if xy is None:
            continue
        pose_map[nid] = xy
        g.add_node(nid, pose=xy)
    for link in place_graph.get("links", []):
        s = str(link.get("source"))
        t = str(link.get("target"))
        if s in pose_map and t in pose_map:
            g.add_edge(s, t, weight=_euclid(pose_map[s], pose_map[t]))
    return g, pose_map


def _wilson(success: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    if total == 0:
        return 0.0, 0.0, 0.0
    p = success / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    halfw = (z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)) / denom
    return p, max(0.0, center - halfw), min(1.0, center + halfw)


def _nearest_node(target: tuple[float, float], pose_map: dict[str, tuple[float, float]]) -> str | None:
    best_id = None
    best_d = math.inf
    for nid, xy in pose_map.items():
        d = _euclid(target, xy)
        if d < best_d:
            best_d = d
            best_id = nid
    return best_id


def simulate(
    pred: dict[str, Any],
    gt: dict[str, Any],
    *,
    goal_labels: list[str],
    n_trials: int,
    seed: int,
    label_radius: float,
    terminal_radius: float,
) -> dict[str, Any]:
    place = pred.get("place_graph") or {}
    sg = pred.get("semantic_graph") or {}
    gt_sg = gt.get("semantic_graph") or {}

    g, pose_map = _build_place_graph(place)
    if not pose_map:
        return {"error": "empty_place_graph", "trials": [], "success_rate": None}

    gt_by_label: dict[str, list[dict[str, Any]]] = {}
    for n in gt_sg.get("nodes", []):
        cat = _norm_label(str(n.get("category") or n.get("label") or ""))
        xy = _xy(n.get("pose", []))
        if xy is None:
            continue
        gt_by_label.setdefault(cat, []).append({"id": str(n.get("id", cat)), "xy": xy})

    pred_by_label: dict[str, list[tuple[float, float]]] = {}
    for n in sg.get("nodes", []):
        cat = _norm_label(str(n.get("category") or n.get("label") or ""))
        xy = _xy(n.get("pose", []))
        if xy is None:
            continue
        pred_by_label.setdefault(cat, []).append(xy)

    rng = random.Random(seed)
    place_ids = sorted(pose_map.keys())
    start_id = place_ids[0]
    start_xy = pose_map[start_id]

    trials: list[dict[str, Any]] = []
    successes = 0
    ratios: list[float] = []

    for i in range(n_trials):
        label = goal_labels[i % len(goal_labels)]
        gt_instances = gt_by_label.get(label, [])
        if not gt_instances:
            trials.append({
                "trial": i,
                "label": label,
                "success": False,
                "reason": "no_gt_instance",
            })
            continue

        gt_inst = rng.choice(gt_instances)
        goal_xy = gt_inst["xy"]
        goal_node = _nearest_node(goal_xy, pose_map)
        if goal_node is None:
            trials.append({"trial": i, "label": label, "success": False, "reason": "no_place_node"})
            continue

        try:
            path = nx.shortest_path(g, start_id, goal_node, weight="weight")
            path_len = nx.shortest_path_length(g, start_id, goal_node, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            trials.append({"trial": i, "label": label, "success": False, "reason": "no_path"})
            continue

        pred_instances = pred_by_label.get(label, [])
        label_ok = any(_euclid(p, goal_xy) <= label_radius for p in pred_instances)

        terminal_xy = pose_map[goal_node]
        terminal_ok = _euclid(terminal_xy, goal_xy) <= terminal_radius

        success = label_ok and terminal_ok
        euclid_dist = _euclid(start_xy, goal_xy)
        ratio = (path_len / euclid_dist) if euclid_dist > 1e-3 else None

        trials.append({
            "trial": i,
            "label": label,
            "gt_id": gt_inst["id"],
            "start_node": start_id,
            "goal_node": goal_node,
            "path_len_m": path_len,
            "euclid_m": euclid_dist,
            "path_ratio": ratio,
            "path_hops": len(path),
            "label_ok": label_ok,
            "terminal_ok": terminal_ok,
            "success": success,
            "reason": "ok" if success else ("label_miss" if not label_ok else "terminal_miss"),
        })
        if success:
            successes += 1
            if ratio is not None and math.isfinite(ratio):
                ratios.append(ratio)

    rate, lo, hi = _wilson(successes, n_trials)
    ratios_sorted = sorted(ratios)
    p95 = ratios_sorted[min(len(ratios_sorted) - 1, math.ceil(0.95 * len(ratios_sorted)) - 1)] if ratios_sorted else None

    return {
        "n_trials": n_trials,
        "success": successes,
        "success_rate": rate,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "mean_path_ratio": (sum(ratios) / len(ratios)) if ratios else None,
        "p95_path_ratio": p95,
        "seed": seed,
        "goal_labels": goal_labels,
        "label_radius_m": label_radius,
        "terminal_radius_m": terminal_radius,
        "trials": trials,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prediction", required=True, type=Path, help="stcm.json")
    ap.add_argument("--ground-truth", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--goal-labels", nargs="+", required=True)
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-radius", type=float, default=1.0)
    ap.add_argument("--terminal-radius", type=float, default=1.5)
    args = ap.parse_args()

    pred = _load(args.prediction)
    gt = _load(args.ground_truth)
    result = simulate(
        pred,
        gt,
        goal_labels=args.goal_labels,
        n_trials=args.n_trials,
        seed=args.seed,
        label_radius=args.label_radius,
        terminal_radius=args.terminal_radius,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")
    if "error" in result:
        print(f"  error={result['error']}")
    else:
        print(f"  success={result['success']}/{result['n_trials']} "
              f"rate={result['success_rate']:.3f} "
              f"CI=[{result['wilson_lo']:.3f},{result['wilson_hi']:.3f}] "
              f"mean_ratio={result['mean_path_ratio']}")


if __name__ == "__main__":
    main()
