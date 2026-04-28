#!/usr/bin/env python3
"""Natural-language grounding accuracy on STCM object graph.

Loads a command set with ground-truth target object ids and scores a
deterministic template grounder against it. Reports top-1 and top-2 accuracy
broken down by command subset (simple / disambiguation / compositional).

Command YAML schema:
  scene: meeting
  commands:
    - id: m_simple_01
      subset: simple
      text: "go to the chair"
      target_label: chair
      gt_object_id: chair_inst_1
    - id: m_disambig_01
      subset: disambiguation
      text: "go to the chair near the desk"
      target_label: chair
      relation:
        type: near
        anchor_label: desk
      gt_object_id: chair_inst_7
    - id: m_compose_01
      subset: compositional
      text: "go to the trash bin between the door and the water fountain"
      target_label: trash bin
      relation:
        type: between
        anchor_labels: [door, water fountain]
      gt_object_id: trash bin_inst_2

Grounder logic:
  - candidates = predicted objects with label == target_label
  - score = -dist_to_anchor (near) or -(d_a + d_b + |d_a - d_b|) (between)
    or 0 (no relation -> first-match wins, broken arbitrarily by id)
  - rank candidates by score desc; top-1 / top-2 hit if gt_object_id appears

Usage:
  python3 scripts/eval/grounding.py \
      --prediction output/stcm.json \
      --commands configs/eval/commands_meeting.yaml \
      --output output/grounding.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml


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


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


_LABEL_ALIASES = {
    "trash bins": "trash bin",
    "trash_bin": "trash bin",
    "trashbin": "trash bin",
    # GT uses "meeting table set" for a composite of one table + up to 4 chairs.
    # The predicted graph stores the centerpiece as "table" or "desk" plus
    # neighbouring "chair" nodes. We collapse "table" / "desk" / "elevator sliding
    # door" onto the GT names so the downstream metrics see the table component
    # of the set as the dominant predicted instance.
    "table": "meeting table set",
    "desk": "meeting table set",
    "tables": "meeting table set",
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


def _objects_by_label(pred: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    sg = pred.get("semantic_graph") or {}
    out: dict[str, list[dict[str, Any]]] = {}
    for n in sg.get("nodes", []):
        raw = str(n.get("category") or n.get("label") or "")
        label = _norm_label(raw)
        xy = _xy(n.get("pose", []))
        if xy is None:
            continue
        out.setdefault(label, []).append({
            "id": str(n.get("id", "")),
            "instance_id": n.get("instance_id"),
            "label": label,
            "xy": xy,
        })
    return out


def _nearest_anchor(cxy: tuple[float, float], label: str,
                    by_label: dict[str, list[dict[str, Any]]],
                    exclude_id: str | None = None) -> tuple[float, float] | None:
    """Return pose of the anchor instance closest to candidate.

    For multi-instance anchor labels (e.g. "desk"), we pick the instance closest
    to the candidate so that "chair near desk" ranks each chair against the
    desk it is actually near, instead of always against the first desk."""
    best: tuple[float, float] | None = None
    best_d = math.inf
    for c in by_label.get(label, []):
        if exclude_id is not None and c.get("id") == exclude_id:
            continue
        d = _dist(cxy, c["xy"])
        if d < best_d:
            best_d = d
            best = c["xy"]
    return best


def _score_candidate(cand: dict[str, Any], relation: dict[str, Any] | None,
                     by_label: dict[str, list[dict[str, Any]]]) -> float:
    if not relation:
        return 0.0
    rtype = relation.get("type")
    cxy = cand["xy"]
    cid = cand.get("id")
    if rtype == "near":
        anchor = _nearest_anchor(cxy, relation.get("anchor_label", ""), by_label, exclude_id=cid)
        if anchor is None:
            return -1e6
        return -_dist(cxy, anchor)
    if rtype == "between":
        labels = relation.get("anchor_labels", []) or []
        if len(labels) < 2:
            return -1e6
        a = _nearest_anchor(cxy, labels[0], by_label, exclude_id=cid)
        b = _nearest_anchor(cxy, labels[1], by_label, exclude_id=cid)
        if a is None or b is None:
            return -1e6
        da = _dist(cxy, a)
        db = _dist(cxy, b)
        return -(da + db + abs(da - db))
    return 0.0


def _normalize_relation(relation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not relation:
        return relation
    out = dict(relation)
    if "anchor_label" in out:
        out["anchor_label"] = _norm_label(out["anchor_label"])
    if "anchor_labels" in out:
        out["anchor_labels"] = [_norm_label(a) for a in out["anchor_labels"]]
    return out


def _ground(cmd: dict[str, Any], by_label: dict[str, list[dict[str, Any]]]) -> list[str]:
    label = _norm_label(cmd.get("target_label", ""))
    cands = list(by_label.get(label, []))
    relation = _normalize_relation(cmd.get("relation"))
    scored = [(c["id"], _score_candidate(c, relation, by_label)) for c in cands]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [cid for cid, _ in scored]


def _resolve_gt_pose(gt_id: str, gt_nodes: list[dict[str, Any]]) -> tuple[float, float] | None:
    for n in gt_nodes:
        if str(n.get("id", "")) == gt_id:
            return _xy(n.get("pose", []))
    return None


def _gt_id_match(predicted_id: str, gt_id: str,
                 pred_xy: tuple[float, float] | None,
                 gt_xy: tuple[float, float] | None,
                 radius_m: float = 1.0) -> bool:
    """Predicted instance matches GT if exact id equality OR same label and pose
    within radius_m of the GT pose. Predicted instance ids encode label, but the
    numeric suffix is system-internal and not aligned with GT id numbering, so
    fall back to spatial proximity."""
    if predicted_id == gt_id:
        return True
    if pred_xy is None or gt_xy is None:
        return False
    return math.hypot(pred_xy[0] - gt_xy[0], pred_xy[1] - gt_xy[1]) <= radius_m


def evaluate(pred: dict[str, Any], commands: list[dict[str, Any]],
             gt: dict[str, Any] | None = None,
             match_radius_m: float = 1.0) -> dict[str, Any]:
    by_label = _objects_by_label(pred)
    pred_xy_by_id: dict[str, tuple[float, float]] = {}
    for cands in by_label.values():
        for c in cands:
            pred_xy_by_id[c["id"]] = c["xy"]
    gt_nodes = (gt or {}).get("semantic_graph", {}).get("nodes", []) if gt else []

    rows: list[dict[str, Any]] = []
    subset_totals: dict[str, dict[str, int]] = {}

    for cmd in commands:
        ranked = _ground(cmd, by_label)
        gt_id = str(cmd.get("gt_object_id", ""))
        gt_xy = _resolve_gt_pose(gt_id, gt_nodes)
        top1 = bool(ranked) and _gt_id_match(ranked[0], gt_id, pred_xy_by_id.get(ranked[0]), gt_xy, match_radius_m)
        top2 = any(_gt_id_match(rid, gt_id, pred_xy_by_id.get(rid), gt_xy, match_radius_m) for rid in ranked[:2])
        subset = str(cmd.get("subset", "unknown"))

        rows.append({
            "id": cmd.get("id"),
            "subset": subset,
            "text": cmd.get("text"),
            "gt_object_id": gt_id,
            "ranked": ranked[:5],
            "top1": top1,
            "top2": top2,
            "n_candidates": len(ranked),
        })
        st = subset_totals.setdefault(subset, {"n": 0, "top1": 0, "top2": 0})
        st["n"] += 1
        st["top1"] += int(top1)
        st["top2"] += int(top2)

    overall = {
        "n": len(rows),
        "top1": sum(1 for r in rows if r["top1"]),
        "top2": sum(1 for r in rows if r["top2"]),
    }
    overall["top1_acc"] = overall["top1"] / overall["n"] if overall["n"] else 0.0
    overall["top2_acc"] = overall["top2"] / overall["n"] if overall["n"] else 0.0

    for st in subset_totals.values():
        st["top1_acc"] = st["top1"] / st["n"] if st["n"] else 0.0
        st["top2_acc"] = st["top2"] / st["n"] if st["n"] else 0.0

    return {"overall": overall, "by_subset": subset_totals, "trials": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prediction", required=True, type=Path)
    ap.add_argument("--ground-truth", type=Path,
                    help="Optional GT graph; enables spatial id-match fallback.")
    ap.add_argument("--match-radius", type=float, default=1.0)
    ap.add_argument("--commands", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    pred = json.loads(args.prediction.read_text())
    gt = json.loads(args.ground_truth.read_text()) if args.ground_truth else None
    spec = yaml.safe_load(args.commands.read_text())
    commands = spec.get("commands", [])

    result = evaluate(pred, commands, gt=gt, match_radius_m=args.match_radius)
    result["scene"] = spec.get("scene")
    result["commands_path"] = str(args.commands)
    result["prediction_path"] = str(args.prediction)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"  overall top1={result['overall']['top1_acc']:.3f}  "
          f"top2={result['overall']['top2_acc']:.3f}")
    for subset, st in result["by_subset"].items():
        print(f"  {subset}: top1={st['top1_acc']:.3f} top2={st['top2_acc']:.3f} n={st['n']}")


if __name__ == "__main__":
    main()
