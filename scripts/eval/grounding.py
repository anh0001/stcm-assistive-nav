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

# Topology bonus added to "near" / "between" scores when the candidate shares
# (or is one hop from) the anchor's place-graph node. Sized to act as a
# tiebreaker between geometrically-similar candidates without overpowering
# clearly-better xy matches (typical inter-instance gaps are 1–5 m).
PLACE_TOPOLOGY_BONUS_M = 1.5


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


def _norm_label(label: str, aliases: dict[str, str] | None = None) -> str:
    key = str(label).strip().lower()
    table = aliases if aliases is not None else _LABEL_ALIASES
    return table.get(key, key)


def _load_alias_map(path: Path | None) -> dict[str, str]:
    """Merge `_LABEL_ALIASES` with optional canonical→[aliases] file.

    File format mirrors `configs/eval/label_aliases.json`:
        { "trash bin": ["trash bins", "trashbin"], ... }
    Returns an alias→canonical lookup with the file aliases winning on
    conflict so downstream graphs (which may emit "trash bins") collapse to
    the GT label."""
    merged: dict[str, str] = dict(_LABEL_ALIASES)
    if path is None:
        return merged
    try:
        raw = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return merged
    if not isinstance(raw, dict):
        return merged
    for canonical, alias_list in raw.items():
        if not isinstance(alias_list, list):
            continue
        canon_key = str(canonical).strip().lower()
        merged[canon_key] = canon_key
        for alias in alias_list:
            merged[str(alias).strip().lower()] = canon_key
    return merged


def _objects_by_label(pred: dict[str, Any],
                      aliases: dict[str, str] | None = None,
                      object_to_place: dict[str, str] | None = None
                      ) -> dict[str, list[dict[str, Any]]]:
    sg = pred.get("semantic_graph") or {}
    out: dict[str, list[dict[str, Any]]] = {}
    for n in sg.get("nodes", []):
        raw = str(n.get("category") or n.get("label") or "")
        label = _norm_label(raw, aliases=aliases)
        xy = _xy(n.get("pose", []))
        if xy is None:
            continue
        oid = str(n.get("id", ""))
        out.setdefault(label, []).append({
            "id": oid,
            "instance_id": n.get("instance_id"),
            "label": label,
            "xy": xy,
            "place": (object_to_place or {}).get(oid),
        })
    return out


def _build_place_index(pred: dict[str, Any]) -> dict[str, Any]:
    """Build a place-graph index for topology-aware scoring.

    Returns: {
        "places":        {place_id: (x, y)},
        "neighbors":     {place_id: set(neighbor_id)},
        "graph":         networkx.Graph or None,
    }
    Falls back to empty structures when the place graph or networkx is
    unavailable so the grounder degrades gracefully to geometric-only."""
    pg = pred.get("place_graph") or {}
    places: dict[str, tuple[float, float]] = {}
    for n in pg.get("nodes", []):
        pid = str(n.get("id", ""))
        xy = _xy(n.get("pose", []))
        if pid and xy is not None:
            places[pid] = xy
    neighbors: dict[str, set[str]] = {pid: set() for pid in places}
    for e in pg.get("links", []) or pg.get("edges", []):
        src = str(e.get("source", ""))
        dst = str(e.get("target", ""))
        if src in neighbors and dst in neighbors:
            neighbors[src].add(dst)
            neighbors[dst].add(src)
    graph = None
    if places:
        try:
            import networkx as nx
            graph = nx.Graph()
            graph.add_nodes_from(places.keys())
            for pid, nbrs in neighbors.items():
                for nb in nbrs:
                    graph.add_edge(pid, nb)
        except ImportError:
            graph = None
    return {"places": places, "neighbors": neighbors, "graph": graph}


def _assign_objects_to_places(pred: dict[str, Any],
                              place_index: dict[str, Any]) -> dict[str, str]:
    """Map each object node id to its nearest place-graph node id (xy)."""
    places = place_index["places"]
    if not places:
        return {}
    sg = pred.get("semantic_graph") or {}
    mapping: dict[str, str] = {}
    place_items = list(places.items())
    for n in sg.get("nodes", []):
        oid = str(n.get("id", ""))
        oxy = _xy(n.get("pose", []))
        if not oid or oxy is None:
            continue
        best_pid = None
        best_d = math.inf
        for pid, pxy in place_items:
            d = _dist(oxy, pxy)
            if d < best_d:
                best_d = d
                best_pid = pid
        if best_pid is not None:
            mapping[oid] = best_pid
    return mapping


def _nearest_anchor(cxy: tuple[float, float], label: str,
                    by_label: dict[str, list[dict[str, Any]]],
                    exclude_id: str | None = None
                    ) -> dict[str, Any] | None:
    """Return the anchor instance (full record) closest to candidate.

    For multi-instance anchor labels (e.g. "desk"), we pick the instance closest
    to the candidate so that "chair near desk" ranks each chair against the
    desk it is actually near, instead of always against the first desk. The
    full record is returned so downstream callers can read the anchor's
    place-graph node for topology-aware scoring."""
    best: dict[str, Any] | None = None
    best_d = math.inf
    for c in by_label.get(label, []):
        if exclude_id is not None and c.get("id") == exclude_id:
            continue
        d = _dist(cxy, c["xy"])
        if d < best_d:
            best_d = d
            best = c
    return best


def _topology_bonus(cand_place: str | None, anchor_place: str | None,
                    place_index: dict[str, Any] | None) -> float:
    """Bonus when candidate and anchor share a place node or are 1-hop apart.

    Returns 0.0 when topology is unavailable so the grounder collapses to
    geometric-only scoring."""
    if not place_index or not cand_place or not anchor_place:
        return 0.0
    if cand_place == anchor_place:
        return PLACE_TOPOLOGY_BONUS_M
    return 0.0


def _between_topology_bonus(cand_place: str | None,
                            anchor_a_place: str | None,
                            anchor_b_place: str | None,
                            place_index: dict[str, Any] | None) -> float:
    """Bonus when the candidate co-locates with *both* anchors at the same
    place node. We only reward strict co-location because the "shortest
    path" variant over-rewarded any candidate sharing one anchor's place
    in early calibration on the JACIII command sets."""
    if (not place_index or not cand_place
            or not anchor_a_place or not anchor_b_place):
        return 0.0
    if cand_place == anchor_a_place and cand_place == anchor_b_place:
        return PLACE_TOPOLOGY_BONUS_M
    return 0.0


def _score_candidate(cand: dict[str, Any], relation: dict[str, Any] | None,
                     by_label: dict[str, list[dict[str, Any]]],
                     place_index: dict[str, Any] | None = None) -> float:
    if not relation:
        return 0.0
    rtype = relation.get("type")
    cxy = cand["xy"]
    cid = cand.get("id")
    if rtype == "near":
        anchor = _nearest_anchor(cxy, relation.get("anchor_label", ""),
                                 by_label, exclude_id=cid)
        if anchor is None:
            return -1e6
        bonus = _topology_bonus(cand.get("place"), anchor.get("place"),
                                place_index)
        return -_dist(cxy, anchor["xy"]) + bonus
    if rtype == "between":
        labels = relation.get("anchor_labels", []) or []
        if len(labels) < 2:
            return -1e6
        a = _nearest_anchor(cxy, labels[0], by_label, exclude_id=cid)
        b = _nearest_anchor(cxy, labels[1], by_label, exclude_id=cid)
        if a is None or b is None:
            return -1e6
        da = _dist(cxy, a["xy"])
        db = _dist(cxy, b["xy"])
        bonus = _between_topology_bonus(cand.get("place"), a.get("place"),
                                        b.get("place"), place_index)
        return -(da + db + abs(da - db)) + bonus
    return 0.0


def _normalize_relation(relation: dict[str, Any] | None,
                        aliases: dict[str, str] | None = None
                        ) -> dict[str, Any] | None:
    if not relation:
        return relation
    out = dict(relation)
    if "anchor_label" in out:
        out["anchor_label"] = _norm_label(out["anchor_label"], aliases=aliases)
    if "anchor_labels" in out:
        out["anchor_labels"] = [_norm_label(a, aliases=aliases)
                                for a in out["anchor_labels"]]
    return out


def _ground(cmd: dict[str, Any], by_label: dict[str, list[dict[str, Any]]],
            aliases: dict[str, str] | None = None,
            place_index: dict[str, Any] | None = None) -> list[str]:
    label = _norm_label(cmd.get("target_label", ""), aliases=aliases)
    cands = list(by_label.get(label, []))
    relation = _normalize_relation(cmd.get("relation"), aliases=aliases)
    scored = [(c["id"], _score_candidate(c, relation, by_label, place_index))
              for c in cands]
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
             match_radius_m: float = 1.0,
             aliases: dict[str, str] | None = None) -> dict[str, Any]:
    place_index = _build_place_index(pred)
    object_to_place = _assign_objects_to_places(pred, place_index)
    by_label = _objects_by_label(pred, aliases=aliases,
                                 object_to_place=object_to_place)
    pred_xy_by_id: dict[str, tuple[float, float]] = {}
    for cands in by_label.values():
        for c in cands:
            pred_xy_by_id[c["id"]] = c["xy"]
    gt_nodes = (gt or {}).get("semantic_graph", {}).get("nodes", []) if gt else []

    rows: list[dict[str, Any]] = []
    subset_totals: dict[str, dict[str, int]] = {}

    for cmd in commands:
        ranked = _ground(cmd, by_label, aliases=aliases,
                         place_index=place_index)
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
    ap.add_argument("--label-aliases", type=Path,
                    default=Path("configs/eval/label_aliases.json"),
                    help="Optional canonical→[aliases] JSON merged with the "
                         "built-in alias table. Pass an empty path to "
                         "disable file aliases.")
    args = ap.parse_args()

    pred = json.loads(args.prediction.read_text())
    gt = json.loads(args.ground_truth.read_text()) if args.ground_truth else None
    spec = yaml.safe_load(args.commands.read_text())
    commands = spec.get("commands", [])
    alias_path = args.label_aliases if args.label_aliases and str(args.label_aliases) else None
    aliases = _load_alias_map(alias_path)

    result = evaluate(pred, commands, gt=gt, match_radius_m=args.match_radius,
                      aliases=aliases)
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
