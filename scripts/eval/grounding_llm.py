#!/usr/bin/env python3
"""Claude-offline LLM grounding backend for STCM (AE-3 LLM ablation).

The reviewer requires a paired comparison between the no-LLM template
grounder (`scripts/eval/grounding.py`) and an LLM-based grounder on the
SAME predicted graph and SAME command set. Reproducibility requires
fixed temperature, archived prompts, and archived responses.

Workflow (no online API key required):

  Phase 1 -- request:
    python3 scripts/eval/grounding_llm.py --phase request \
        --prediction output/stcm.json \
        --commands configs/eval/commands_meeting.yaml \
        --output output/grounding_llm/meeting_request.json

  Phase 2 -- have Claude (Sonnet 4.6, temperature=0) read the request file
  and produce a response file. The Claude session writes the response file
  itself; no API key is involved. The expected response schema is in
  `_RESPONSE_SCHEMA_DOC` below and is also embedded in the request file.

  Phase 3 -- score:
    python3 scripts/eval/grounding_llm.py --phase score \
        --prediction output/stcm.json \
        --commands configs/eval/commands_meeting.yaml \
        --responses output/grounding_llm/meeting_response.json \
        --output output/grounding_llm/meeting_grounding.json

The score-phase output schema is byte-compatible with `scripts/eval/grounding.py`
so that `scripts/eval/render_tables.py` can consume either.

Pairing contract: same prediction graph, same command file, same alias map.
The only difference between LLM and no-LLM runs is which ranking comes from
the grounder; everything else is held constant.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from grounding import (  # noqa: E402  -- local sibling import
    _anchors_present,
    _assign_objects_to_places,
    _build_place_index,
    _classify_failure,
    _gold_referent_present,
    _gt_id_match,
    _load_alias_map,
    _norm_label,
    _normalize_relation,
    _objects_by_label,
    _resolve_gt_pose,
    _xy,
)


_RESPONSE_SCHEMA_DOC = """\
Response file schema (the LLM must produce this exact structure):

{
  "backend": "claude-sonnet-4-6",
  "temperature": 0,
  "scene": "<same as request.scene>",
  "rankings": [
    {
      "command_id": "m_simple_01",
      "ranked_object_ids": ["chair_1_0", "chair_2_0", ...]
    },
    ...
  ]
}

Rules:
- One entry per command in the request.
- ranked_object_ids contains predicted object ids drawn ONLY from
  request.candidates[command_id]. No invented ids.
- Order is most-likely first; the scorer reads top-1 and top-2.
- An empty list is allowed when no candidate is plausible.
- temperature must be 0 for the reviewer-mandated determinism check.
"""


def _candidates_for_command(cmd: dict[str, Any],
                            by_label: dict[str, list[dict[str, Any]]],
                            aliases: dict[str, str]
                            ) -> list[dict[str, Any]]:
    """Return the candidate object pool the LLM must rank for one command.

    Two modes:
      - target_label set: pre-filter to objects whose normalized label matches.
        Mirrors the template grounder so LLM and template see the same pool;
        LLM's job is to RANK.
      - target_label null/empty (functional / intent commands): expose ALL
        objects across labels. The LLM must perform retrieval AND ranking,
        i.e. map intent text to a label, then to an instance. The template
        grounder cannot do this and returns [] -- exactly the no-LLM baseline
        behavior the AE-3 ablation wants."""
    raw = cmd.get("target_label")
    label = _norm_label(raw, aliases=aliases) if raw else ""
    pool: list[dict[str, Any]] = []
    if not label:
        for cands in by_label.values():
            for c in cands:
                pool.append({
                    "id": c["id"],
                    "label": c["label"],
                    "xy": [c["xy"][0], c["xy"][1]],
                    "place": c.get("place"),
                })
        return pool
    for c in by_label.get(label, []):
        pool.append({
            "id": c["id"],
            "label": c["label"],
            "xy": [c["xy"][0], c["xy"][1]],
            "place": c.get("place"),
        })
    return pool


def _anchors_for_command(cmd: dict[str, Any],
                         by_label: dict[str, list[dict[str, Any]]],
                         aliases: dict[str, str]
                         ) -> dict[str, list[dict[str, Any]]]:
    relation = _normalize_relation(cmd.get("relation"), aliases=aliases) or {}
    anchor_pool: dict[str, list[dict[str, Any]]] = {}
    labels: list[str] = []
    if "anchor_label" in relation:
        labels.append(relation["anchor_label"])
    if "anchor_labels" in relation:
        labels.extend(relation["anchor_labels"])
    for lab in labels:
        if lab in anchor_pool:
            continue
        anchor_pool[lab] = [
            {"id": a["id"], "xy": [a["xy"][0], a["xy"][1]], "place": a.get("place")}
            for a in by_label.get(lab, [])
        ]
    return anchor_pool


def _emit_request(pred: dict[str, Any], commands: list[dict[str, Any]],
                  scene: str | None,
                  aliases: dict[str, str]) -> dict[str, Any]:
    place_index = _build_place_index(pred)
    object_to_place = _assign_objects_to_places(pred, place_index)
    by_label = _objects_by_label(pred, aliases=aliases,
                                 object_to_place=object_to_place)

    items: list[dict[str, Any]] = []
    for cmd in commands:
        raw_label = cmd.get("target_label")
        label_norm = _norm_label(raw_label, aliases=aliases) if raw_label else None
        items.append({
            "command_id": cmd.get("id"),
            "subset": cmd.get("subset"),
            "text": cmd.get("text"),
            "target_label": label_norm,
            "relation": _normalize_relation(cmd.get("relation"),
                                            aliases=aliases),
            "candidates": _candidates_for_command(cmd, by_label, aliases),
            "anchors": _anchors_for_command(cmd, by_label, aliases),
        })

    place_summary = []
    for pid, xy in place_index["places"].items():
        place_summary.append({
            "id": pid,
            "xy": [xy[0], xy[1]],
            "neighbors": sorted(place_index["neighbors"].get(pid, set())),
        })

    return {
        "schema_version": 1,
        "backend_required": "claude-sonnet-4-6",
        "temperature_required": 0,
        "scene": scene,
        "instructions": (
            "You are a spatial-language grounder for an indoor robot. For each "
            "command, pick the candidate object id that best satisfies the "
            "command. Use only ids drawn from candidates[command_id]. Use "
            "relation semantics (near, between) and the place-graph topology "
            "to disambiguate. Output the schema in `response_schema`."
        ),
        "response_schema": _RESPONSE_SCHEMA_DOC,
        "place_graph": place_summary,
        "commands": items,
    }


def _evaluate(pred: dict[str, Any], commands: list[dict[str, Any]],
              rankings_by_id: dict[str, list[str]],
              gt: dict[str, Any] | None,
              aliases: dict[str, str],
              match_radius_m: float) -> dict[str, Any]:
    place_index = _build_place_index(pred)
    object_to_place = _assign_objects_to_places(pred, place_index)
    by_label = _objects_by_label(pred, aliases=aliases,
                                 object_to_place=object_to_place)
    pred_xy_by_id: dict[str, tuple[float, float]] = {}
    valid_ids: set[str] = set()
    for cands in by_label.values():
        for c in cands:
            pred_xy_by_id[c["id"]] = c["xy"]
            valid_ids.add(c["id"])
    gt_nodes = (gt or {}).get("semantic_graph", {}).get("nodes", []) if gt else []

    rows: list[dict[str, Any]] = []
    subset_totals: dict[str, dict[str, int]] = {}

    for cmd in commands:
        cid = cmd.get("id")
        ranked = list(rankings_by_id.get(str(cid), []))
        gt_id = str(cmd.get("gt_object_id", ""))
        gt_xy = _resolve_gt_pose(gt_id, gt_nodes)
        top1 = bool(ranked) and _gt_id_match(
            ranked[0], gt_id, pred_xy_by_id.get(ranked[0]), gt_xy,
            match_radius_m,
        )
        top2 = any(
            _gt_id_match(rid, gt_id, pred_xy_by_id.get(rid), gt_xy,
                         match_radius_m)
            for rid in ranked[:2]
        )
        subset = str(cmd.get("subset", "unknown"))

        rel = _normalize_relation(cmd.get("relation"), aliases=aliases)
        gold_present = _gold_referent_present(gt_xy, pred_xy_by_id, match_radius_m)
        anchors_present = _anchors_present(rel, by_label)
        eligible = gold_present and anchors_present
        ranker_picked_invalid = bool(ranked) and (ranked[0] not in valid_ids)
        failure_source = _classify_failure(
            top1=top1,
            gold_present=gold_present,
            anchors_present=anchors_present,
            candidates_pool_empty=(len(ranked) == 0),
            ranker_picked_invalid=ranker_picked_invalid,
        )

        rows.append({
            "id": cid,
            "subset": subset,
            "text": cmd.get("text"),
            "gt_object_id": gt_id,
            "ranked": ranked[:5],
            "top1": top1,
            "top2": top2,
            "n_candidates": len(ranked),
            "gold_referent_present": gold_present,
            "anchor_nodes_present": anchors_present,
            "eligible_grounding_given_perception": eligible,
            "failure_source": failure_source,
        })
        st = subset_totals.setdefault(
            subset,
            {"n": 0, "top1": 0, "top2": 0,
             "n_eligible": 0, "top1_eligible": 0},
        )
        st["n"] += 1
        st["top1"] += int(top1)
        st["top2"] += int(top2)
        st["n_eligible"] += int(eligible)
        st["top1_eligible"] += int(eligible and top1)

    overall = {
        "n": len(rows),
        "top1": sum(1 for r in rows if r["top1"]),
        "top2": sum(1 for r in rows if r["top2"]),
        "n_eligible": sum(1 for r in rows if r["eligible_grounding_given_perception"]),
        "top1_eligible": sum(1 for r in rows
                             if r["eligible_grounding_given_perception"]
                             and r["top1"]),
    }
    overall["top1_acc"] = overall["top1"] / overall["n"] if overall["n"] else 0.0
    overall["top2_acc"] = overall["top2"] / overall["n"] if overall["n"] else 0.0
    overall["top1_acc_given_perception"] = (
        overall["top1_eligible"] / overall["n_eligible"]
        if overall["n_eligible"] else 0.0
    )

    failure_counts: dict[str, int] = {}
    for r in rows:
        failure_counts[r["failure_source"]] = failure_counts.get(r["failure_source"], 0) + 1
    overall["failure_counts"] = failure_counts

    for st in subset_totals.values():
        st["top1_acc"] = st["top1"] / st["n"] if st["n"] else 0.0
        st["top2_acc"] = st["top2"] / st["n"] if st["n"] else 0.0
        st["top1_acc_given_perception"] = (
            st["top1_eligible"] / st["n_eligible"]
            if st["n_eligible"] else 0.0
        )

    return {"overall": overall, "by_subset": subset_totals, "trials": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=["request", "score"])
    ap.add_argument("--prediction", required=True, type=Path)
    ap.add_argument("--commands", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--responses", type=Path,
                    help="Score phase only: path to LLM response JSON.")
    ap.add_argument("--ground-truth", type=Path)
    ap.add_argument("--match-radius", type=float, default=1.0)
    ap.add_argument("--label-aliases", type=Path,
                    default=Path("configs/eval/label_aliases.json"))
    ap.add_argument("--backend", default="claude-sonnet-4-6",
                    help="Recorded in score-phase output for provenance.")
    args = ap.parse_args()

    pred = json.loads(args.prediction.read_text())
    spec = yaml.safe_load(args.commands.read_text())
    commands = spec.get("commands", [])
    alias_path = args.label_aliases if args.label_aliases and str(args.label_aliases) else None
    aliases = _load_alias_map(alias_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.phase == "request":
        bundle = _emit_request(pred, commands, spec.get("scene"), aliases)
        args.output.write_text(json.dumps(bundle, indent=2) + "\n")
        print(f"wrote request bundle: {args.output} "
              f"({len(bundle['commands'])} commands)")
        return

    if args.responses is None:
        ap.error("--responses is required for --phase score")
    response = json.loads(args.responses.read_text())
    if response.get("temperature", None) != 0:
        raise SystemExit(
            "LLM response temperature must be 0 for reviewer reproducibility; "
            f"got {response.get('temperature')!r}"
        )
    rankings_by_id: dict[str, list[str]] = {}
    for entry in response.get("rankings", []):
        cid = str(entry.get("command_id"))
        rankings_by_id[cid] = [str(x) for x in entry.get("ranked_object_ids", [])]

    gt = json.loads(args.ground_truth.read_text()) if args.ground_truth else None
    result = _evaluate(pred, commands, rankings_by_id, gt, aliases,
                       args.match_radius)
    result["scene"] = spec.get("scene")
    result["commands_path"] = str(args.commands)
    result["prediction_path"] = str(args.prediction)
    result["responses_path"] = str(args.responses)
    result["backend"] = args.backend
    result["temperature"] = response.get("temperature")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")
    ov = result["overall"]
    print(f"  overall top1={ov['top1_acc']:.3f}  "
          f"top2={ov['top2_acc']:.3f}  "
          f"top1|perception={ov['top1_acc_given_perception']:.3f} "
          f"(n_eligible={ov['n_eligible']}/{ov['n']})")
    print(f"  failure_counts={ov['failure_counts']}")
    for subset, st in result["by_subset"].items():
        print(f"  {subset}: top1={st['top1_acc']:.3f} "
              f"top2={st['top2_acc']:.3f} "
              f"top1|perc={st['top1_acc_given_perception']:.3f} "
              f"n={st['n']} n_elig={st['n_eligible']}")


if __name__ == "__main__":
    main()
