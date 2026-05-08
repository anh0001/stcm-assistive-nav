#!/usr/bin/env python3
"""Benchmark predicted STCM object nodes against a ground-truth STCM graph."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "stcm"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stcm.core.munkres import Munkres


DEFAULT_PREDICTION = REPO_ROOT / "output" / "stcm.json"
DEFAULT_GROUND_TRUTH = REPO_ROOT / "configs" / "experiments" / "ground_truth" / "meeting_stcm_gt.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "output" / "stcm_benchmark.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "output" / "stcm_benchmark.csv"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _semantic_nodes(data: dict[str, Any]) -> list[dict[str, Any]]:
    if "semantic_graph" in data:
        semantic = data.get("semantic_graph") or {}
    else:
        semantic = data
    return list(semantic.get("nodes", []) or [])


def _node_id(node: dict[str, Any], index: int) -> str:
    for key in ("id", "instance_id", "name", "object_id"):
        value = node.get(key)
        if value is not None:
            return str(value)
    return f"node_{index}"


def _node_label(node: dict[str, Any]) -> str:
    for key in ("category", "label", "class"):
        value = node.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _node_pose(node: dict[str, Any]) -> list[float] | None:
    pose = node.get("pose")
    if not isinstance(pose, list) or len(pose) < 2:
        return None
    try:
        z_val = float(pose[2]) if len(pose) >= 3 else 0.0
        return [float(pose[0]), float(pose[1]), z_val]
    except (TypeError, ValueError):
        return None


def _normalized_nodes(nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = []
    invalid = []
    for index, node in enumerate(nodes):
        pose = _node_pose(node)
        record = {
            "id": _node_id(node, index),
            "label": _node_label(node),
            "pose": pose,
        }
        if pose is None:
            invalid.append(record)
        else:
            valid.append(record)
    return valid, invalid


def _xy_distance(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _xyz_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2.0 * precision * recall / (precision + recall)


def _distance_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean_m": None,
            "median_m": None,
            "rmse_m": None,
            "p95_m": None,
            "max_m": None,
        }
    sorted_values = sorted(float(value) for value in values)
    count = len(sorted_values)
    mid = count // 2
    if count % 2:
        median = sorted_values[mid]
    else:
        median = (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    p95_index = min(count - 1, math.ceil(0.95 * count) - 1)
    return {
        "mean_m": sum(sorted_values) / count,
        "median_m": median,
        "rmse_m": math.sqrt(sum(value * value for value in sorted_values) / count),
        "p95_m": sorted_values[p95_index],
        "max_m": sorted_values[-1],
    }


def _match_label(
    label: str,
    gt_nodes: list[dict[str, Any]],
    pred_nodes: list[dict[str, Any]],
    threshold_m: float,
) -> dict[str, Any]:
    if not gt_nodes and not pred_nodes:
        return {
            "label": label,
            "gt_count": 0,
            "pred_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "matched_pairs": [],
            "false_positive_nodes": [],
            "false_negative_gt_nodes": [],
            "xy_error": _distance_summary([]),
            "xyz_error": _distance_summary([]),
            "count_error": 0,
        }

    matches = []
    if gt_nodes and pred_nodes:
        cost_matrix = np.array([
            [_xy_distance(gt_node["pose"], pred_node["pose"]) for pred_node in pred_nodes]
            for gt_node in gt_nodes
        ], dtype=float)
        for gt_index, pred_index in Munkres().compute(cost_matrix):
            if gt_index >= len(gt_nodes) or pred_index >= len(pred_nodes):
                continue
            xy_error = cost_matrix[gt_index][pred_index]
            if xy_error <= threshold_m:
                gt_node = gt_nodes[gt_index]
                pred_node = pred_nodes[pred_index]
                matches.append(
                    {
                        "label": label,
                        "gt_id": gt_node["id"],
                        "pred_id": pred_node["id"],
                        "gt_pose": gt_node["pose"],
                        "pred_pose": pred_node["pose"],
                        "xy_error_m": xy_error,
                        "xyz_error_m": _xyz_distance(gt_node["pose"], pred_node["pose"]),
                    }
                )

    matched_gt_ids = {match["gt_id"] for match in matches}
    matched_pred_ids = {match["pred_id"] for match in matches}
    false_negative_gt_nodes = [
        {"id": node["id"], "label": node["label"], "pose": node["pose"]}
        for node in gt_nodes
        if node["id"] not in matched_gt_ids
    ]
    false_positive_nodes = [
        {"id": node["id"], "label": node["label"], "pose": node["pose"]}
        for node in pred_nodes
        if node["id"] not in matched_pred_ids
    ]

    tp = len(matches)
    fp = len(false_positive_nodes)
    fn = len(false_negative_gt_nodes)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return {
        "label": label,
        "gt_count": len(gt_nodes),
        "pred_count": len(pred_nodes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "matched_pairs": matches,
        "false_positive_nodes": false_positive_nodes,
        "false_negative_gt_nodes": false_negative_gt_nodes,
        "xy_error": _distance_summary([match["xy_error_m"] for match in matches]),
        "xyz_error": _distance_summary([match["xyz_error_m"] for match in matches]),
        "count_error": len(pred_nodes) - len(gt_nodes),
    }


def _group_by_label(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        grouped.setdefault(node["label"], []).append(node)
    return grouped


def _label_key(label: str) -> str:
    return str(label).strip().lower()


def _normalize_label_aliases(label_aliases: dict[str, Any] | None) -> dict[str, set[str]]:
    normalized: dict[str, set[str]] = {}
    for gt_label, aliases in (label_aliases or {}).items():
        allowed = {_label_key(gt_label)}
        if isinstance(aliases, str):
            allowed.add(_label_key(aliases))
        else:
            allowed.update(_label_key(alias) for alias in aliases or [])
        normalized[_label_key(gt_label)] = {label for label in allowed if label}
    return normalized


def _is_acceptable_label_pair(
    gt_label: str,
    pred_label: str,
    label_aliases: dict[str, set[str]] | None,
) -> bool:
    gt_key = _label_key(gt_label)
    pred_key = _label_key(pred_label)
    if gt_key == pred_key:
        return True
    if not label_aliases:
        return False
    return pred_key in label_aliases.get(gt_key, set())


def _resolve_threshold(label: str, threshold: Any, default: float = 1.0) -> float:
    """Resolve match threshold for a GT label.

    `threshold` can be either a single float (uniform threshold) or a dict
    mapping label → meters (per-class threshold). Dict supports `_default`
    key as fallback for unlisted labels.
    """
    if isinstance(threshold, dict):
        if label in threshold:
            return float(threshold[label])
        return float(threshold.get("_default", default))
    return float(threshold)


def _max_threshold(threshold: Any, default: float = 1.0) -> float:
    if isinstance(threshold, dict):
        if not threshold:
            return float(default)
        return float(max(threshold.values()))
    return float(threshold)


def _match_with_label_aliases(
    *,
    gt_nodes: list[dict[str, Any]],
    pred_nodes: list[dict[str, Any]],
    threshold_m: Any,
    label_aliases: dict[str, set[str]],
) -> dict[str, Any]:
    labels = sorted({node["label"] for node in gt_nodes} | {node["label"] for node in pred_nodes})
    matches = []
    if gt_nodes and pred_nodes:
        reject_cost = float(_max_threshold(threshold_m)) + 1_000_000.0
        cost_matrix = np.full((len(gt_nodes), len(pred_nodes)), reject_cost, dtype=float)
        acceptable = np.zeros((len(gt_nodes), len(pred_nodes)), dtype=bool)
        for gt_index, gt_node in enumerate(gt_nodes):
            for pred_index, pred_node in enumerate(pred_nodes):
                if not _is_acceptable_label_pair(gt_node["label"], pred_node["label"], label_aliases):
                    continue
                acceptable[gt_index, pred_index] = True
                cost_matrix[gt_index, pred_index] = _xy_distance(gt_node["pose"], pred_node["pose"])

        for gt_index, pred_index in Munkres().compute(cost_matrix):
            if gt_index >= len(gt_nodes) or pred_index >= len(pred_nodes):
                continue
            xy_error = cost_matrix[gt_index][pred_index]
            gt_label_for_thresh = gt_nodes[gt_index]["label"]
            per_pair_threshold = _resolve_threshold(gt_label_for_thresh, threshold_m)
            if acceptable[gt_index, pred_index] and xy_error <= per_pair_threshold:
                gt_node = gt_nodes[gt_index]
                pred_node = pred_nodes[pred_index]
                matches.append(
                    {
                        "label": gt_node["label"],
                        "gt_id": gt_node["id"],
                        "pred_id": pred_node["id"],
                        "gt_label": gt_node["label"],
                        "pred_label": pred_node["label"],
                        "gt_pose": gt_node["pose"],
                        "pred_pose": pred_node["pose"],
                        "xy_error_m": xy_error,
                        "xyz_error_m": _xyz_distance(gt_node["pose"], pred_node["pose"]),
                    }
                )

    matched_gt_ids = {match["gt_id"] for match in matches}
    matched_pred_ids = {match["pred_id"] for match in matches}
    false_negative_gt_nodes = [
        {"id": node["id"], "label": node["label"], "pose": node["pose"]}
        for node in gt_nodes
        if node["id"] not in matched_gt_ids
    ]
    false_positive_nodes = [
        {"id": node["id"], "label": node["label"], "pose": node["pose"]}
        for node in pred_nodes
        if node["id"] not in matched_pred_ids
    ]

    per_label = {}
    for label in labels:
        label_gt_nodes = [node for node in gt_nodes if node["label"] == label]
        label_pred_nodes = [node for node in pred_nodes if node["label"] == label]
        label_matches = [match for match in matches if match["gt_label"] == label]
        label_false_negatives = [
            {"id": node["id"], "label": node["label"], "pose": node["pose"]}
            for node in label_gt_nodes
            if node["id"] not in matched_gt_ids
        ]
        label_false_positives = [
            {"id": node["id"], "label": node["label"], "pose": node["pose"]}
            for node in label_pred_nodes
            if node["id"] not in matched_pred_ids
        ]
        tp = len(label_matches)
        fp = len(label_false_positives)
        fn = len(label_false_negatives)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        per_label[label] = {
            "label": label,
            "gt_count": len(label_gt_nodes),
            "pred_count": len(label_pred_nodes),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "matched_pairs": label_matches,
            "false_positive_nodes": label_false_positives,
            "false_negative_gt_nodes": label_false_negatives,
            "xy_error": _distance_summary([match["xy_error_m"] for match in label_matches]),
            "xyz_error": _distance_summary([match["xyz_error_m"] for match in label_matches]),
            "count_error": len(label_pred_nodes) - len(label_gt_nodes),
        }

    return {
        "per_label": per_label,
        "matched_pairs": matches,
        "false_positive_nodes": false_positive_nodes,
        "false_negative_gt_nodes": false_negative_gt_nodes,
    }


def _composite_cover_false_positives(
    *,
    gt_nodes: list[dict[str, Any]],
    false_positive_nodes: list[dict[str, Any]],
    composite_covers: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not composite_covers:
        return []

    covered = []
    covered_ids = set()
    gt_by_label = _group_by_label(gt_nodes)
    for gt_label, rule in composite_covers.items():
        if not isinstance(rule, dict):
            continue
        try:
            radius_m = float(rule.get("radius_m", 0.0))
        except (TypeError, ValueError):
            continue
        if radius_m <= 0.0:
            continue
        allowed_labels = {
            _label_key(label)
            for label in (
                rule.get("covered_pred_labels")
                or rule.get("pred_labels")
                or rule.get("labels")
                or []
            )
        }
        if not allowed_labels:
            allowed_labels = {_label_key(gt_label)}
        for gt_node in gt_by_label.get(str(gt_label), []):
            for pred_node in false_positive_nodes:
                pred_id = pred_node["id"]
                if pred_id in covered_ids or _label_key(pred_node["label"]) not in allowed_labels:
                    continue
                distance = _xy_distance(gt_node["pose"], pred_node["pose"])
                if distance <= radius_m:
                    covered_ids.add(pred_id)
                    covered.append(
                        {
                            **pred_node,
                            "covered_by_gt_id": gt_node["id"],
                            "covered_by_gt_label": gt_node["label"],
                            "xy_error_m": distance,
                        }
                    )
    return covered


def _duplicate_pairs(nodes: list[dict[str, Any]], threshold_m: float) -> list[dict[str, Any]]:
    pairs = []
    grouped = _group_by_label(nodes)
    for label, entries in sorted(grouped.items()):
        for index, node_a in enumerate(entries):
            for node_b in entries[index + 1 :]:
                distance = _xy_distance(node_a["pose"], node_b["pose"])
                if distance <= threshold_m:
                    pairs.append(
                        {
                            "label": label,
                            "node_a": node_a["id"],
                            "node_b": node_b["id"],
                            "xy_distance_m": distance,
                        }
                    )
    return pairs


def _nearest_wrong_label(
    gt_nodes: list[dict[str, Any]],
    pred_nodes: list[dict[str, Any]],
    threshold_m: float,
    label_aliases: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    near = []
    for gt_node in gt_nodes:
        best = None
        for pred_node in pred_nodes:
            if _is_acceptable_label_pair(gt_node["label"], pred_node["label"], label_aliases):
                continue
            distance = _xy_distance(gt_node["pose"], pred_node["pose"])
            if distance <= threshold_m and (best is None or distance < best["xy_error_m"]):
                best = {
                    "gt_id": gt_node["id"],
                    "gt_label": gt_node["label"],
                    "pred_id": pred_node["id"],
                    "pred_label": pred_node["label"],
                    "xy_error_m": distance,
                }
        if best is not None:
            near.append(best)
    return near


def evaluate_graphs(
    *,
    prediction_path: Path,
    ground_truth_path: Path,
    match_threshold_m: Any = 1.0,
    duplicate_threshold_m: float = 0.5,
    wrong_label_threshold_m: float = 1.0,
    label_aliases: dict[str, Any] | None = None,
    composite_covers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """`match_threshold_m` may be float (uniform) or dict[label,float] with
    optional `_default` key (per-class). Dict accommodates large objects
    where GT is annotated on outer surface but predictions land at mask
    centroid (e.g. trailer wall vs. trailer center)."""
    prediction_path = Path(prediction_path).expanduser()
    ground_truth_path = Path(ground_truth_path).expanduser()
    pred_raw, pred_invalid = _normalized_nodes(_semantic_nodes(_load_json(prediction_path)))
    gt_raw, gt_invalid = _normalized_nodes(_semantic_nodes(_load_json(ground_truth_path)))
    pred_by_label = _group_by_label(pred_raw)
    gt_by_label = _group_by_label(gt_raw)
    labels = sorted(set(pred_by_label) | set(gt_by_label))
    normalized_aliases = _normalize_label_aliases(label_aliases)
    if normalized_aliases:
        match_result = _match_with_label_aliases(
            gt_nodes=gt_raw,
            pred_nodes=pred_raw,
            threshold_m=match_threshold_m,
            label_aliases=normalized_aliases,
        )
        per_label = match_result["per_label"]
        matched_pairs = match_result["matched_pairs"]
        false_positive_nodes = match_result["false_positive_nodes"]
        false_negative_gt_nodes = match_result["false_negative_gt_nodes"]
    else:
        per_label = {
            label: _match_label(
                label,
                gt_by_label.get(label, []),
                pred_by_label.get(label, []),
                _resolve_threshold(label, match_threshold_m),
            )
            for label in labels
        }
        matched_pairs = [
            match for item in per_label.values() for match in item["matched_pairs"]
        ]
        false_positive_nodes = [
            node for item in per_label.values() for node in item["false_positive_nodes"]
        ]
        false_negative_gt_nodes = [
            node for item in per_label.values() for node in item["false_negative_gt_nodes"]
        ]

    covered_false_positive_nodes = _composite_cover_false_positives(
        gt_nodes=gt_raw,
        false_positive_nodes=false_positive_nodes,
        composite_covers=composite_covers,
    )
    covered_false_positive_ids = {node["id"] for node in covered_false_positive_nodes}
    if covered_false_positive_ids:
        false_positive_nodes = [
            node for node in false_positive_nodes if node["id"] not in covered_false_positive_ids
        ]
        for label, payload in per_label.items():
            label_false_positives = [
                node
                for node in payload["false_positive_nodes"]
                if node["id"] not in covered_false_positive_ids
            ]
            payload["false_positive_nodes"] = label_false_positives
            payload["fp"] = len(label_false_positives)
            payload["precision"] = _safe_ratio(payload["tp"], payload["tp"] + payload["fp"])
            payload["f1"] = _f1(payload["precision"], payload["recall"])

    tp = sum(item["tp"] for item in per_label.values())
    fp = len(false_positive_nodes)
    fn = len(false_negative_gt_nodes)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    xy_errors = [
        match["xy_error_m"]
        for item in per_label.values()
        for match in item["matched_pairs"]
    ]
    xyz_errors = [
        match["xyz_error_m"]
        for item in per_label.values()
        for match in item["matched_pairs"]
    ]
    duplicate_pairs = _duplicate_pairs(pred_raw, duplicate_threshold_m)
    if isinstance(match_threshold_m, dict):
        metric_name_thresh_str = "per-class"
    else:
        metric_name_thresh_str = f"{float(match_threshold_m):g}m"
    return {
        "metric_name": f"STCM Object Map F1@{metric_name_thresh_str}",
        "prediction_path": str(prediction_path),
        "ground_truth_path": str(ground_truth_path),
        "match_policy": {
            "label": "category_or_label_exact_match"
            if not normalized_aliases
            else "category_or_label_exact_match_with_configured_gt_aliases",
            "label_aliases": {label: sorted(aliases) for label, aliases in normalized_aliases.items()},
            "composite_covers": composite_covers or {},
            "position": "xy_distance_m",
            "assignment": "one_to_one_hungarian_per_label"
            if not normalized_aliases
            else "one_to_one_hungarian_global_with_label_constraints",
            "match_threshold_m": match_threshold_m,
            "duplicate_threshold_m": duplicate_threshold_m,
            "wrong_label_threshold_m": wrong_label_threshold_m,
        },
        "summary": {
            "gt_nodes": len(gt_raw),
            "pred_nodes": len(pred_raw),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "xy_error": _distance_summary(xy_errors),
            "xyz_error": _distance_summary(xyz_errors),
            "count_error": len(pred_raw) - len(gt_raw),
            "duplicate_pair_count": len(duplicate_pairs),
            "invalid_prediction_nodes": len(pred_invalid),
            "invalid_ground_truth_nodes": len(gt_invalid),
            "covered_false_positive_nodes": len(covered_false_positive_nodes),
        },
        "per_label": per_label,
        "matched_pairs": matched_pairs,
        "false_positive_nodes": false_positive_nodes,
        "covered_false_positive_nodes": covered_false_positive_nodes,
        "false_negative_gt_nodes": false_negative_gt_nodes,
        "wrong_label_near_gt": _nearest_wrong_label(
            gt_raw,
            pred_raw,
            wrong_label_threshold_m,
            normalized_aliases,
        ),
        "duplicate_pairs": duplicate_pairs,
        "invalid_prediction_nodes": pred_invalid,
        "invalid_ground_truth_nodes": gt_invalid,
    }


def _row(scope: str, label: str, payload: dict[str, Any]) -> dict[str, Any]:
    xy = payload.get("xy_error") or {}
    xyz = payload.get("xyz_error") or {}
    return {
        "scope": scope,
        "label": label,
        "gt_nodes": payload.get("gt_count", payload.get("gt_nodes")),
        "pred_nodes": payload.get("pred_count", payload.get("pred_nodes")),
        "tp": payload.get("tp"),
        "fp": payload.get("fp"),
        "fn": payload.get("fn"),
        "precision": payload.get("precision"),
        "recall": payload.get("recall"),
        "f1": payload.get("f1"),
        "xy_error_mean_m": xy.get("mean_m"),
        "xy_error_median_m": xy.get("median_m"),
        "xy_error_rmse_m": xy.get("rmse_m"),
        "xy_error_p95_m": xy.get("p95_m"),
        "xy_error_max_m": xy.get("max_m"),
        "xyz_error_mean_m": xyz.get("mean_m"),
        "xyz_error_rmse_m": xyz.get("rmse_m"),
        "count_error": payload.get("count_error"),
    }


def write_csv(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope",
        "label",
        "gt_nodes",
        "pred_nodes",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "xy_error_mean_m",
        "xy_error_median_m",
        "xy_error_rmse_m",
        "xy_error_p95_m",
        "xy_error_max_m",
        "xyz_error_mean_m",
        "xyz_error_rmse_m",
        "count_error",
    ]
    rows = [_row("overall", "__overall__", result["summary"])]
    rows.extend(_row("label", label, payload) for label, payload in result["per_label"].items())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(text: str) -> Path:
    path = Path(text).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", default=str(DEFAULT_PREDICTION))
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH))
    parser.add_argument("--match-threshold-m", type=float, default=1.0)
    parser.add_argument("--duplicate-threshold-m", type=float, default=0.5)
    parser.add_argument("--wrong-label-threshold-m", type=float, default=1.0)
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument(
        "--label-aliases",
        default=None,
        help="Optional JSON file mapping GT label -> list of acceptable predicted labels.",
    )
    return parser.parse_args()


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> int:
    args = _parse_args()
    aliases = None
    if args.label_aliases:
        aliases = json.loads(_resolve_path(args.label_aliases).read_text())
    result = evaluate_graphs(
        prediction_path=_resolve_path(args.prediction),
        ground_truth_path=_resolve_path(args.ground_truth),
        match_threshold_m=args.match_threshold_m,
        duplicate_threshold_m=args.duplicate_threshold_m,
        wrong_label_threshold_m=args.wrong_label_threshold_m,
        label_aliases=aliases,
    )
    output_json = _resolve_path(args.output_json)
    output_csv = _resolve_path(args.output_csv)
    _write_json(output_json, result)
    write_csv(output_csv, result)
    summary = result["summary"]
    print(
        f"{result['metric_name']}: "
        f"TP={summary['tp']} FP={summary['fp']} FN={summary['fn']} "
        f"precision={_format_metric(summary['precision'])} "
        f"recall={_format_metric(summary['recall'])} "
        f"f1={_format_metric(summary['f1'])}"
    )
    print(output_json.relative_to(REPO_ROOT))
    print(output_csv.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
