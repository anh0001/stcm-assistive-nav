#!/usr/bin/env python3
"""Filter semantic graph nodes using category or spatial rules.

Rules JSON example:
{
  "remove_categories": ["cardboard box"],
  "remove_ids": ["chair_91_0"],
  "remove_near": [
    {"category": "chair", "xy": [7.9, 3.2], "radius": 0.5}
  ],
  "remove_inside_bbox": [
    {"category": "chair", "xmin": 7.5, "xmax": 8.5, "ymin": 2.5, "ymax": 3.7}
  ],
  "remove_closest": [
    {"xy": [7.9, 3.2], "category": "chair", "max_distance": 0.8}
  ],
  "keep_inside_bbox": {"xmin": 0, "xmax": 12, "ymin": -2, "ymax": 8}
}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_PATH = REPO_ROOT / "output" / "semantic_graph.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "output" / "semantic_graph.filtered.json"


@dataclass
class RemoveNearRule:
    xy: Tuple[float, float]
    radius: float
    category: Optional[str] = None

    def matches(self, category: str, pose: Sequence[float]) -> bool:
        if len(pose) < 2:
            return False
        if self.category is not None and self.category != category:
            return False
        dx = float(pose[0]) - self.xy[0]
        dy = float(pose[1]) - self.xy[1]
        return math.hypot(dx, dy) <= self.radius


@dataclass
class BBoxRule:
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    ymin: Optional[float] = None
    ymax: Optional[float] = None
    zmin: Optional[float] = None
    zmax: Optional[float] = None

    def outside(self, pose: Sequence[float]) -> bool:
        if len(pose) < 2:
            return False
        x = float(pose[0])
        y = float(pose[1])
        z = float(pose[2]) if len(pose) > 2 else None
        if self.xmin is not None and x < self.xmin:
            return True
        if self.xmax is not None and x > self.xmax:
            return True
        if self.ymin is not None and y < self.ymin:
            return True
        if self.ymax is not None and y > self.ymax:
            return True
        if z is not None and self.zmin is not None and z < self.zmin:
            return True
        if z is not None and self.zmax is not None and z > self.zmax:
            return True
        return False

    def inside(self, pose: Sequence[float]) -> bool:
        if len(pose) < 2:
            return False
        x = float(pose[0])
        y = float(pose[1])
        z = float(pose[2]) if len(pose) > 2 else None
        if self.xmin is not None and x < self.xmin:
            return False
        if self.xmax is not None and x > self.xmax:
            return False
        if self.ymin is not None and y < self.ymin:
            return False
        if self.ymax is not None and y > self.ymax:
            return False
        if self.zmin is not None and z is None:
            return False
        if self.zmax is not None and z is None:
            return False
        if z is not None and self.zmin is not None and z < self.zmin:
            return False
        if z is not None and self.zmax is not None and z > self.zmax:
            return False
        return True


@dataclass
class RemoveInsideRule:
    bbox: BBoxRule
    category: Optional[str] = None

    def matches(self, category: str, pose: Sequence[float]) -> bool:
        if self.category is not None and self.category != category:
            return False
        return self.bbox.inside(pose)


@dataclass
class RemoveClosestRule:
    xy: Tuple[float, float]
    category: Optional[str] = None
    max_distance: Optional[float] = None

    def matches_category(self, category: str) -> bool:
        return self.category is None or self.category == category


@dataclass
class FilterRules:
    remove_categories: set[str] = field(default_factory=set)
    remove_ids: set[str] = field(default_factory=set)
    remove_near: List[RemoveNearRule] = field(default_factory=list)
    remove_inside_bbox: List[RemoveInsideRule] = field(default_factory=list)
    remove_outside_bbox: Optional[BBoxRule] = None
    keep_inside_bbox: Optional[BBoxRule] = None
    remove_closest: List[RemoveClosestRule] = field(default_factory=list)

    def has_any(self) -> bool:
        return bool(
            self.remove_categories
            or self.remove_ids
            or self.remove_near
            or self.remove_inside_bbox
            or self.remove_outside_bbox
            or self.keep_inside_bbox
            or self.remove_closest
        )


def _node_id(node: Dict[str, Any], index: int) -> str:
    for key in ("id", "name"):
        value = node.get(key)
        if value is not None:
            return str(value)
    return f"node_{index}"


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _parse_remove_near(value: str) -> RemoveNearRule:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) == 3:
        category = None
        x_str, y_str, r_str = parts
    elif len(parts) == 4:
        category = parts[0]
        x_str, y_str, r_str = parts[1:]
    else:
        raise ValueError("Expected format 'x,y,radius' or 'category,x,y,radius'.")
    return RemoveNearRule(
        xy=(float(x_str), float(y_str)),
        radius=float(r_str),
        category=category,
    )


def _parse_remove_closest(value: str) -> RemoveClosestRule:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in (2, 3, 4):
        raise ValueError("Expected format 'x,y', 'x,y,max_distance', 'category,x,y', or 'category,x,y,max_distance'.")

    category: Optional[str] = None
    max_distance: Optional[float] = None
    if len(parts) == 2:
        x_str, y_str = parts
    elif len(parts) == 3:
        if _is_float(parts[0]) and _is_float(parts[1]) and _is_float(parts[2]):
            x_str, y_str, max_str = parts
            max_distance = float(max_str)
        else:
            category = parts[0]
            x_str, y_str = parts[1:]
    else:
        category = parts[0]
        x_str, y_str, max_str = parts[1:]
        max_distance = float(max_str)

    return RemoveClosestRule(
        xy=(float(x_str), float(y_str)),
        category=category,
        max_distance=max_distance,
    )


def _parse_bbox_values(values: Sequence[float]) -> BBoxRule:
    if len(values) not in (4, 6):
        raise ValueError("Expected 4 values (xmin,xmax,ymin,ymax) or 6 values (xmin,xmax,ymin,ymax,zmin,zmax).")
    xmin, xmax, ymin, ymax = values[:4]
    zmin, zmax = (values[4], values[5]) if len(values) == 6 else (None, None)
    return BBoxRule(
        xmin=float(xmin),
        xmax=float(xmax),
        ymin=float(ymin),
        ymax=float(ymax),
        zmin=None if zmin is None else float(zmin),
        zmax=None if zmax is None else float(zmax),
    )


def _parse_remove_inside(value: str) -> RemoveInsideRule:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) not in (4, 5, 6, 7):
        raise ValueError(
            "Expected format 'xmin,xmax,ymin,ymax', 'xmin,xmax,ymin,ymax,zmin,zmax', "
            "'category,xmin,xmax,ymin,ymax', or 'category,xmin,xmax,ymin,ymax,zmin,zmax'."
        )

    category: Optional[str] = None
    if _is_float(parts[0]):
        values = [float(value) for value in parts]
    else:
        category = parts[0]
        values = [float(value) for value in parts[1:]]

    return RemoveInsideRule(bbox=_parse_bbox_values(values), category=category)


def _parse_rules_file(path: Path) -> FilterRules:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    rules = FilterRules()

    for category in data.get("remove_categories", []):
        rules.remove_categories.add(str(category))

    for node_id in data.get("remove_ids", []):
        rules.remove_ids.add(str(node_id))

    for item in data.get("remove_near", []):
        if not isinstance(item, dict):
            raise ValueError("remove_near entries must be objects with xy/radius.")
        xy = item.get("xy")
        if xy is None or len(xy) < 2:
            raise ValueError("remove_near entries must include xy with at least 2 values.")
        radius = item.get("radius")
        if radius is None:
            raise ValueError("remove_near entries must include radius.")
        category = item.get("category")
        if isinstance(category, list):
            for cat in category:
                rules.remove_near.append(
                    RemoveNearRule(xy=(float(xy[0]), float(xy[1])), radius=float(radius), category=str(cat))
                )
        else:
            rules.remove_near.append(
                RemoveNearRule(xy=(float(xy[0]), float(xy[1])), radius=float(radius), category=category)
            )

    inside_bboxes = data.get("remove_inside_bbox", [])
    if isinstance(inside_bboxes, dict):
        inside_bboxes = [inside_bboxes]
    for bbox in inside_bboxes:
        if isinstance(bbox, dict):
            rules.remove_inside_bbox.append(
                RemoveInsideRule(
                    bbox=BBoxRule(
                        xmin=bbox.get("xmin"),
                        xmax=bbox.get("xmax"),
                        ymin=bbox.get("ymin"),
                        ymax=bbox.get("ymax"),
                        zmin=bbox.get("zmin"),
                        zmax=bbox.get("zmax"),
                    ),
                    category=bbox.get("category"),
                )
            )
        elif isinstance(bbox, list):
            rules.remove_inside_bbox.append(RemoveInsideRule(bbox=_parse_bbox_values(bbox)))
        elif bbox:
            raise ValueError("remove_inside_bbox must be a dict or list.")

    outside_bbox = data.get("remove_outside_bbox")
    if outside_bbox:
        if isinstance(outside_bbox, dict):
            rules.remove_outside_bbox = BBoxRule(
                xmin=outside_bbox.get("xmin"),
                xmax=outside_bbox.get("xmax"),
                ymin=outside_bbox.get("ymin"),
                ymax=outside_bbox.get("ymax"),
                zmin=outside_bbox.get("zmin"),
                zmax=outside_bbox.get("zmax"),
            )
        elif isinstance(outside_bbox, list):
            rules.remove_outside_bbox = _parse_bbox_values(outside_bbox)
        else:
            raise ValueError("remove_outside_bbox must be a dict or list.")

    keep_bbox = data.get("keep_inside_bbox")
    if keep_bbox:
        if isinstance(keep_bbox, dict):
            rules.keep_inside_bbox = BBoxRule(
                xmin=keep_bbox.get("xmin"),
                xmax=keep_bbox.get("xmax"),
                ymin=keep_bbox.get("ymin"),
                ymax=keep_bbox.get("ymax"),
                zmin=keep_bbox.get("zmin"),
                zmax=keep_bbox.get("zmax"),
            )
        elif isinstance(keep_bbox, list):
            rules.keep_inside_bbox = _parse_bbox_values(keep_bbox)
        else:
            raise ValueError("keep_inside_bbox must be a dict or list.")

    for item in data.get("remove_closest", []):
        if not isinstance(item, dict):
            raise ValueError("remove_closest entries must be objects with xy.")
        xy = item.get("xy")
        if xy is None or len(xy) < 2:
            raise ValueError("remove_closest entries must include xy with at least 2 values.")
        rules.remove_closest.append(
            RemoveClosestRule(
                xy=(float(xy[0]), float(xy[1])),
                category=item.get("category"),
                max_distance=item.get("max_distance"),
            )
        )

    return rules


def _merge_rules(base: FilterRules, extra: FilterRules) -> FilterRules:
    base.remove_categories |= extra.remove_categories
    base.remove_ids |= extra.remove_ids
    base.remove_near.extend(extra.remove_near)
    base.remove_inside_bbox.extend(extra.remove_inside_bbox)
    if extra.remove_outside_bbox:
        base.remove_outside_bbox = extra.remove_outside_bbox
    if extra.keep_inside_bbox:
        base.keep_inside_bbox = extra.keep_inside_bbox
    base.remove_closest.extend(extra.remove_closest)
    return base


def _evaluate_removals(
    nodes: List[Dict[str, Any]],
    rules: FilterRules,
) -> Dict[str, List[str]]:
    removed: Dict[str, List[str]] = {}

    for idx, node in enumerate(nodes):
        node_id = _node_id(node, idx)
        category = str(node.get("category", "unknown"))
        pose = node.get("pose") or []
        reasons: List[str] = []

        if node_id in rules.remove_ids:
            reasons.append("id")
        if category in rules.remove_categories:
            reasons.append("category")

        if rules.remove_outside_bbox and pose:
            if rules.remove_outside_bbox.outside(pose):
                reasons.append("outside_bbox")

        if rules.keep_inside_bbox and pose:
            if not rules.keep_inside_bbox.inside(pose):
                reasons.append("outside_keep_bbox")

        if rules.remove_inside_bbox and pose:
            for rule in rules.remove_inside_bbox:
                if rule.matches(category, pose):
                    reasons.append("inside_bbox")
                    break

        if pose:
            for rule in rules.remove_near:
                if rule.matches(category, pose):
                    reasons.append(f"near({rule.xy[0]:.2f},{rule.xy[1]:.2f})")
                    break

        if reasons:
            removed[node_id] = reasons

    if rules.remove_closest:
        remaining_ids = {node_id for node_id in (_node_id(n, i) for i, n in enumerate(nodes))} - set(removed)
        for rule in rules.remove_closest:
            best_id = None
            best_distance = None
            for idx, node in enumerate(nodes):
                node_id = _node_id(node, idx)
                if node_id not in remaining_ids:
                    continue
                category = str(node.get("category", "unknown"))
                if not rule.matches_category(category):
                    continue
                pose = node.get("pose") or []
                if len(pose) < 2:
                    continue
                dx = float(pose[0]) - rule.xy[0]
                dy = float(pose[1]) - rule.xy[1]
                distance = math.hypot(dx, dy)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_id = node_id
            if best_id is None:
                continue
            if rule.max_distance is not None and best_distance is not None and best_distance > rule.max_distance:
                continue
            removed.setdefault(best_id, []).append(
                f"closest({rule.xy[0]:.2f},{rule.xy[1]:.2f})"
            )
            remaining_ids.discard(best_id)

    return removed


def _links_use_indices(links: List[Dict[str, Any]], node_ids: set[str]) -> bool:
    for link in links:
        for endpoint in ("source", "target"):
            value = link.get(endpoint)
            if isinstance(value, int) and str(value) not in node_ids:
                return True
    return False


def _endpoint_to_index(endpoint: Any, id_to_index: Dict[str, int]) -> Optional[int]:
    if isinstance(endpoint, int):
        return endpoint
    if endpoint is None:
        return None
    return id_to_index.get(str(endpoint))


def _filter_links(
    links: List[Dict[str, Any]],
    node_ids: List[str],
    removed_ids: set[str],
) -> List[Dict[str, Any]]:
    if not links:
        return []

    node_id_set = set(node_ids)
    uses_indices = _links_use_indices(links, node_id_set)

    if uses_indices:
        kept_indices = [idx for idx, node_id in enumerate(node_ids) if node_id not in removed_ids]
        old_to_new = {old: new for new, old in enumerate(kept_indices)}
        id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        filtered_links: List[Dict[str, Any]] = []
        for link in links:
            source = _endpoint_to_index(link.get("source"), id_to_index)
            target = _endpoint_to_index(link.get("target"), id_to_index)
            if source is None or target is None:
                continue
            if source not in old_to_new or target not in old_to_new:
                continue
            new_link = dict(link)
            new_link["source"] = old_to_new[source]
            new_link["target"] = old_to_new[target]
            filtered_links.append(new_link)
        return filtered_links

    kept_ids = set(node_ids) - removed_ids
    return [
        link
        for link in links
        if str(link.get("source")) in kept_ids and str(link.get("target")) in kept_ids
    ]


def filter_graph(
    data: Dict[str, Any],
    rules: FilterRules,
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    nodes = list(data.get("nodes", []))
    links_key = "links" if "links" in data else "edges"
    links = list(data.get(links_key, []))

    node_ids = [_node_id(node, idx) for idx, node in enumerate(nodes)]
    removed = _evaluate_removals(nodes, rules)
    kept_nodes = [
        node for idx, node in enumerate(nodes) if _node_id(node, idx) not in removed
    ]
    removed_ids = set(removed.keys())
    filtered_links = _filter_links(links, node_ids, removed_ids)

    filtered_data = dict(data)
    filtered_data["nodes"] = kept_nodes
    filtered_data[links_key] = filtered_links
    return filtered_data, removed


def _list_nodes(nodes: List[Dict[str, Any]]) -> None:
    for idx, node in enumerate(nodes):
        node_id = _node_id(node, idx)
        category = node.get("category", "unknown")
        pose = node.get("pose")
        if pose and len(pose) >= 3:
            pose_str = f"({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})"
        elif pose and len(pose) >= 2:
            pose_str = f"({pose[0]:.2f}, {pose[1]:.2f})"
        else:
            pose_str = "n/a"
        print(f"- {node_id} | {category} | {pose_str}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter false-positive nodes from a semantic graph JSON.")
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        default=str(DEFAULT_GRAPH_PATH),
        help=f"Path to semantic graph JSON (default: {DEFAULT_GRAPH_PATH})",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output graph JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--rules",
        dest="rules_path",
        help="Path to a JSON rules file (see script docstring for schema).",
    )
    parser.add_argument(
        "--remove-category",
        action="append",
        default=[],
        help="Remove all nodes matching this category (repeatable).",
    )
    parser.add_argument(
        "--remove-id",
        action="append",
        default=[],
        help="Remove nodes by id (repeatable).",
    )
    parser.add_argument(
        "--remove-near",
        action="append",
        default=[],
        help="Remove nodes near a 2D position: 'x,y,radius' or 'category,x,y,radius'.",
    )
    parser.add_argument(
        "--remove-inside",
        action="append",
        default=[],
        help=(
            "Remove nodes inside bbox: 'xmin,xmax,ymin,ymax', 'xmin,xmax,ymin,ymax,zmin,zmax', "
            "'category,xmin,xmax,ymin,ymax', or 'category,xmin,xmax,ymin,ymax,zmin,zmax'."
        ),
    )
    parser.add_argument(
        "--remove-outside",
        dest="remove_outside",
        help="Remove nodes outside bbox: 'xmin,xmax,ymin,ymax' or 'xmin,xmax,ymin,ymax,zmin,zmax'.",
    )
    parser.add_argument(
        "--keep-inside",
        dest="keep_inside",
        help="Keep only nodes inside bbox: 'xmin,xmax,ymin,ymax' or 'xmin,xmax,ymin,ymax,zmin,zmax'.",
    )
    parser.add_argument(
        "--remove-closest",
        action="append",
        default=[],
        help=(
            "Remove the closest node to a 2D position: 'x,y', 'x,y,max_distance', "
            "'category,x,y', or 'category,x,y,max_distance'."
        ),
    )
    parser.add_argument(
        "--list-nodes",
        action="store_true",
        help="Print node ids, categories, and positions before filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing an output file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each removed node and the reason it was filtered.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input_path).expanduser()
    if not input_path.exists():
        print(f"Input graph not found: {input_path}")
        return 2

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    nodes = list(data.get("nodes", []))
    if args.list_nodes:
        _list_nodes(nodes)

    rules = FilterRules()
    if args.rules_path:
        rules = _merge_rules(rules, _parse_rules_file(Path(args.rules_path).expanduser()))

    for category in args.remove_category:
        rules.remove_categories.add(str(category))
    for node_id in args.remove_id:
        rules.remove_ids.add(str(node_id))
    for spec in args.remove_near:
        rules.remove_near.append(_parse_remove_near(spec))
    for spec in args.remove_inside:
        rules.remove_inside_bbox.append(_parse_remove_inside(spec))
    if args.remove_outside:
        values = [float(value.strip()) for value in args.remove_outside.split(",")]
        rules.remove_outside_bbox = _parse_bbox_values(values)
    if args.keep_inside:
        values = [float(value.strip()) for value in args.keep_inside.split(",")]
        rules.keep_inside_bbox = _parse_bbox_values(values)
    for spec in args.remove_closest:
        rules.remove_closest.append(_parse_remove_closest(spec))

    if not rules.has_any():
        print("No filter rules provided. Use --rules or --remove-* options.")
        return 1

    filtered_data, removed = filter_graph(data, rules)

    total_nodes = len(nodes)
    kept_nodes = len(filtered_data.get("nodes", []))
    removed_nodes = len(removed)
    total_links = len(data.get("links", data.get("edges", [])) or [])
    kept_links = len(filtered_data.get("links", filtered_data.get("edges", [])) or [])

    print(f"Filtered nodes: {removed_nodes} removed, {kept_nodes} kept (from {total_nodes} total).")
    print(f"Filtered links: {total_links - kept_links} removed, {kept_links} kept.")

    if args.verbose and removed:
        for node_id, reasons in removed.items():
            print(f"- removed {node_id} ({', '.join(reasons)})")

    if args.dry_run:
        return 0

    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(filtered_data, handle, indent=4)
    print(f"✓ Wrote filtered graph to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
