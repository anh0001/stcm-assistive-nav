#!/usr/bin/env python3
"""Paper-figure renderer: single 2D map panel from an STCM-style graph JSON.

Drops the side panels (semantic object graph + place graph) used by
``plot_positions_2d.py``. Output is tuned for direct paper inclusion:
large axis labels, ≥10pt legend, ≥300 DPI.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "stcm"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stcm.map_utils import read_stcm_json  # noqa: E402


def _pose_to_xy(pose) -> np.ndarray | None:
    if pose is None:
        return None
    try:
        return np.array([float(pose[0]), float(pose[1])])
    except Exception:
        return None


def _place_marker_size(visits, base: float = 240.0) -> float:
    try:
        v = max(1, int(visits))
    except Exception:
        v = 1
    return base * (1.0 + 0.18 * (v - 1))


def _merge_nodes(
    nodes: List[Tuple[str, np.ndarray]], radius: float
) -> List[Tuple[str, np.ndarray]]:
    """Greedy spatial merge for same-class nodes within radius (meters)."""
    if not nodes or radius <= 0:
        return nodes
    remaining = list(nodes)
    merged: List[Tuple[str, np.ndarray]] = []
    while remaining:
        seed_id, seed_pos = remaining.pop(0)
        cluster = [(seed_id, seed_pos)]
        rest: List[Tuple[str, np.ndarray]] = []
        for nid, pos in remaining:
            if np.linalg.norm(pos - seed_pos) <= radius:
                cluster.append((nid, pos))
            else:
                rest.append((nid, pos))
        remaining = rest
        centroid = np.mean(np.array([p for _, p in cluster]), axis=0)
        merged.append((cluster[0][0], centroid))
    return merged


def _collect_object_place_links(llm_links, object_positions, place_positions):
    out = []
    if not isinstance(llm_links, list):
        return out
    for link in llm_links:
        if not isinstance(link, dict):
            continue
        obj = link.get("object_id") or link.get("object")
        place = link.get("place_id") or link.get("place")
        if obj in object_positions and place in place_positions:
            out.append((obj, place, link))
    return out


def render(
    graph_path: Path,
    out_path: Path,
    title: str,
    dpi: int = 300,
    show_place_graph: bool = True,
    show_object_place_links: bool = True,
    legend_loc: str = "upper right",
    merge_rules: Dict[str, float] | None = None,
) -> None:
    payload = read_stcm_json(str(graph_path))
    semantic_graph = payload["semantic_graph"]
    place_graph = payload["place_graph"] if payload["is_stcm"] else nx.Graph()
    llm_summary = payload.get("llm") if payload["is_stcm"] else {}

    category_nodes: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    node_positions: Dict[str, np.ndarray] = {}
    for node_id, data in semantic_graph.nodes(data=True):
        cat = data.get("category", "unknown")
        pos = _pose_to_xy(data.get("pose"))
        if pos is None:
            continue
        node_positions[node_id] = pos
        category_nodes[cat].append((node_id, pos))

    place_positions: Dict[str, np.ndarray] = {}
    place_nodes: List[Tuple[str, np.ndarray, Dict]] = []
    if show_place_graph and place_graph.number_of_nodes() > 0:
        for node_id, data in place_graph.nodes(data=True):
            pos = _pose_to_xy(data.get("pose"))
            if pos is None:
                continue
            place_positions[node_id] = pos
            place_nodes.append((node_id, pos, data))

    has_place_graph = bool(place_nodes)
    object_place_links = []
    if show_object_place_links and node_positions and place_positions:
        llm_links = (
            llm_summary.get("object_place_links")
            if isinstance(llm_summary, dict)
            else []
        )
        object_place_links = _collect_object_place_links(
            llm_links, node_positions, place_positions
        )

    if merge_rules:
        for cls, radius in merge_rules.items():
            if cls in category_nodes:
                before = len(category_nodes[cls])
                category_nodes[cls] = _merge_nodes(category_nodes[cls], radius)
                after = len(category_nodes[cls])
                print(f"  merged {cls}: {before} -> {after} @ r={radius}m")

    categories = sorted(category_nodes.keys()) or ["unknown"]
    cmap = plt.cm.get_cmap("tab20", max(1, len(categories)))
    category_colors = {c: cmap(i) for i, c in enumerate(categories)}

    fig, ax = plt.subplots(figsize=(11.5, 8.5))

    place_edge_drawn = False
    link_drawn = False

    if has_place_graph:
        for a, b, _ in place_graph.edges(data=True):
            if a not in place_positions or b not in place_positions:
                continue
            pa, pb = place_positions[a], place_positions[b]
            ax.plot(
                [pa[0], pb[0]], [pa[1], pb[1]],
                color="#0f766e", linewidth=1.4, alpha=0.55,
                linestyle="--", zorder=1,
            )
            place_edge_drawn = True

    for obj_id, place_id, _ in object_place_links:
        if obj_id not in node_positions or place_id not in place_positions:
            continue
        pa, pb = node_positions[obj_id], place_positions[place_id]
        ax.plot(
            [pa[0], pb[0]], [pa[1], pb[1]],
            color="#c2410c", linewidth=1.8, alpha=0.65,
            linestyle=":", zorder=0,
        )
        link_drawn = True

    for category in categories:
        nodes = category_nodes.get(category, [])
        if not nodes:
            continue
        coords = np.array([pos for _, pos in nodes])
        label = f"{category} ({len(nodes)})"
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=230, alpha=0.88,
            color=category_colors[category],
            label=label, edgecolors="black", linewidths=1.3,
            marker="s", zorder=3,
        )

    if has_place_graph:
        sizes = [_place_marker_size(d.get("visits")) for _, _, d in place_nodes]
        ax.scatter(
            [p[0] for _, p, _ in place_nodes],
            [p[1] for _, p, _ in place_nodes],
            s=sizes, alpha=0.85,
            color="#0f766e", edgecolors="black", linewidths=1.3,
            marker="o", label=f"places ({len(place_nodes)})", zorder=4,
        )

    # Origin star
    ax.scatter(
        [0.0], [0.0],
        marker="*", s=420, color="#dc2626",
        edgecolors="black", linewidths=1.2,
        label="Origin (0, 0)", zorder=5,
    )

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if place_edge_drawn:
        h, = ax.plot([], [], color="#0f766e", linewidth=1.4, linestyle="--")
        legend_handles.append(h); legend_labels.append("place edges")
    if link_drawn:
        h, = ax.plot([], [], color="#c2410c", linewidth=1.8, linestyle=":")
        legend_handles.append(h); legend_labels.append("object-place links")

    ax.set_xlabel("X Position (m)", fontsize=16)
    ax.set_ylabel("Y Position (m)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3, linestyle="--")

    if legend_handles:
        ncol = 2 if len(legend_labels) > 8 else 1
        ax.legend(
            handles=legend_handles, labels=legend_labels,
            loc=legend_loc, fontsize=11, ncol=ncol,
            framealpha=0.92, markerscale=0.85,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  (objects={semantic_graph.number_of_nodes()}, places={place_graph.number_of_nodes() if has_place_graph else 0})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("graph_path", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-place-graph", action="store_true")
    p.add_argument("--no-object-place-links", action="store_true")
    p.add_argument("--legend-loc", default="upper right")
    p.add_argument(
        "--merge",
        action="append",
        default=[],
        help="Same-class greedy merge: LABEL:RADIUS_M. Repeatable.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    merge_rules: Dict[str, float] = {}
    for spec in args.merge:
        if ":" not in spec:
            raise SystemExit(f"--merge expects LABEL:RADIUS, got {spec!r}")
        label, raw = spec.rsplit(":", 1)
        merge_rules[label] = float(raw)
    render(
        args.graph_path,
        args.output,
        args.title,
        dpi=args.dpi,
        show_place_graph=not args.no_place_graph,
        show_object_place_links=not args.no_object_place_links,
        legend_loc=args.legend_loc,
        merge_rules=merge_rules,
    )


if __name__ == "__main__":
    main()
