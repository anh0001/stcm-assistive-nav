#!/usr/bin/env python3
"""Plot STCM maps on a 2D map."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "stcm"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stcm.map_utils import read_stcm_json

DEFAULT_GRAPH_PATH = REPO_ROOT / "stcm.json"
DEFAULT_FIGURE_PATH = REPO_ROOT / "output" / "stcm_map_2d.png"


def _unique_positions(points: Iterable[np.ndarray], tolerance: float) -> List[np.ndarray]:
    unique: List[np.ndarray] = []
    for pose in points:
        if not unique:
            unique.append(pose)
            continue
        if all(np.linalg.norm(pose[:2] - other[:2]) >= tolerance for other in unique):
            unique.append(pose)
    return unique


def _edge_label(edge_data: Dict) -> str | None:
    if not edge_data:
        return None
    for key in ("relationship", "relation", "type"):
        value = edge_data.get(key)
        if value:
            return str(value)
    if "weight" in edge_data:
        return f"w={edge_data['weight']}"
    return None


def _pose_to_xy(pose):
    if pose is None or len(pose) < 2:
        return None
    return np.array(pose[:2], dtype=float)


def _place_marker_size(visits, base=150.0):
    if visits is None:
        return base
    try:
        value = float(visits)
    except (TypeError, ValueError):
        return base
    return min(520.0, base + 70.0 * math.log1p(max(value, 0.0)))


def _collect_object_place_links(llm_links, object_positions, place_positions):
    links = []
    if llm_links:
        for link in llm_links:
            obj_id = link.get("object_id")
            place_id = link.get("place_id")
            if obj_id in object_positions and place_id in place_positions:
                links.append((obj_id, place_id, link.get("distance")))
        if links:
            return links
    for obj_id, obj_pos in object_positions.items():
        best_place = None
        best_distance = None
        for place_id, place_pos in place_positions.items():
            distance = float(np.linalg.norm(obj_pos - place_pos))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_place = place_id
        if best_place is not None:
            links.append((obj_id, best_place, best_distance))
    return links


def plot_positions(
    graph_path: Path,
    save_path: Path | None = None,
    dedup_threshold: float = 0.1,
    dpi: int = 150,
    show_plot: bool = True,
    show_object_labels: bool = False,
    show_place_labels: bool = True,
    show_place_graph: bool = True,
    show_object_place_links: bool = True,
    show_place_ids: bool = False,
    show_place_visits: bool = False,
    show_place_semantic_labels: bool = True,
) -> None:
    payload = read_stcm_json(str(graph_path))
    semantic_graph = payload["semantic_graph"]
    place_graph = payload["place_graph"] if payload["is_stcm"] else nx.Graph()
    llm_summary = payload.get("llm") if payload["is_stcm"] else {}

    category_nodes: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    node_positions: Dict[str, np.ndarray] = {}
    categories = set()

    for node_id, data in semantic_graph.nodes(data=True):
        category = data.get("category", "unknown")
        categories.add(category)
        position = _pose_to_xy(data.get("pose"))
        if position is None:
            continue
        node_positions[node_id] = position
        category_nodes[category].append((node_id, position))

    if not categories:
        categories.add("unknown")

    place_nodes: List[Tuple[str, np.ndarray, Dict]] = []
    place_positions: Dict[str, np.ndarray] = {}
    if show_place_graph and place_graph.number_of_nodes() > 0:
        for node_id, data in place_graph.nodes(data=True):
            position = _pose_to_xy(data.get("pose"))
            if position is None:
                continue
            place_positions[node_id] = position
            place_nodes.append((node_id, position, data))

    has_place_graph = bool(place_nodes)
    object_place_links = []
    if show_object_place_links and node_positions and place_positions:
        llm_links = llm_summary.get("object_place_links") if isinstance(llm_summary, dict) else []
        object_place_links = _collect_object_place_links(llm_links, node_positions, place_positions)

    cmap = plt.cm.get_cmap("tab20", max(1, len(categories)))
    category_colors = {cat: cmap(idx) for idx, cat in enumerate(sorted(categories))}

    if has_place_graph:
        fig = plt.figure(figsize=(18, 9))
        grid = fig.add_gridspec(
            2, 2, width_ratios=[2.3, 1.2], height_ratios=[1.0, 1.0], wspace=0.2, hspace=0.25
        )
        map_ax = fig.add_subplot(grid[:, 0])
        graph_ax = fig.add_subplot(grid[0, 1])
        place_ax = fig.add_subplot(grid[1, 1])
    else:
        fig, (map_ax, graph_ax) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [2.0, 1.25]}
        )
        place_ax = None

    title = "STCM Map Overview" if payload["is_stcm"] else "Semantic Graph Overview"
    fig.suptitle(f"{title} ({graph_path.name})", fontsize=16, fontweight="bold", y=0.98)
    subtitle = (
        f"Objects: {semantic_graph.number_of_nodes()} | "
        f"Object edges: {semantic_graph.number_of_edges()}"
    )
    if has_place_graph:
        subtitle += (
            f" | Places: {place_graph.number_of_nodes()} | "
            f"Place edges: {place_graph.number_of_edges()}"
        )
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=10)

    object_edge_drawn = False
    place_edge_drawn = False
    link_drawn = False

    if has_place_graph:
        for node_a, node_b, _ in place_graph.edges(data=True):
            if node_a not in place_positions or node_b not in place_positions:
                continue
            pos_a = place_positions[node_a]
            pos_b = place_positions[node_b]
            map_ax.plot(
                [pos_a[0], pos_b[0]],
                [pos_a[1], pos_b[1]],
                color="#0f766e",
                linewidth=1.2,
                alpha=0.55,
                linestyle="--",
                zorder=1,
            )
            place_edge_drawn = True

    # Object edges are now hidden by default
    # for node_a, node_b, data in semantic_graph.edges(data=True):
    #     if node_a not in node_positions or node_b not in node_positions:
    #         continue
    #     pos_a = node_positions[node_a]
    #     pos_b = node_positions[node_b]
    #     map_ax.plot(
    #         [pos_a[0], pos_b[0]],
    #         [pos_a[1], pos_b[1]],
    #         color="#555555",
    #         linewidth=1.1,
    #         alpha=0.5,
    #         zorder=0,
    #     )
    #     object_edge_drawn = True
    #     label = _edge_label(data)
    #     if label:
    #         midpoint = (pos_a + pos_b) / 2.0
    #         map_ax.text(
    #             midpoint[0],
    #             midpoint[1],
    #             label,
    #             fontsize=8,
    #             color="#444444",
    #             ha="center",
    #             va="center",
    #             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, linewidth=0),
    #         )

    if object_place_links:
        for obj_id, place_id, _ in object_place_links:
            if obj_id not in node_positions or place_id not in place_positions:
                continue
            pos_a = node_positions[obj_id]
            pos_b = place_positions[place_id]
            map_ax.plot(
                [pos_a[0], pos_b[0]],
                [pos_a[1], pos_b[1]],
                color="#c2410c",
                linewidth=1.8,
                alpha=0.65,
                linestyle=":",
                zorder=0,
            )
            link_drawn = True

    for category in sorted(category_nodes):
        nodes = category_nodes[category]
        coords = np.array([pos for _, pos in nodes])
        unique = _unique_positions(coords, dedup_threshold) if len(nodes) > 1 else coords
        label = f"{category} ({len(nodes)} nodes"
        if len(unique) != len(nodes):
            label += f", {len(unique)} unique"
        label += ")"
        map_ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=170,
            alpha=0.82,
            color=category_colors[category],
            label=label,
            edgecolors="black",
            linewidths=1.2,
            marker="s",
            zorder=3,
        )
        if show_object_labels:
            for node_id, pos in nodes:
                map_ax.annotate(
                    str(node_id),
                    (pos[0], pos[1]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color="#202020",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, linewidth=0),
                )

    if has_place_graph:
        place_sizes = [_place_marker_size(data.get("visits")) for _, _, data in place_nodes]
        map_ax.scatter(
            [pos[0] for _, pos, _ in place_nodes],
            [pos[1] for _, pos, _ in place_nodes],
            s=place_sizes,
            alpha=0.85,
            color="#0f766e",
            edgecolors="black",
            linewidths=1.2,
            marker="o",
            label=f"places ({len(place_nodes)} nodes)",
            zorder=4,
        )
        if show_place_labels:
            for node_id, pos, data in place_nodes:
                label_parts = []
                if show_place_ids:
                    label_parts.append(str(node_id))
                if show_place_semantic_labels:
                    place_label = data.get("label")
                    if place_label:
                        label_parts.append(str(place_label))
                if show_place_visits:
                    visits = data.get("visits")
                    if visits is not None:
                        label_parts.append(f"visits={int(visits)}")
                if label_parts:
                    label = "\n".join(label_parts)
                    map_ax.annotate(
                        label,
                        (pos[0], pos[1]),
                        xytext=(8, -10),
                        textcoords="offset points",
                        fontsize=8,
                        color="#123a3a",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65, linewidth=0),
                    )

    map_ax.scatter(
        0,
        0,
        c="red",
        s=260,
        marker="*",
        label="Origin (0, 0)",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )
    map_ax.annotate(
        "Origin",
        (0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

    map_ax.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
    map_ax.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
    map_ax.set_title(
        "STCM" if has_place_graph else "Object Positions in World Frame",
        fontsize=14,
        fontweight="bold",
    )
    map_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    map_ax.axhline(0, color="#888888", linewidth=0.8, alpha=0.4)
    map_ax.axvline(0, color="#888888", linewidth=0.8, alpha=0.4)
    map_ax.axis("equal")

    legend_handles, legend_labels = map_ax.get_legend_handles_labels()
    if object_edge_drawn:
        handle, = map_ax.plot([], [], color="#555555", linewidth=1.1, label="object edges")
        legend_handles.append(handle)
        legend_labels.append("object edges")
    if place_edge_drawn:
        handle, = map_ax.plot([], [], color="#0f766e", linewidth=1.2, linestyle="--", label="place edges")
        legend_handles.append(handle)
        legend_labels.append("place edges")
    if link_drawn:
        handle, = map_ax.plot([], [], color="#c2410c", linewidth=1.0, linestyle=":", label="object-place links")
        legend_handles.append(handle)
        legend_labels.append("object-place links")

    if legend_handles:
        ncol = 2 if len(legend_labels) > 7 else 1
        map_ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper right",
            fontsize=9,
            ncol=ncol,
            framealpha=0.9,
        )

    if semantic_graph.number_of_nodes() > 0:
        layout = nx.spring_layout(semantic_graph, seed=42)
        node_colors = [
            category_colors.get(
                semantic_graph.nodes[n].get("category", "unknown"), (0.6, 0.6, 0.6, 1.0)
            )
            for n in semantic_graph.nodes
        ]
        nx.draw_networkx_nodes(
            semantic_graph,
            layout,
            node_color=node_colors,
            node_size=680,
            linewidths=1.2,
            edgecolors="black",
            ax=graph_ax,
        )
        nx.draw_networkx_edges(semantic_graph, layout, ax=graph_ax, width=1.4)
        nx.draw_networkx_labels(
            semantic_graph,
            layout,
            labels={n: semantic_graph.nodes[n].get("id", n) for n in semantic_graph.nodes},
            font_size=7,
            ax=graph_ax,
        )
        edge_labels = {
            (u, v): label
            for u, v, data in semantic_graph.edges(data=True)
            if (label := _edge_label(data)) is not None
        }
        if edge_labels:
            nx.draw_networkx_edge_labels(
                semantic_graph, layout, edge_labels=edge_labels, font_size=8, ax=graph_ax
            )
        graph_ax.set_title("Semantic Object Graph", fontsize=13, fontweight="bold")
        graph_ax.axis("off")
    else:
        graph_ax.text(
            0.5, 0.5, "Graph is empty", ha="center", va="center", transform=graph_ax.transAxes, fontsize=12
        )
        graph_ax.axis("off")

    if place_ax is not None:
        if place_graph.number_of_nodes() > 0:
            layout = nx.spring_layout(place_graph, seed=13)
            nx.draw_networkx_nodes(
                place_graph,
                layout,
                node_color="#0f766e",
                node_size=520,
                linewidths=1.2,
                edgecolors="black",
                ax=place_ax,
            )
            nx.draw_networkx_edges(place_graph, layout, ax=place_ax, width=1.2)
            nx.draw_networkx_labels(
                place_graph,
                layout,
                labels={n: str(n) for n in place_graph.nodes},
                font_size=8,
                ax=place_ax,
            )
            place_ax.set_title("Place Graph", fontsize=13, fontweight="bold")
            place_ax.axis("off")
        else:
            place_ax.text(
                0.5, 0.5, "Place graph is empty", ha="center", va="center", transform=place_ax.transAxes, fontsize=12
            )
            place_ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✓ Saved STCM map visualization to: {save_path}")

    if show_plot:
        try:
            plt.show()
        except Exception:
            print("  (Display not available, plot saved to file)")
    else:
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize STCM maps (semantic + place graphs).")
    parser.add_argument(
        "graph_path",
        nargs="?",
        default=str(DEFAULT_GRAPH_PATH),
        help=f"Path to STCM or semantic graph JSON (default: {DEFAULT_GRAPH_PATH})",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default=str(DEFAULT_FIGURE_PATH),
        help=f"Output image path (default: {DEFAULT_FIGURE_PATH})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing the figure to disk (useful for quick inspection).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip launching a window with the visualization.",
    )
    parser.add_argument(
        "--no-place-graph",
        action="store_true",
        help="Hide place nodes/edges even if present in the STCM file.",
    )
    parser.add_argument(
        "--no-object-place-links",
        action="store_true",
        help="Hide object-to-place association links.",
    )
    parser.add_argument(
        "--show-object-labels",
        action="store_true",
        help="Annotate object IDs on the spatial map.",
    )
    parser.add_argument(
        "--hide-place-labels",
        action="store_true",
        help="Hide place node labels on the spatial map.",
    )
    parser.add_argument(
        "--show-place-ids",
        action="store_true",
        help="Show place node IDs (place_00, place_01, etc.) on the spatial map.",
    )
    parser.add_argument(
        "--show-place-visits",
        action="store_true",
        help="Show visit counts for place nodes on the spatial map.",
    )
    parser.add_argument(
        "--hide-place-semantic-labels",
        action="store_true",
        help="Hide semantic labels (e.g., 'water fountain') from place nodes.",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.1,
        help="Minimum distance (meters) used to count unique poses per category.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI when saving the figure.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    graph_file = Path(args.graph_path).expanduser()
    if not graph_file.exists() and str(graph_file) == str(DEFAULT_GRAPH_PATH):
        fallback = REPO_ROOT / "output" / "stcm.json"
        if fallback.exists():
            graph_file = fallback
    if not graph_file.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    output_path = None if args.no_save else Path(args.output_path).expanduser()
    plot_positions(
        graph_file,
        save_path=output_path,
        dedup_threshold=float(args.dedup_threshold),
        dpi=int(args.dpi),
        show_plot=not args.no_show,
        show_object_labels=args.show_object_labels,
        show_place_labels=not args.hide_place_labels,
        show_place_graph=not args.no_place_graph,
        show_object_place_links=not args.no_object_place_links,
        show_place_ids=args.show_place_ids,
        show_place_visits=args.show_place_visits,
        show_place_semantic_labels=not args.hide_place_semantic_labels,
    )
