#!/usr/bin/env python3
"""Plot semantic object positions and relationships on a 2D map."""

from __future__ import annotations

import argparse
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

from stcm.map_utils import read_graph_json

DEFAULT_GRAPH_PATH = REPO_ROOT / "output" / "stcm.json"
DEFAULT_FIGURE_PATH = REPO_ROOT / "output" / "object_positions_2d.png"


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


def plot_positions(
    graph_path: Path,
    save_path: Path | None = None,
    dedup_threshold: float = 0.1,
    dpi: int = 150,
    show_plot: bool = True,
) -> None:
    graph = read_graph_json(str(graph_path))

    category_nodes: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    node_positions: Dict[str, np.ndarray] = {}
    categories = set()

    for node_id, data in graph.nodes(data=True):
        category = data.get("category", "unknown")
        categories.add(category)
        pose = data.get("pose")
        if pose is None or len(pose) < 2:
            continue
        position = np.array(pose[:2], dtype=float)
        node_positions[node_id] = position
        category_nodes[category].append((node_id, position))

    if not categories:
        categories.add("unknown")

    cmap = plt.cm.get_cmap("tab20", max(1, len(categories)))
    category_colors = {cat: cmap(idx) for idx, cat in enumerate(sorted(categories))}

    fig, (map_ax, graph_ax) = plt.subplots(
        1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [2.0, 1.25]}
    )
    fig.suptitle(f"Semantic Graph Overview ({graph_path.name})", fontsize=16, fontweight="bold")

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
            s=180,
            alpha=0.75,
            color=category_colors[category],
            label=label,
            edgecolors="black",
            linewidths=1.5,
        )
        for node_id, pos in nodes:
            map_ax.annotate(
                f"{node_id}\n({pos[0]:.2f}, {pos[1]:.2f})",
                (pos[0], pos[1]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
                color="#202020",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, linewidth=0),
            )

    for node_a, node_b, data in graph.edges(data=True):
        if node_a not in node_positions or node_b not in node_positions:
            continue
        pos_a = node_positions[node_a]
        pos_b = node_positions[node_b]
        map_ax.plot(
            [pos_a[0], pos_b[0]],
            [pos_a[1], pos_b[1]],
            color="#555555",
            linewidth=1.2,
            alpha=0.6,
            zorder=0,
        )
        label = _edge_label(data)
        if label:
            midpoint = (pos_a + pos_b) / 2.0
            map_ax.text(
                midpoint[0],
                midpoint[1],
                label,
                fontsize=8,
                color="#444444",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, linewidth=0),
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
    map_ax.set_title("Object Positions in World Frame", fontsize=14, fontweight="bold")
    map_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    map_ax.axhline(0, color="#888888", linewidth=0.8, alpha=0.4)
    map_ax.axvline(0, color="#888888", linewidth=0.8, alpha=0.4)
    map_ax.axis("equal")
    map_ax.legend(loc="upper right", fontsize=9)

    if graph.number_of_nodes() > 0:
        layout = nx.spring_layout(graph, seed=42)
        node_colors = [
            category_colors.get(graph.nodes[n].get("category", "unknown"), (0.6, 0.6, 0.6, 1.0))
            for n in graph.nodes
        ]
        nx.draw_networkx_nodes(
            graph,
            layout,
            node_color=node_colors,
            node_size=700,
            linewidths=1.5,
            edgecolors="black",
            ax=graph_ax,
        )
        nx.draw_networkx_edges(graph, layout, ax=graph_ax, width=1.5)
        nx.draw_networkx_labels(
            graph,
            layout,
            labels={n: graph.nodes[n].get("id", n) for n in graph.nodes},
            font_size=8,
            ax=graph_ax,
        )
        edge_labels = {
            (u, v): label
            for u, v, data in graph.edges(data=True)
            if (label := _edge_label(data)) is not None
        }
        if edge_labels:
            nx.draw_networkx_edge_labels(graph, layout, edge_labels=edge_labels, font_size=8, ax=graph_ax)
        graph_ax.set_title("Semantic Graph Connectivity", fontsize=14, fontweight="bold")
        graph_ax.axis("off")
    else:
        graph_ax.text(
            0.5, 0.5, "Graph is empty", ha="center", va="center", transform=graph_ax.transAxes, fontsize=12
        )
        graph_ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"✓ Saved semantic graph visualization to: {save_path}")

    if show_plot:
        try:
            plt.show()
        except Exception:
            print("  (Display not available, plot saved to file)")
    else:
        plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize semantic graph nodes and relationships.")
    parser.add_argument(
        "graph_path",
        nargs="?",
        default=str(DEFAULT_GRAPH_PATH),
        help=f"Path to semantic graph JSON (default: {DEFAULT_GRAPH_PATH})",
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
    if not graph_file.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    output_path = None if args.no_save else Path(args.output_path).expanduser()
    plot_positions(
        graph_file,
        save_path=output_path,
        dedup_threshold=float(args.dedup_threshold),
        dpi=int(args.dpi),
        show_plot=not args.no_show,
    )
