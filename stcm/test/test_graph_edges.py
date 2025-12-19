#!/usr/bin/env python3
"""Smoke tests for spatial edge creation in the semantic graph."""

from pathlib import Path
import sys

import networkx as nx

TEST_DIR = Path(__file__).resolve().parent
PKG_ROOT = TEST_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from stcm.map_utils import update_graph_edges


def _build_graph():
    graph = nx.Graph()
    graph.add_node("node_a", pose=[0.0, 0.0, 0.0])
    graph.add_node("node_b", pose=[0.5, 0.0, 0.0])
    graph.add_node("node_c", pose=[2.0, 0.0, 0.0])
    return graph


def test_update_graph_edges_threshold():
    graph = _build_graph()
    update_graph_edges(graph, edge_distance_threshold=1.0)

    assert graph.has_edge("node_a", "node_b")
    assert not graph.has_edge("node_a", "node_c")
    assert not graph.has_edge("node_b", "node_c")

    distance = graph.edges["node_a", "node_b"].get("distance")
    assert distance is not None
    assert abs(distance - 0.5) < 1e-6


def test_update_graph_edges_skips_missing_pose():
    graph = nx.Graph()
    graph.add_node("node_a", pose=[0.0, 0.0, 0.0])
    graph.add_node("node_b")

    update_graph_edges(graph, edge_distance_threshold=1.0)

    assert graph.number_of_edges() == 0


if __name__ == "__main__":
    test_update_graph_edges_threshold()
    test_update_graph_edges_skips_missing_pose()
    print("OK")
