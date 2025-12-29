#!/usr/bin/env python3

"""Smoke test for the topological place GNG module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running this file directly (python stcm/test/test_place_gng.py ...)
TEST_DIR = Path(__file__).resolve().parent
PKG_ROOT = TEST_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from stcm.core.place_gng import PlaceGng


def test_place_gng_basic() -> None:
    gng = PlaceGng(
        enabled=True,
        distance_threshold=1.0,
        eps_w=0.1,
        eps_n=0.01,
        max_edge_age=2,
        semantic_alpha=0.5,
        semantic_aggregation="max",
        use_second_best_edge=True,
        use_transition_edges=True,
        update_semantics_when_empty=False,
        labels=["table", "chair"],
    )

    update = gng.update(np.array([0.0, 0.0]), labels=["table"], scores=[0.8])
    assert update is not None
    assert gng.graph.number_of_nodes() == 1
    node_id = next(iter(gng.graph.nodes))
    assert gng.graph.nodes[node_id]["label"] == "table"

    gng.update(np.array([0.2, 0.1]), labels=["chair"], scores=[0.9])
    assert gng.graph.number_of_nodes() == 1
    assert gng.graph.nodes[node_id]["visits"] == 2

    gng.update(np.array([2.5, 0.0]), labels=["chair"], scores=[0.9])
    assert gng.graph.number_of_nodes() == 2
    assert gng.graph.number_of_edges() >= 1

    edge_data = next(iter(gng.graph.edges(data=True)))[2]
    assert "traversals" in edge_data


if __name__ == "__main__":
    test_place_gng_basic()
    print("PlaceGng smoke test passed.")
