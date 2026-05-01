"""Unit tests for scripts/eval/grounding.py.

Covers alias loading, place-graph "near" same-place tiebreak, and "between"
strict co-location bonus. Run with:

    PYTHONPATH=. pytest stcm/test/test_grounding.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

import grounding  # noqa: E402


def _make_pred(objects: list[dict], place_nodes: list[dict],
               place_edges: list[tuple[str, str]]) -> dict:
    return {
        "semantic_graph": {
            "nodes": [
                {"id": o["id"], "category": o["label"], "pose": list(o["pose"]) + [0.0]}
                for o in objects
            ],
            "links": [],
        },
        "place_graph": {
            "nodes": [
                {"id": p["id"], "pose": list(p["pose"]) + [0.0]}
                for p in place_nodes
            ],
            "links": [{"source": s, "target": t} for s, t in place_edges],
        },
    }


def test_alias_file_collapses_predicted_label(tmp_path: Path) -> None:
    """`configs/eval/label_aliases.json`-style file should make predicted
    "trash bins" group with GT command label "trash bin"."""
    alias_file = tmp_path / "aliases.json"
    alias_file.write_text(json.dumps({"trash bin": ["trash bins", "trashbin"]}))
    aliases = grounding._load_alias_map(alias_file)
    assert aliases["trash bins"] == "trash bin"
    assert aliases["trashbin"] == "trash bin"
    # built-in aliases preserved
    assert aliases["meeting_table_set"] == "meeting table set"


def test_near_same_place_breaks_geometric_tie() -> None:
    """Two chairs equidistant from the desk; the chair sharing the desk's
    place node should win top-1."""
    objects = [
        {"id": "chair_inst_1", "label": "chair", "pose": (1.0, 0.0)},   # place_a
        {"id": "chair_inst_2", "label": "chair", "pose": (-1.0, 0.0)},  # place_b
        {"id": "desk_inst_0",  "label": "desk",  "pose": (0.95, 0.0)},  # place_a
    ]
    place_nodes = [
        {"id": "place_a", "pose": (1.0, 0.0)},
        {"id": "place_b", "pose": (-1.0, 0.0)},
    ]
    pred = _make_pred(objects, place_nodes, [])
    cmd = {
        "id": "t1", "subset": "disambiguation", "target_label": "chair",
        "relation": {"type": "near", "anchor_label": "desk"},
        "gt_object_id": "chair_inst_1",
    }
    aliases = grounding._load_alias_map(None)
    out = grounding.evaluate(pred, [cmd], aliases=aliases)
    assert out["trials"][0]["ranked"][0] == "chair_inst_1"


def test_between_requires_strict_co_location() -> None:
    """The "between" topology bonus only fires when the candidate shares a
    place node with *both* anchors. A candidate at one anchor's place must
    not get the bonus."""
    objects = [
        {"id": "chair_inst_1", "label": "chair", "pose": (0.0, 0.0)},   # midpoint
        {"id": "chair_inst_2", "label": "chair", "pose": (0.0, 0.05)},  # also midpoint, sharing place
        {"id": "table_inst_0", "label": "meeting table set", "pose": (-2.0, 0.0)},
        {"id": "shoes_inst_0", "label": "shoes", "pose": (2.0, 0.0)},
    ]
    place_nodes = [
        {"id": "place_mid", "pose": (0.0, 0.0)},
        {"id": "place_left", "pose": (-2.0, 0.0)},
        {"id": "place_right", "pose": (2.0, 0.0)},
    ]
    pred = _make_pred(objects, place_nodes, [
        ("place_left", "place_mid"), ("place_mid", "place_right")
    ])
    cmd = {
        "id": "t2", "subset": "compositional", "target_label": "chair",
        "relation": {
            "type": "between",
            "anchor_labels": ["meeting table set", "shoes"],
        },
        "gt_object_id": "chair_inst_1",
    }
    aliases = grounding._load_alias_map(None)
    out = grounding.evaluate(pred, [cmd], aliases=aliases)
    # Both midpoint chairs are valid; tie broken alphabetically.
    assert out["trials"][0]["ranked"][:2] == ["chair_inst_1", "chair_inst_2"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
