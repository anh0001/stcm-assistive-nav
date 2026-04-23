#!/usr/bin/env python3

"""Tests for the STCM GT annotation backend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Allow running directly from the repo root.
TEST_DIR = Path(__file__).resolve().parent
PKG_ROOT = TEST_DIR.parent
REPO_ROOT = PKG_ROOT.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from stcm.gt_annotation import AnnotationSession, RosbagRgbIndex


MEETING_GT = REPO_ROOT / "configs" / "experiments" / "ground_truth" / "meeting_stcm_gt.json"
MEETING_BAG = Path("/media/dl-box/STREAM1/ranger_recording_20251215_163827_uncompressed")


class _FakeBagIndex:
    def __init__(self) -> None:
        self.frames = [type("Frame", (), {"frame_index": 0, "timestamp_ns": 0, "robot_pose": (0.0, 0.0, 0.0)})()]
        self.trajectory = [(0.0, 0.0), (1.0, 1.0)]

    def nearest_frame_index(self, x: float, y: float) -> int:
        return 0

    def get_frame_image(self, frame_index: int, *, max_width: int = 900, max_height: int = 520) -> np.ndarray:
        return np.zeros((120, 160, 3), dtype=np.uint8)


def test_annotation_save_preserves_stcm_container(tmp_path: Path) -> None:
    output_path = tmp_path / "meeting_gt_draft.json"
    session = AnnotationSession(
        input_json=MEETING_GT,
        output_json=output_path,
        rosbag_index=_FakeBagIndex(),
    )

    original_place_nodes = session.place_graph.number_of_nodes()
    original_place_edges = session.place_graph.number_of_edges()
    original_metadata = dict(session.metadata)

    first_id = session.objects[0]["id"]
    second_id = session.objects[1]["id"]
    session.semantic_edges = [(first_id, second_id, {"distance": 1.23})]
    session.select_object(first_id)
    session.delete_selected_object(confirmed=True)
    session.objects.append({"id": "test_object_1_0", "category": "test object", "pose": [1.5, 2.5, 0.0]})

    status = session.save(output_path)
    assert "Saved draft" in status

    with output_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert set(payload.keys()) >= {"stcm_version", "semantic_graph", "place_graph", "llm", "metadata"}
    assert payload["metadata"] == original_metadata
    assert len(payload["place_graph"]["nodes"]) == original_place_nodes
    assert len(payload["place_graph"]["links"]) == original_place_edges
    assert all(
        link["source"] != first_id and link["target"] != first_id
        for link in payload["semantic_graph"].get("links", [])
    )
    semantic_ids = {node["id"] for node in payload["semantic_graph"]["nodes"]}
    assert "test_object_1_0" in semantic_ids
    assert payload["llm"]["summary"]["object_count"] == len(payload["semantic_graph"]["nodes"])
    assert "object_place_links" in payload["llm"]


def test_rosbag_rgb_index_resolves_meeting_robot_poses() -> None:
    if not MEETING_BAG.exists():
        raise AssertionError(f"Expected meeting rosbag at {MEETING_BAG}")

    index = RosbagRgbIndex(MEETING_BAG)
    assert len(index.frames) > 0
    assert any(frame.robot_pose is not None for frame in index.frames)

    frame = next(frame for frame in index.frames if frame.robot_pose is not None)
    image = index.get_frame_image(frame.frame_index, max_width=320, max_height=240)
    assert image.ndim == 3
    assert image.shape[2] == 3
