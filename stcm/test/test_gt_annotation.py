#!/usr/bin/env python3

"""Tests for the STCM GT annotation backend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running directly from the repo root.
TEST_DIR = Path(__file__).resolve().parent
PKG_ROOT = TEST_DIR.parent
REPO_ROOT = PKG_ROOT.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from stcm.gt_annotation import (
    AnnotationFrameBundle,
    AnnotationSession,
    FrameRecord,
    RosbagRgbIndex,
    map_preview_point_to_original,
    resolve_rosbag_storage_id,
)
from stcm.map_utils import pose_in_map_frame_from_projected_mode


MEETING_GT = REPO_ROOT / "configs" / "experiments" / "ground_truth" / "meeting_stcm_gt.json"
MEETING_BAG = Path("/media/dl-box/STREAM1/ranger_recording_20251215_163827_uncompressed")
LIVING_LAB_MCAP_BAG = Path("/media/dl-box/STREAM1/robotics_living_lab_20260417_171737")


class _StubSamPredictor:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = np.asarray(mask, dtype=np.uint8)

    def predict_point(self, image, point_xy, *, multimask_output=True):
        return self.mask.copy(), 0.99


class _FakeBagIndex:
    def __init__(
        self,
        *,
        original_rgb: np.ndarray | None = None,
        preview_rgb: np.ndarray | None = None,
        projected_cloud: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        intrinsics: dict[str, float] | None = None,
        rt_camera: np.ndarray | None = None,
        rt_base: np.ndarray | None = None,
        rt_projected: np.ndarray | None = None,
    ) -> None:
        original_rgb = np.zeros((120, 160, 3), dtype=np.uint8) if original_rgb is None else original_rgb
        preview_rgb = np.zeros((120, 160, 3), dtype=np.uint8) if preview_rgb is None else preview_rgb
        rt_camera = np.eye(4, dtype=float) if rt_camera is None else rt_camera
        rt_base = np.eye(4, dtype=float) if rt_base is None else rt_base
        rt_projected = np.eye(4, dtype=float) if rt_projected is None else rt_projected
        self._bundle = AnnotationFrameBundle(
            frame_index=0,
            timestamp_ns=0,
            original_rgb=original_rgb,
            preview_rgb=preview_rgb,
            depth=depth,
            projected_cloud=projected_cloud,
            intrinsics=intrinsics,
            rt_camera=rt_camera,
            rt_base=rt_base,
            rt_projected=rt_projected,
        )
        self.frames = [
            FrameRecord(
                frame_index=0,
                message_id=0,
                timestamp_ns=0,
                robot_pose=(0.0, 0.0, 0.0),
                depth_message_id=None,
                projected_cloud_message_id=None,
                rt_camera=rt_camera,
                rt_base=rt_base,
                rt_projected=rt_projected,
            )
        ]
        self.trajectory = [(0.0, 0.0), (1.0, 1.0)]

    def nearest_frame_index(self, x: float, y: float) -> int:
        return 0

    def get_frame_image(self, frame_index: int, *, max_width: int = 900, max_height: int = 520) -> np.ndarray:
        return self._bundle.preview_rgb.copy()

    def get_frame_bundle(self, frame_index: int, *, max_width: int = 900, max_height: int = 520) -> AnnotationFrameBundle:
        return self._bundle

    def preview_to_original_point(self, frame_index: int, x_pixel: int, y_pixel: int) -> tuple[int, int]:
        return map_preview_point_to_original(
            self._bundle.preview_rgb.shape,
            self._bundle.original_rgb.shape,
            x_pixel,
            y_pixel,
        )


def _cloud_dtype():
    return np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("u", np.float32),
            ("v", np.float32),
        ]
    )


def test_map_preview_point_to_original_handles_resized_preview() -> None:
    original_shape = (1860, 2880, 3)
    preview_shape = (520, 806, 3)
    x, y = map_preview_point_to_original(preview_shape, original_shape, 403, 260)
    assert x == 1440
    assert y == 930


def test_resolve_rosbag_storage_id_reads_metadata(tmp_path: Path) -> None:
    bag_path = tmp_path / "sample_bag"
    bag_path.mkdir()
    (bag_path / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n"
        "  storage_identifier: mcap\n",
        encoding="utf-8",
    )

    assert resolve_rosbag_storage_id(bag_path, "auto") == "mcap"
    assert resolve_rosbag_storage_id(bag_path, "sqlite3") == "sqlite3"


def test_projected_mode_pose_uses_densest_voxel() -> None:
    cloud = np.zeros(4, dtype=_cloud_dtype())
    cloud["x"] = [1.01, 1.04, 1.08, 2.5]
    cloud["y"] = [0.02, 0.03, 0.04, 1.5]
    cloud["z"] = [0.41, 0.42, 0.44, 3.0]
    cloud["u"] = [2.0, 2.0, 2.0, 2.0]
    cloud["v"] = [1.0, 1.0, 1.0, 1.0]

    segment = np.zeros((4, 4), dtype=np.uint8)
    segment[1, 2] = 1

    result = pose_in_map_frame_from_projected_mode(
        cloud,
        np.eye(4),
        np.eye(4),
        segment=segment,
        voxel_size=0.2,
    )

    assert result is not None
    assert result["voxel_count"] == 3
    assert np.allclose(result["pose"], [1.0433333, 0.03, 0.4233333], atol=1e-4)


def test_projected_mode_pose_breaks_ties_with_smaller_depth() -> None:
    cloud = np.zeros(4, dtype=_cloud_dtype())
    cloud["x"] = [1.01, 1.04, 2.01, 2.04]
    cloud["y"] = [0.00, 0.01, 0.00, 0.01]
    cloud["z"] = [1.0, 1.1, 2.0, 2.1]
    cloud["u"] = [2.0, 2.0, 2.0, 2.0]
    cloud["v"] = [1.0, 1.0, 1.0, 1.0]

    segment = np.zeros((4, 4), dtype=np.uint8)
    segment[1, 2] = 1

    result = pose_in_map_frame_from_projected_mode(
        cloud,
        np.eye(4),
        np.eye(4),
        segment=segment,
        rt_camera=np.eye(4),
        voxel_size=0.2,
    )

    assert result is not None
    assert result["voxel_count"] == 2
    assert np.allclose(result["pose"], [1.025, 0.005, 1.05], atol=1e-5)


def test_projected_mode_pose_returns_none_when_mask_has_no_points() -> None:
    cloud = np.zeros(1, dtype=_cloud_dtype())
    cloud["x"] = [1.0]
    cloud["y"] = [0.0]
    cloud["z"] = [1.0]
    cloud["u"] = [2.0]
    cloud["v"] = [1.0]

    segment = np.zeros((4, 4), dtype=np.uint8)
    result = pose_in_map_frame_from_projected_mode(
        cloud,
        np.eye(4),
        np.eye(4),
        segment=segment,
        voxel_size=0.2,
    )

    assert result is None


def test_annotation_rgb_click_add_uses_projected_lidar_mode_pose(tmp_path: Path) -> None:
    original_rgb = np.zeros((6, 6, 3), dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[2, 3] = 1

    cloud = np.zeros(4, dtype=_cloud_dtype())
    cloud["x"] = [1.00, 1.03, 1.07, 3.0]
    cloud["y"] = [0.02, 0.03, 0.04, 0.0]
    cloud["z"] = [0.41, 0.43, 0.44, 0.8]
    cloud["u"] = [3.0, 3.0, 3.0, 3.0]
    cloud["v"] = [2.0, 2.0, 2.0, 2.0]

    session = AnnotationSession(
        input_json=MEETING_GT,
        output_json=tmp_path / "draft.json",
        rosbag_index=_FakeBagIndex(original_rgb=original_rgb, preview_rgb=original_rgb, projected_cloud=cloud),
        sam_predictor=_StubSamPredictor(mask),
        lidar_voxel_size=0.2,
    )

    before_count = len(session.objects)
    status = session.rgb_click_add(3, 2, category_hint="chair")

    assert "via projected_lidar" in status
    assert len(session.objects) == before_count + 1
    created = session.get_selected_object()
    assert created is not None
    assert created["category"] == "chair"
    assert np.allclose(created["pose"], [1.0333333, 0.03, 0.4266667], atol=1e-4)


def test_annotation_rgb_click_add_falls_back_to_depth(tmp_path: Path) -> None:
    original_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[0, 0] = 2.0

    session = AnnotationSession(
        input_json=MEETING_GT,
        output_json=tmp_path / "draft.json",
        rosbag_index=_FakeBagIndex(
            original_rgb=original_rgb,
            preview_rgb=original_rgb,
            projected_cloud=None,
            depth=depth,
            intrinsics={"fx": 1.0, "fy": 1.0, "px": 0.0, "py": 0.0},
        ),
        sam_predictor=_StubSamPredictor(mask),
    )

    before_count = len(session.objects)
    status = session.rgb_click_add(0, 0, category_hint="cup")

    assert "via depth_fallback" in status
    assert len(session.objects) == before_count + 1
    created = session.get_selected_object()
    assert created is not None
    assert np.allclose(created["pose"], [0.0, 0.0, 2.0], atol=1e-6)


def test_annotation_rgb_click_add_does_not_create_object_without_pose(tmp_path: Path) -> None:
    original_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 1
    depth = np.zeros((4, 4), dtype=np.float32)

    session = AnnotationSession(
        input_json=MEETING_GT,
        output_json=tmp_path / "draft.json",
        rosbag_index=_FakeBagIndex(
            original_rgb=original_rgb,
            preview_rgb=original_rgb,
            projected_cloud=None,
            depth=depth,
            intrinsics={"fx": 1.0, "fy": 1.0, "px": 0.0, "py": 0.0},
        ),
        sam_predictor=_StubSamPredictor(mask),
    )

    before_count = len(session.objects)
    status = session.rgb_click_add(1, 1, category_hint="cup")

    assert "no valid 3D pose" in status
    assert len(session.objects) == before_count


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


def test_rosbag_rgb_index_loads_living_lab_mcap_bag() -> None:
    if not LIVING_LAB_MCAP_BAG.exists():
        pytest.skip(f"External MCAP bag not available: {LIVING_LAB_MCAP_BAG}")

    index = RosbagRgbIndex(
        LIVING_LAB_MCAP_BAG,
        storage_id="mcap",
        camera_frame="camera_link",
    )

    assert index.storage_id == "mcap"
    assert len(index.frames) > 0
    assert any(frame.robot_pose is not None for frame in index.frames)
    assert any(frame.projected_cloud_message_id is not None for frame in index.frames)

    image = index.get_frame_image(0, max_width=320, max_height=240)
    assert image.ndim == 3
    assert image.shape[2] == 3
