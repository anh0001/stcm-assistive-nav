#!/usr/bin/python3
"""Projected LiDAR pose test and GDINO/SAM export helper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image as PILImg
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import torch
import yaml

# Allow running this file directly (python stcm/test/test_projected_lidar_pose.py ...)
TEST_DIR = Path(__file__).resolve().parent
PKG_ROOT = TEST_DIR.parent
REPO_ROOT = PKG_ROOT.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from stcm.core.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from stcm.core.vision_utils import annotate, filter as filter_detections, filter_large_boxes, overlay_masks
from stcm.image_listener import ImageListener
from stcm.map_utils import pose_in_map_frame, pose_in_map_frame_from_projected

DEFAULT_CONFIG_PATH = PKG_ROOT / "config" / "semantic_mapping_params.yaml"
DEFAULT_OUTPUT_SUBDIR = "gdino_sam_pose"


def _resolve_config_path(config_path: Path) -> Path:
    if config_path.is_absolute() and config_path.exists():
        return config_path
    if config_path.exists():
        return config_path.resolve()
    candidate = REPO_ROOT / config_path
    return candidate.resolve()


def _load_yaml_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    return data


def _expanduser_if_set(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def _select_by_indices(container, indices):
    if hasattr(container, "index_select"):
        if not indices:
            return container[:0]
        idx_tensor = torch.as_tensor(indices, dtype=torch.long, device=container.device)
        return container.index_select(0, idx_tensor)
    if isinstance(container, np.ndarray):
        return container[indices]
    return [container[i] for i in indices]


class ProjectedLidarPoseExporter(Node):
    def __init__(self, config_path: Path, output_subdir: str) -> None:
        super().__init__("projected_lidar_pose_exporter")

        self.config_path = config_path
        self.config = _load_yaml_config(config_path)

        self.use_sim_time = bool(self.config.get("use_sim_time", False))
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, self.use_sim_time)])

        self.rgb_topic = self.config.get("rgb_topic", "/camera/image_raw")
        self.depth_topic = self.config.get("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.camera_info_topic = self.config.get("camera_info_topic", "/camera/camera_info")
        self.camera_frame = self.config.get("camera_frame", "camera_color_optical_frame")
        self.base_frame = self.config.get("base_frame", "base_link")
        self.world_frame = self.config.get("world_frame", "map")
        self.use_projected_lidar = bool(self.config.get("use_projected_lidar", False))
        self.projected_lidar_topic = self.config.get("projected_lidar_topic", "/lidar_points_projected")
        self.projected_lidar_frame = self.config.get("projected_lidar_frame", "")
        self.projected_lidar_timeout_sec = float(self.config.get("projected_lidar_timeout_sec", 2.0))
        self.reset_tf_on_time_jump = bool(self.config.get("reset_tf_on_time_jump", True))

        self.text_prompt = self.config.get("text_prompt", "objects")
        self.box_threshold = float(self.config.get("box_threshold", 0.35))
        self.text_threshold = float(self.config.get("text_threshold", 0.35))
        self.filter_conf_bound = float(self.config.get("filter_conf_bound", 1.0))
        self.filter_y_val = float(self.config.get("filter_y_val", 1.0))
        self.filter_percent_width = float(self.config.get("filter_percent_width", 0.9))
        self.filter_percent_height = float(self.config.get("filter_percent_height", 0.9))
        self.filter_percent_area = float(self.config.get("filter_percent_area", 0.005))
        self.filter_enabled = bool(self.config.get("filter_enabled", True))
        self.processing_period = float(self.config.get("processing_period", 1.0))

        self.output_dir = self._prepare_output_dir(output_subdir)
        self._last_stamp_ns = None

        self.listener = ImageListener(
            self,
            rgb_topic=self.rgb_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
            base_frame=self.base_frame,
            camera_frame=self.camera_frame,
            world_frame=self.world_frame,
            use_projected_lidar=self.use_projected_lidar,
            projected_lidar_topic=self.projected_lidar_topic,
            projected_lidar_frame=self.projected_lidar_frame,
            projected_lidar_timeout_sec=self.projected_lidar_timeout_sec,
            reset_tf_on_time_jump=self.reset_tf_on_time_jump,
        )

        self.gdino = GroundingDINOObjectPredictor(
            checkpoint_path=_expanduser_if_set(self.config.get("groundingdino_checkpoint"))
        )
        self.sam = SegmentAnythingPredictor(
            checkpoint_path=_expanduser_if_set(self.config.get("mobilesam_checkpoint"))
        )

        self.timer = self.create_timer(self.processing_period, self._process_frame)
        self.get_logger().info(
            f"Exporter ready (rgb: {self.rgb_topic}, output: {self.output_dir})"
        )

    def _prepare_output_dir(self, output_subdir: str) -> Path:
        graph_output_path = Path(self.config.get("graph_output_path", "output/stcm.json"))
        if not graph_output_path.is_absolute():
            graph_output_path = (REPO_ROOT / graph_output_path).resolve()
        output_root = graph_output_path.parent
        run_id = time.strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / output_subdir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _stamp_to_ns(self, stamp) -> int | None:
        if stamp is None:
            return None
        if not hasattr(stamp, "sec") or not hasattr(stamp, "nanosec"):
            return None
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _process_frame(self) -> None:
        frames = self.listener.get_latest_frames()
        if frames is None:
            return

        stamp_ns = self._stamp_to_ns(frames.get("stamp"))
        if stamp_ns is not None and stamp_ns == self._last_stamp_ns:
            return
        self._last_stamp_ns = stamp_ns

        rgb_image = frames["rgb"].astype(np.uint8)
        depth_image = frames["depth"]
        rt_camera = frames["rt_camera"]
        rt_base = frames["rt_base"]
        projected_cloud = frames.get("projected_cloud")
        rt_projected = frames.get("rt_projected")

        img_pil = PILImg.fromarray(rgb_image[:, :, (2, 1, 0)])
        bboxes, phrases, gdino_conf = self.gdino.predict(
            img_pil, self.text_prompt, self.box_threshold, self.text_threshold
        )
        if not phrases:
            return

        bboxes, gdino_conf, phrases, skip_detection = filter_detections(
            bboxes,
            gdino_conf,
            phrases,
            self.filter_conf_bound,
            self.filter_y_val,
            self.filter_percent_width,
            self.filter_percent_height,
            self.filter_percent_area,
            self.filter_enabled,
        )
        if skip_detection or not phrases:
            return

        width = rgb_image.shape[1]
        height = rgb_image.shape[0]
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, width, height)

        image_pil_bboxes, masks = self.sam.predict(img_pil, image_pil_bboxes)
        if image_pil_bboxes is None or masks is None:
            return
        image_pil_bboxes, keep_index = filter_large_boxes(image_pil_bboxes, width, height, threshold=0.5)
        keep_index_np = keep_index.cpu().numpy() if hasattr(keep_index, "cpu") else np.asarray(keep_index)
        if not np.any(keep_index_np):
            return
        keep_indices = np.where(keep_index_np)[0].tolist()
        masks = _select_by_indices(masks, keep_indices)
        gdino_conf = _select_by_indices(gdino_conf, keep_indices)
        phrases = [phrases[idx] for idx in keep_indices]

        poses = []
        valid_indices = []
        for idx, mask in enumerate(masks):
            mask_np = mask[0].detach().cpu().numpy()
            if self.use_projected_lidar and projected_cloud is not None and rt_projected is not None:
                pose = pose_in_map_frame_from_projected(
                    projected_cloud,
                    rt_projected,
                    rt_base,
                    segment=mask_np,
                    rt_camera=rt_camera,
                )
            else:
                pose = pose_in_map_frame(
                    rt_camera,
                    rt_base,
                    depth_image,
                    segment=mask_np,
                    intrinsics=frames["intrinsics"],
                )
            if pose is None:
                continue
            valid_indices.append(idx)
            poses.append(pose)

        if not valid_indices:
            return

        if len(valid_indices) < len(phrases):
            masks = _select_by_indices(masks, valid_indices)
            gdino_conf = _select_by_indices(gdino_conf, valid_indices)
            image_pil_bboxes = _select_by_indices(image_pil_bboxes, valid_indices)
            phrases = [phrases[idx] for idx in valid_indices]

        phrases_with_xyz = [
            f"{phrase} xyz=({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f})"
            for phrase, pose in zip(phrases, poses)
        ]

        annotated = annotate(overlay_masks(img_pil, masks), image_pil_bboxes, gdino_conf, phrases_with_xyz)

        if stamp_ns is None:
            stamp_ns = int(time.time() * 1e9)
        output_path = self.output_dir / f"gdino_sam_{stamp_ns}.png"
        annotated.save(output_path)
        self.get_logger().info(f"Saved detection image: {output_path}")


def test_pose_in_map_frame_from_projected():
    dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("u", np.float32),
            ("v", np.float32),
        ]
    )
    cloud = np.zeros(4, dtype=dtype)
    cloud["x"] = [1.0, 3.0, 5.0, 1.0]
    cloud["y"] = [0.0, 0.0, 0.0, 0.0]
    cloud["z"] = [1.0, 2.0, 0.0, 0.5]
    cloud["u"] = [2.0, 2.0, 4.0, 2.0]
    cloud["v"] = [1.0, 1.0, 4.0, 1.0]

    segment = np.zeros((5, 5), dtype=np.uint8)
    segment[1, 2] = 1

    rt_cloud = np.eye(4)
    rt_base = np.eye(4)
    rt_base[:3, 3] = np.array([5.0, -1.0, 0.0], dtype=np.float32)

    pose = pose_in_map_frame_from_projected(
        cloud, rt_cloud, rt_base, segment=segment, rt_camera=np.eye(4)
    )
    assert pose is not None
    assert np.allclose(pose, [6.0, -1.0, 0.5], atol=1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GDINO+SAM detections with XYZ overlays.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to semantic_mapping_params.yaml",
    )
    parser.add_argument(
        "--output-subdir",
        default=DEFAULT_OUTPUT_SUBDIR,
        help="Subfolder name created under the output directory.",
    )
    parser.add_argument(
        "--run-unit-test",
        action="store_true",
        help="Run the projected LiDAR pose unit test instead of the ROS exporter.",
    )
    args = parser.parse_args()

    if args.run_unit_test:
        test_pose_in_map_frame_from_projected()
        print("OK")
        return

    config_path = _resolve_config_path(args.config)
    rclpy.init()
    node = ProjectedLidarPoseExporter(config_path, args.output_subdir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
