"""ROS 2 node that builds a semantic graph from RGB-D detections."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import ros2_numpy as ros_numpy
import rosbag2_py
import rclpy
from PIL import Image as PILImg
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException
from visualization_msgs.msg import Marker, MarkerArray

from ..core.perception import (
    DepthAnythingPredictor,
    GroundingDINOObjectPredictor,
    SegmentAnythingPredictor,
)
from ..core.gng_instance_manager import GngInstanceManager
from ..core.place_gng import PlaceGng
from ..core.vision_utils import annotate, filter, filter_large_boxes, overlay_masks
from ..image_listener import ImageListener
from ..map_utils import (
    is_nearby_in_map,
    pose_in_map_frame,
    pose_in_map_frame_from_projected,
    save_graph_json,
    save_stcm_json,
    update_graph_edges,
)
from ..ros_utils import ros_qt_to_rt


class SemanticMapBuilder(Node):
    """Main ROS node that drives the semantic graph construction."""

    def __init__(self) -> None:
        super().__init__("semantic_map_builder")

        self.use_sim_time = bool(self.get_parameter("use_sim_time").value)
        self.rgb_topic = self.declare_parameter("rgb_topic", "/head_camera/rgb/image_raw").value
        self.depth_topic = self.declare_parameter("depth_topic", "/head_camera/depth_registered/image_raw").value
        self.camera_info_topic = self.declare_parameter("camera_info_topic", "/head_camera/rgb/camera_info").value
        self.camera_frame = self.declare_parameter("camera_frame", "head_camera_rgb_optical_frame").value
        self._rgb_frame_override = self.camera_frame.strip() if self.camera_frame else None
        self.base_frame = self.declare_parameter("base_frame", "base_link").value
        self.world_frame = self.declare_parameter("world_frame", "map").value
        self.use_projected_lidar = bool(self.declare_parameter("use_projected_lidar", False).value)
        self.projected_lidar_topic = self.declare_parameter(
            "projected_lidar_topic", "/lidar_points_projected"
        ).value
        self.projected_lidar_frame = self.declare_parameter("projected_lidar_frame", "").value
        self.projected_lidar_timeout = float(
            self.declare_parameter("projected_lidar_timeout_sec", 2.0).value
        )
        self.synchronizer_slop = float(self.declare_parameter("synchronizer_slop_sec", 0.1).value)
        self.reset_tf_on_time_jump = bool(self.declare_parameter("reset_tf_on_time_jump", True).value)

        labels = self.declare_parameter("target_labels", ["table", "door", "chair"]).value
        thresholds = self.declare_parameter("target_label_thresholds", [2.0, 2.0, 0.6]).value
        self.distance_thresholds = self._build_threshold_map(labels, thresholds)
        if len(labels) != len(thresholds):
            self.get_logger().warning(
                "target_label_thresholds length does not match target_labels; "
                "extra labels will reuse the last threshold."
            )
        self.get_logger().info(f"Target label merge radii: {self.distance_thresholds}")
        self._label_lookup = self._build_label_lookup(self.distance_thresholds.keys())
        self._unknown_phrase_cache: set[str] = set()
        self.text_prompt = self.declare_parameter("text_prompt", "table . door . chair .").value
        self.box_threshold = float(self.declare_parameter("box_threshold", 0.55).value)
        self.text_threshold = float(self.declare_parameter("text_threshold", 0.55).value)
        self.filter_conf_bound = float(self.declare_parameter("filter_conf_bound", 1.0).value)
        self.filter_y_val = float(self.declare_parameter("filter_y_val", 1.0).value)
        self.filter_percent_width = float(self.declare_parameter("filter_percent_width", 0.9).value)
        self.filter_percent_height = float(self.declare_parameter("filter_percent_height", 0.9).value)
        self.filter_percent_area = float(self.declare_parameter("filter_percent_area", 0.005).value)
        self.filter_enabled = bool(self.declare_parameter("filter_enabled", True).value)
        self.processing_period = float(self.declare_parameter("processing_period", 1.0).value)
        self.edge_distance_threshold = float(self.declare_parameter("edge_distance_threshold", 3.0).value)
        self.gng_enabled = bool(self.declare_parameter("gng_enabled", False).value)
        self.gng_per_label = bool(self.declare_parameter("gng_per_label", True).value)
        self.gng_max_nodes = int(self.declare_parameter("gng_max_nodes", 1000).value)
        self.gng_lambda = int(self.declare_parameter("gng_lambda", 200).value)
        self.gng_max_age = int(self.declare_parameter("gng_max_age", 200).value)
        self.gng_eps_w = float(self.declare_parameter("gng_eps_w", 0.05).value)
        self.gng_eps_n = float(self.declare_parameter("gng_eps_n", 0.0006).value)
        self.gng_alpha = float(self.declare_parameter("gng_alpha", 0.95).value)
        self.gng_beta = float(self.declare_parameter("gng_beta", 0.9995).value)
        self.gng_min_observations_to_commit = int(
            self.declare_parameter("gng_min_observations_to_commit", 3).value
        )
        self.gng_cluster_merge_distance = float(
            self.declare_parameter("gng_cluster_merge_distance", 0.5).value
        )
        self.gng_outlier_gate_meters = float(
            self.declare_parameter("gng_outlier_gate_meters", 0.0).value
        )
        self.place_gng_enabled = bool(self.declare_parameter("place_gng_enabled", False).value)
        self.place_gng_distance_threshold = float(
            self.declare_parameter("place_gng_distance_threshold", 1.5).value
        )
        self.place_gng_eps_w = float(self.declare_parameter("place_gng_eps_w", 0.1).value)
        self.place_gng_eps_n = float(self.declare_parameter("place_gng_eps_n", 0.01).value)
        self.place_gng_max_edge_age = int(self.declare_parameter("place_gng_max_edge_age", 50).value)
        self.place_gng_semantic_alpha = float(
            self.declare_parameter("place_gng_semantic_alpha", 0.1).value
        )
        self.place_gng_semantic_aggregation = self.declare_parameter(
            "place_gng_semantic_aggregation", "max"
        ).value
        self.place_gng_use_second_best_edge = bool(
            self.declare_parameter("place_gng_use_second_best_edge", True).value
        )
        self.place_gng_use_transition_edges = bool(
            self.declare_parameter("place_gng_use_transition_edges", True).value
        )
        self.place_gng_update_when_empty = bool(
            self.declare_parameter("place_gng_update_when_empty", False).value
        )
        self.place_gng_output_path = Path(
            self.declare_parameter("place_gng_output_path", "stcm.json").value
        )
        self.graph_path = Path(self.declare_parameter("graph_output_path", "stcm.json").value)
        self.groundingdino_checkpoint = self.declare_parameter("groundingdino_checkpoint", "").value
        self.mobilesam_checkpoint = self.declare_parameter("mobilesam_checkpoint", "").value
        self.depth_anything_checkpoint = self.declare_parameter("depth_anything_checkpoint", "").value
        self.use_depth_anything_fallback = bool(
            self.declare_parameter("use_depth_anything_fallback", True).value
        )
        self.depth_anything_max_depth = float(
            self.declare_parameter("depth_anything_max_depth", 5.0).value
        )
        self.offline_sequential = bool(self.declare_parameter("offline_sequential", False).value)
        self.rosbag_path = self.declare_parameter("rosbag_path", "").value
        self.rosbag_storage_id = self.declare_parameter("rosbag_storage_id", "sqlite3").value
        self.offline_frame_stride = int(self.declare_parameter("offline_frame_stride", 1).value)
        if self.offline_frame_stride < 1:
            self.get_logger().warning("offline_frame_stride < 1; defaulting to 1.")
            self.offline_frame_stride = 1

        self.pose_history: Dict[str, List[List[float]]] = {label: [] for label in self.distance_thresholds}
        self.graph = nx.Graph()
        self.place_graph = nx.Graph()
        self.iteration = 0
        self._offline_frame_counter = 0
        self._gng_manager = None
        self._place_gng = None
        self._tf_buffer: Optional[Buffer] = None
        self._intrinsics: Optional[Dict[str, float]] = None
        self._intrinsics_warned = False
        self._cloud_field_warning_emitted = False
        self._depth_anything = None
        self._depth_anything_failed = False
        self._depth_anything_cache = None
        self._depth_anything_cache_stamp = None

        self.listener = None
        self.timer = None
        if not self.offline_sequential:
            self.listener = ImageListener(
                self,
                rgb_topic=self.rgb_topic,
                depth_topic=self.depth_topic,
                camera_info_topic=self.camera_info_topic,
                base_frame=self.base_frame,
                camera_frame=self.camera_frame,
                world_frame=self.world_frame,
                slop_seconds=self.synchronizer_slop,
                use_projected_lidar=self.use_projected_lidar,
                projected_lidar_topic=self.projected_lidar_topic,
                projected_lidar_frame=self.projected_lidar_frame,
                projected_lidar_timeout_sec=self.projected_lidar_timeout,
                reset_tf_on_time_jump=self.reset_tf_on_time_jump,
            )

        self.gdino = GroundingDINOObjectPredictor(
            checkpoint_path=self._expanduser_if_set(self.groundingdino_checkpoint)
        )
        self.sam = SegmentAnythingPredictor(
            checkpoint_path=self._expanduser_if_set(self.mobilesam_checkpoint)
        )

        if self.gng_enabled:
            self._gng_manager = GngInstanceManager(
                enabled=self.gng_enabled,
                per_label=self.gng_per_label,
                max_nodes=self.gng_max_nodes,
                lambda_=self.gng_lambda,
                max_age=self.gng_max_age,
                eps_w=self.gng_eps_w,
                eps_n=self.gng_eps_n,
                alpha=self.gng_alpha,
                beta=self.gng_beta,
                min_observations_to_commit=self.gng_min_observations_to_commit,
                cluster_merge_distance=self.gng_cluster_merge_distance,
                outlier_gate_meters=self.gng_outlier_gate_meters,
                logger=self.get_logger(),
            )
            if not self._gng_manager.enabled:
                self.get_logger().warning(
                    "gng_enabled was set, but GNG bindings are unavailable; falling back to distance merge."
                )

        if self.place_gng_enabled:
            self._place_gng = PlaceGng(
                enabled=self.place_gng_enabled,
                distance_threshold=self.place_gng_distance_threshold,
                eps_w=self.place_gng_eps_w,
                eps_n=self.place_gng_eps_n,
                max_edge_age=self.place_gng_max_edge_age,
                semantic_alpha=self.place_gng_semantic_alpha,
                semantic_aggregation=self.place_gng_semantic_aggregation,
                use_second_best_edge=self.place_gng_use_second_best_edge,
                use_transition_edges=self.place_gng_use_transition_edges,
                update_semantics_when_empty=self.place_gng_update_when_empty,
                labels=list(self.distance_thresholds.keys()),
                graph=self.place_graph,
                logger=self.get_logger(),
            )

        self.marker_pub = self.create_publisher(MarkerArray, "semantic_graph/nodes", 10)
        self.place_marker_pub = None
        if self.place_gng_enabled:
            self.place_marker_pub = self.create_publisher(MarkerArray, "semantic_graph/place_graph", 10)
        self.image_pub = self.create_publisher(Image, "semantic_graph/segmented_image", 10)
        if not self.offline_sequential:
            self.timer = self.create_timer(self.processing_period, self._process_frame)
        else:
            self.get_logger().info("Offline sequential mode enabled; using rosbag2 reader.")

        self.get_logger().info(f"Semantic map builder ready (labels: {', '.join(self.distance_thresholds.keys())})")

    def _build_threshold_map(self, labels, thresholds):
        thresholds = list(thresholds) if isinstance(thresholds, (list, tuple)) else [thresholds]
        threshold_map = {}
        for idx, label in enumerate(labels):
            threshold_value = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
            threshold_map[label] = float(threshold_value)
        return threshold_map

    def _build_label_lookup(self, labels):
        lookup = {}
        token_lookup = {}
        for label in labels:
            normalized = self._normalize_label_key(label)
            if not normalized:
                continue
            if normalized in lookup and lookup[normalized] != label:
                self.get_logger().warning(
                    f"Duplicate normalized label '{normalized}' maps to both '{lookup[normalized]}' and '{label}'"
                )
            lookup[normalized] = label
            token_lookup[normalized] = normalized.split()
        self._label_token_lookup = token_lookup
        return lookup

    def _normalize_label_key(self, text: str) -> str:
        tokens = []
        es_suffixes = ("ches", "shes", "sses", "xes", "zes")
        for raw_token in text.split():
            token = raw_token.strip(" .,;:()[]{}-_\"'\n\t").lower()
            if not token:
                continue
            if len(token) > 3 and token.endswith("ies"):
                token = token[:-3] + "y"
            elif len(token) > 4 and token.endswith(es_suffixes):
                token = token[:-2]
            elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
            tokens.append(token)
        return " ".join(tokens)

    def _canonicalize_phrase(self, phrase: str) -> str | None:
        normalized_phrase = self._normalize_label_key(phrase)

        if normalized_phrase in self._label_lookup:
            return self._label_lookup[normalized_phrase]

        phrase_tokens = normalized_phrase.split()
        for normalized_label, original_label in self._label_lookup.items():
            label_tokens = self._label_token_lookup.get(normalized_label, [])
            if all(token in phrase_tokens for token in label_tokens):
                return original_label

        if normalized_phrase and normalized_phrase not in self._unknown_phrase_cache:
            self._unknown_phrase_cache.add(normalized_phrase)
            self.get_logger().warning(
                f"Skipping detection phrase '{phrase}' (normalized: '{normalized_phrase}') because it does not match any target_labels."
            )
        return None

    @staticmethod
    def _expanduser_if_set(value: str | None) -> Path | None:
        if not value:
            return None
        return Path(value).expanduser()

    @staticmethod
    def _stamp_key(stamp) -> tuple[int, int] | None:
        if stamp is None:
            return None
        return (int(stamp.sec), int(stamp.nanosec))

    def _scale_depth_anything(self, depth_raw: np.ndarray, depth_reference: np.ndarray | None) -> np.ndarray | None:
        depth_raw = depth_raw.astype(np.float32, copy=False)
        valid_raw = np.isfinite(depth_raw) & (depth_raw > 0.0)
        if not np.any(valid_raw):
            return None

        if depth_reference is not None:
            ref_valid = np.isfinite(depth_reference) & (depth_reference > 0.0)
            if np.any(ref_valid):
                raw_median = np.median(depth_raw[valid_raw])
                ref_median = np.median(depth_reference[ref_valid])
                if raw_median > 0.0 and np.isfinite(ref_median):
                    scale = ref_median / raw_median
                    if np.isfinite(scale) and scale > 0.0:
                        scaled = depth_raw * scale
                        scaled = np.where(valid_raw, scaled, 0.0)
                        return scaled.astype(np.float32)

        max_val = np.max(depth_raw[valid_raw])
        if not np.isfinite(max_val) or max_val <= 0.0:
            return None
        max_depth = self.depth_anything_max_depth if self.depth_anything_max_depth > 0.0 else 5.0
        scaled = (depth_raw / max_val) * max_depth
        scaled = np.where(valid_raw, scaled, 0.0)
        return scaled.astype(np.float32)

    def _get_depth_anything_depth(
        self,
        img_pil: PILImg.Image,
        depth_reference: np.ndarray | None,
        stamp,
    ) -> np.ndarray | None:
        if not self.use_depth_anything_fallback:
            return None
        if not self.depth_anything_checkpoint:
            return None
        if self._depth_anything_failed:
            return None

        stamp_key = self._stamp_key(stamp)
        if (
            stamp_key is not None
            and self._depth_anything_cache_stamp == stamp_key
            and self._depth_anything_cache is not None
        ):
            return self._depth_anything_cache

        if self._depth_anything is None:
            ckpt = self._expanduser_if_set(self.depth_anything_checkpoint) or self.depth_anything_checkpoint
            try:
                self._depth_anything = DepthAnythingPredictor(ckpt)
            except Exception as exc:
                self.get_logger().error(f"Depth Anything initialization failed: {exc}")
                self._depth_anything_failed = True
                return None

        try:
            _, depth_raw = self._depth_anything.predict(img_pil)
        except Exception as exc:
            self.get_logger().error(f"Depth Anything prediction failed: {exc}")
            self._depth_anything_failed = True
            return None

        scaled = self._scale_depth_anything(depth_raw, depth_reference)
        if scaled is None:
            return None
        if stamp_key is not None:
            self._depth_anything_cache_stamp = stamp_key
            self._depth_anything_cache = scaled
        return scaled

    def _filter_to_target_labels(self, boxes, masks, scores, phrases):
        valid_indices = []
        canonical_phrases = []
        for idx, phrase in enumerate(phrases):
            canonical_label = self._canonicalize_phrase(phrase)
            if canonical_label is None:
                continue
            canonical_phrases.append(canonical_label)
            valid_indices.append(idx)

        if not valid_indices:
            return None

        def _select(container):
            if hasattr(container, "index_select"):
                index_tensor = torch.as_tensor(valid_indices, dtype=torch.long, device=container.device)
                return container.index_select(0, index_tensor)
            if isinstance(container, np.ndarray):
                return container[valid_indices]
            return [container[i] for i in valid_indices]

        filtered_boxes = _select(boxes)
        filtered_masks = _select(masks)
        filtered_scores = _select(scores)
        return filtered_boxes, filtered_masks, filtered_scores, canonical_phrases

    def _process_frame(self) -> None:
        if self.listener is None:
            return

        frames = self.listener.get_latest_frames()
        if frames is None:
            self.get_logger().debug("No frames available from listener")
            return

        self._process_frame_data(frames)

    def _process_frame_data(self, frames) -> None:
        self.get_logger().info("Processing frame - running detection")
        rgb_image = frames["rgb"].astype(np.uint8)
        depth_image = frames.get("depth")
        rt_camera = frames.get("rt_camera")
        rt_base = frames.get("rt_base")
        intrinsics = frames.get("intrinsics")
        projected_cloud = frames.get("projected_cloud")
        rt_projected = frames.get("rt_projected")

        img_pil = PILImg.fromarray(rgb_image[:, :, (2, 1, 0)])
        place_labels = None
        place_scores = None

        bboxes, phrases, gdino_conf = self.gdino.predict(
            img_pil, self.text_prompt, self.box_threshold, self.text_threshold
        )
        self.get_logger().info(f"GroundingDINO detected {len(phrases)} objects: {phrases}")

        bboxes, gdino_conf, phrases, skip_detection = filter(
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
        self.get_logger().info(f"After filtering: {len(phrases)} objects remain - {phrases}")
        if skip_detection or len(phrases) == 0:
            self.get_logger().debug("Skipping frame: no detections after filtering")
            if self._maybe_update_place_graph(rt_base):
                self._publish_place_graph_markers()
            return

        width = rgb_image.shape[1]
        height = rgb_image.shape[0]
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, width, height)

        self.get_logger().info(f"Running SAM segmentation on {len(image_pil_bboxes)} boxes...")
        try:
            image_pil_bboxes, masks = self.sam.predict(img_pil, image_pil_bboxes)
            self.get_logger().info(f"SAM segmentation completed, got {len(masks)} masks")
        except Exception as e:
            self.get_logger().error(f"SAM segmentation failed: {e}")
            if self._maybe_update_place_graph(rt_base):
                self._publish_place_graph_markers()
            return

        self.get_logger().info("Filtering large boxes...")
        image_pil_bboxes, keep_index = filter_large_boxes(image_pil_bboxes, width, height, threshold=0.5)
        # Convert PyTorch tensor to numpy for np.any() and indexing
        keep_index_np = keep_index.numpy() if hasattr(keep_index, "numpy") else keep_index
        if not np.any(keep_index_np):
            self.get_logger().info("All boxes filtered out as too large")
            if self._maybe_update_place_graph(rt_base):
                self._publish_place_graph_markers()
            return
        masks = masks[keep_index]
        gdino_conf = gdino_conf[keep_index]
        selected_idx = np.where(keep_index_np)[0]
        phrases = [phrases[i] for i in selected_idx]
        self.get_logger().info(f"After large box filter: {len(phrases)} detections remain")

        self.get_logger().info("Filtering to target labels...")
        filtered = self._filter_to_target_labels(image_pil_bboxes, masks, gdino_conf, phrases)
        self.get_logger().info("Target label filtering complete")
        if filtered is None:
            self.get_logger().debug("Detections did not match any target_labels; skipping frame.")
            if self._maybe_update_place_graph(rt_base):
                self._publish_place_graph_markers()
            return
        image_pil_bboxes, masks, gdino_conf, phrases = filtered
        place_labels = phrases
        place_scores = gdino_conf

        graph_updated = False
        projected_ready = (
            self.use_projected_lidar and projected_cloud is not None and rt_projected is not None
        )
        depth_ready = depth_image is not None
        fallback_ready = (
            self.use_depth_anything_fallback and bool(self.depth_anything_checkpoint) and rt_camera is not None
        )

        if projected_ready and rt_base is not None:
            self.get_logger().info(f"Updating graph with {len(phrases)} detections (projected LiDAR mode)...")
            self.get_logger().info("Converting masks to numpy...")
            masks_numpy = masks.cpu().numpy()
            self.get_logger().info(f"Masks converted: shape {masks_numpy.shape}")
            self._update_graph(
                masks_numpy,
                phrases,
                gdino_conf,
                rt_camera,
                rt_base,
                depth_image,
                intrinsics,
                projected_cloud=projected_cloud,
                rt_projected=rt_projected,
                img_pil=img_pil,
                stamp=frames.get("stamp"),
            )
            graph_updated = True
            self.get_logger().info("Graph update complete")
        elif rt_base is not None and rt_camera is not None and (depth_ready or fallback_ready):
            if intrinsics is None and not self._intrinsics_warned:
                self.get_logger().warning("Camera intrinsics missing; using defaults for graph projection.")
                self._intrinsics_warned = True
            self._update_graph(
                masks.cpu().numpy(),
                phrases,
                gdino_conf,
                rt_camera,
                rt_base,
                depth_image,
                intrinsics,
                projected_cloud=None,
                rt_projected=None,
                img_pil=img_pil,
                stamp=frames.get("stamp"),
            )
            graph_updated = True
        else:
            missing = []
            if rt_base is None:
                missing.append("rt_base")
            if self.use_projected_lidar and projected_cloud is None:
                missing.append("projected_cloud")
            if self.use_projected_lidar and rt_projected is None:
                missing.append("rt_projected")
            if depth_image is None and not fallback_ready:
                missing.append("depth_image")
            if rt_camera is None:
                missing.append("rt_camera")
            if missing:
                self.get_logger().warning(f"Skipping graph update due to missing data: {', '.join(missing)}")

        if graph_updated:
            update_graph_edges(self.graph, self.edge_distance_threshold)
            self.iteration += 1
        self._publish_segmentation(img_pil, image_pil_bboxes, gdino_conf, phrases, masks, frames)
        self._publish_graph_markers()
        if self._maybe_update_place_graph(rt_base, place_labels, place_scores):
            self._publish_place_graph_markers()

    def _pose_from_sources(
        self,
        mask,
        rt_camera,
        rt_base,
        depth_image,
        intrinsics,
        projected_cloud,
        rt_projected,
        img_pil,
        stamp,
    ):
        pose = None
        depth_method = None

        if self.use_projected_lidar and projected_cloud is not None and rt_projected is not None:
            pose = pose_in_map_frame_from_projected(
                projected_cloud,
                rt_projected,
                rt_base,
                segment=mask[0],
                rt_camera=rt_camera,
            )
            depth_method = "LiDAR"

        if pose is None and depth_image is not None and rt_camera is not None:
            pose = pose_in_map_frame(
                rt_camera, rt_base, depth_image, segment=mask[0], intrinsics=intrinsics
            )
            depth_method = "RGB-D"

        if pose is None and rt_camera is not None and img_pil is not None:
            depth_anything = self._get_depth_anything_depth(img_pil, depth_image, stamp)
            if depth_anything is not None:
                pose = pose_in_map_frame(
                    rt_camera, rt_base, depth_anything, segment=mask[0], intrinsics=intrinsics
                )
                depth_method = "Depth Anything"

        return pose, depth_method

    @staticmethod
    def _time_from_header(msg) -> int:
        stamp = msg.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _update_intrinsics_from_msg(self, msg: CameraInfo) -> None:
        intrinsics = np.array(msg.k).reshape(3, 3)
        self._intrinsics = {
            "fx": float(intrinsics[0, 0]),
            "fy": float(intrinsics[1, 1]),
            "px": float(intrinsics[0, 2]),
            "py": float(intrinsics[1, 2]),
        }
        self._intrinsics_warned = False

    def _add_tf_message(self, msg: TFMessage, is_static: bool) -> None:
        if self._tf_buffer is None:
            return
        for transform in msg.transforms:
            try:
                if is_static:
                    self._tf_buffer.set_transform_static(transform, "offline_bag")
                else:
                    self._tf_buffer.set_transform(transform, "offline_bag")
            except Exception as exc:
                self.get_logger().warning(f"Failed to insert TF transform: {exc}")

    def _lookup_tf(self, target_frame: str, source_frame: str, stamp: Time) -> Optional[np.ndarray]:
        if self._tf_buffer is None:
            return None
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=0.2),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            try:
                transform = self._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.2),
                )
                self.get_logger().warning(
                    f"TF lookup failed at {stamp.nanoseconds} ({source_frame} -> {target_frame}): {exc}. Using latest."
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as exc_latest:
                self.get_logger().warning(
                    f"TF lookup failed ({source_frame} -> {target_frame}): {exc_latest}"
                )
                return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        trans = [translation.x, translation.y, translation.z]
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        return ros_qt_to_rt(quat, trans)

    @staticmethod
    def _select_nearest_message(queue, target_stamp_ns: int) -> Optional[Tuple[int, object, int]]:
        best_stamp = None
        best_msg = None
        best_diff = None
        for stamp_ns, msg in queue:
            diff = abs(stamp_ns - target_stamp_ns)
            if best_diff is None or diff < best_diff or (diff == best_diff and stamp_ns < best_stamp):
                best_stamp = stamp_ns
                best_msg = msg
                best_diff = diff
        if best_msg is None:
            return None
        return best_stamp, best_msg, int(best_diff)

    @staticmethod
    def _trim_queue(queue, cutoff_ns: int) -> None:
        while len(queue) > 1 and queue[1][0] <= cutoff_ns:
            queue.popleft()

    def _parse_depth_image(self, depth_msg: Image) -> Optional[np.ndarray]:
        if depth_msg.encoding == "32FC1":
            depth_cv = ros_numpy.numpify(depth_msg)
            depth_cv[np.isnan(depth_cv)] = 0.0
            return depth_cv
        if depth_msg.encoding == "16UC1":
            depth_cv = ros_numpy.numpify(depth_msg).astype(np.float32)
            depth_cv /= 1000.0
            return depth_cv
        self.get_logger().error(f"Unsupported depth type: {depth_msg.encoding}")
        return None

    def _parse_pointcloud2_all_fields(self, cloud: PointCloud2) -> Optional[np.ndarray]:
        type_map = {
            1: np.int8,
            2: np.uint8,
            3: np.int16,
            4: np.uint16,
            5: np.int32,
            6: np.uint32,
            7: np.float32,
            8: np.float64,
        }
        names = []
        formats = []
        offsets = []
        for field in cloud.fields:
            np_dtype = type_map.get(field.datatype)
            if np_dtype is None:
                self.get_logger().warning(
                    f"Unknown datatype {field.datatype} for field {field.name}"
                )
                continue
            count = field.count if field.count > 0 else 1
            dtype_entry = np_dtype if count == 1 else np.dtype((np_dtype, count))
            names.append(field.name)
            formats.append(dtype_entry)
            offsets.append(field.offset)

        if not names:
            return None

        dtype = np.dtype(
            {"names": names, "formats": formats, "offsets": offsets, "itemsize": cloud.point_step}
        )
        try:
            cloud_array = np.frombuffer(cloud.data, dtype=dtype)
            if cloud.is_bigendian:
                cloud_array = cloud_array.byteswap().newbyteorder()
            return cloud_array
        except Exception as exc:
            self.get_logger().error(f"Failed to parse PointCloud2: {exc}")
            return None

    def _extract_cloud_xyz(self, cloud_array, field_names):
        field_check = set(field_names) if field_names else set()

        if {"x", "y", "z"}.issubset(field_check):
            points = np.stack((cloud_array["x"], cloud_array["y"], cloud_array["z"]), axis=1)
            return points.astype(np.float32, copy=False)
        if {"x_lidar", "y_lidar", "z_lidar"}.issubset(field_check):
            points = np.stack(
                (cloud_array["x_lidar"], cloud_array["y_lidar"], cloud_array["z_lidar"]),
                axis=1,
            )
            return points.astype(np.float32, copy=False)
        return None

    def _project_cloud_to_depth(
        self,
        cloud_array: np.ndarray,
        cloud_frame: Optional[str],
        cloud_stamp: Time,
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        field_names = cloud_array.dtype.names or ()

        if "u" not in field_names or "v" not in field_names:
            if not self._cloud_field_warning_emitted:
                self.get_logger().error(
                    f"Projected cloud missing required 'u'/'v' fields. Available fields: {list(field_names)}"
                )
                self._cloud_field_warning_emitted = True
            return None

        xyz = self._extract_cloud_xyz(cloud_array, field_names)
        if xyz is None:
            if not self._cloud_field_warning_emitted:
                self.get_logger().error("Projected cloud missing XYZ data.")
                self._cloud_field_warning_emitted = True
            return None

        if not cloud_frame:
            self.get_logger().error("Projected cloud frame is unset.")
            return None

        if cloud_frame != self.camera_frame:
            transform = self._lookup_tf(self.camera_frame, cloud_frame, cloud_stamp)
            if transform is None:
                return None
            xyz = (transform[:3, :3] @ xyz.T).T
            xyz += transform[:3, 3]

        depth_vals = xyz[:, 2].astype(np.float32, copy=False)
        u_coords = np.asarray(cloud_array["u"], dtype=np.float32)
        v_coords = np.asarray(cloud_array["v"], dtype=np.float32)

        valid = (
            np.isfinite(depth_vals)
            & (depth_vals > 0.0)
            & np.isfinite(u_coords)
            & np.isfinite(v_coords)
        )
        if not np.any(valid):
            return None

        depth_vals = depth_vals[valid]
        u_coords = u_coords[valid]
        v_coords = v_coords[valid]

        u_idx = np.rint(u_coords).astype(np.int32)
        v_idx = np.rint(v_coords).astype(np.int32)
        inside = (
            (u_idx >= 0)
            & (u_idx < width)
            & (v_idx >= 0)
            & (v_idx < height)
        )
        if not np.any(inside):
            return None

        u_idx = u_idx[inside]
        v_idx = v_idx[inside]
        depth_vals = depth_vals[inside]

        depth_img = np.full((height, width), np.inf, dtype=np.float32)
        np.minimum.at(depth_img, (v_idx, u_idx), depth_vals)
        depth_img[~np.isfinite(depth_img)] = 0.0
        depth_img[depth_img == np.inf] = 0.0
        return depth_img

    def _build_offline_frames(
        self,
        rgb_msg: Image,
        rgb_stamp_ns: int,
        pending_depth,
        pending_cloud,
        slop_ns: int,
    ):
        rgb_image = ros_numpy.numpify(rgb_msg)
        if rgb_image is None:
            return None

        if self._intrinsics is None and not self._intrinsics_warned:
            self.get_logger().warning("CameraInfo not seen yet; graph updates may be skipped.")
            self._intrinsics_warned = True

        frame_stamp = Time.from_msg(rgb_msg.header.stamp)
        rt_camera = self._lookup_tf(self.base_frame, self.camera_frame, frame_stamp)
        rt_base = self._lookup_tf(self.world_frame, self.base_frame, frame_stamp)

        projected_cloud = None
        rt_projected = None
        depth_image = None

        if self.use_projected_lidar and pending_cloud:
            selection = self._select_nearest_message(pending_cloud, rgb_stamp_ns)
            if selection is not None:
                cloud_stamp_ns, cloud_msg, diff_ns = selection
                if slop_ns > 0 and diff_ns > slop_ns:
                    self.get_logger().warning(
                        f"Projected LiDAR delta {diff_ns / 1e9:.3f}s exceeds slop {slop_ns / 1e9:.3f}s"
                    )
                projected_cloud = self._parse_pointcloud2_all_fields(cloud_msg)
                if projected_cloud is not None:
                    cloud_frame = self.projected_lidar_frame or cloud_msg.header.frame_id
                    rt_projected = None
                    if cloud_frame:
                        rt_projected = self._lookup_tf(
                            self.base_frame,
                            cloud_frame,
                            Time.from_msg(cloud_msg.header.stamp),
                        )
                    depth_image = self._project_cloud_to_depth(
                        projected_cloud,
                        cloud_frame,
                        Time.from_msg(cloud_msg.header.stamp),
                        rgb_msg.width,
                        rgb_msg.height,
                    )
            self._trim_queue(pending_cloud, rgb_stamp_ns - slop_ns)

        if depth_image is None and self.depth_topic:
            selection = self._select_nearest_message(pending_depth, rgb_stamp_ns)
            if selection is not None:
                depth_stamp_ns, depth_msg, diff_ns = selection
                if slop_ns > 0 and diff_ns > slop_ns:
                    self.get_logger().warning(
                        f"Depth delta {diff_ns / 1e9:.3f}s exceeds slop {slop_ns / 1e9:.3f}s"
                    )
                depth_image = self._parse_depth_image(depth_msg)
            self._trim_queue(pending_depth, rgb_stamp_ns - slop_ns)

        return {
            "rgb": rgb_image,
            "depth": depth_image,
            "frame_id": self._rgb_frame_override or rgb_msg.header.frame_id,
            "stamp": rgb_msg.header.stamp,
            "rt_camera": rt_camera,
            "rt_base": rt_base,
            "projected_cloud": projected_cloud,
            "rt_projected": rt_projected,
            "intrinsics": self._intrinsics,
        }

    def _should_process_offline_frame(self) -> bool:
        if self.offline_frame_stride <= 1:
            self._offline_frame_counter += 1
            return True
        should_process = self._offline_frame_counter % self.offline_frame_stride == 0
        self._offline_frame_counter += 1
        return should_process

    def _drain_offline_queue(
        self,
        pending_rgb,
        pending_depth,
        pending_cloud,
        current_time_ns: int,
        slop_ns: int,
        final: bool = False,
    ) -> int:
        processed = 0
        while pending_rgb:
            rgb_stamp_ns, rgb_msg = pending_rgb[0]
            if not final and current_time_ns is not None and rgb_stamp_ns + slop_ns > current_time_ns:
                break
            should_process = self._should_process_offline_frame()
            pending_rgb.popleft()
            if not should_process:
                self._trim_queue(pending_depth, rgb_stamp_ns - slop_ns)
                self._trim_queue(pending_cloud, rgb_stamp_ns - slop_ns)
                continue
            frames = self._build_offline_frames(
                rgb_msg, rgb_stamp_ns, pending_depth, pending_cloud, slop_ns
            )
            if frames is None:
                continue
            self._process_frame_data(frames)
            processed += 1
        return processed

    def run_offline_bag(self) -> None:
        if not self.rosbag_path:
            self.get_logger().error("offline_sequential is set but rosbag_path is empty.")
            return
        bag_path = Path(self.rosbag_path).expanduser()
        if not bag_path.exists():
            self.get_logger().error(f"Rosbag path does not exist: {bag_path}")
            return

        self._offline_frame_counter = 0
        if self.offline_frame_stride > 1:
            self.get_logger().info(
                f"Offline stride enabled: processing every {self.offline_frame_stride} RGB frames."
            )
        self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self._intrinsics = None
        self._intrinsics_warned = False
        self._cloud_field_warning_emitted = False

        reader = rosbag2_py.SequentialReader()
        storage_id = self.rosbag_storage_id or "sqlite3"
        storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        try:
            reader.open(storage_options, converter_options)
        except Exception as exc:
            self.get_logger().error(f"Failed to open rosbag: {exc}")
            return

        topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
        tf_topic = "/tf"
        tf_static_topic = "/tf_static"

        required_topics = {self.rgb_topic, self.camera_info_topic, tf_topic, tf_static_topic}
        optional_topics = {self.depth_topic, self.projected_lidar_topic}
        type_cache = {}
        for topic in required_topics | optional_topics:
            if not topic:
                continue
            type_name = topic_types.get(topic)
            if type_name is None:
                if topic in required_topics:
                    self.get_logger().warning(f"Topic '{topic}' not found in rosbag.")
                continue
            try:
                type_cache[topic] = get_message(type_name)
            except (AttributeError, ModuleNotFoundError) as exc:
                self.get_logger().warning(f"Failed to load message type for '{topic}': {exc}")

        if self.rgb_topic not in type_cache:
            self.get_logger().error(f"RGB topic '{self.rgb_topic}' missing; offline processing aborted.")
            return

        pending_rgb = deque()
        pending_depth = deque()
        pending_cloud = deque()
        slop_ns = int(self.synchronizer_slop * 1e9)
        current_time_ns = None
        processed_frames = 0

        while reader.has_next() and rclpy.ok():
            topic, data, timestamp = reader.read_next()
            current_time_ns = timestamp
            msg_type = type_cache.get(topic)
            if msg_type is None:
                continue
            msg = deserialize_message(data, msg_type)

            if topic == tf_topic:
                self._add_tf_message(msg, is_static=False)
            elif topic == tf_static_topic:
                self._add_tf_message(msg, is_static=True)
            elif topic == self.camera_info_topic:
                self._update_intrinsics_from_msg(msg)
            elif topic == self.rgb_topic:
                pending_rgb.append((self._time_from_header(msg), msg))
            elif topic == self.depth_topic:
                pending_depth.append((self._time_from_header(msg), msg))
            elif topic == self.projected_lidar_topic:
                pending_cloud.append((self._time_from_header(msg), msg))

            processed_frames += self._drain_offline_queue(
                pending_rgb,
                pending_depth,
                pending_cloud,
                current_time_ns,
                slop_ns,
                final=False,
            )

        if pending_rgb:
            processed_frames += self._drain_offline_queue(
                pending_rgb,
                pending_depth,
                pending_cloud,
                current_time_ns,
                slop_ns,
                final=True,
            )

        # Print prominent completion messages
        self.get_logger().info("=" * 80)
        self.get_logger().info("ROSBAG PROCESSING COMPLETE")
        self.get_logger().info(f"Total frames processed: {processed_frames}")
        self.get_logger().info(f"Total nodes in graph: {self.graph.number_of_nodes()}")
        self.get_logger().info(f"Total edges in graph: {self.graph.number_of_edges()}")

        # Save the graph immediately after processing
        self._save_graphs()

        self.get_logger().info("You can now stop the process with Ctrl+C")
        self.get_logger().info("=" * 80)

    def _update_graph(
        self,
        masks,
        phrases,
        scores,
        rt_camera,
        rt_base,
        depth_image,
        intrinsics,
        projected_cloud=None,
        rt_projected=None,
        img_pil=None,
        stamp=None,
    ):
        # Log transform chain once at startup
        if not hasattr(self, "_debug_logged"):
            if self.use_projected_lidar and rt_projected is not None:
                depth_source = "PROJECTED LIDAR"
            elif depth_image is not None:
                depth_source = "RGB-D DEPTH"
            else:
                depth_source = "DEPTH ANYTHING"
            self.get_logger().info(f"Transform chain: camera→base→map | Depth source: {depth_source}")
            self._debug_logged = True

        label_iter = {label: 0 for label in self.distance_thresholds}
        self.get_logger().info(f"Processing {len(masks)} masks for graph update...")
        for idx, mask in enumerate(masks):
            label = phrases[idx]
            self.get_logger().info(f"  Mask {idx+1}/{len(masks)}: label='{label}', shape={mask.shape}")
            if label not in self.distance_thresholds:
                self.get_logger().info(f"    Skipping '{label}' (not in distance_thresholds)")
                continue
            self.get_logger().info(f"    Calculating 3D pose for '{label}'...")
            pose, depth_method = self._pose_from_sources(
                mask,
                rt_camera,
                rt_base,
                depth_image,
                intrinsics,
                projected_cloud,
                rt_projected,
                img_pil,
                stamp,
            )
            self.get_logger().info(f"    Pose calculation complete: pose={'valid' if pose is not None else 'None'}")
            if pose is None:
                self.get_logger().warning(f"Failed to calculate 3D position for '{label}'")
                continue

            if self._gng_manager is not None and self._gng_manager.enabled:
                self.get_logger().info(f"    Running GNG update for '{label}'...")
                score = None
                if scores is not None:
                    score = scores[idx]
                    if hasattr(score, "item"):
                        score = float(score.item())
                    else:
                        score = float(score)
                self.get_logger().info(f"    Calling GNG manager.update() with pose={pose}, score={score}")
                try:
                    assignment = self._gng_manager.update(label, np.asarray(pose), score, stamp)
                    self.get_logger().info(f"    GNG update complete: assignment={'committed' if assignment and assignment.committed else 'not committed'}")
                except Exception as e:
                    self.get_logger().error(f"    GNG update failed: {e}")
                    continue
                if assignment is None or not assignment.committed:
                    self.get_logger().info(f"    Skipping '{label}' - not yet committed (needs {self.gng_min_observations_to_commit} observations)")
                    continue
                node_id = assignment.instance_id
                pose_list = assignment.centroid.tolist()
                if self.graph.has_node(node_id):
                    node_data = self.graph.nodes[node_id]
                    node_data["pose"] = pose_list
                    node_data["robot_pose"] = rt_base.tolist()
                    node_data["stability"] = assignment.stability
                    continue
                self.graph.add_node(
                    node_id,
                    id=node_id,
                    instance_id=assignment.instance_id,
                    pose=pose_list,
                    robot_pose=rt_base.tolist(),
                    category=label,
                    stability=assignment.stability,
                )
                self.get_logger().info(
                    f"Added '{label}' at [{pose_list[0]:.2f}, {pose_list[1]:.2f}, {pose_list[2]:.2f}] "
                    f"({depth_method})"
                )
                continue

            pose_history, is_nearby = is_nearby_in_map(
                self.pose_history[label],
                pose,
                threshold=self.distance_thresholds[label],
            )
            self.pose_history[label] = pose_history
            if is_nearby:
                continue

            node_id = f"{label}_{self.iteration}_{label_iter[label]}"
            self.graph.add_node(
                node_id,
                id=node_id,
                pose=pose,
                robot_pose=rt_base.tolist(),
                category=label,
            )
            self.get_logger().info(
                f"Added '{label}' at [{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}] ({depth_method})"
            )
            label_iter[label] += 1

    def _publish_segmentation(self, img_pil, boxes, scores, phrases, masks, frames):
        annotated = annotate(overlay_masks(img_pil, masks), boxes, scores, phrases)
        annotated_np = np.array(annotated)

        msg = ros_numpy.msgify(Image, annotated_np, encoding="rgb8")
        msg.header.stamp = frames["stamp"]
        msg.header.frame_id = frames["frame_id"]
        self.image_pub.publish(msg)

    def _create_marker(self, pose, category, marker_id):
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = category
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = pose[0]
        marker.pose.position.y = pose[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0

        if category == "table":
            marker.color.b = 1.0
        elif category == "chair":
            marker.color.g = 1.0
        elif category == "door":
            marker.color.r = 1.0
        else:
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
        return marker

    def _publish_graph_markers(self):
        marker_array = MarkerArray()
        node_id = 0
        for _, data in self.graph.nodes(data=True):
            marker_array.markers.append(self._create_marker(data["pose"], data["category"], node_id))
            node_id += 1
        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    def _maybe_update_place_graph(self, rt_base, labels=None, scores=None) -> bool:
        if self._place_gng is None or not self.place_gng_enabled:
            return False
        if rt_base is None:
            return False
        position = np.asarray(rt_base[:2, 3], dtype=np.float64)
        score_list = self._coerce_scores(scores)
        label_list = list(labels) if labels is not None else None
        update = self._place_gng.update(position, labels=label_list, scores=score_list)
        return update is not None

    @staticmethod
    def _coerce_scores(scores):
        if scores is None:
            return None
        if isinstance(scores, (list, tuple)):
            return [float(s.item()) if hasattr(s, "item") else float(s) for s in scores]
        if hasattr(scores, "tolist"):
            values = scores.tolist()
            if isinstance(values, list):
                return [float(v) for v in values]
            return [float(values)]
        return [float(scores)]

    def _publish_place_graph_markers(self):
        if self.place_marker_pub is None:
            return
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        now = self.get_clock().now().to_msg()
        edge_marker = Marker()
        edge_marker.header.frame_id = self.world_frame
        edge_marker.header.stamp = now
        edge_marker.ns = "place_edges"
        edge_marker.id = 0
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.05
        edge_marker.color.r = 0.7
        edge_marker.color.g = 0.7
        edge_marker.color.b = 0.7
        edge_marker.color.a = 0.8
        for node_a, node_b in self.place_graph.edges():
            pose_a = self.place_graph.nodes[node_a].get("pose")
            pose_b = self.place_graph.nodes[node_b].get("pose")
            if pose_a is None or pose_b is None:
                continue
            if len(pose_a) < 2 or len(pose_b) < 2:
                continue
            point_a = Point()
            point_a.x = float(pose_a[0])
            point_a.y = float(pose_a[1])
            point_a.z = 0.0
            point_b = Point()
            point_b.x = float(pose_b[0])
            point_b.y = float(pose_b[1])
            point_b.z = 0.0
            edge_marker.points.extend([point_a, point_b])
        if edge_marker.points:
            marker_array.markers.append(edge_marker)

        marker_id = 0
        for _, data in self.place_graph.nodes(data=True):
            pose = data.get("pose")
            if pose is None or len(pose) < 2:
                continue
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = now
            marker.ns = "place_nodes"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(pose[0])
            marker.pose.position.y = float(pose[1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.1
            marker.color.r = 0.2
            marker.color.g = 0.7
            marker.color.b = 1.0
            marker.color.a = 0.9
            marker_array.markers.append(marker)
            marker_id += 1

        if marker_array.markers:
            self.place_marker_pub.publish(marker_array)

    def destroy_node(self):
        if self._gng_manager is not None:
            self._gng_manager.shutdown()
        self._save_graphs()
        super().destroy_node()

    def _save_graphs(self):
        metadata = {
            "world_frame": self.world_frame,
            "base_frame": self.base_frame,
            "place_gng_enabled": bool(self.place_gng_enabled),
        }
        place_graph = self.place_graph if self.place_gng_enabled and self._place_gng is not None else None
        save_stcm_json(self.graph, place_graph=place_graph, file=str(self.graph_path), metadata=metadata)
        self.get_logger().info(f"STCM graph saved to: {self.graph_path.resolve()}")
        if (
            place_graph is not None
            and self.place_gng_output_path.resolve() != self.graph_path.resolve()
        ):
            save_graph_json(self.place_graph, file=str(self.place_gng_output_path))
            self.get_logger().info(
                f"Place graph saved to: {self.place_gng_output_path.resolve()}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapBuilder()
    try:
        if node.offline_sequential:
            node.run_offline_bag()
            # After offline processing completes, keep reminding user to exit
            node.get_logger().info("Offline processing done. Waiting for Ctrl+C to exit...")
            import time
            while rclpy.ok():
                time.sleep(5)
                node.get_logger().info(">>> ROSBAG PROCESSING COMPLETE - Press Ctrl+C to exit <<<")
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Only shutdown if context is still valid
        if rclpy.ok():
            rclpy.shutdown()
