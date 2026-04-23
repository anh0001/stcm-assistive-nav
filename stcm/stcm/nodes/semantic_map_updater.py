"""ROS 2 node that keeps an existing semantic graph up to date."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import ros2_numpy as ros_numpy
import rclpy
import torch
from PIL import Image as PILImg
from rclpy.node import Node
from geometry_msgs.msg import Point as RosPoint
from sensor_msgs.msg import Image
from shapely.geometry import Point as ShapelyPoint, Polygon
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker, MarkerArray

from ..core.perception import (
    DepthAnythingPredictor,
    GroundingDINOObjectPredictor,
    SegmentAnythingPredictor,
)
from ..core.gng_instance_manager import GngInstanceManager
from ..core.nyu_grounded_backend import NyuGroundedRgbdProposalBackend
from ..core.place_gng import PlaceGng
from ..core.vision_utils import annotate, filter, filter_large_boxes, filter_xyxy, overlay_masks
from ..image_listener import ImageListener
from ..map_utils import (
    get_fov_points_in_map,
    is_nearby_in_map,
    pose_in_map_frame,
    pose_in_map_frame_from_projected,
    read_graph_json,
    read_stcm_json,
    save_graph_json,
    save_stcm_json,
    update_graph_edges,
)


class SemanticMapUpdater(Node):
    """Maintains an existing semantic graph based on the current scene."""

    def __init__(self) -> None:
        super().__init__("semantic_map_updater")

        self.use_sim_time = bool(self.get_parameter("use_sim_time").value)
        self.rgb_topic = self.declare_parameter("rgb_topic", "/head_camera/rgb/image_raw").value
        self.depth_topic = self.declare_parameter("depth_topic", "/head_camera/depth_registered/image_raw").value
        self.camera_info_topic = self.declare_parameter("camera_info_topic", "/head_camera/rgb/camera_info").value
        self.camera_frame = self.declare_parameter("camera_frame", "head_camera_rgb_optical_frame").value
        self.base_frame = self.declare_parameter("base_frame", "base_link").value
        self.world_frame = self.declare_parameter("world_frame", "map").value
        self.use_projected_lidar = bool(self.declare_parameter("use_projected_lidar", False).value)
        self.projected_lidar_topic = self.declare_parameter(
            "projected_lidar_topic", "/lidar_points_projected"
        ).value
        self.projected_lidar_frame = self.declare_parameter("projected_lidar_frame", "").value
        self.pause_topic = self.declare_parameter("pause_topic", "").value
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
        self.text_prompt = self.declare_parameter("text_prompt", "table . door . chair .").value
        self.box_threshold = float(self.declare_parameter("box_threshold", 0.55).value)
        self.text_threshold = float(self.declare_parameter("text_threshold", 0.55).value)
        self.perception_backend = str(self.declare_parameter("perception_backend", "legacy").value).strip().lower()
        self.nyu_grounded_repo_path = self.declare_parameter("nyu_grounded_repo_path", "").value
        self.nyu_prompt_bank_path = self.declare_parameter("nyu_prompt_bank_path", "").value
        self.nyu_gdino_model_id = self.declare_parameter(
            "nyu_gdino_model_id", "IDEA-Research/grounding-dino-base"
        ).value
        self.nyu_sam_backend = self.declare_parameter("nyu_sam_backend", "mobilesam").value
        self.nyu_sam_model_type = self.declare_parameter("nyu_sam_model_type", "vit_t").value
        self.label_rerank_enabled = bool(self.declare_parameter("label_rerank_enabled", False).value)
        self.label_rerank_model = self.declare_parameter(
            "label_rerank_model",
            "openai/clip-vit-base-patch32",
        ).value
        self.label_margin_min = float(self.declare_parameter("label_margin_min", 0.1).value)
        self.filter_conf_bound = float(self.declare_parameter("filter_conf_bound", 1.0).value)
        self.filter_y_val = float(self.declare_parameter("filter_y_val", 0.8).value)
        self.filter_percent_width = float(self.declare_parameter("filter_percent_width", 0.8).value)
        self.filter_percent_height = float(self.declare_parameter("filter_percent_height", 0.8).value)
        self.filter_percent_area = float(self.declare_parameter("filter_percent_area", 0.01).value)
        self.filter_enabled = bool(self.declare_parameter("filter_enabled", True).value)
        self.processing_period = float(self.declare_parameter("processing_period", 1.5).value)
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
        self.instance_label_voting_enabled = bool(
            self.declare_parameter("instance_label_voting_enabled", False).value
        )
        self.cross_label_merge_distance_m = float(
            self.declare_parameter("cross_label_merge_distance_m", self.gng_cluster_merge_distance).value
        )
        self.cross_label_merge_min_cosine = float(
            self.declare_parameter("cross_label_merge_min_cosine", 0.25).value
        )
        self.instance_label_switch_margin = float(
            self.declare_parameter("instance_label_switch_margin", 0.15).value
        )
        self.instance_label_switch_min_observations = int(
            self.declare_parameter("instance_label_switch_min_observations", 2).value
        )
        self.place_gng_enabled = bool(self.declare_parameter("place_gng_enabled", False).value)
        self.place_gng_distance_threshold = float(
            self.declare_parameter("place_gng_distance_threshold", 1.5).value
        )
        self.place_gng_eps_w = float(self.declare_parameter("place_gng_eps_w", 0.1).value)
        self.place_gng_eps_n = float(self.declare_parameter("place_gng_eps_n", 0.01).value)
        self.place_gng_max_edge_age = int(self.declare_parameter("place_gng_max_edge_age", 50).value)
        self.place_gng_max_nodes = int(self.declare_parameter("place_gng_max_nodes", 0).value)
        self.place_gng_lambda = int(self.declare_parameter("place_gng_lambda", 100).value)
        self.place_gng_alpha = float(self.declare_parameter("place_gng_alpha", 0.95).value)
        self.place_gng_beta = float(self.declare_parameter("place_gng_beta", 0.9995).value)
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
        self.place_gng_input_path = Path(
            self.declare_parameter("place_gng_input_path", "stcm.json").value
        )
        self.place_gng_output_path = Path(
            self.declare_parameter("place_gng_output_path", "stcm.json").value
        )
        self.graph_input_path = Path(self.declare_parameter("graph_input_path", "stcm.json").value)
        self.graph_output_path = Path(self.declare_parameter("graph_output_path", "stcm.json").value)
        self.groundingdino_checkpoint = self.declare_parameter("groundingdino_checkpoint", "").value
        self.mobilesam_checkpoint = self.declare_parameter("mobilesam_checkpoint", "").value
        self.depth_anything_checkpoint = self.declare_parameter("depth_anything_checkpoint", "").value
        self.use_depth_anything_fallback = bool(
            self.declare_parameter("use_depth_anything_fallback", True).value
        )
        self.depth_anything_max_depth = float(
            self.declare_parameter("depth_anything_max_depth", 5.0).value
        )

        self.pose_history: Dict[str, List[List[float]]] = {label: [] for label in self.distance_thresholds}
        self.graph = read_graph_json(str(self.graph_input_path)) if self.graph_input_path.exists() else nx.Graph()
        if self.place_gng_enabled:
            if self.place_gng_input_path.exists():
                payload = read_stcm_json(str(self.place_gng_input_path))
                self.place_graph = (
                    payload["place_graph"] if payload["is_stcm"] else payload["semantic_graph"]
                )
            elif self.graph_input_path.exists():
                payload = read_stcm_json(str(self.graph_input_path))
                self.place_graph = payload["place_graph"] if payload["is_stcm"] else nx.Graph()
            else:
                self.place_graph = nx.Graph()
        else:
            self.place_graph = nx.Graph()
        self.pause = False
        self._gng_manager = None
        self._place_gng = None
        self._depth_anything = None
        self._depth_anything_failed = False
        self._depth_anything_cache = None
        self._depth_anything_cache_stamp = None
        self._proposal_backend = None

        if self.pause_topic:
            self.create_subscription(Int32, self.pause_topic, self._pause_callback, 10)

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
            reset_tf_on_time_jump=self.reset_tf_on_time_jump,
        )

        self.gdino = None
        self.sam = None
        if self.perception_backend == "nyu_grounded_rgbd":
            self._proposal_backend = NyuGroundedRgbdProposalBackend(
                repo_path=self.nyu_grounded_repo_path,
                prompt_bank_path=self.nyu_prompt_bank_path,
                gdino_model_id=self.nyu_gdino_model_id,
                sam_backend=self.nyu_sam_backend,
                sam_model_type=self.nyu_sam_model_type,
                sam_checkpoint=self._expanduser_if_set(self.mobilesam_checkpoint),
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                label_rerank_enabled=self.label_rerank_enabled,
                label_rerank_model=self.label_rerank_model,
                label_margin_min=self.label_margin_min,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.get_logger().info(
                f"Using perception backend '{self.perception_backend}' via {self.nyu_grounded_repo_path}"
            )
        else:
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
                instance_label_voting_enabled=self.instance_label_voting_enabled,
                cross_label_merge_distance_m=self.cross_label_merge_distance_m,
                cross_label_merge_min_cosine=self.cross_label_merge_min_cosine,
                instance_label_switch_margin=self.instance_label_switch_margin,
                instance_label_switch_min_observations=self.instance_label_switch_min_observations,
                logger=self.get_logger(),
            )
            if self._gng_manager.enabled:
                self._gng_manager.seed_from_graph(self.graph)
            else:
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
                gng_max_nodes=self.place_gng_max_nodes,
                gng_lambda=self.place_gng_lambda,
                gng_alpha=self.place_gng_alpha,
                gng_beta=self.place_gng_beta,
                semantic_alpha=self.place_gng_semantic_alpha,
                semantic_aggregation=self.place_gng_semantic_aggregation,
                use_second_best_edge=self.place_gng_use_second_best_edge,
                use_transition_edges=self.place_gng_use_transition_edges,
                update_semantics_when_empty=self.place_gng_update_when_empty,
                labels=list(self.distance_thresholds.keys()),
                graph=self.place_graph,
                logger=self.get_logger(),
            )
            if not self._place_gng.enabled:
                self.get_logger().warning(
                    "place_gng_enabled was set, but GNG bindings are unavailable; disabling place graph."
                )
                self.place_gng_enabled = False

        self.marker_pub = self.create_publisher(MarkerArray, "semantic_graph/nodes", 10)
        self.place_marker_pub = None
        if self.place_gng_enabled:
            self.place_marker_pub = self.create_publisher(MarkerArray, "semantic_graph/place_graph", 10)
        self.image_pub = self.create_publisher(Image, "semantic_graph/segmented_image", 10)
        self.timer = self.create_timer(self.processing_period, self._process_frame)

    def _pause_callback(self, msg: Int32) -> None:
        self.pause = bool(msg.data)

    def _build_threshold_map(self, labels, thresholds):
        thresholds = list(thresholds) if isinstance(thresholds, (list, tuple)) else [thresholds]
        threshold_map = {}
        for idx, label in enumerate(labels):
            threshold_value = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
            threshold_map[label] = float(threshold_value)
        return threshold_map

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

    def _run_proposal_backend(self, rgb_image: np.ndarray):
        rgb_for_model = rgb_image[:, :, (2, 1, 0)]
        img_pil = PILImg.fromarray(rgb_for_model)

        if self._proposal_backend is not None:
            batch = self._proposal_backend.detect_and_segment(rgb_for_model)
            image_pil_bboxes = batch.boxes_xyxy
            masks = batch.masks
            gdino_conf = batch.scores
            phrases = batch.phrases
            self.get_logger().info(
                f"Proposal backend '{self.perception_backend}' detected {len(phrases)} objects: {phrases}"
            )
            image_pil_bboxes, gdino_conf, phrases, skip_detection, keep_mask = filter_xyxy(
                image_pil_bboxes,
                gdino_conf,
                phrases,
                self.filter_conf_bound,
                self.filter_y_val,
                self.filter_percent_width,
                self.filter_percent_height,
                self.filter_percent_area,
                self.filter_enabled,
                image_width=rgb_image.shape[1],
                image_height=rgb_image.shape[0],
                return_mask=True,
            )
            if keep_mask is not None:
                index_tensor = torch.nonzero(keep_mask, as_tuple=False).flatten().to(device=masks.device)
                masks = masks.index_select(0, index_tensor)
            return img_pil, image_pil_bboxes, masks, gdino_conf, phrases, skip_detection

        bboxes, phrases, gdino_conf = self.gdino.predict(
            img_pil, self.text_prompt, self.box_threshold, self.text_threshold
        )
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
        if skip_detection:
            return img_pil, None, None, gdino_conf, phrases, True

        width = rgb_image.shape[1]
        height = rgb_image.shape[0]
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, width, height)
        image_pil_bboxes, masks = self.sam.predict(img_pil, image_pil_bboxes)
        return img_pil, image_pil_bboxes, masks, gdino_conf, phrases, False

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

        if self.use_projected_lidar and projected_cloud is not None and rt_projected is not None:
            pose = pose_in_map_frame_from_projected(
                projected_cloud,
                rt_projected,
                rt_base,
                segment=mask[0],
                rt_camera=rt_camera,
            )

        if pose is None and depth_image is not None and rt_camera is not None:
            pose = pose_in_map_frame(
                rt_camera, rt_base, depth_image, segment=mask[0], intrinsics=intrinsics
            )

        if pose is None and rt_camera is not None and img_pil is not None:
            depth_anything = self._get_depth_anything_depth(img_pil, depth_image, stamp)
            if depth_anything is not None:
                pose = pose_in_map_frame(
                    rt_camera, rt_base, depth_anything, segment=mask[0], intrinsics=intrinsics
                )

        return pose

    def _process_frame(self) -> None:
        if self.pause:
            return

        frames = self.listener.get_latest_frames()
        if frames is None:
            return

        rgb_image = frames["rgb"].astype(np.uint8)
        depth_image = frames["depth"]
        rt_camera = frames["rt_camera"]
        rt_base = frames["rt_base"]
        intrinsics = frames["intrinsics"]
        projected_cloud = frames.get("projected_cloud")
        rt_projected = frames.get("rt_projected")

        img_pil, image_pil_bboxes, masks, gdino_conf, phrases, skip_detection = self._run_proposal_backend(
            rgb_image
        )
        place_labels = None
        place_scores = None

        if skip_detection:
            self._prune_nodes_in_fov(depth_image, rt_camera, rt_base, intrinsics)
            if self._maybe_update_place_graph(rt_base):
                self._publish_place_graph_markers()
            return
        if len(phrases) == 0:
            if self._maybe_update_place_graph(rt_base):
                self._publish_place_graph_markers()
            return

        width = rgb_image.shape[1]
        height = rgb_image.shape[0]
        image_pil_bboxes, keep_index = filter_large_boxes(image_pil_bboxes, width, height, threshold=0.5)
        if not np.any(keep_index):
            if self._maybe_update_place_graph(rt_base):
                self._publish_place_graph_markers()
            return
        masks = masks[keep_index]
        gdino_conf = gdino_conf[keep_index]
        selected_idx = np.where(keep_index)[0]
        phrases = [phrases[i] for i in selected_idx]
        place_labels = phrases
        place_scores = gdino_conf

        detected_poses = {label: [] for label in self.distance_thresholds}
        mask_array = masks.cpu().numpy()
        for idx, mask in enumerate(mask_array):
            label = phrases[idx]
            if label not in detected_poses:
                continue
            pose = self._pose_from_sources(
                mask,
                rt_camera,
                rt_base,
                depth_image,
                intrinsics,
                projected_cloud,
                rt_projected,
                img_pil,
                frames.get("stamp"),
            )
            if pose is None:
                continue
            detected_poses[label].append(pose)

        self._remove_missing_nodes(depth_image, rt_camera, rt_base, detected_poses, intrinsics)
        self._add_new_nodes(
            mask_array,
            phrases,
            gdino_conf,
            rt_camera,
            rt_base,
            depth_image,
            intrinsics,
            projected_cloud,
            rt_projected,
            img_pil,
            frames.get("stamp"),
        )
        update_graph_edges(self.graph, self.edge_distance_threshold)

        annotated = annotate(overlay_masks(img_pil, masks), image_pil_bboxes, gdino_conf, phrases)
        msg = ros_numpy.msgify(Image, np.array(annotated), encoding="rgb8")
        msg.header.stamp = frames["stamp"]
        msg.header.frame_id = frames["frame_id"]
        self.image_pub.publish(msg)
        self._publish_graph_markers()
        if self._maybe_update_place_graph(rt_base, place_labels, place_scores):
            self._publish_place_graph_markers()

    def _remove_missing_nodes(self, depth_image, rt_camera, rt_base, detected_poses, intrinsics):
        fov_points = get_fov_points_in_map(depth_image, rt_camera, rt_base, intrinsics)
        polygon = Polygon([(p[0], p[1]) for p in fov_points])
        nodes_to_remove = []
        for node_name, data in self.graph.nodes(data=True):
            label = data.get("category")
            if label not in detected_poses or label not in self.distance_thresholds:
                continue
            pose = list(data["pose"])
            pose[2] = 0.0
            if not polygon.contains(ShapelyPoint(pose)):
                continue
            detections = detected_poses[label]
            if not detections:
                nodes_to_remove.append(node_name)
                continue
            distances = np.linalg.norm(np.array(detections)[:, :2] - np.array(pose)[:2], axis=1)
            if np.all(distances > self.distance_thresholds[label]):
                nodes_to_remove.append(node_name)

        for node_name in nodes_to_remove:
            self.graph.remove_node(node_name)

    def _add_new_nodes(
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
        label_iter = {label: 0 for label in self.distance_thresholds}
        for idx, mask in enumerate(masks):
            label = phrases[idx]
            if label not in self.distance_thresholds:
                continue
            pose = self._pose_from_sources(
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
            if pose is None:
                continue
            if self._gng_manager is not None and self._gng_manager.enabled:
                score = None
                if scores is not None:
                    score = scores[idx]
                    if hasattr(score, "item"):
                        score = float(score.item())
                    else:
                        score = float(score)
                assignment = self._gng_manager.update(label, np.asarray(pose), score, stamp)
                if assignment is None or not assignment.committed:
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
                label_iter[label] += 1
                continue

            pose_history, is_nearby = is_nearby_in_map(
                self.pose_history[label],
                pose,
                threshold=self.distance_thresholds[label],
            )
            self.pose_history[label] = pose_history
            if is_nearby:
                continue

            node_id = f"new_{label}_{label_iter[label]}"
            self.graph.add_node(
                node_id,
                id=node_id,
                pose=pose,
                robot_pose=rt_base.tolist(),
                category=label,
            )
            label_iter[label] += 1

    def _prune_nodes_in_fov(self, depth_image, rt_camera, rt_base, intrinsics):
        fov_points = get_fov_points_in_map(depth_image, rt_camera, rt_base, intrinsics)
        polygon = Polygon([(p[0], p[1]) for p in fov_points])
        nodes_to_remove = []
        for node_name, data in self.graph.nodes(data=True):
            if "new" in node_name:
                continue
            pose = list(data["pose"])
            pose[2] = 0.0
            if polygon.contains(ShapelyPoint(pose)):
                nodes_to_remove.append(node_name)
        for node_name in nodes_to_remove:
            self.graph.remove_node(node_name)

    def _publish_graph_markers(self):
        marker_array = MarkerArray()
        marker_id = 0
        for _, data in self.graph.nodes(data=True):
            marker_array.markers.append(self._create_marker(data["pose"], data["category"], marker_id))
            marker_id += 1
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
            point_a = RosPoint()
            point_a.x = float(pose_a[0])
            point_a.y = float(pose_a[1])
            point_a.z = 0.0
            point_b = RosPoint()
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
            marker.color.r = marker.color.g = marker.color.b = 0.5
        return marker

    def destroy_node(self):
        if self._gng_manager is not None:
            self._gng_manager.shutdown()
        if self._place_gng is not None:
            self._place_gng.shutdown()
        self._save_graphs()
        super().destroy_node()

    def _save_graphs(self):
        metadata = {
            "world_frame": self.world_frame,
            "base_frame": self.base_frame,
            "place_gng_enabled": bool(self.place_gng_enabled),
        }
        place_graph = self.place_graph if self.place_gng_enabled and self._place_gng is not None else None
        save_stcm_json(self.graph, place_graph=place_graph, file=str(self.graph_output_path), metadata=metadata)
        self.get_logger().info(f"STCM graph saved to {self.graph_output_path.resolve()}")
        if (
            place_graph is not None
            and self.place_gng_output_path.resolve() != self.graph_output_path.resolve()
        ):
            save_graph_json(self.place_graph, file=str(self.place_gng_output_path))
            self.get_logger().info(
                f"Place graph saved to {self.place_gng_output_path.resolve()}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapUpdater()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Only shutdown if context is still valid
        if rclpy.ok():
            rclpy.shutdown()
