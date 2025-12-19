"""ROS 2 node that builds a semantic graph from RGB-D detections."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import torch
import ros2_numpy as ros_numpy
import rclpy
from PIL import Image as PILImg
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

from ..core.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from ..core.vision_utils import annotate, filter, filter_large_boxes, overlay_masks
from ..image_listener import ImageListener
from ..map_utils import is_nearby_in_map, pose_in_map_frame, save_graph_json, update_graph_edges


class SemanticMapBuilder(Node):
    """Main ROS node that drives the semantic graph construction."""

    def __init__(self) -> None:
        super().__init__("semantic_map_builder")

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
        self.projected_lidar_timeout = float(
            self.declare_parameter("projected_lidar_timeout_sec", 2.0).value
        )
        self.reset_tf_on_time_jump = bool(self.declare_parameter("reset_tf_on_time_jump", True).value)

        labels = self.declare_parameter("target_labels", ["table", "door", "chair"]).value
        thresholds = self.declare_parameter("target_label_thresholds", [2.0, 2.0, 0.6]).value
        self.distance_thresholds = self._build_threshold_map(labels, thresholds)
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
        self.graph_path = Path(self.declare_parameter("graph_output_path", "graph.json").value)
        self.groundingdino_checkpoint = self.declare_parameter("groundingdino_checkpoint", "").value
        self.mobilesam_checkpoint = self.declare_parameter("mobilesam_checkpoint", "").value
        self.depth_anything_checkpoint = self.declare_parameter("depth_anything_checkpoint", "").value

        self.pose_history: Dict[str, List[List[float]]] = {label: [] for label in self.distance_thresholds}
        self.graph = nx.Graph()
        self.iteration = 0

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
            projected_lidar_timeout_sec=self.projected_lidar_timeout,
            reset_tf_on_time_jump=self.reset_tf_on_time_jump,
        )

        self.gdino = GroundingDINOObjectPredictor(
            checkpoint_path=self._expanduser_if_set(self.groundingdino_checkpoint)
        )
        self.sam = SegmentAnythingPredictor(
            checkpoint_path=self._expanduser_if_set(self.mobilesam_checkpoint)
        )

        self.marker_pub = self.create_publisher(MarkerArray, "semantic_graph/nodes", 10)
        self.image_pub = self.create_publisher(Image, "semantic_graph/segmented_image", 10)
        self.timer = self.create_timer(self.processing_period, self._process_frame)

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
                    "Duplicate normalized label '%s' maps to both '%s' and '%s'", normalized, lookup[normalized], label
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
                "Skipping detection phrase '%s' (normalized: '%s') because it does not match any target_labels.",
                phrase,
                normalized_phrase,
            )
        return None

    @staticmethod
    def _expanduser_if_set(value: str | None) -> Path | None:
        if not value:
            return None
        return Path(value).expanduser()

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
        frames = self.listener.get_latest_frames()
        if frames is None:
            self.get_logger().debug("No frames available from listener")
            return

        self.get_logger().info("Processing frame - running detection")
        rgb_image = frames["rgb"].astype(np.uint8)
        depth_image = frames["depth"]
        rt_camera = frames["rt_camera"]
        rt_base = frames["rt_base"]

        img_pil = PILImg.fromarray(rgb_image[:, :, (2, 1, 0)])

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
            return

        width = rgb_image.shape[1]
        height = rgb_image.shape[0]
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, width, height)

        image_pil_bboxes, masks = self.sam.predict(img_pil, image_pil_bboxes)
        image_pil_bboxes, keep_index = filter_large_boxes(image_pil_bboxes, width, height, threshold=0.5)
        # Convert PyTorch tensor to numpy for np.any() and indexing
        keep_index_np = keep_index.numpy() if hasattr(keep_index, 'numpy') else keep_index
        if not np.any(keep_index_np):
            return
        masks = masks[keep_index]
        gdino_conf = gdino_conf[keep_index]
        selected_idx = np.where(keep_index_np)[0]
        phrases = [phrases[i] for i in selected_idx]

        filtered = self._filter_to_target_labels(image_pil_bboxes, masks, gdino_conf, phrases)
        if filtered is None:
            self.get_logger().debug("Detections did not match any target_labels; skipping frame.")
            return
        image_pil_bboxes, masks, gdino_conf, phrases = filtered

        self._update_graph(
            masks.cpu().numpy(), phrases, rt_camera, rt_base, depth_image, frames["intrinsics"]
        )
        update_graph_edges(self.graph, self.edge_distance_threshold)
        self._publish_segmentation(img_pil, image_pil_bboxes, gdino_conf, phrases, masks, frames)
        self._publish_graph_markers()
        self.iteration += 1

    def _update_graph(self, masks, phrases, rt_camera, rt_base, depth_image, intrinsics):
        label_iter = {label: 0 for label in self.distance_thresholds}
        for idx, mask in enumerate(masks):
            label = phrases[idx]
            if label not in self.distance_thresholds:
                continue
            pose = pose_in_map_frame(
                rt_camera, rt_base, depth_image, segment=mask[0], intrinsics=intrinsics
            )
            if pose is None:
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

    def destroy_node(self):
        save_graph_json(self.graph, file=str(self.graph_path))
        self.get_logger().info(f"Semantic graph saved to {self.graph_path.resolve()}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Only shutdown if context is still valid
        if rclpy.ok():
            rclpy.shutdown()
