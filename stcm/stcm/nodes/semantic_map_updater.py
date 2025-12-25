"""ROS 2 node that keeps an existing semantic graph up to date."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import ros2_numpy as ros_numpy
import rclpy
from PIL import Image as PILImg
from rclpy.node import Node
from sensor_msgs.msg import Image
from shapely.geometry import Point, Polygon
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker, MarkerArray

from ..core.perception import (
    DepthAnythingPredictor,
    GroundingDINOObjectPredictor,
    SegmentAnythingPredictor,
)
from ..core.gng_instance_manager import GngInstanceManager
from ..core.vision_utils import annotate, filter, filter_large_boxes, overlay_masks
from ..image_listener import ImageListener
from ..map_utils import (
    get_fov_points_in_map,
    is_nearby_in_map,
    pose_in_map_frame,
    pose_in_map_frame_from_projected,
    read_graph_json,
    save_graph_json,
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
        self.graph_input_path = Path(self.declare_parameter("graph_input_path", "graph.json").value)
        self.graph_output_path = Path(self.declare_parameter("graph_output_path", "graph_updated.json").value)
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
        self.pause = False
        self._gng_manager = None
        self._depth_anything = None
        self._depth_anything_failed = False
        self._depth_anything_cache = None
        self._depth_anything_cache_stamp = None

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
            if self._gng_manager.enabled:
                self._gng_manager.seed_from_graph(self.graph)
            else:
                self.get_logger().warning(
                    "gng_enabled was set, but GNG bindings are unavailable; falling back to distance merge."
                )

        self.marker_pub = self.create_publisher(MarkerArray, "semantic_graph/nodes", 10)
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

        img_pil = PILImg.fromarray(rgb_image[:, :, (2, 1, 0)])

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
            self._prune_nodes_in_fov(depth_image, rt_camera, rt_base, intrinsics)
            return
        if len(phrases) == 0:
            return

        width = rgb_image.shape[1]
        height = rgb_image.shape[0]
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, width, height)
        image_pil_bboxes, masks = self.sam.predict(img_pil, image_pil_bboxes)
        image_pil_bboxes, keep_index = filter_large_boxes(image_pil_bboxes, width, height, threshold=0.5)
        if not np.any(keep_index):
            return
        masks = masks[keep_index]
        gdino_conf = gdino_conf[keep_index]
        selected_idx = np.where(keep_index)[0]
        phrases = [phrases[i] for i in selected_idx]

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
            if not polygon.contains(Point(pose)):
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
            if polygon.contains(Point(pose)):
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
        save_graph_json(self.graph, file=str(self.graph_output_path))
        self.get_logger().info(f"Updated graph saved to {self.graph_output_path.resolve()}")
        super().destroy_node()


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
