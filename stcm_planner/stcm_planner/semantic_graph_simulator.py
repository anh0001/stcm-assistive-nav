#!/usr/bin/python3
"""Simulate 2D language plans from a semantic graph JSON in RViz."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from stcm_planner.code_parsing.code_parser import CodeParser
from stcm_planner.llm_backend.enums import LanguageModel, NavQueryRunMode, SystemMode
from stcm_planner.llm_backend.llm_query_langchain import LLMQueryHandler
from stcm.map_utils import read_graph_json


def _coerce_pose(pose: Sequence[float]) -> List[float]:
    if pose is None:
        return []
    if len(pose) == 2:
        return [float(pose[0]), float(pose[1]), 0.0]
    return [float(pose[0]), float(pose[1]), float(pose[2])]


class SemanticGraphSimulator(Node):
    """ROS 2 node that visualizes LLM plans on top of a semantic graph."""

    def __init__(self) -> None:
        super().__init__("semantic_graph_simulator")

        self.graph_path = Path(
            self.declare_parameter("graph_path", "output/semantic_graph.json").value
        ).expanduser()
        self.environment_name = self.declare_parameter(
            "environment_name", "a room"
        ).value
        self.frame_id = self.declare_parameter("frame_id", "map").value
        self.run_mode = NavQueryRunMode(
            self.declare_parameter("run_mode", "use_tools").value
        )
        self.model = LanguageModel(self.declare_parameter("model", "mistral").value)
        self.grid_resolution = float(
            self.declare_parameter("grid_resolution", 0.25).value
        )
        self.grid_padding = float(self.declare_parameter("grid_padding", 1.0).value)
        self.object_size = float(self.declare_parameter("object_size", 0.35).value)
        self.robot_speed = float(self.declare_parameter("robot_speed", 0.5).value)
        self.simulation_hz = float(
            self.declare_parameter("simulation_hz", 10.0).value
        )
        self.reload_graph_on_query = bool(
            self.declare_parameter("reload_graph_on_query", True).value
        )

        start_pose = self.declare_parameter("start_pose", [0.0, 0.0]).value
        if isinstance(start_pose, (list, tuple)) and len(start_pose) >= 2:
            self.robot_pose = np.array([float(start_pose[0]), float(start_pose[1]), 0.0])
        else:
            self.robot_pose = np.array([0.0, 0.0, 0.0])

        self.llm_handler = LLMQueryHandler(
            model=self.model,
            run_mode=self.run_mode,
            system_mode=SystemMode.LIVE_NAVIGATION,
        )
        self.code_parser = CodeParser()

        self.object_dict: Dict[str, Dict] = {}
        self._object_markers = MarkerArray()
        self._path_marker: Marker | None = None
        self._robot_marker: Marker | None = None
        self._pending_waypoints: List[np.ndarray] = []
        self._target_ids: List[List[str]] = []

        self.object_pub = self.create_publisher(MarkerArray, "semantic_graph_sim/nodes", 10)
        self.path_pub = self.create_publisher(Marker, "semantic_graph_sim/path", 10)
        self.robot_pub = self.create_publisher(Marker, "semantic_graph_sim/robot", 10)
        self.query_sub = self.create_subscription(
            String, "/stcm_planner_query", self.handle_language_query, 10
        )

        self.marker_timer = self.create_timer(1.0, self._publish_static_markers)
        self.sim_timer = self.create_timer(
            1.0 / max(self.simulation_hz, 1.0), self._step_sim
        )

        self._load_graph()
        self._robot_marker = self._make_robot_marker(self.robot_pose)

    def _load_graph(self) -> None:
        if not self.graph_path.exists():
            self.get_logger().warning(f"Graph file not found: {self.graph_path}")
            self.object_dict = {}
            self._object_markers = MarkerArray()
            return

        graph = read_graph_json(str(self.graph_path))
        self.object_dict = self._build_object_dict(graph)
        self._object_markers = self._build_object_markers(self.object_dict, set())
        self.get_logger().info(f"Loaded {len(self.object_dict)} objects from {self.graph_path}")

    def _build_object_dict(self, graph) -> Dict[str, Dict]:
        object_dict: Dict[str, Dict] = {}
        for node_id, data in sorted(
            graph.nodes(data=True), key=lambda item: str(item[0])
        ):
            pose = _coerce_pose(data.get("pose"))
            if not pose:
                continue
            category = data.get("category", "object")
            dimensions = data.get("dimensions")
            if dimensions is None or len(dimensions) < 3:
                dimensions = [self.object_size, self.object_size, self.object_size]
            else:
                dimensions = [float(dimensions[0]), float(dimensions[1]), float(dimensions[2])]
            largest_face = float(
                data.get("largest_face", dimensions[0] * dimensions[1])
            )
            object_dict[str(node_id)] = {
                "name": str(category),
                "caption": str(data.get("caption", category)),
                "centroid": [pose[0], pose[1], pose[2]],
                "dimensions": dimensions,
                "heading": float(data.get("heading", 0.0)),
                "largest_face": largest_face,
            }
        return object_dict

    def _compute_grid(
        self, points: Iterable[Sequence[float]]
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        pts = np.array([p[:2] for p in points], dtype=float)
        if pts.size == 0:
            return (1, 1), np.array([0.0, 0.0])
        min_x, min_y = pts.min(axis=0) - self.grid_padding
        max_x, max_y = pts.max(axis=0) + self.grid_padding
        width = max(1, int(math.ceil((max_x - min_x) / self.grid_resolution)))
        height = max(1, int(math.ceil((max_y - min_y) / self.grid_resolution)))
        return (height, width), np.array([min_x, min_y], dtype=float)

    def _convert_to_grid(
        self, points: Iterable[Sequence[float]], origin: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        grid_points: List[Tuple[int, int, int]] = []
        for point in points:
            grid_x = int(math.floor((float(point[0]) - origin[0]) / self.grid_resolution))
            grid_y = int(math.floor((float(point[1]) - origin[1]) / self.grid_resolution))
            grid_z = int(math.floor(float(point[2]) / self.grid_resolution))
            grid_points.append((grid_x, grid_y, grid_z))
        return grid_points

    def _build_llm_objects(
        self, object_dict: Dict[str, Dict]
    ) -> Tuple[List[List], Dict[int, str], Tuple[int, int], np.ndarray]:
        object_ids = list(object_dict.keys())
        centroids = [object_dict[obj_id]["centroid"] for obj_id in object_ids]
        all_points = centroids + [self.robot_pose.tolist()]
        grid_shape, origin = self._compute_grid(all_points)
        grid_coords = self._convert_to_grid(centroids, origin)
        object_id_map = {relative_id: absolute_id for relative_id, absolute_id in enumerate(object_ids)}

        objects_llm: List[List] = []
        for relative_id, absolute_id in object_id_map.items():
            obj = object_dict[absolute_id]
            grid_coord = grid_coords[relative_id]
            objects_llm.append(
                [
                    relative_id,
                    obj["name"],
                    obj["caption"],
                    grid_coord[0],
                    grid_coord[1],
                    grid_coord[2],
                    obj["largest_face"],
                ]
            )
        return objects_llm, object_id_map, grid_shape, origin

    def _build_object_markers(
        self, object_dict: Dict[str, Dict], targets: set[str]
    ) -> MarkerArray:
        marker_array = MarkerArray()
        palette = [
            (0.2, 0.6, 0.9),
            (0.9, 0.5, 0.2),
            (0.2, 0.8, 0.4),
            (0.8, 0.2, 0.3),
            (0.6, 0.6, 0.2),
            (0.3, 0.3, 0.7),
        ]
        for idx, (obj_id, obj) in enumerate(object_dict.items()):
            if obj_id in targets:
                color = (0.1, 0.9, 0.1)
            else:
                color = palette[hash(obj["name"]) % len(palette)]
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "objects"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(obj["centroid"][0])
            marker.pose.position.y = float(obj["centroid"][1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(obj["dimensions"][0])
            marker.scale.y = float(obj["dimensions"][1])
            marker.scale.z = float(obj["dimensions"][2])
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.85
            marker_array.markers.append(marker)

            text_marker = Marker()
            text_marker.header.frame_id = self.frame_id
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "labels"
            text_marker.id = idx
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(obj["centroid"][0])
            text_marker.pose.position.y = float(obj["centroid"][1])
            text_marker.pose.position.z = float(obj["dimensions"][2]) + 0.2
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.3
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.9
            text_marker.text = str(obj["name"])
            marker_array.markers.append(text_marker)
        return marker_array

    def _make_path_marker(self, points: Iterable[Sequence[float]]) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 0.9
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=0.05) for p in points]
        return marker

    def _make_robot_marker(self, pose: np.ndarray) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robot"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(pose[0])
        marker.pose.position.y = float(pose[1])
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.9
        marker.color.a = 0.9
        return marker

    def _publish_static_markers(self) -> None:
        if self._object_markers.markers:
            self.object_pub.publish(self._object_markers)
        if self._path_marker is not None:
            self.path_pub.publish(self._path_marker)
        if self._robot_marker is not None:
            self.robot_pub.publish(self._robot_marker)

    def handle_language_query(self, msg: String) -> None:
        if self.reload_graph_on_query:
            self._load_graph()
        if not self.object_dict:
            self.get_logger().warning("No objects loaded from graph.")
            return

        query = msg.data.strip()
        if not query:
            return

        self.get_logger().info(f"Query received: {query}")

        objects_llm, object_id_map, grid_shape, origin = self._build_llm_objects(
            self.object_dict
        )
        robot_grid_coords = self._convert_to_grid([self.robot_pose], origin)
        robot_grid = np.array([[float(coord) for coord in robot_grid_coords[0]]], dtype=float)

        output_code = self.llm_handler.generate_query(
            self.environment_name,
            grid_shape,
            robot_grid,
            objects_llm,
            query,
            self.object_dict,
        )
        waypoints, target_ids = self.code_parser.parse_code(
            output_code, self.object_dict, object_id_map
        )
        if not len(waypoints):
            self.get_logger().warning("No waypoints parsed from LLM output.")
            return

        self._pending_waypoints = [np.array([pt[0], pt[1], 0.0]) for pt in waypoints]
        self._target_ids = target_ids
        target_set = {str(obj_id) for sublist in target_ids for obj_id in sublist}
        self._object_markers = self._build_object_markers(self.object_dict, target_set)

        path_points = [self.robot_pose.tolist()] + [pt.tolist() for pt in self._pending_waypoints]
        self._path_marker = self._make_path_marker(path_points)
        self.get_logger().info(f"Planned {len(self._pending_waypoints)} waypoint(s).")

    def _step_sim(self) -> None:
        if not self._pending_waypoints:
            return

        target = self._pending_waypoints[0]
        direction = target[:2] - self.robot_pose[:2]
        distance = float(np.linalg.norm(direction))
        step = self.robot_speed / max(self.simulation_hz, 1.0)
        if distance <= step:
            self.robot_pose[:2] = target[:2]
            self._pending_waypoints.pop(0)
        else:
            direction /= distance
            self.robot_pose[0] += direction[0] * step
            self.robot_pose[1] += direction[1] * step

        self._robot_marker = self._make_robot_marker(self.robot_pose)


def main() -> None:
    rclpy.init()
    node = SemanticGraphSimulator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
