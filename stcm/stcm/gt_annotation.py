"""Backend utilities for STCM ground-truth inspection and annotation."""

from __future__ import annotations

import colorsys
import copy
import math
import sqlite3
import threading
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
from cv_bridge import CvBridge
from networkx.readwrite import json_graph
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from .map_utils import (
    _build_llm_summary,
    pose_in_map_frame,
    pose_in_map_frame_from_projected_mode,
    read_stcm_json,
)


DEFAULT_RGB_TOPIC = "/camera/image_raw"
DEFAULT_DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
DEFAULT_CAMERA_INFO_TOPIC = "/camera/camera_info"
DEFAULT_TF_TOPIC = "/tf"
DEFAULT_TF_STATIC_TOPIC = "/tf_static"
DEFAULT_PROJECTED_LIDAR_TOPIC = "/lidar_points_projected"
DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"
DEFAULT_PROJECTED_LIDAR_FRAME = "lidar_link"
DEFAULT_WORLD_FRAME = "map"
DEFAULT_BASE_FRAME = "base_footprint"
DEFAULT_SYNC_SLOP_SEC = 2.0
DEFAULT_LIDAR_VOXEL_SIZE = 0.15
MAP_IMAGE_SIZE = (980, 720)
RGB_PREVIEW_MAX_WIDTH = 900
RGB_PREVIEW_MAX_HEIGHT = 520


@dataclass(frozen=True)
class FrameRecord:
    """Indexed RGB frame metadata for annotation browsing."""

    frame_index: int
    message_id: int
    timestamp_ns: int
    robot_pose: tuple[float, float, float] | None
    depth_message_id: int | None
    projected_cloud_message_id: int | None
    rt_camera: np.ndarray | None
    rt_base: np.ndarray | None
    rt_projected: np.ndarray | None


@dataclass(frozen=True)
class TransformRecord:
    """A single transform edge in the TF tree."""

    parent_frame: str
    child_frame: str
    matrix: np.ndarray
    stamp_ns: int
    is_static: bool


@dataclass(frozen=True)
class MapViewport:
    """World-to-image conversion helpers for the 2D annotation canvas."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: int
    height: int

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        span_x = max(self.max_x - self.min_x, 1e-6)
        span_y = max(self.max_y - self.min_y, 1e-6)
        px = int(round((float(x) - self.min_x) / span_x * (self.width - 1)))
        py = int(round((self.max_y - float(y)) / span_y * (self.height - 1)))
        return px, py

    def pixel_to_world(self, px: int, py: int) -> tuple[float, float]:
        span_x = max(self.max_x - self.min_x, 1e-6)
        span_y = max(self.max_y - self.min_y, 1e-6)
        x = self.min_x + (float(px) / max(self.width - 1, 1)) * span_x
        y = self.max_y - (float(py) / max(self.height - 1, 1)) * span_y
        return x, y


@dataclass(frozen=True)
class AnnotationFrameBundle:
    """Frame-local data required for RGB-click annotation."""

    frame_index: int
    timestamp_ns: int
    original_rgb: np.ndarray
    preview_rgb: np.ndarray
    depth: np.ndarray | None
    projected_cloud: np.ndarray | None
    intrinsics: dict[str, float] | None
    rt_camera: np.ndarray | None
    rt_base: np.ndarray | None
    rt_projected: np.ndarray | None


def _quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(4, dtype=float)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    mat = np.eye(4, dtype=float)
    mat[0, 0] = 1.0 - (yy + zz)
    mat[0, 1] = xy - wz
    mat[0, 2] = xz + wy
    mat[1, 0] = xy + wz
    mat[1, 1] = 1.0 - (xx + zz)
    mat[1, 2] = yz - wx
    mat[2, 0] = xz - wy
    mat[2, 1] = yz + wx
    mat[2, 2] = 1.0 - (xx + yy)
    return mat


def _transform_to_matrix(transform_msg) -> np.ndarray:
    translation = transform_msg.translation
    rotation = transform_msg.rotation
    matrix = _quaternion_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
    matrix[:3, 3] = [float(translation.x), float(translation.y), float(translation.z)]
    return matrix


def _ensure_pose3(pose: list[float] | tuple[float, ...] | None) -> list[float] | None:
    if pose is None or len(pose) < 2:
        return None
    z_val = pose[2] if len(pose) >= 3 else 0.0
    return [float(pose[0]), float(pose[1]), float(z_val)]


def _category_color(category: str) -> tuple[int, int, int]:
    bucket = (sum(ord(ch) for ch in str(category)) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(bucket, 0.55, 0.92)
    return int(r * 255), int(g * 255), int(b * 255)


def _coerce_pose_xy(data: dict[str, Any]) -> tuple[float, float] | None:
    pose = data.get("pose")
    if not isinstance(pose, (list, tuple)) or len(pose) < 2:
        return None
    return float(pose[0]), float(pose[1])


def _resize_image(image: np.ndarray, *, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    if scale >= 1.0:
        return image
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def map_preview_point_to_original(
    preview_shape: tuple[int, int] | tuple[int, int, int],
    original_shape: tuple[int, int] | tuple[int, int, int],
    x_pixel: int,
    y_pixel: int,
) -> tuple[int, int]:
    """Map a click on the preview image back to the original RGB resolution."""
    preview_h, preview_w = int(preview_shape[0]), int(preview_shape[1])
    orig_h, orig_w = int(original_shape[0]), int(original_shape[1])
    if preview_w <= 0 or preview_h <= 0 or orig_w <= 0 or orig_h <= 0:
        return 0, 0
    x_clipped = max(0, min(int(x_pixel), preview_w - 1))
    y_clipped = max(0, min(int(y_pixel), preview_h - 1))
    orig_x = int(round(x_clipped * (orig_w / preview_w)))
    orig_y = int(round(y_clipped * (orig_h / preview_h)))
    return min(orig_w - 1, max(0, orig_x)), min(orig_h - 1, max(0, orig_y))


def _overlay_click_mask(image_rgb: np.ndarray, mask: np.ndarray | None, point_xy: tuple[int, int]) -> np.ndarray:
    overlay = image_rgb.copy()
    if mask is not None and np.any(mask):
        mask_bool = np.asarray(mask, dtype=bool)
        tinted = overlay.copy()
        tinted[mask_bool] = (0.35 * tinted[mask_bool] + 0.65 * np.array([41, 121, 255])).astype(np.uint8)
        overlay = tinted
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 240, 0), 3)
    cv2.circle(overlay, (int(point_xy[0]), int(point_xy[1])), 9, (255, 70, 70), -1)
    cv2.circle(overlay, (int(point_xy[0]), int(point_xy[1])), 13, (255, 255, 255), 2)
    return overlay


def _parse_pointcloud2_all_fields(cloud_msg) -> np.ndarray | None:
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
    for field in cloud_msg.fields:
        np_dtype = type_map.get(field.datatype)
        if np_dtype is None:
            continue
        count = field.count if field.count > 0 else 1
        dtype_entry = np_dtype if count == 1 else np.dtype((np_dtype, count))
        names.append(field.name)
        formats.append(dtype_entry)
        offsets.append(field.offset)
    if not names:
        return None
    dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": cloud_msg.point_step}
    )
    cloud_array = np.frombuffer(cloud_msg.data, dtype=dtype)
    if cloud_msg.is_bigendian:
        cloud_array = cloud_array.byteswap().newbyteorder()
    return cloud_array


class RosbagRgbIndex:
    """Sequential rosbag index plus random-access RGB frame loading."""

    def __init__(
        self,
        bag_path: str | Path,
        *,
        rgb_topic: str = DEFAULT_RGB_TOPIC,
        depth_topic: str = DEFAULT_DEPTH_TOPIC,
        camera_info_topic: str = DEFAULT_CAMERA_INFO_TOPIC,
        tf_topic: str = DEFAULT_TF_TOPIC,
        tf_static_topic: str = DEFAULT_TF_STATIC_TOPIC,
        projected_lidar_topic: str = DEFAULT_PROJECTED_LIDAR_TOPIC,
        projected_lidar_frame: str = DEFAULT_PROJECTED_LIDAR_FRAME,
        camera_frame: str = DEFAULT_CAMERA_FRAME,
        synchronizer_slop_sec: float = DEFAULT_SYNC_SLOP_SEC,
        world_frame: str = DEFAULT_WORLD_FRAME,
        base_frame: str = DEFAULT_BASE_FRAME,
        storage_id: str = "sqlite3",
    ) -> None:
        self.bag_path = Path(bag_path).expanduser()
        self.rgb_topic = str(rgb_topic)
        self.depth_topic = str(depth_topic or "")
        self.camera_info_topic = str(camera_info_topic)
        self.tf_topic = str(tf_topic)
        self.tf_static_topic = str(tf_static_topic)
        self.projected_lidar_topic = str(projected_lidar_topic or "")
        self.projected_lidar_frame = str(projected_lidar_frame or "")
        self.camera_frame = str(camera_frame or "")
        self.world_frame = str(world_frame)
        self.base_frame = str(base_frame)
        self.storage_id = str(storage_id)
        self.synchronizer_slop_sec = max(0.0, float(synchronizer_slop_sec))
        self._sync_slop_ns = int(self.synchronizer_slop_sec * 1e9)
        self._bridge = CvBridge()
        self._frames: list[FrameRecord] = []
        self._message_type_cache: dict[str, Any] = {}
        self._dynamic_transforms: dict[str, TransformRecord] = {}
        self._static_transforms: dict[str, TransformRecord] = {}
        self._db_path = self._resolve_db_path()
        self._thread_local = threading.local()
        self._cache_lock = threading.Lock()
        self._conn = self._open_connection()
        self._rgb_rows, self._rgb_timestamps = self._load_topic_rows(self.rgb_topic)
        self._depth_rows, self._depth_timestamps = self._load_topic_rows(self.depth_topic)
        self._projected_rows, self._projected_timestamps = self._load_topic_rows(self.projected_lidar_topic)
        self._rgb_message_type = None
        self._depth_message_type = None
        self._projected_message_type = None
        self.image_width: int | None = None
        self.image_height: int | None = None
        self.intrinsics: dict[str, float] | None = None
        self._frame_bundle_cache: dict[int, AnnotationFrameBundle] = {}
        self._build_index()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _get_thread_connection(self) -> sqlite3.Connection:
        conn = getattr(self._thread_local, "conn", None)
        if conn is None:
            conn = self._open_connection()
            self._thread_local.conn = conn
        return conn

    def _resolve_db_path(self) -> Path:
        if self.bag_path.is_file():
            return self.bag_path
        candidates = sorted(self.bag_path.glob("*.db3"))
        if not candidates:
            raise FileNotFoundError(f"No .db3 file found under rosbag path: {self.bag_path}")
        return candidates[0]

    def _load_topic_rows(self, topic_name: str) -> tuple[list[tuple[int, int]], list[int]]:
        if not topic_name:
            return [], []
        cursor = self._conn.execute(
            """
            SELECT m.id, m.timestamp
            FROM messages AS m
            JOIN topics AS t ON t.id = m.topic_id
            WHERE t.name = ?
            ORDER BY m.timestamp, m.id
            """,
            (topic_name,),
        )
        rows = [(int(row["id"]), int(row["timestamp"])) for row in cursor.fetchall()]
        return rows, [timestamp for _, timestamp in rows]

    @property
    def frames(self) -> list[FrameRecord]:
        return self._frames

    @property
    def trajectory(self) -> list[tuple[float, float]]:
        coords = []
        for frame in self._frames:
            if frame.robot_pose is not None:
                coords.append((frame.robot_pose[0], frame.robot_pose[1]))
        return coords

    def _build_index(self) -> None:
        if self.storage_id != "sqlite3":
            raise RuntimeError(
                f"GT annotation backend currently supports sqlite3 rosbag storage only, got '{self.storage_id}'"
            )
        topic_types = {
            str(row["name"]): str(row["type"])
            for row in self._conn.execute("SELECT name, type FROM topics")
        }

        required_topics = [self.rgb_topic, self.tf_topic, self.tf_static_topic]
        optional_topics = [self.camera_info_topic, self.depth_topic, self.projected_lidar_topic]
        for topic in required_topics + optional_topics:
            if not topic:
                continue
            type_name = topic_types.get(topic)
            if type_name is None:
                if topic in required_topics:
                    raise RuntimeError(f"Required topic '{topic}' not found in rosbag {self.bag_path}")
                continue
            self._message_type_cache[topic] = get_message(type_name)

        self._rgb_message_type = self._message_type_cache[self.rgb_topic]
        self._depth_message_type = self._message_type_cache.get(self.depth_topic)
        self._projected_message_type = self._message_type_cache.get(self.projected_lidar_topic)

        aux_rows = list(
            self._conn.execute(
                """
                SELECT t.name AS topic_name, m.timestamp, m.data
                FROM messages AS m
                JOIN topics AS t ON t.id = m.topic_id
                WHERE t.name IN (?, ?, ?)
                ORDER BY m.timestamp, m.id
                """,
                (self.tf_topic, self.tf_static_topic, self.camera_info_topic),
            )
        )

        aux_index = 0
        rgb_index = 0
        while aux_index < len(aux_rows) or rgb_index < len(self._rgb_rows):
            next_aux_ts = int(aux_rows[aux_index]["timestamp"]) if aux_index < len(aux_rows) else None
            next_rgb_ts = int(self._rgb_rows[rgb_index][1]) if rgb_index < len(self._rgb_rows) else None

            if next_rgb_ts is not None and (next_aux_ts is None or next_rgb_ts <= next_aux_ts):
                message_id, timestamp_ns = self._rgb_rows[rgb_index]
                rgb_index += 1
                rt_camera = self._resolve_transform_matrix(self.base_frame, self.camera_frame) if self.camera_frame else None
                rt_base = self._resolve_transform_matrix(self.world_frame, self.base_frame)
                rt_projected = (
                    self._resolve_transform_matrix(self.base_frame, self.projected_lidar_frame)
                    if self.projected_lidar_frame
                    else None
                )
                robot_pose = None
                if rt_base is not None:
                    x, y, z = rt_base[:3, 3]
                    robot_pose = (float(x), float(y), float(z))
                self._frames.append(
                    FrameRecord(
                        frame_index=len(self._frames),
                        message_id=int(message_id),
                        timestamp_ns=int(timestamp_ns),
                        robot_pose=robot_pose,
                        depth_message_id=self._nearest_message_id(
                            self._depth_rows, self._depth_timestamps, timestamp_ns
                        ),
                        projected_cloud_message_id=self._nearest_message_id(
                            self._projected_rows, self._projected_timestamps, timestamp_ns
                        ),
                        rt_camera=rt_camera.copy() if rt_camera is not None else None,
                        rt_base=rt_base.copy() if rt_base is not None else None,
                        rt_projected=rt_projected.copy() if rt_projected is not None else None,
                    )
                )
                continue

            row = aux_rows[aux_index]
            aux_index += 1
            topic = str(row["topic_name"])
            msg_type = self._message_type_cache.get(topic)
            if msg_type is None:
                continue
            msg = deserialize_message(row["data"], msg_type)
            if topic == self.tf_static_topic:
                self._ingest_tf_message(msg, is_static=True)
            elif topic == self.tf_topic:
                self._ingest_tf_message(msg, is_static=False)
            elif topic == self.camera_info_topic and getattr(msg, "width", None):
                self.image_width = int(msg.width)
                self.image_height = int(msg.height)
                k = list(getattr(msg, "k", []))
                if len(k) >= 9:
                    self.intrinsics = {
                        "fx": float(k[0]),
                        "fy": float(k[4]),
                        "px": float(k[2]),
                        "py": float(k[5]),
                    }

        if not self._frames:
            raise RuntimeError(f"No RGB frames found on topic '{self.rgb_topic}'")

    def _ingest_tf_message(self, msg, *, is_static: bool) -> None:
        for transform in getattr(msg, "transforms", []):
            record = TransformRecord(
                parent_frame=str(transform.header.frame_id),
                child_frame=str(transform.child_frame_id),
                matrix=_transform_to_matrix(transform.transform),
                stamp_ns=int(transform.header.stamp.sec) * 1_000_000_000 + int(transform.header.stamp.nanosec),
                is_static=bool(is_static),
            )
            if is_static:
                self._static_transforms[record.child_frame] = record
            else:
                self._dynamic_transforms[record.child_frame] = record

    def _resolve_transform_matrix(self, target_frame: str, source_frame: str) -> np.ndarray | None:
        if not target_frame or not source_frame:
            return None
        if target_frame == source_frame:
            return np.eye(4, dtype=float)
        chain: list[np.ndarray] = []
        visited: set[str] = set()
        current = str(source_frame)
        while current != target_frame:
            if current in visited:
                return None
            visited.add(current)
            transform = self._dynamic_transforms.get(current) or self._static_transforms.get(current)
            if transform is None:
                return None
            chain.append(transform.matrix)
            current = transform.parent_frame
        matrix = np.eye(4, dtype=float)
        for edge in reversed(chain):
            matrix = matrix @ edge
        return matrix

    def _nearest_message_id(
        self,
        rows: list[tuple[int, int]],
        timestamps: list[int],
        target_timestamp_ns: int,
    ) -> int | None:
        if not rows:
            return None
        index = bisect_left(timestamps, int(target_timestamp_ns))
        candidates: list[tuple[int, int]] = []
        if index < len(rows):
            candidates.append(rows[index])
        if index > 0:
            candidates.append(rows[index - 1])
        best_id = None
        best_delta = None
        for message_id, timestamp_ns in candidates:
            delta = abs(int(timestamp_ns) - int(target_timestamp_ns))
            if delta > self._sync_slop_ns:
                continue
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_id = int(message_id)
        return best_id

    def _load_message(self, message_id: int, msg_type) -> Any:
        conn = self._get_thread_connection()
        cursor = conn.execute("SELECT data FROM messages WHERE id = ?", (int(message_id),))
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Message id {message_id} not found")
        return deserialize_message(row["data"], msg_type)

    def _decode_rgb(self, message_id: int) -> np.ndarray:
        msg = self._load_message(message_id, self._rgb_message_type)
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        return np.asarray(image)

    def _decode_depth(self, message_id: int | None) -> np.ndarray | None:
        if message_id is None or self._depth_message_type is None:
            return None
        msg = self._load_message(message_id, self._depth_message_type)
        depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(depth)
        if msg.encoding == "16UC1":
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32, copy=False)
        depth[np.isnan(depth)] = 0.0
        return depth

    def _decode_projected_cloud(self, message_id: int | None) -> np.ndarray | None:
        if message_id is None or self._projected_message_type is None:
            return None
        msg = self._load_message(message_id, self._projected_message_type)
        return _parse_pointcloud2_all_fields(msg)

    def get_frame_bundle(
        self,
        frame_index: int,
        *,
        max_width: int = RGB_PREVIEW_MAX_WIDTH,
        max_height: int = RGB_PREVIEW_MAX_HEIGHT,
    ) -> AnnotationFrameBundle:
        frame_index = int(frame_index)
        with self._cache_lock:
            cached = self._frame_bundle_cache.get(frame_index)
        if cached is not None:
            return cached

        frame = self._frames[frame_index]
        original_rgb = self._decode_rgb(frame.message_id)
        preview_rgb = _resize_image(original_rgb, max_width=max_width, max_height=max_height)
        bundle = AnnotationFrameBundle(
            frame_index=frame_index,
            timestamp_ns=int(frame.timestamp_ns),
            original_rgb=original_rgb,
            preview_rgb=preview_rgb,
            depth=self._decode_depth(frame.depth_message_id),
            projected_cloud=self._decode_projected_cloud(frame.projected_cloud_message_id),
            intrinsics=copy.deepcopy(self.intrinsics) if self.intrinsics is not None else None,
            rt_camera=frame.rt_camera.copy() if frame.rt_camera is not None else None,
            rt_base=frame.rt_base.copy() if frame.rt_base is not None else None,
            rt_projected=frame.rt_projected.copy() if frame.rt_projected is not None else None,
        )
        with self._cache_lock:
            self._frame_bundle_cache[frame_index] = bundle
            if len(self._frame_bundle_cache) > 12:
                oldest = sorted(self._frame_bundle_cache)[:4]
                for key in oldest:
                    self._frame_bundle_cache.pop(key, None)
        return bundle

    def get_frame_image(
        self,
        frame_index: int,
        *,
        max_width: int = RGB_PREVIEW_MAX_WIDTH,
        max_height: int = RGB_PREVIEW_MAX_HEIGHT,
    ) -> np.ndarray:
        bundle = self.get_frame_bundle(frame_index, max_width=max_width, max_height=max_height)
        return bundle.preview_rgb

    def preview_to_original_point(self, frame_index: int, x_pixel: int, y_pixel: int) -> tuple[int, int]:
        bundle = self.get_frame_bundle(frame_index)
        return map_preview_point_to_original(
            bundle.preview_rgb.shape,
            bundle.original_rgb.shape,
            int(x_pixel),
            int(y_pixel),
        )

    def nearest_frame_index(self, x: float, y: float) -> int:
        best_index = 0
        best_distance = math.inf
        for frame in self._frames:
            if frame.robot_pose is None:
                continue
            dx = float(frame.robot_pose[0]) - float(x)
            dy = float(frame.robot_pose[1]) - float(y)
            distance = dx * dx + dy * dy
            if distance < best_distance:
                best_distance = distance
                best_index = frame.frame_index
        return int(best_index)


class AnnotationSession:
    """Editable session state for semantic-object GT authoring."""

    def __init__(
        self,
        *,
        input_json: str | Path,
        output_json: str | Path,
        rosbag_index: RosbagRgbIndex,
        sam_predictor: Any | None = None,
        lidar_voxel_size: float = DEFAULT_LIDAR_VOXEL_SIZE,
    ) -> None:
        self.input_json = Path(input_json).expanduser()
        self.output_json = Path(output_json).expanduser()
        self.rosbag_index = rosbag_index
        self._sam_predictor = sam_predictor
        self.lidar_voxel_size = max(float(lidar_voxel_size), 1e-6)
        self._rgb_preview_override: np.ndarray | None = None

        payload = read_stcm_json(str(self.input_json))
        self.semantic_graph = payload["semantic_graph"]
        self.place_graph = payload["place_graph"]
        self.metadata = copy.deepcopy(payload.get("metadata") or {})
        self.semantic_edges = [
            (str(node_a), str(node_b), dict(data or {}))
            for node_a, node_b, data in self.semantic_graph.edges(data=True)
        ]
        self.objects: list[dict[str, Any]] = []
        for node_id, data in sorted(self.semantic_graph.nodes(data=True), key=lambda item: str(item[0])):
            entry = copy.deepcopy(dict(data or {}))
            entry["id"] = str(entry.get("id", node_id))
            pose = _ensure_pose3(entry.get("pose"))
            if pose is None:
                continue
            entry["pose"] = pose
            entry["category"] = str(entry.get("category", entry.get("label", "object")))
            self.objects.append(entry)

        self.current_frame_index = 0
        self.selected_object_id: str | None = self.objects[0]["id"] if self.objects else None
        if self.selected_object_id is not None:
            selected = self.get_selected_object()
            if selected is not None:
                self.current_frame_index = self.rosbag_index.nearest_frame_index(
                    selected["pose"][0], selected["pose"][1]
                )

    def _clear_rgb_preview_override(self) -> None:
        self._rgb_preview_override = None

    def _load_sam_predictor(self):
        if self._sam_predictor is None:
            from .core.perception import SegmentAnythingPredictor

            self._sam_predictor = SegmentAnythingPredictor()
        return self._sam_predictor

    def build_viewport(self, width: int = MAP_IMAGE_SIZE[0], height: int = MAP_IMAGE_SIZE[1]) -> MapViewport:
        xs: list[float] = []
        ys: list[float] = []
        for obj in self.objects:
            pose = _ensure_pose3(obj.get("pose"))
            if pose:
                xs.append(pose[0])
                ys.append(pose[1])
        for _, data in self.place_graph.nodes(data=True):
            xy = _coerce_pose_xy(data)
            if xy:
                xs.append(xy[0])
                ys.append(xy[1])
        for x, y in self.rosbag_index.trajectory:
            xs.append(float(x))
            ys.append(float(y))
        xs.extend([0.0])
        ys.extend([0.0])

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        pad_x = max(1.0, 0.12 * max(max_x - min_x, 1.0))
        pad_y = max(1.0, 0.12 * max(max_y - min_y, 1.0))
        return MapViewport(
            min_x=min_x - pad_x,
            max_x=max_x + pad_x,
            min_y=min_y - pad_y,
            max_y=max_y + pad_y,
            width=int(width),
            height=int(height),
        )

    def get_selected_object(self) -> dict[str, Any] | None:
        if self.selected_object_id is None:
            return None
        for obj in self.objects:
            if str(obj["id"]) == str(self.selected_object_id):
                return obj
        return None

    def list_categories(self) -> list[str]:
        categories = sorted({str(obj.get("category", "object")) for obj in self.objects})
        return categories

    def build_object_table(
        self,
        *,
        search_text: str = "",
        category_filter: str = "All",
        sort_field: str = "id",
        descending: bool = False,
    ) -> tuple[list[list[Any]], list[str]]:
        search = str(search_text or "").strip().lower()
        rows: list[tuple[str, list[Any]]] = []
        for obj in self.objects:
            obj_id = str(obj["id"])
            category = str(obj.get("category", "object"))
            if category_filter not in ("", "All") and category != category_filter:
                continue
            haystack = f"{obj_id} {category}".lower()
            if search and search not in haystack:
                continue
            pose = _ensure_pose3(obj.get("pose")) or [0.0, 0.0, 0.0]
            rows.append(
                (
                    obj_id,
                    [
                        obj_id,
                        category,
                        round(float(pose[0]), 3),
                        round(float(pose[1]), 3),
                        round(float(pose[2]), 3),
                    ],
                )
            )

        key_index = {"id": 0, "category": 1, "x": 2, "y": 3, "z": 4}.get(sort_field, 0)
        rows.sort(key=lambda item: item[1][key_index], reverse=bool(descending))
        return [item[1] for item in rows], [item[0] for item in rows]

    def render_map(
        self,
        *,
        search_text: str = "",
        category_filter: str = "All",
        viewport: MapViewport | None = None,
    ) -> np.ndarray:
        viewport = viewport or self.build_viewport()
        image = Image.new("RGB", (viewport.width, viewport.height), (247, 248, 250))
        draw = ImageDraw.Draw(image)

        self._draw_grid(draw, viewport)
        self._draw_trajectory(draw, viewport)
        self._draw_place_graph(draw, viewport)
        self._draw_semantic_objects(
            draw,
            viewport,
            search_text=search_text,
            category_filter=category_filter,
        )
        self._draw_robot_marker(draw, viewport)
        self._draw_origin(draw, viewport)
        self._draw_header(draw, viewport)
        return np.asarray(image)

    def _draw_grid(self, draw: ImageDraw.ImageDraw, viewport: MapViewport) -> None:
        span_x = max(viewport.max_x - viewport.min_x, 1.0)
        span_y = max(viewport.max_y - viewport.min_y, 1.0)
        step = max(1.0, round(max(span_x, span_y) / 10.0))
        for value in np.arange(math.floor(viewport.min_x), math.ceil(viewport.max_x) + step, step):
            x0, y0 = viewport.world_to_pixel(float(value), viewport.min_y)
            x1, y1 = viewport.world_to_pixel(float(value), viewport.max_y)
            draw.line((x0, y0, x1, y1), fill=(228, 231, 236), width=1)
        for value in np.arange(math.floor(viewport.min_y), math.ceil(viewport.max_y) + step, step):
            x0, y0 = viewport.world_to_pixel(viewport.min_x, float(value))
            x1, y1 = viewport.world_to_pixel(viewport.max_x, float(value))
            draw.line((x0, y0, x1, y1), fill=(228, 231, 236), width=1)

    def _draw_trajectory(self, draw: ImageDraw.ImageDraw, viewport: MapViewport) -> None:
        points = [viewport.world_to_pixel(x, y) for x, y in self.rosbag_index.trajectory]
        if len(points) >= 2:
            draw.line(points, fill=(122, 127, 133), width=3)

    def _draw_place_graph(self, draw: ImageDraw.ImageDraw, viewport: MapViewport) -> None:
        positions: dict[str, tuple[int, int]] = {}
        for node_id, data in self.place_graph.nodes(data=True):
            xy = _coerce_pose_xy(data)
            if xy is None:
                continue
            positions[str(node_id)] = viewport.world_to_pixel(xy[0], xy[1])
        for node_a, node_b in self.place_graph.edges():
            pos_a = positions.get(str(node_a))
            pos_b = positions.get(str(node_b))
            if pos_a and pos_b:
                draw.line((pos_a[0], pos_a[1], pos_b[0], pos_b[1]), fill=(15, 118, 110), width=2)
        for pos in positions.values():
            radius = 4
            draw.ellipse(
                (pos[0] - radius, pos[1] - radius, pos[0] + radius, pos[1] + radius),
                fill=(15, 118, 110),
                outline=(255, 255, 255),
                width=1,
            )

    def _draw_semantic_objects(
        self,
        draw: ImageDraw.ImageDraw,
        viewport: MapViewport,
        *,
        search_text: str,
        category_filter: str,
    ) -> None:
        search = str(search_text or "").strip().lower()
        for obj in self.objects:
            obj_id = str(obj["id"])
            category = str(obj.get("category", "object"))
            if category_filter not in ("", "All") and category != category_filter:
                continue
            if search and search not in f"{obj_id} {category}".lower():
                continue
            pose = _ensure_pose3(obj.get("pose"))
            if pose is None:
                continue
            px, py = viewport.world_to_pixel(pose[0], pose[1])
            color = _category_color(category)
            size = 9
            bbox = (px - size, py - size, px + size, py + size)
            draw.rectangle(bbox, fill=color, outline=(18, 18, 18), width=2)
            if obj_id == self.selected_object_id:
                outer = (px - size - 4, py - size - 4, px + size + 4, py + size + 4)
                draw.rectangle(outer, outline=(255, 214, 10), width=3)
                label = f"{obj_id} ({category})"
                draw.text((px + 14, py - 12), label, fill=(27, 31, 35))

    def _draw_robot_marker(self, draw: ImageDraw.ImageDraw, viewport: MapViewport) -> None:
        if not self.rosbag_index.frames:
            return
        frame = self.rosbag_index.frames[self.current_frame_index]
        if frame.robot_pose is None:
            return
        px, py = viewport.world_to_pixel(frame.robot_pose[0], frame.robot_pose[1])
        radius = 8
        draw.ellipse(
            (px - radius, py - radius, px + radius, py + radius),
            fill=(220, 38, 38),
            outline=(255, 255, 255),
            width=2,
        )
        draw.text((px + 12, py + 10), f"frame {frame.frame_index}", fill=(145, 0, 0))

    def _draw_origin(self, draw: ImageDraw.ImageDraw, viewport: MapViewport) -> None:
        px, py = viewport.world_to_pixel(0.0, 0.0)
        draw.line((px - 8, py, px + 8, py), fill=(0, 0, 0), width=2)
        draw.line((px, py - 8, px, py + 8), fill=(0, 0, 0), width=2)
        draw.text((px + 10, py + 6), "origin", fill=(30, 30, 30))

    def _draw_header(self, draw: ImageDraw.ImageDraw, viewport: MapViewport) -> None:
        total_frames = len(self.rosbag_index.frames)
        draw.rounded_rectangle((14, 12, 322, 70), radius=12, fill=(255, 255, 255), outline=(220, 223, 228))
        draw.text((26, 22), f"Objects: {len(self.objects)}", fill=(27, 31, 35))
        draw.text((26, 42), f"Frames: {total_frames} | Places: {self.place_graph.number_of_nodes()}", fill=(27, 31, 35))
        frame = self.rosbag_index.frames[self.current_frame_index]
        draw.text((190, 22), f"Selected: {self.selected_object_id or 'none'}", fill=(27, 31, 35))
        draw.text((190, 42), f"t={frame.timestamp_ns / 1e9:.2f}s", fill=(27, 31, 35))

    def selected_object_fields(self) -> tuple[str, str, float, float, float]:
        obj = self.get_selected_object()
        if obj is None:
            return "", "", 0.0, 0.0, 0.0
        pose = _ensure_pose3(obj.get("pose")) or [0.0, 0.0, 0.0]
        return (
            str(obj["id"]),
            str(obj.get("category", "object")),
            float(pose[0]),
            float(pose[1]),
            float(pose[2]),
        )

    def selected_frame_image(self) -> np.ndarray:
        if self._rgb_preview_override is not None:
            return self._rgb_preview_override
        return self.rosbag_index.get_frame_image(self.current_frame_index)

    def set_frame_index(self, frame_index: int) -> str:
        self._clear_rgb_preview_override()
        frame_index = max(0, min(int(frame_index), len(self.rosbag_index.frames) - 1))
        self.current_frame_index = frame_index
        frame = self.rosbag_index.frames[frame_index]
        if frame.robot_pose is None:
            return f"Frame {frame_index}: RGB loaded, robot pose unavailable."
        return (
            f"Frame {frame_index}: robot at "
            f"({frame.robot_pose[0]:.2f}, {frame.robot_pose[1]:.2f}, {frame.robot_pose[2]:.2f}) m."
        )

    def select_object(self, object_id: str) -> str:
        self._clear_rgb_preview_override()
        self.selected_object_id = str(object_id)
        obj = self.get_selected_object()
        if obj is None:
            return f"Object '{object_id}' not found."
        pose = _ensure_pose3(obj.get("pose")) or [0.0, 0.0, 0.0]
        self.current_frame_index = self.rosbag_index.nearest_frame_index(pose[0], pose[1])
        return (
            f"Selected '{obj['id']}' ({obj.get('category', 'object')}) at "
            f"({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}) m."
        )

    def select_table_row(self, row_index: int, table_ids: list[str]) -> str:
        if row_index < 0 or row_index >= len(table_ids):
            return "No table row selected."
        return self.select_object(table_ids[row_index])

    def map_click(
        self,
        x_pixel: int,
        y_pixel: int,
        *,
        mode: str,
        category_hint: str = "object",
        viewport: MapViewport | None = None,
    ) -> str:
        self._clear_rgb_preview_override()
        viewport = viewport or self.build_viewport()
        world_x, world_y = viewport.pixel_to_world(int(x_pixel), int(y_pixel))
        mode = str(mode or "inspect")
        if mode == "add":
            new_id = self._generate_unique_id(category_hint or "object")
            self.objects.append(
                {
                    "id": new_id,
                    "category": str(category_hint or "object"),
                    "pose": [float(world_x), float(world_y), 0.0],
                }
            )
            self.selected_object_id = new_id
            self.current_frame_index = self.rosbag_index.nearest_frame_index(world_x, world_y)
            return f"Added '{new_id}' at ({world_x:.2f}, {world_y:.2f}, 0.00) m."
        if mode == "move":
            obj = self.get_selected_object()
            if obj is None:
                return "Select an object before moving it."
            existing_pose = _ensure_pose3(obj.get("pose")) or [0.0, 0.0, 0.0]
            obj["pose"] = [float(world_x), float(world_y), float(existing_pose[2])]
            self.current_frame_index = self.rosbag_index.nearest_frame_index(world_x, world_y)
            return f"Moved '{obj['id']}' to ({world_x:.2f}, {world_y:.2f}, {obj['pose'][2]:.2f}) m."
        nearest_id = self._find_object_near_pixel(x_pixel, y_pixel, viewport=viewport)
        if nearest_id is not None:
            return self.select_object(nearest_id)
        return f"Map at ({world_x:.2f}, {world_y:.2f}) m. No nearby object selected."

    def rgb_click_add(self, x_pixel: int, y_pixel: int, *, category_hint: str = "object") -> str:
        category_hint = str(category_hint or "").strip()
        if not category_hint:
            return "RGB click add failed: object category is required."

        bundle = self.rosbag_index.get_frame_bundle(self.current_frame_index)
        orig_x, orig_y = self.rosbag_index.preview_to_original_point(self.current_frame_index, x_pixel, y_pixel)

        predictor = self._load_sam_predictor()
        mask, sam_score = predictor.predict_point(bundle.original_rgb, (orig_x, orig_y))
        if mask is None or not np.any(mask):
            point_overlay = _overlay_click_mask(bundle.original_rgb, None, (orig_x, orig_y))
            self._rgb_preview_override = _resize_image(
                point_overlay,
                max_width=RGB_PREVIEW_MAX_WIDTH,
                max_height=RGB_PREVIEW_MAX_HEIGHT,
            )
            return (
                f"RGB click add failed: SAM did not produce a mask at "
                f"preview=({int(x_pixel)}, {int(y_pixel)}) original=({orig_x}, {orig_y})."
            )

        mask = np.asarray(mask, dtype=np.uint8)
        overlay_rgb = _overlay_click_mask(bundle.original_rgb, mask, (orig_x, orig_y))
        self._rgb_preview_override = _resize_image(
            overlay_rgb,
            max_width=RGB_PREVIEW_MAX_WIDTH,
            max_height=RGB_PREVIEW_MAX_HEIGHT,
        )

        pose_result = None
        pose_source = ""
        if bundle.projected_cloud is not None and bundle.rt_projected is not None and bundle.rt_base is not None:
            pose_result = pose_in_map_frame_from_projected_mode(
                bundle.projected_cloud,
                bundle.rt_projected,
                bundle.rt_base,
                segment=mask,
                rt_camera=bundle.rt_camera,
                voxel_size=self.lidar_voxel_size,
            )
            if pose_result is not None:
                pose_source = "projected_lidar"

        pose = None
        if pose_result is not None:
            pose = pose_result["pose"]
        elif bundle.depth is not None and bundle.rt_camera is not None and bundle.rt_base is not None:
            pose = pose_in_map_frame(
                bundle.rt_camera,
                bundle.rt_base,
                bundle.depth.copy(),
                segment=mask.astype(np.float32),
                intrinsics=bundle.intrinsics,
            )
            if pose is not None:
                pose_source = "depth_fallback"

        if pose is None:
            return (
                f"RGB click add failed: no valid 3D pose for "
                f"preview=({int(x_pixel)}, {int(y_pixel)}) original=({orig_x}, {orig_y})."
            )

        new_id = self._generate_unique_id(category_hint)
        self.objects.append(
            {
                "id": new_id,
                "category": category_hint,
                "pose": [float(pose[0]), float(pose[1]), float(pose[2])],
            }
        )
        self.selected_object_id = new_id

        score_text = f"SAM={sam_score:.3f}" if sam_score is not None else "SAM=n/a"
        if pose_result is not None:
            return (
                f"Added '{new_id}' from RGB click preview=({int(x_pixel)}, {int(y_pixel)}) "
                f"original=({orig_x}, {orig_y}) -> "
                f"({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}) m via {pose_source} "
                f"(voxel_count={pose_result['voxel_count']}, point_count={pose_result['point_count']}, "
                f"voxel={pose_result['voxel_index']}, {score_text})."
            )
        return (
            f"Added '{new_id}' from RGB click preview=({int(x_pixel)}, {int(y_pixel)}) "
            f"original=({orig_x}, {orig_y}) -> "
            f"({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}) m via {pose_source} ({score_text})."
        )

    def _find_object_near_pixel(
        self,
        x_pixel: int,
        y_pixel: int,
        *,
        viewport: MapViewport,
        max_distance_px: float = 22.0,
    ) -> str | None:
        best_id = None
        best_distance = float(max_distance_px)
        for obj in self.objects:
            pose = _ensure_pose3(obj.get("pose"))
            if pose is None:
                continue
            px, py = viewport.world_to_pixel(pose[0], pose[1])
            distance = math.hypot(px - int(x_pixel), py - int(y_pixel))
            if distance <= best_distance:
                best_distance = distance
                best_id = str(obj["id"])
        return best_id

    def update_selected_object(
        self,
        *,
        new_id: str,
        category: str,
        x: float,
        y: float,
        z: float,
    ) -> str:
        self._clear_rgb_preview_override()
        obj = self.get_selected_object()
        if obj is None:
            return "No object selected."
        new_id = str(new_id or "").strip()
        category = str(category or "").strip()
        if not new_id:
            return "Object id is required."
        if not category:
            return "Object category is required."
        if new_id != obj["id"] and any(str(other["id"]) == new_id for other in self.objects):
            return f"Object id '{new_id}' already exists."
        old_id = str(obj["id"])
        obj["id"] = new_id
        obj["category"] = category
        obj["pose"] = [float(x), float(y), float(z)]
        self.selected_object_id = new_id
        if old_id != new_id:
            updated_edges = []
            for src, dst, data in self.semantic_edges:
                src = new_id if src == old_id else src
                dst = new_id if dst == old_id else dst
                updated_edges.append((src, dst, data))
            self.semantic_edges = updated_edges
        self.current_frame_index = self.rosbag_index.nearest_frame_index(float(x), float(y))
        return f"Updated '{new_id}'."

    def delete_selected_object(self, *, confirmed: bool) -> str:
        self._clear_rgb_preview_override()
        obj = self.get_selected_object()
        if obj is None:
            return "No object selected."
        if not confirmed:
            return "Enable delete confirmation before removing the selected object."
        doomed_id = str(obj["id"])
        self.objects = [item for item in self.objects if str(item["id"]) != doomed_id]
        self.semantic_edges = [
            (src, dst, data)
            for src, dst, data in self.semantic_edges
            if src != doomed_id and dst != doomed_id
        ]
        self.selected_object_id = self.objects[0]["id"] if self.objects else None
        return f"Deleted '{doomed_id}'."

    def _generate_unique_id(self, category: str) -> str:
        slug = str(category or "object").strip() or "object"
        existing = {str(obj["id"]) for obj in self.objects}
        counter = 1
        while True:
            candidate = f"{slug}_{counter}_0"
            if candidate not in existing:
                return candidate
            counter += 1

    def save(self, output_path: str | Path | None = None) -> str:
        output_path = Path(output_path).expanduser() if output_path else self.output_json
        ids = [str(obj["id"]).strip() for obj in self.objects]
        if any(not obj_id for obj_id in ids):
            return "Save failed: every object must have a non-empty id."
        if len(set(ids)) != len(ids):
            return "Save failed: object ids must be unique."
        if any(not str(obj.get("category", "")).strip() for obj in self.objects):
            return "Save failed: every object must have a category."

        semantic_graph = nx.Graph()
        for obj in self.objects:
            node_data = copy.deepcopy(obj)
            node_id = str(node_data["id"])
            pose = _ensure_pose3(node_data.get("pose"))
            if pose is None:
                return f"Save failed: object '{node_id}' has an invalid pose."
            node_data["pose"] = pose
            node_data["id"] = node_id
            node_data["category"] = str(node_data.get("category", "object"))
            semantic_graph.add_node(node_id, **node_data)

        valid_ids = set(semantic_graph.nodes)
        for src, dst, data in self.semantic_edges:
            if src in valid_ids and dst in valid_ids and src != dst:
                semantic_graph.add_edge(src, dst, **dict(data or {}))

        payload = {
            "stcm_version": "1.0",
            "semantic_graph": json_graph.node_link_data(semantic_graph, edges="links"),
            "place_graph": json_graph.node_link_data(self.place_graph, edges="links")
            if self.place_graph is not None
            else {"directed": False, "multigraph": False, "graph": {}, "nodes": [], "links": []},
            "llm": _build_llm_summary(semantic_graph, self.place_graph),
            "metadata": copy.deepcopy(self.metadata),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            import json

            json.dump(payload, handle, indent=4)
            handle.write("\n")
        self.output_json = output_path
        return f"Saved draft to {output_path}"


def default_output_json_path(input_json: str | Path) -> Path:
    input_json = Path(input_json).expanduser()
    return input_json.with_name(f"{input_json.stem}_draft{input_json.suffix}")
