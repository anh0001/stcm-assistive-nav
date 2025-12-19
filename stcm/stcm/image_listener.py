"""ROS 2 RGB-D listener utilities."""

from __future__ import annotations

import threading
from typing import Optional

import message_filters
import numpy as np
import ros2_numpy as ros_numpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from .ros_utils import ros_qt_to_rt


class ImageListener:
    """Synchronizes RGB and depth streams while tracking TF transforms."""

    def __init__(
        self,
        node,
        rgb_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        base_frame: str,
        camera_frame: str,
        world_frame: str,
        queue_size: int = 5,
        slop_seconds: float = 0.1,
        use_projected_lidar: bool = False,
        projected_lidar_topic: str = "/lidar_points_projected",
        projected_lidar_frame: str | None = None,
        projected_lidar_timeout_sec: float = 2.0,
    ) -> None:
        self._node = node
        self._lock = threading.Lock()
        self._cv_bridge = CvBridge()

        self.im: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.rgb_frame_id: Optional[str] = None
        self.rgb_frame_stamp = None

        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self._depth_topic_name = depth_topic or "<depth topic unavailable>"
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        self._rgb_frame_override = camera_frame.strip() if camera_frame else None
        self.world_frame = world_frame
        self.use_projected_lidar = bool(use_projected_lidar)
        self.projected_lidar_topic = projected_lidar_topic
        self._projected_topic_name = projected_lidar_topic or "<projected lidar topic unavailable>"
        self.projected_lidar_frame = projected_lidar_frame or None
        self._cloud_field_warning_emitted = False
        self.projected_lidar_timeout_sec = max(0.0, float(projected_lidar_timeout_sec))
        self._projected_lidar_timeout_ns = int(self.projected_lidar_timeout_sec * 1e9) if self.projected_lidar_timeout_sec > 0.0 else 0
        self._lidar_depth_active = False
        self._cloud_missing_warned = False
        self._cloud_timeout_warned = False
        self._last_cloud_time: Optional[Time] = None

        self.intrinsics: Optional[np.ndarray] = None
        self.fx = None
        self.fy = None
        self.px = None
        self.py = None

        self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self._tf_listener = TransformListener(self._tf_buffer, node)

        self._camera_info_ready = threading.Event()
        self._camera_info_sub = node.create_subscription(
            CameraInfo, camera_info_topic, self._camera_info_callback, qos_profile_sensor_data
        )

        self._rgb_depth_sub = None
        self._rgb_cloud_sub = None
        self._depth_sub = None
        self._cloud_sub = None
        self._depth_synchronizer = None
        self._cloud_synchronizer = None

        if depth_topic:
            self._rgb_depth_sub = message_filters.Subscriber(
                node, Image, rgb_topic, qos_profile=qos_profile_sensor_data
            )
            self._depth_sub = message_filters.Subscriber(
                node, Image, depth_topic, qos_profile=qos_profile_sensor_data
            )
            self._depth_synchronizer = message_filters.ApproximateTimeSynchronizer(
                [self._rgb_depth_sub, self._depth_sub],
                queue_size=queue_size,
                slop=slop_seconds,
                allow_headerless=False,
            )
            self._depth_synchronizer.registerCallback(self._callback_rgbd)

        if self.use_projected_lidar:
            self._rgb_cloud_sub = message_filters.Subscriber(
                node, Image, rgb_topic, qos_profile=qos_profile_sensor_data
            )
            self._cloud_sub = message_filters.Subscriber(
                node, PointCloud2, projected_lidar_topic, qos_profile=qos_profile_sensor_data
            )
            self._cloud_synchronizer = message_filters.ApproximateTimeSynchronizer(
                [self._rgb_cloud_sub, self._cloud_sub],
                queue_size=queue_size,
                slop=slop_seconds,
                allow_headerless=False,
            )
            self._cloud_synchronizer.registerCallback(self._callback_rgb_cloud)

        if self._depth_synchronizer is None and self._cloud_synchronizer is None:
            raise ValueError(
                "ImageListener requires either a depth topic or a projected LiDAR topic to be enabled."
            )

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        intrinsics = np.array(msg.k).reshape(3, 3)
        with self._lock:
            self.intrinsics = intrinsics
            self.fx = float(intrinsics[0, 0])
            self.fy = float(intrinsics[1, 1])
            self.px = float(intrinsics[0, 2])
            self.py = float(intrinsics[1, 2])
        self._camera_info_ready.set()
        # Only need the intrinsics once
        self._node.destroy_subscription(self._camera_info_sub)
        self._camera_info_sub = None

    def _record_projected_cloud_sample(self, stamp_msg) -> None:
        if not self.use_projected_lidar:
            return
        self._lidar_depth_active = True
        self._last_cloud_time = Time.from_msg(stamp_msg)
        if self._cloud_missing_warned or self._cloud_timeout_warned:
            self._node.get_logger().info(
                "Projected LiDAR topic '%s' is active; prioritizing fused depth.",
                self._projected_topic_name,
            )
        self._cloud_missing_warned = False
        self._cloud_timeout_warned = False

    def _projected_cloud_is_recent(self) -> bool:
        if not self.use_projected_lidar or not self._lidar_depth_active:
            return False
        if self._last_cloud_time is None:
            return False
        if self._projected_lidar_timeout_ns == 0:
            return True
        now = self._node.get_clock().now()
        elapsed = now - self._last_cloud_time
        return elapsed.nanoseconds <= self._projected_lidar_timeout_ns

    def _handle_projected_lidar_unavailable(self, reason: str) -> None:
        if not self.use_projected_lidar:
            return
        if reason == "missing" and not self._cloud_missing_warned:
            self._node.get_logger().warning(
                "Projected LiDAR topic '%s' has not produced any data; falling back to depth topic '%s'.",
                self._projected_topic_name,
                self._depth_topic_name,
            )
            self._cloud_missing_warned = True
        elif reason == "timeout" and not self._cloud_timeout_warned:
            self._node.get_logger().warning(
                "Projected LiDAR topic '%s' has not updated within %.1f s; reusing depth topic '%s'.",
                self._projected_topic_name,
                self.projected_lidar_timeout_sec,
                self._depth_topic_name,
            )
            self._cloud_timeout_warned = True

    def _lookup_tf(self, target_frame: str, source_frame: str, stamp: Time | None = None) -> Optional[np.ndarray]:
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp or Time(),
                timeout=Duration(seconds=0.2),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            self._node.get_logger().warning(f"TF lookup failed ({source_frame} -> {target_frame}): {exc}")
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        trans = [translation.x, translation.y, translation.z]
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        return ros_qt_to_rt(quat, trans)

    def _callback_rgbd(self, rgb: Image, depth: Image) -> None:
        if not self._camera_info_ready.is_set():
            return

        if self.use_projected_lidar:
            if self._lidar_depth_active:
                if self._projected_cloud_is_recent():
                    return
                self._lidar_depth_active = False
                self._handle_projected_lidar_unavailable("timeout")
            else:
                self._handle_projected_lidar_unavailable("missing")

        frame_stamp = Time.from_msg(rgb.header.stamp)
        rt_camera = self._lookup_tf(self.base_frame, self.camera_frame, frame_stamp)
        rt_base = self._lookup_tf(self.world_frame, self.base_frame, frame_stamp)

        if rt_camera is None or rt_base is None:
            return

        if depth.encoding == "32FC1":
            depth_cv = ros_numpy.numpify(depth)
            depth_cv[np.isnan(depth_cv)] = 0.0
        elif depth.encoding == "16UC1":
            depth_cv = ros_numpy.numpify(depth).astype(np.float32)
            depth_cv /= 1000.0
        else:
            self._node.get_logger().error("Unsupported depth type: %s", depth.encoding)
            return

        rgb_image = ros_numpy.numpify(rgb)

        with self._lock:
            self.im = rgb_image.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = self._rgb_frame_override or rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = rt_camera
            self.RT_base = rt_base

    def _callback_rgb_cloud(self, rgb: Image, cloud: PointCloud2) -> None:
        if not self._camera_info_ready.is_set():
            self._node.get_logger().debug("RGB-Cloud callback: waiting for camera info")
            return

        frame_stamp = Time.from_msg(rgb.header.stamp)
        rt_camera = self._lookup_tf(self.base_frame, self.camera_frame, frame_stamp)
        rt_base = self._lookup_tf(self.world_frame, self.base_frame, frame_stamp)

        if rt_camera is None or rt_base is None:
            self._node.get_logger().debug("RGB-Cloud callback: TF lookup failed")
            return

        depth_cv = self._project_cloud_to_depth(cloud, rgb.width, rgb.height)
        if depth_cv is None:
            self._node.get_logger().debug("RGB-Cloud callback: cloud projection failed")
            return

        self._node.get_logger().info("RGB-Cloud callback: successfully processed frame")

        rgb_image = ros_numpy.numpify(rgb)
        with self._lock:
            self.im = rgb_image.copy()
            self.depth = depth_cv
            self.rgb_frame_id = self._rgb_frame_override or rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = rt_camera
            self.RT_base = rt_base
        self._record_projected_cloud_sample(rgb.header.stamp)

    def _parse_pointcloud2_all_fields(self, cloud: PointCloud2) -> Optional[np.ndarray]:
        """Parse PointCloud2 message preserving ALL fields including custom u/v."""
        import struct

        # Build numpy dtype from PointCloud2 fields
        dtype_list = []
        for field in cloud.fields:
            # Map PointCloud2 datatypes to numpy dtypes
            type_map = {
                1: np.int8, 2: np.uint8,
                3: np.int16, 4: np.uint16,
                5: np.int32, 6: np.uint32,
                7: np.float32, 8: np.float64
            }
            np_dtype = type_map.get(field.datatype)
            if np_dtype is None:
                self._node.get_logger().warn(f"Unknown datatype {field.datatype} for field {field.name}")
                continue

            dtype_list.append((field.name, np_dtype))

        if not dtype_list:
            return None

        # Create structured array from raw data
        dtype = np.dtype(dtype_list)
        try:
            # Reshape raw data into structured array
            cloud_array = np.frombuffer(cloud.data, dtype=dtype)
            return cloud_array
        except Exception as exc:
            self._node.get_logger().error(f"Failed to parse PointCloud2: {exc}")
            return None

    def _project_cloud_to_depth(self, cloud: PointCloud2, width: int, height: int) -> Optional[np.ndarray]:
        # Parse PointCloud2 with all fields preserved
        cloud_array = self._parse_pointcloud2_all_fields(cloud)
        if cloud_array is None:
            if not self._cloud_field_warning_emitted:
                self._node.get_logger().error("Unable to parse projected cloud")
                self._cloud_field_warning_emitted = True
            return None

        field_names = cloud_array.dtype.names or ()

        if "u" not in field_names or "v" not in field_names:
            if not self._cloud_field_warning_emitted:
                self._node.get_logger().error(
                    f"Projected cloud missing required 'u'/'v' fields. Available fields: {list(field_names)}"
                )
                self._cloud_field_warning_emitted = True
            return None

        xyz = self._extract_cloud_xyz(cloud_array, field_names)
        if xyz is None:
            if not self._cloud_field_warning_emitted:
                self._node.get_logger().error("Projected cloud missing XYZ data.")
                self._cloud_field_warning_emitted = True
            return None

        cloud_frame = self.projected_lidar_frame or cloud.header.frame_id
        if not cloud_frame:
            self._node.get_logger().error("Projected cloud frame is unset.")
            return None

        if cloud_frame != self.camera_frame:
            cloud_stamp = Time.from_msg(cloud.header.stamp)
            transform = self._lookup_tf(self.camera_frame, cloud_frame, cloud_stamp)
            if transform is None:
                return None
            xyz = (transform[:3, :3] @ xyz.T).T
            xyz += transform[:3, 3]

        depth_vals = xyz[:, 2].astype(np.float32, copy=False)

        # Handle both dict and structured array formats
        if isinstance(cloud_array, dict):
            u_coords = np.asarray(cloud_array["u"], dtype=np.float32)
            v_coords = np.asarray(cloud_array["v"], dtype=np.float32)
        else:
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

    def _extract_cloud_xyz(self, cloud_array, field_names):
        # Handle both dict and set/tuple of field names
        if isinstance(field_names, set):
            field_check = field_names
        else:
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

    def get_latest_frames(self):
        with self._lock:
            if self.im is None or self.depth is None:
                return None
            return {
                "rgb": self.im.copy(),
                "depth": self.depth.copy(),
                "frame_id": self.rgb_frame_id,
                "stamp": self.rgb_frame_stamp,
                "rt_camera": self.RT_camera.copy(),
                "rt_base": self.RT_base.copy(),
                "height": self.height,
                "width": self.width,
                "intrinsics": {
                    "fx": self.fx,
                    "fy": self.fy,
                    "px": self.px,
                    "py": self.py,
                },
            }
