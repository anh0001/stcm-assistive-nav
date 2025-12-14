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
from sensor_msgs.msg import CameraInfo, Image
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
    ) -> None:
        self._node = node
        self._lock = threading.Lock()
        self._cv_bridge = CvBridge()

        self.im: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.rgb_frame_id: Optional[str] = None
        self.rgb_frame_stamp = None

        self.base_frame = base_frame
        self.camera_frame = camera_frame
        self.world_frame = world_frame

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

        self._rgb_sub = message_filters.Subscriber(
            node, Image, rgb_topic, qos_profile=qos_profile_sensor_data
        )
        self._depth_sub = message_filters.Subscriber(
            node, Image, depth_topic, qos_profile=qos_profile_sensor_data
        )

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub], queue_size=queue_size, slop=slop_seconds, allow_headerless=False
        )
        self._synchronizer.registerCallback(self._callback_rgbd)

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

    def _lookup_tf(self, target_frame: str, source_frame: str) -> Optional[np.ndarray]:
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            self._node.get_logger().warning("TF lookup failed (%s -> %s): %s", source_frame, target_frame, exc)
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        trans = [translation.x, translation.y, translation.z]
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        return ros_qt_to_rt(quat, trans)

    def _callback_rgbd(self, rgb: Image, depth: Image) -> None:
        if not self._camera_info_ready.is_set():
            return

        rt_camera = self._lookup_tf(self.base_frame, self.camera_frame)
        rt_base = self._lookup_tf(self.world_frame, self.base_frame)

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
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp
            self.height = depth_cv.shape[0]
            self.width = depth_cv.shape[1]
            self.RT_camera = rt_camera
            self.RT_base = rt_base

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
