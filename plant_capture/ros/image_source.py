"""ImageSource backed by sensor_msgs/Image and CameraInfo subscriptions."""

from __future__ import annotations

import threading

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from ..core.model import Intrinsics, StampedImage
from .msg_conversions import intrinsics_from_camera_info, stamp_to_seconds


class RosImageSource:
    """Keeps the latest image; `wait_for_image_after` blocks until one newer
    than a given time arrives. Callbacks run on the executor thread, the scan
    runs on the main thread, so access is guarded by a condition variable."""

    def __init__(self, node: Node, image_topic: str, camera_info_topic: str) -> None:
        self._bridge = CvBridge()
        self._cond = threading.Condition()
        self._latest: StampedImage | None = None
        self._intrinsics: Intrinsics | None = None
        node.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, camera_info_topic, self._on_info, qos_profile_sensor_data)

    def _on_image(self, msg: Image) -> None:
        frame = StampedImage(self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"),
                             stamp_to_seconds(msg.header.stamp))
        with self._cond:
            self._latest = frame
            self._cond.notify_all()

    def _on_info(self, msg: CameraInfo) -> None:
        with self._cond:
            if self._intrinsics is None:
                self._intrinsics = intrinsics_from_camera_info(msg)
                self._cond.notify_all()

    def wait_for_image_after(self, t: float, timeout_s: float) -> StampedImage | None:
        with self._cond:
            ok = self._cond.wait_for(lambda: self._latest is not None and self._latest.stamp >= t, timeout_s)
            return self._latest if ok else None

    def wait_for_intrinsics(self, timeout_s: float) -> Intrinsics | None:
        with self._cond:
            self._cond.wait_for(lambda: self._intrinsics is not None, timeout_s)
            return self._intrinsics
