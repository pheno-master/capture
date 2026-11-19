"""PoseSource backed by tf2: the camera pose comes from the robot's forward
kinematics (robot_state_publisher) plus the static hand–eye transform."""

from __future__ import annotations

import numpy as np
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from ..core.model import StampedPose
from .msg_conversions import seconds_to_nanoseconds, stamp_to_seconds, transform_to_matrix


class TfPoseSource:
    def __init__(self, node: Node, base_frame: str, camera_frame: str, timeout_s: float = 0.5) -> None:
        self._node = node
        self._base = base_frame
        self._camera = camera_frame
        self._timeout = Duration(seconds=timeout_s)
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, node)  # keep a reference

    def camera_pose_at(self, stamp: float) -> StampedPose | None:
        T, used = self._lookup(self._base, self._camera, stamp)
        return None if T is None else StampedPose(T, used)

    def static_transform(self, parent: str, child: str) -> np.ndarray | None:
        """Latest transform parent ← child, e.g. the hand–eye ee_link ← camera."""
        T, _ = self._lookup(parent, child, None)
        return T

    def _lookup(self, parent: str, child: str, stamp: float | None) -> tuple[np.ndarray | None, float]:
        when = Time() if stamp is None else Time(nanoseconds=seconds_to_nanoseconds(stamp))
        try:
            tf = self._buffer.lookup_transform(parent, child, when, timeout=self._timeout)
        except TransformException as exc:
            self._node.get_logger().warning(f"tf {parent} <- {child}: {exc}")
            return None, 0.0
        return transform_to_matrix(tf.transform), stamp_to_seconds(tf.header.stamp)
