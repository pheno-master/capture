"""Conversions from ROS message fields to core types.

Deliberately free of rclpy imports: functions only read message attributes,
so tests can pass simple stand-in objects (e.g. types.SimpleNamespace).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core.geometry import make_transform, matrix_to_quat, quat_to_matrix
from ..core.model import Intrinsics

SUPPORTED_DISTORTION_MODELS = ("plumb_bob", "")


def stamp_to_seconds(stamp: Any) -> float:
    """builtin_interfaces/Time → float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


def seconds_to_nanoseconds(t: float) -> int:
    """float seconds → integer nanoseconds (for rclpy.time.Time)."""
    return int(round(t * 1e9))


def transform_to_matrix(transform: Any) -> np.ndarray:
    """geometry_msgs/Transform → 4x4 matrix."""
    t, q = transform.translation, transform.rotation
    return make_transform(quat_to_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z]))


def matrix_to_pose_fields(T: np.ndarray) -> dict[str, tuple[float, ...]]:
    """4x4 matrix → position and orientation (x, y, z, w), ready to copy into
    a geometry_msgs/Pose."""
    return {"position": tuple(float(v) for v in T[:3, 3]), "orientation": matrix_to_quat(T[:3, :3])}


def intrinsics_from_camera_info(msg: Any) -> Intrinsics:
    """sensor_msgs/CameraInfo → Intrinsics. Rejects distortion models the Go
    side cannot undistort."""
    model = getattr(msg, "distortion_model", "")
    if model not in SUPPORTED_DISTORTION_MODELS:
        raise ValueError(f"distortion model {model!r} not supported (use plumb_bob)")
    return Intrinsics.from_k(list(msg.k), list(msg.d), msg.width, msg.height)


def joint_state_to_dict(msg: Any) -> Mapping[str, float]:
    """sensor_msgs/JointState → {name: position}."""
    return {name: float(pos) for name, pos in zip(msg.name, msg.position)}
