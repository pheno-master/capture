"""Abstractions the core depends on (dependency inversion).

Each protocol is deliberately small (interface segregation): the ROS adapters
in capture.ros implement them for the real robot, and the tests
implement them with in-memory fakes.
"""

from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np

from .model import CapturedFrame, StampedImage, StampedPose, ViewFailure


class Clock(Protocol):
    def now(self) -> float:
        """Current time in seconds, on the same clock as image and pose stamps."""

    def sleep(self, seconds: float) -> None: ...


class ImageSource(Protocol):
    def wait_for_image_after(self, t: float, timeout_s: float) -> StampedImage | None:
        """Return the first image stamped at or after `t`, or None on timeout."""


class PoseSource(Protocol):
    def camera_pose_at(self, stamp: float) -> StampedPose | None:
        """Camera pose (T_world_cam) at `stamp`, or None if unavailable."""


class JointSource(Protocol):
    def joint_positions(self) -> Mapping[str, float]:
        """Latest joint positions by joint name (radians or metres)."""


class Mover(Protocol):
    def move_camera_to(self, T_world_cam: np.ndarray) -> bool:
        """Plan and execute a motion that puts the camera at T_world_cam.

        Returns False if the pose is unreachable or execution failed. Must
        block until the motion has finished.
        """


class FrameSink(Protocol):
    def add_frame(self, frame: CapturedFrame) -> object:
        """Store a captured frame."""

    def record_failure(self, failure: ViewFailure) -> None:
        """Store a planned view that produced no frame."""
