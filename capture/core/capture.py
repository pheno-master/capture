"""Capturing one synchronised image–pose pair while the arm is stationary.

Strategy (stop-and-shoot): after a motion finishes, wait `settle_time_s`
for vibrations to die out, then take the first image stamped after that
moment and look up the camera pose at the image's own timestamp. Because the
arm is not moving, a small timing error between image and pose costs almost
nothing, which removes most synchronisation risk.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import is_rigid
from .interfaces import Clock, ImageSource, JointSource, PoseSource
from .model import CapturedFrame, CaptureFailure, CaptureOutcome


@dataclass(frozen=True)
class CaptureSettings:
    settle_time_s: float = 0.5  # wait after motion before accepting images
    image_timeout_s: float = 2.0  # how long to wait for a fresh image
    max_sync_dt_s: float = 0.05  # max |pose stamp − image stamp|

    def __post_init__(self) -> None:
        if self.settle_time_s < 0 or self.image_timeout_s <= 0 or self.max_sync_dt_s < 0:
            raise ValueError(f"invalid capture settings: {self}")


class FrameCapturer:
    """Combines an image source, a pose source and a joint source into one
    validated CapturedFrame. Has no knowledge of ROS or of the disk."""

    def __init__(
        self,
        images: ImageSource,
        poses: PoseSource,
        joints: JointSource,
        clock: Clock,
        settings: CaptureSettings = CaptureSettings(),
    ) -> None:
        self._images = images
        self._poses = poses
        self._joints = joints
        self._clock = clock
        self._settings = settings

    def capture(self) -> CaptureOutcome:
        """Capture once. Call only after the arm has stopped moving."""
        s = self._settings
        self._clock.sleep(s.settle_time_s)
        earliest = self._clock.now()

        image = self._images.wait_for_image_after(earliest, s.image_timeout_s)
        if image is None:
            return CaptureFailure(f"no image within {s.image_timeout_s:.1f} s")

        pose = self._poses.camera_pose_at(image.stamp)
        if pose is None:
            return CaptureFailure(f"no camera pose available at t={image.stamp:.3f}")

        dt = abs(pose.stamp - image.stamp)
        if dt > s.max_sync_dt_s:
            return CaptureFailure(f"pose is {dt * 1000:.0f} ms from the image (limit {s.max_sync_dt_s * 1000:.0f} ms)")

        if not is_rigid(pose.T_world_cam, tol=1e-4):
            return CaptureFailure("camera pose is not a rigid transform")

        return CapturedFrame(
            image=image.image,
            stamp=image.stamp,
            T_world_cam=pose.T_world_cam,
            joints=dict(self._joints.joint_positions()),
        )
