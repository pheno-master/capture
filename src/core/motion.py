"""Mover decorators that add behaviour without changing the robot adapter
(open/closed principle)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry import make_transform
from .interfaces import Mover


def roll_about_optical_axis(T_world_cam: np.ndarray, roll_deg: float) -> np.ndarray:
    """Rotate the camera about its own +Z (viewing) axis. The camera still
    looks at the same point; only the image orientation changes."""
    a = np.radians(roll_deg)
    c, s = np.cos(a), np.sin(a)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return T_world_cam @ make_transform(Rz, np.zeros(3))


class RollSearchMover:
    """Tries several rolls about the optical axis and keeps the first one the
    arm can reach. A look-at view does not care about roll, but a fixed roll
    can make an otherwise reachable view impossible for a 6-joint arm.

    The pose that was actually reached is available as `last_reached`, but the
    capture uses the measured pose from tf anyway, so nothing else needs it.
    """

    def __init__(self, inner: Mover, rolls_deg: Sequence[float] = (0.0, 90.0, -90.0, 180.0)) -> None:
        if not rolls_deg:
            raise ValueError("need at least one roll angle")
        self._inner = inner
        self._rolls = tuple(rolls_deg)
        self.last_reached: np.ndarray | None = None

    def move_camera_to(self, T_world_cam: np.ndarray) -> bool:
        for roll in self._rolls:
            target = roll_about_optical_axis(T_world_cam, roll)
            if self._inner.move_camera_to(target):
                self.last_reached = target
                return True
        self.last_reached = None
        return False
