"""In-memory implementations of the core interfaces, for tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from capture.core.model import CapturedFrame, StampedImage, StampedPose, ViewFailure


class FakeClock:
    """Manual clock: sleep() advances time instantly."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeImageSource:
    """Produces a new image every `period_s`, stamped on the fake clock."""

    def __init__(self, clock: FakeClock, period_s: float = 0.1, available: bool = True,
                 shape: tuple[int, ...] = (6, 8, 3)) -> None:
        self.clock = clock
        self.period = period_s
        self.available = available
        self.shape = shape

    def wait_for_image_after(self, t: float, timeout_s: float) -> StampedImage | None:
        if not self.available:
            self.clock.advance(timeout_s)
            return None
        stamp = np.ceil(t / self.period) * self.period  # next frame time ≥ t
        self.clock.t = max(self.clock.t, stamp)
        return StampedImage(np.full(self.shape, int(stamp * 10) % 256, np.uint8), float(stamp))


class FakePoseSource:
    def __init__(self, pose_fn: Callable[[float], StampedPose | None]) -> None:
        self.pose_fn = pose_fn

    @classmethod
    def exact(cls, T: np.ndarray) -> "FakePoseSource":
        return cls(lambda stamp: StampedPose(T, stamp))

    def camera_pose_at(self, stamp: float) -> StampedPose | None:
        return self.pose_fn(stamp)


class FakeJoints:
    def __init__(self, joints: dict[str, float] | None = None) -> None:
        self.joints = joints or {"joint1": 0.1, "joint2": -0.2}

    def joint_positions(self) -> dict[str, float]:
        return dict(self.joints)


class FakeMover:
    """Reaches a pose if `reachable(T)` is true; motion takes `duration_s`."""

    def __init__(self, clock: FakeClock, reachable: Callable[[np.ndarray], bool] = lambda T: True,
                 duration_s: float = 2.0) -> None:
        self.clock = clock
        self.reachable = reachable
        self.duration = duration_s
        self.requests: list[np.ndarray] = []

    def move_camera_to(self, T_world_cam: np.ndarray) -> bool:
        self.requests.append(T_world_cam)
        self.clock.advance(self.duration)
        return self.reachable(T_world_cam)


@dataclass
class MemorySink:
    frames: list[CapturedFrame] = field(default_factory=list)
    failures: list[ViewFailure] = field(default_factory=list)

    def add_frame(self, frame: CapturedFrame) -> None:
        self.frames.append(frame)

    def record_failure(self, failure: ViewFailure) -> None:
        self.failures.append(failure)


class MemoryImageWriter:
    def __init__(self) -> None:
        self.written: dict[Path, np.ndarray] = {}

    def write(self, path: Path, image: np.ndarray) -> None:
        self.written[Path(path)] = image
