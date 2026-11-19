"""Plain data types passed between components. No behaviour beyond
(de)serialisation and validation lives here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .geometry import Matrix4, is_rigid

DISTORTION_LENGTH = 5  # OpenCV plumb_bob order: k1, k2, p1, p2, k3


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics with OpenCV distortion. Matches intrinsics.json."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple[float, ...] = (0.0,) * DISTORTION_LENGTH

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError(f"invalid intrinsics: {self}")
        if len(self.distortion) != DISTORTION_LENGTH:
            raise ValueError(f"distortion needs {DISTORTION_LENGTH} values (k1 k2 p1 p2 k3), got {len(self.distortion)}")

    @classmethod
    def from_k(cls, K: Sequence[float], distortion: Sequence[float], width: int, height: int) -> "Intrinsics":
        """Build from a row-major 3x3 camera matrix (9 values) and distortion.

        Shorter distortion vectors are zero-padded; longer ones are rejected,
        since the Go side only models k1 k2 p1 p2 k3.
        """
        if len(K) != 9:
            raise ValueError(f"K needs 9 values, got {len(K)}")
        d = list(distortion)
        if len(d) > DISTORTION_LENGTH:
            if any(abs(v) > 0 for v in d[DISTORTION_LENGTH:]):
                raise ValueError(f"unsupported distortion model with {len(d)} non-zero coefficients")
            d = d[:DISTORTION_LENGTH]
        d += [0.0] * (DISTORTION_LENGTH - len(d))
        return cls(fx=float(K[0]), fy=float(K[4]), cx=float(K[2]), cy=float(K[5]),
                   width=int(width), height=int(height), distortion=tuple(float(v) for v in d))

    def to_dict(self) -> dict[str, Any]:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "width": self.width, "height": self.height, "distortion": list(self.distortion)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Intrinsics":
        return cls(fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"], width=d["width"],
                   height=d["height"], distortion=tuple(d.get("distortion", (0.0,) * DISTORTION_LENGTH)))


@dataclass(frozen=True)
class StampedImage:
    """An image (BGR uint8, OpenCV convention) with its capture time in seconds."""

    image: np.ndarray
    stamp: float


@dataclass(frozen=True)
class StampedPose:
    """A camera pose (T_world_cam) and the time it is valid for."""

    T_world_cam: Matrix4
    stamp: float


@dataclass(frozen=True)
class CapturedFrame:
    """One synchronised image–pose pair, ready to be written."""

    image: np.ndarray
    stamp: float
    T_world_cam: Matrix4
    joints: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CaptureFailure:
    """Why a capture attempt produced no frame."""

    reason: str


CaptureOutcome = CapturedFrame | CaptureFailure


@dataclass(frozen=True)
class FrameRecord:
    """One entry of poses.json."""

    id: int
    image: str  # path relative to the scan folder
    stamp: float
    T_world_cam: Matrix4
    joints: Mapping[str, float] = field(default_factory=dict)
    mask: str | None = None

    def __post_init__(self) -> None:
        if not is_rigid(np.asarray(self.T_world_cam), tol=1e-4):
            raise ValueError(f"frame {self.id}: T_world_cam is not a rigid transform")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "image": self.image,
            "stamp": self.stamp,
            "T_world_cam": [float(v) for v in np.asarray(self.T_world_cam).reshape(-1)],
            "joints": dict(self.joints),
        }
        if self.mask:
            d["mask"] = self.mask
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FrameRecord":
        T = np.asarray(d["T_world_cam"], float)
        if T.size != 16:
            raise ValueError(f"frame {d.get('id')}: T_world_cam has {T.size} values, want 16")
        return cls(id=int(d["id"]), image=d["image"], stamp=float(d["stamp"]),
                   T_world_cam=T.reshape(4, 4), joints=d.get("joints", {}), mask=d.get("mask"))


@dataclass(frozen=True)
class ViewFailure:
    """A planned view that produced no frame, and why."""

    view_index: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"view_index": self.view_index, "reason": self.reason}


@dataclass
class ScanSummary:
    """Outcome and timing of a scan; written into metadata.json."""

    planned: int = 0
    captured: int = 0
    failures: list[ViewFailure] = field(default_factory=list)
    motion_time_s: float = 0.0
    capture_time_s: float = 0.0
    total_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "planned": self.planned,
            "captured": self.captured,
            "failures": [f.to_dict() for f in self.failures],
            "motion_time_s": round(self.motion_time_s, 3),
            "capture_time_s": round(self.capture_time_s, 3),
            "total_time_s": round(self.total_time_s, 3),
        }
