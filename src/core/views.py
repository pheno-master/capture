"""Static view planning: camera poses on a hemisphere around the plant.

This is the fixed capture path (condition A in the RPP). Elevation is
measured from the horizontal plane: 0° looks at the plant from the side,
90° looks straight down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import Matrix4, look_at


@dataclass(frozen=True)
class HemisphereSpec:
    center: tuple[float, float, float]  # plant centre in the world frame (m)
    radius: float  # camera distance from the centre (m)
    elevations_deg: tuple[float, ...] = (20.0, 45.0, 70.0)
    views_per_ring: int | tuple[int, ...] = 12  # one value for all rings, or one per ring
    azimuth_offset_deg: float = 0.0  # rotates every ring (to avoid an unreachable azimuth)
    azimuth_range_deg: tuple[float, float] = (0.0, 360.0)  # restrict to an arc if the arm cannot reach all round
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    serpentine: bool = True  # alternate ring direction to shorten arm motion

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if not self.elevations_deg:
            raise ValueError("need at least one elevation")
        if isinstance(self.views_per_ring, tuple) and len(self.views_per_ring) != len(self.elevations_deg):
            raise ValueError("views_per_ring must have one value per elevation")

    def ring_sizes(self) -> tuple[int, ...]:
        if isinstance(self.views_per_ring, int):
            return (self.views_per_ring,) * len(self.elevations_deg)
        return self.views_per_ring


def azimuths_deg(count: int, start_deg: float, stop_deg: float, offset_deg: float = 0.0) -> np.ndarray:
    """`count` evenly spaced azimuths. A full circle excludes the end point
    (0° and 360° are the same view); an arc includes both ends."""
    if count < 1:
        return np.empty(0)
    full_circle = np.isclose((stop_deg - start_deg) % 360.0, 0.0)
    values = np.linspace(start_deg, stop_deg, count, endpoint=not full_circle or count == 1)
    return values + offset_deg


def point_on_sphere(center: np.ndarray, radius: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    return center + radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def ring_poses(spec: HemisphereSpec, elevation_deg: float, count: int) -> list[Matrix4]:
    """Camera poses on one ring, all looking at the centre."""
    center = np.asarray(spec.center, float)
    up = np.asarray(spec.up, float)
    return [
        look_at(point_on_sphere(center, spec.radius, az, elevation_deg), center, up)
        for az in azimuths_deg(count, *spec.azimuth_range_deg, spec.azimuth_offset_deg)
    ]


def serpentine(rings: Sequence[Sequence[Matrix4]]) -> list[Matrix4]:
    """Flatten rings, reversing every other one, so the arm does not swing
    back to the start azimuth when it changes ring."""
    out: list[Matrix4] = []
    for i, ring in enumerate(rings):
        out.extend(reversed(ring) if i % 2 else ring)
    return out


def hemisphere_views(spec: HemisphereSpec) -> list[Matrix4]:
    """All planned camera poses (T_world_cam), in capture order."""
    rings = [ring_poses(spec, el, n) for el, n in zip(spec.elevations_deg, spec.ring_sizes())]
    if spec.serpentine:
        return serpentine(rings)
    return [pose for ring in rings for pose in ring]
