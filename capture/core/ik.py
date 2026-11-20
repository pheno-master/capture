"""Robot-independent helpers for inverse-kinematics retries."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def ik_seeds(joint_names: Sequence[str], attempts: int, limit_rad: float,
             rng: np.random.Generator) -> list[dict[str, float] | None]:
    """Starting configurations for repeated IK calls.

    The first seed is None, meaning "start from the robot's current state"
    (usually the closest solution, so the shortest motion). The others are
    uniform random joint positions in [-limit, +limit], which lets a local
    solver such as KDL escape a seed it cannot converge from.
    """
    if attempts < 1:
        raise ValueError("need at least one IK attempt")
    seeds: list[dict[str, float] | None] = [None]
    for _ in range(attempts - 1):
        seeds.append({n: float(rng.uniform(-limit_rad, limit_rad)) for n in joint_names})
    return seeds


def select_joints(solution: Mapping[str, float], wanted: Sequence[str]) -> dict[str, float]:
    """Keep only the arm joints of an IK solution (it may also contain the
    gripper). Raises if a wanted joint is missing, which means the joint names
    in the config do not match the robot."""
    missing = [n for n in wanted if n not in solution]
    if missing:
        raise KeyError(f"IK solution has no joint(s) {missing}; available: {sorted(solution)}")
    return {n: float(solution[n]) for n in wanted}
