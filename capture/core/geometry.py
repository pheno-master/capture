"""Rigid-transform helpers shared by the whole package.

Conventions (identical to the Go side):
    * Units are metres.
    * 4x4 homogeneous matrices, points are column vectors: p' = T @ p.
    * Names follow T_<to>_<from>: T_world_cam maps camera coordinates to world.
    * Camera frame is the optical frame: +Z forward, +X right, +Y down.
    * Quaternions are (x, y, z, w), the ROS order, unless a name says otherwise.
"""

from __future__ import annotations

import numpy as np

Matrix4 = np.ndarray  # shape (4, 4), float64
Vector3 = np.ndarray  # shape (3,), float64

_FALLBACK_UPS = (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> Matrix4:
    """Build a 4x4 rigid transform from a 3x3 rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def rigid_inverse(T: Matrix4) -> Matrix4:
    """Invert a rigid transform [R t; 0 1] as [Rᵀ -Rᵀt; 0 1]."""
    R_t = T[:3, :3].T
    return make_transform(R_t, -R_t @ T[:3, 3])


def is_rigid(T: np.ndarray, tol: float = 1e-6) -> bool:
    """True if T is a 4x4 proper rigid transform (orthonormal rotation, det +1)."""
    if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1], atol=tol):
        return False
    R = T[:3, :3]
    return bool(np.allclose(R @ R.T, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1) < tol)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = _FALLBACK_UPS[0]) -> Matrix4:
    """Camera-to-world pose of a camera at `eye` looking at `target`.

    The result uses the optical convention: +Z points at the target, +X to the
    right in the image and +Y down. `up` is the world direction that should
    appear upwards in the image. If `up` is (nearly) parallel to the viewing
    direction (a top-down view with up = +Z), another axis is used instead.
    """
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-12:
        raise ValueError("eye and target coincide")
    z = forward / norm
    for candidate in (np.asarray(up, float), *_FALLBACK_UPS):
        x = np.cross(z, candidate)
        if np.linalg.norm(x) > 1e-6:
            break
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_transform(np.column_stack((x, y, z)), eye)


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """3x3 rotation from a quaternion in ROS order (x, y, z, w)."""
    q = np.array([x, y, z, w], float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) from a 3x3 rotation, with w >= 0."""
    m = np.asarray(R, float)
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return tuple(float(v) for v in q)  # type: ignore[return-value]


def camera_target_to_ee_target(T_world_cam: Matrix4, T_ee_cam: Matrix4) -> Matrix4:
    """End-effector pose that puts the camera at T_world_cam.

    T_ee_cam is the hand–eye calibration (camera pose in the end-effector
    frame): T_world_cam = T_world_ee @ T_ee_cam  ⇒  T_world_ee = T_world_cam @ T_ee_cam⁻¹.
    """
    return T_world_cam @ rigid_inverse(T_ee_cam)
