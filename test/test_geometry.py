import numpy as np
import pytest

from capture.core.geometry import (
    camera_target_to_ee_target, is_rigid, look_at, make_transform, matrix_to_quat, quat_to_matrix, rigid_inverse
)


def rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def test_rigid_inverse_round_trip():
    T = make_transform(rot([1, 2, 3], 40), [0.1, -0.2, 0.3])
    assert np.allclose(T @ rigid_inverse(T), np.eye(4))
    assert is_rigid(T)


def test_is_rigid_rejects_mirror_and_bad_last_row():
    T = np.eye(4)
    T[0, 0] = -1
    assert not is_rigid(T)
    T = np.eye(4)
    T[3, 0] = 1
    assert not is_rigid(T)


def test_look_at_optical_convention():
    eye, target = np.array([0.5, 0.0, 0.3]), np.array([0.0, 0.0, 0.1])
    T = look_at(eye, target)
    assert is_rigid(T)
    z_cam = T[:3, 2]
    assert np.allclose(z_cam, (target - eye) / np.linalg.norm(target - eye))  # +Z at target
    # World up must point "up" in the image, i.e. towards camera −Y.
    up_in_cam = rigid_inverse(T)[:3, :3] @ np.array([0, 0, 1.0])
    assert up_in_cam[1] < 0


def test_look_at_top_down_falls_back():
    T = look_at(np.array([0, 0, 1.0]), np.zeros(3))
    assert is_rigid(T)
    assert np.allclose(T[:3, 2], [0, 0, -1])


def test_look_at_rejects_coincident_points():
    with pytest.raises(ValueError):
        look_at(np.ones(3), np.ones(3))


@pytest.mark.parametrize("axis,deg", [([1, 0, 0], 0), ([0, 0, 1], 90), ([1, 1, 0], 180), ([0, 1, 0], 180),
                                      ([1, 2, 3], 123), ([-1, 0.5, 2], 359)])
def test_quaternion_round_trip(axis, deg):
    R = rot(axis, deg)
    q = matrix_to_quat(R)
    assert q[3] >= 0
    assert np.allclose(quat_to_matrix(*q), R, atol=1e-12)


def test_camera_target_to_ee_target():
    T_ee_cam = make_transform(rot([0, 0, 1], 90), [0.0, 0.03, 0.05])  # camera mounted on the flange
    T_world_cam = look_at(np.array([0.3, 0.2, 0.4]), np.zeros(3))
    T_world_ee = camera_target_to_ee_target(T_world_cam, T_ee_cam)
    assert np.allclose(T_world_ee @ T_ee_cam, T_world_cam)
