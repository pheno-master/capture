from types import SimpleNamespace

import numpy as np
import pytest

from fakes import MemoryImageWriter
from capture.core.dataset_reader import load_dataset
from capture.core.dataset_writer import DatasetWriter
from capture.core.geometry import look_at, make_transform, quat_to_matrix
from capture.core.model import CapturedFrame, Intrinsics
from capture.ros import msg_conversions as mc
from capture.tools.export_colmap import camera_line, export


def test_colmap_export_recovers_poses(tmp_path, intrinsics):
    poses = [look_at(np.array([0.3 * np.cos(a), 0.3 * np.sin(a), 0.25]), np.zeros(3)) for a in (0.0, 2.0, 4.0)]
    w = DatasetWriter(tmp_path / "scan", intrinsics, MemoryImageWriter())
    for i, T in enumerate(poses):
        w.add_frame(CapturedFrame(np.zeros((2, 2, 3), np.uint8), float(i), T))
    export(load_dataset(tmp_path / "scan"), tmp_path / "colmap")

    lines = [l for l in (tmp_path / "colmap" / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) == 3
    for line, T in zip(lines, poses):
        parts = line.split()
        qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        T_cam_world = make_transform(quat_to_matrix(qx, qy, qz, qw), [tx, ty, tz])
        assert np.allclose(np.linalg.inv(T_cam_world), T, atol=1e-9)
        assert parts[9].endswith(".png")


def test_colmap_camera_model_choice(intrinsics):
    assert camera_line(intrinsics).split()[1] == "OPENCV"
    k3 = Intrinsics(**{**intrinsics.__dict__, "distortion": (0.1, 0, 0, 0, 0.01)})
    assert camera_line(k3).split()[1] == "FULL_OPENCV"


def camera_info(model="plumb_bob", d=(-0.1, 0.02, 0.001, 0.0, 0.0)):
    return SimpleNamespace(distortion_model=model, k=[600, 0, 320, 0, 610, 240, 0, 0, 1], d=list(d),
                           width=640, height=480)


def test_camera_info_to_intrinsics():
    k = mc.intrinsics_from_camera_info(camera_info())
    assert (k.fx, k.fy, k.cx, k.cy, k.width) == (600, 610, 320, 240, 640)
    assert k.distortion == (-0.1, 0.02, 0.001, 0.0, 0.0)
    assert mc.intrinsics_from_camera_info(camera_info(d=())).distortion == (0.0,) * 5


def test_unsupported_distortion_rejected():
    with pytest.raises(ValueError):
        mc.intrinsics_from_camera_info(camera_info(model="rational_polynomial"))
    with pytest.raises(ValueError):
        mc.intrinsics_from_camera_info(camera_info(d=(0, 0, 0, 0, 0, 0.3, 0, 0)))


def test_transform_and_pose_fields_round_trip(pose):
    f = mc.matrix_to_pose_fields(pose)
    (x, y, z), (qx, qy, qz, qw) = f["position"], f["orientation"]
    msg = SimpleNamespace(translation=SimpleNamespace(x=x, y=y, z=z),
                          rotation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw))
    assert np.allclose(mc.transform_to_matrix(msg), pose)


def test_time_conversions():
    assert mc.stamp_to_seconds(SimpleNamespace(sec=12, nanosec=500_000_000)) == 12.5
    assert mc.seconds_to_nanoseconds(12.5) == 12_500_000_000


def test_joint_state():
    msg = SimpleNamespace(name=["a", "b"], position=[0.1, -0.2])
    assert mc.joint_state_to_dict(msg) == {"a": 0.1, "b": -0.2}
