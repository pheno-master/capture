import json

import numpy as np
import pytest

from fakes import MemoryImageWriter
from plant_capture.core import schema
from plant_capture.core.dataset_reader import load_dataset
from plant_capture.core.dataset_writer import DatasetWriter
from plant_capture.core.image_io import Cv2ImageWriter
from plant_capture.core.model import CapturedFrame, FrameRecord, ScanSummary, ViewFailure


def captured(pose, stamp=1.0):
    return CapturedFrame(np.full((4, 5, 3), 7, np.uint8), stamp, pose, {"joint1": 0.5})


def test_write_then_read_round_trip(tmp_path, intrinsics, pose):
    images = MemoryImageWriter()
    w = DatasetWriter(tmp_path / "scan", intrinsics, images, metadata={"plant_id": "p1"})
    r0 = w.add_frame(captured(pose, 1.0))
    r1 = w.add_frame(captured(pose, 2.0))
    assert (r0.id, r1.id) == (0, 1)
    assert r1.image == "frames/000001.png"
    assert set(images.written) == {tmp_path / "scan" / r0.image, tmp_path / "scan" / r1.image}

    ds = load_dataset(tmp_path / "scan")
    assert ds.intrinsics == intrinsics
    assert ds.convention == schema.CONVENTION
    assert [f.stamp for f in ds.frames] == [1.0, 2.0]
    assert np.allclose(ds.frames[0].T_world_cam, pose)
    assert ds.frames[0].joints == {"joint1": 0.5}


def test_poses_written_after_every_frame(tmp_path, intrinsics, pose):
    w = DatasetWriter(tmp_path, intrinsics, MemoryImageWriter())
    assert schema.read_json(tmp_path / schema.POSES_FILE)["frames"] == []
    w.add_frame(captured(pose))
    assert len(schema.read_json(tmp_path / schema.POSES_FILE)["frames"]) == 1


def test_json_matches_go_reader_keys(tmp_path, intrinsics, pose):
    """The keys the Go dataset package reads must be present."""
    DatasetWriter(tmp_path, intrinsics, MemoryImageWriter()).add_frame(captured(pose))
    k = json.loads((tmp_path / "intrinsics.json").read_text())
    assert {"fx", "fy", "cx", "cy", "width", "height", "distortion"} <= set(k)
    f = json.loads((tmp_path / "poses.json").read_text())["frames"][0]
    assert {"id", "image", "stamp", "T_world_cam"} <= set(f) and len(f["T_world_cam"]) == 16


def test_failures_and_summary_in_metadata(tmp_path, intrinsics):
    w = DatasetWriter(tmp_path, intrinsics, MemoryImageWriter(), metadata={"plant_id": "p1"})
    w.record_failure(ViewFailure(3, "unreachable"))
    w.finalize(ScanSummary(planned=5, captured=4, failures=[ViewFailure(3, "unreachable")]))
    meta = load_dataset(tmp_path).metadata
    assert meta["plant_id"] == "p1"
    assert meta["failed_views"] == [{"view_index": 3, "reason": "unreachable"}]
    assert meta["summary"]["captured"] == 4


def test_refuses_non_empty_folder(tmp_path, intrinsics):
    (tmp_path / "old.txt").write_text("x")
    with pytest.raises(FileExistsError):
        DatasetWriter(tmp_path, intrinsics, MemoryImageWriter())
    DatasetWriter(tmp_path, intrinsics, MemoryImageWriter(), overwrite=True)


def test_frame_record_rejects_non_rigid():
    with pytest.raises(ValueError):
        FrameRecord(id=0, image="x.png", stamp=0.0, T_world_cam=np.diag([2.0, 1, 1, 1]))


def test_cv2_writer_writes_png(tmp_path):
    cv2 = pytest.importorskip("cv2")
    img = np.random.default_rng(0).integers(0, 255, (10, 12, 3), dtype=np.uint8)
    Cv2ImageWriter().write(tmp_path / "a" / "x.png", img)
    assert np.array_equal(cv2.imread(str(tmp_path / "a" / "x.png")), img)
