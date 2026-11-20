from datetime import datetime

import pytest

from plant_capture.core.config import RosSettings, ScanConfig, build_dataclass, load_config, scan_dir_name, scan_metadata
from plant_capture.core.views import hemisphere_views


def test_example_config_is_valid(example_config_path):
    cfg = load_config(example_config_path)
    assert isinstance(cfg, ScanConfig)
    assert isinstance(cfg.views.elevations_deg, tuple)
    assert cfg.ros.camera_frame.endswith("optical_frame")
    assert len(hemisphere_views(cfg.views)) > 0


def test_unknown_key_is_rejected():
    data = {"output_root": "/tmp", "plant_id": "p", "views": {"center": [0, 0, 0], "radius": 0.3, "radiuss": 1}}
    with pytest.raises(KeyError, match="views: unknown key"):
        build_dataclass(ScanConfig, data)


def test_defaults_and_nested(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"output_root": "/data", "plant_id": "p7", "views": {"center": [0, 0, 0.1], "radius": 0.3}}')
    cfg = load_config(p)
    assert cfg.ros == RosSettings()
    assert cfg.views.center == (0, 0, 0.1)


def test_metadata_and_dir_name(example_config_path):
    cfg = load_config(example_config_path)
    t = datetime(2026, 11, 20, 14, 3, 12)
    assert scan_dir_name("plant01", t) == "scan_2026-11-20T14-03-12_plant01"
    meta = scan_metadata(cfg, t)
    assert meta["created"] == "2026-11-20T14:03:12"
    assert meta["config"]["views"]["radius"] == cfg.views.radius


def test_invalid_mover_rejected():
    with pytest.raises(ValueError, match="ros.mover"):
        RosSettings(mover="teleport")
