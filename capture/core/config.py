"""Scan configuration loaded from YAML or JSON.

A single generic builder turns nested dictionaries into the frozen
dataclasses used by the core, rejecting unknown keys so a typo in the config
file fails loudly instead of silently using a default.
"""

from __future__ import annotations

import dataclasses
import json
import typing
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .capture import CaptureSettings
from .views import HemisphereSpec


@dataclass(frozen=True)
class RosSettings:
    base_frame: str = "base_link"  # world frame for all poses
    camera_frame: str = "camera_optical_frame"  # must be the OPTICAL frame
    ee_link: str = "link6_flange"  # link MoveIt moves; hand–eye is ee_link → camera_frame
    planning_group: str = "arm"
    image_topic: str = "/camera/image_raw"
    camera_info_topic: str = "/camera/camera_info"
    joint_states_topic: str = "/joint_states"
    tf_timeout_s: float = 0.5
    mover: str = "move_group"  # "move_group" (action to a running move_group) or "moveit_py"
    arm_joints: tuple[str, ...] = ()  # joints of planning_group; enables IK + joint goals (move_group mover)
    ik_attempts: int = 10  # first from the current state, the rest from random seeds
    ik_timeout_s: float = 0.05

    def __post_init__(self) -> None:
        if self.mover not in ("move_group", "moveit_py"):
            raise ValueError(f"ros.mover must be 'move_group' or 'moveit_py', got {self.mover!r}")


@dataclass(frozen=True)
class MotionSettings:
    roll_candidates_deg: tuple[float, ...] = (0.0, 90.0, -90.0, 180.0)


@dataclass(frozen=True)
class ScanConfig:
    output_root: str
    plant_id: str
    views: HemisphereSpec
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    ros: RosSettings = field(default_factory=RosSettings)
    motion: MotionSettings = field(default_factory=MotionSettings)
    notes: str = ""


def build_dataclass(cls: type, data: Mapping[str, Any], path: str = ""):
    """Recursively construct dataclass `cls` from a mapping."""
    if not isinstance(data, Mapping):
        raise TypeError(f"{path or cls.__name__}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"{path or cls.__name__}: unknown key(s) {sorted(unknown)}")
    kwargs = {name: _convert(hints[name], value, f"{path}.{name}".lstrip(".")) for name, value in data.items()}
    return cls(**kwargs)


def _convert(hint: Any, value: Any, path: str) -> Any:
    if dataclasses.is_dataclass(hint):
        return build_dataclass(hint, value, path)
    if isinstance(value, list):  # YAML/JSON lists → tuples (the dataclasses are frozen)
        return tuple(value)
    return value


def load_config(path: Path) -> ScanConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml  # ships with ROS 2

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return build_dataclass(ScanConfig, data)


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """For storing the exact configuration of a scan in metadata.json."""
    return dataclasses.asdict(cfg)


def scan_dir_name(plant_id: str, created: datetime) -> str:
    """Folder name for one scan, e.g. 'scan_2026-11-20T14-03-12_plant01'."""
    return f"scan_{created.strftime('%Y-%m-%dT%H-%M-%S')}_{plant_id}"


def scan_metadata(cfg: ScanConfig, created: datetime) -> dict[str, Any]:
    """Initial metadata.json content: who/what/when plus the full config."""
    return {
        "plant_id": cfg.plant_id,
        "created": created.isoformat(timespec="seconds"),
        "notes": cfg.notes,
        "config": config_to_dict(cfg),
    }
