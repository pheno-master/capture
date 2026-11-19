"""Reading a scan folder (used by exporters, tools and tests)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import schema
from .model import FrameRecord, Intrinsics


@dataclass(frozen=True)
class Dataset:
    root: Path
    intrinsics: Intrinsics
    frames: tuple[FrameRecord, ...]
    convention: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def image_path(self, frame: FrameRecord) -> Path:
        return self.root / frame.image


def load_dataset(root: Path) -> Dataset:
    root = Path(root)
    intrinsics = Intrinsics.from_dict(schema.read_json(root / schema.INTRINSICS_FILE))
    poses = schema.read_json(root / schema.POSES_FILE)
    frames_json = poses["frames"] if isinstance(poses, dict) else poses
    meta_path = root / schema.METADATA_FILE
    return Dataset(
        root=root,
        intrinsics=intrinsics,
        frames=tuple(FrameRecord.from_dict(f) for f in frames_json),
        convention=poses.get("convention", "") if isinstance(poses, dict) else "",
        metadata=schema.read_json(meta_path) if meta_path.exists() else {},
    )
