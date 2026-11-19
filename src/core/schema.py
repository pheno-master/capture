"""The scan-folder format: file names, conventions and JSON I/O.

This is the single place that knows the on-disk layout, so the writer, the
reader and the exporters cannot drift apart. It must stay in sync with
internal/dataset in the Go reconstruction code.

    scan_dir/
      intrinsics.json
      poses.json      {"convention": ..., "schema_version": ..., "frames": [...]}
      metadata.json   free-form scan information and the ScanSummary
      frames/000000.png
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONVENTION = "T_world_cam, row-major, camera optical frame (+Z forward, +X right, +Y down), metres"

INTRINSICS_FILE = "intrinsics.json"
POSES_FILE = "poses.json"
METADATA_FILE = "metadata.json"
FRAMES_DIR = "frames"
MASKS_DIR = "masks"
IMAGE_EXTENSION = ".png"


def frame_image_path(frame_id: int) -> str:
    """Relative path of a frame image, e.g. 'frames/000012.png'."""
    return f"{FRAMES_DIR}/{frame_id:06d}{IMAGE_EXTENSION}"


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (temp file + rename), so a crash mid-scan never
    leaves a half-written poses.json behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
