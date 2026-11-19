"""Writing a scan folder in the format read by the Go reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import schema
from .image_io import ImageWriter
from .model import CapturedFrame, FrameRecord, Intrinsics, ScanSummary, ViewFailure


class DatasetWriter:
    """Implements FrameSink. poses.json is rewritten (atomically) after every
    frame, so an interrupted scan still leaves a usable dataset."""

    def __init__(self, root: Path, intrinsics: Intrinsics, image_writer: ImageWriter,
                 metadata: Mapping[str, Any] | None = None, overwrite: bool = False) -> None:
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()) and not overwrite:
            raise FileExistsError(f"{self.root} is not empty (pass overwrite=True to reuse it)")
        self.root.mkdir(parents=True, exist_ok=True)
        self._image_writer = image_writer
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._records: list[FrameRecord] = []
        self._failures: list[ViewFailure] = []

        schema.write_json(self.root / schema.INTRINSICS_FILE, intrinsics.to_dict())
        self._write_poses()
        self._write_metadata()

    @property
    def records(self) -> tuple[FrameRecord, ...]:
        return tuple(self._records)

    def add_frame(self, frame: CapturedFrame) -> FrameRecord:
        record = FrameRecord(
            id=len(self._records),
            image=schema.frame_image_path(len(self._records)),
            stamp=frame.stamp,
            T_world_cam=frame.T_world_cam,
            joints=frame.joints,
        )
        self._image_writer.write(self.root / record.image, frame.image)
        self._records.append(record)
        self._write_poses()
        return record

    def record_failure(self, failure: ViewFailure) -> None:
        self._failures.append(failure)
        self._write_metadata()

    def finalize(self, summary: ScanSummary) -> None:
        self._metadata["summary"] = summary.to_dict()
        self._write_metadata()

    def _write_poses(self) -> None:
        schema.write_json(self.root / schema.POSES_FILE, {
            "schema_version": schema.SCHEMA_VERSION,
            "convention": schema.CONVENTION,
            "frames": [r.to_dict() for r in self._records],
        })

    def _write_metadata(self) -> None:
        data = dict(self._metadata)
        data["failed_views"] = [f.to_dict() for f in self._failures]
        schema.write_json(self.root / schema.METADATA_FILE, data)
