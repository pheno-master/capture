"""Running a scan: move to each planned view, capture, and store the result."""

from __future__ import annotations

import logging
from typing import Callable, Protocol, Sequence

import numpy as np

from .interfaces import Clock, FrameSink, Mover
from .model import CaptureFailure, CaptureOutcome, ScanSummary, ViewFailure

log = logging.getLogger(__name__)


class Capturer(Protocol):
    def capture(self) -> CaptureOutcome: ...


ProgressCallback = Callable[[int, int, str], None]  # (view index, total, status)


class ScanSession:
    """Orchestrates one scan. Knows the order of steps, nothing about how
    the robot moves, how images arrive, or how frames are stored."""

    def __init__(
        self,
        mover: Mover,
        capturer: Capturer,
        sink: FrameSink,
        clock: Clock,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._mover = mover
        self._capturer = capturer
        self._sink = sink
        self._clock = clock
        self._on_progress = on_progress or _log_progress

    def run(self, views: Sequence[np.ndarray]) -> ScanSummary:
        summary = ScanSummary(planned=len(views))
        start = self._clock.now()
        for index, T_world_cam in enumerate(views):
            status = self._scan_view(index, T_world_cam, summary)
            self._on_progress(index, len(views), status)
        summary.total_time_s = self._clock.now() - start
        return summary

    def _scan_view(self, index: int, T_world_cam: np.ndarray, summary: ScanSummary) -> str:
        reached, motion_s = self._timed(lambda: self._mover.move_camera_to(T_world_cam))
        summary.motion_time_s += motion_s
        if not reached:
            return self._fail(index, "unreachable or motion failed", summary)

        outcome, capture_s = self._timed(self._capturer.capture)
        summary.capture_time_s += capture_s
        if isinstance(outcome, CaptureFailure):
            return self._fail(index, outcome.reason, summary)

        self._sink.add_frame(outcome)
        summary.captured += 1
        return "captured"

    def _fail(self, index: int, reason: str, summary: ScanSummary) -> str:
        failure = ViewFailure(index, reason)
        summary.failures.append(failure)
        self._sink.record_failure(failure)
        return f"skipped: {reason}"

    def _timed(self, fn):
        t0 = self._clock.now()
        result = fn()
        return result, self._clock.now() - t0


def _log_progress(index: int, total: int, status: str) -> None:
    log.info("view %d/%d: %s", index + 1, total, status)
