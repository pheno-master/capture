import numpy as np
import pytest

from fakes import FakeClock, FakeMover, MemorySink
from plant_capture.core.geometry import look_at
from plant_capture.core.model import CapturedFrame, CaptureFailure
from plant_capture.core.motion import RollSearchMover, roll_about_optical_axis
from plant_capture.core.session import ScanSession


class ScriptedCapturer:
    """Returns pre-defined outcomes in order."""

    def __init__(self, clock, outcomes, duration_s=0.7):
        self.clock, self.outcomes, self.duration = clock, list(outcomes), duration_s

    def capture(self):
        self.clock.advance(self.duration)
        return self.outcomes.pop(0)


def frame(T):
    return CapturedFrame(np.zeros((2, 2, 3), np.uint8), 0.0, T)


@pytest.fixture
def views():
    return [look_at(np.array([0.3 * np.cos(a), 0.3 * np.sin(a), 0.3]), np.zeros(3)) for a in np.linspace(0, 3, 4)]


def test_all_views_captured(views):
    clock, sink = FakeClock(), MemorySink()
    capturer = ScriptedCapturer(clock, [frame(T) for T in views])
    summary = ScanSession(FakeMover(clock, duration_s=2.0), capturer, sink, clock, lambda *a: None).run(views)
    assert summary.captured == 4 and summary.planned == 4 and not summary.failures
    assert len(sink.frames) == 4
    assert summary.motion_time_s == pytest.approx(8.0)
    assert summary.capture_time_s == pytest.approx(2.8)
    assert summary.total_time_s == pytest.approx(10.8)


def test_unreachable_and_failed_captures_are_recorded(views):
    clock, sink = FakeClock(), MemorySink()
    unreachable = {id(views[1])}
    mover = FakeMover(clock, reachable=lambda T: id(T) not in unreachable)
    capturer = ScriptedCapturer(clock, [frame(views[0]), CaptureFailure("no image"), frame(views[3])])
    progress = []
    summary = ScanSession(mover, capturer, sink, clock, lambda i, n, s: progress.append(s)).run(views)
    assert summary.captured == 2
    assert [(f.view_index, f.reason) for f in summary.failures] == [(1, "unreachable or motion failed"),
                                                                    (2, "no image")]
    assert sink.failures == summary.failures
    assert progress[0] == "captured" and progress[1].startswith("skipped")


def test_roll_keeps_viewing_direction(pose):
    rolled = roll_about_optical_axis(pose, 90)
    assert np.allclose(rolled[:3, 3], pose[:3, 3])
    assert np.allclose(rolled[:3, 2], pose[:3, 2])
    assert np.allclose(rolled[:3, 0], pose[:3, 1])  # new x = old y after +90°


def test_roll_search_tries_in_order_and_stops(pose):
    target = roll_about_optical_axis(pose, -90)
    inner = FakeMover(FakeClock(), reachable=lambda T: np.allclose(T, target))
    mover = RollSearchMover(inner, rolls_deg=(0, 90, -90, 180))
    assert mover.move_camera_to(pose)
    assert len(inner.requests) == 3  # 0°, 90°, then −90° succeeds; 180° never tried
    assert np.allclose(mover.last_reached, target)


def test_roll_search_gives_up(pose):
    mover = RollSearchMover(FakeMover(FakeClock(), reachable=lambda T: False), rolls_deg=(0, 180))
    assert not mover.move_camera_to(pose)
    assert mover.last_reached is None
