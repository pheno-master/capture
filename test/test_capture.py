import numpy as np
import pytest

from fakes import FakeClock, FakeImageSource, FakeJoints, FakePoseSource
from capture.core.capture import CaptureSettings, FrameCapturer
from capture.core.model import CapturedFrame, CaptureFailure, StampedPose


def make_capturer(pose_source, clock=None, images=None, **settings):
    clock = clock or FakeClock()
    images = images or FakeImageSource(clock)
    return FrameCapturer(images, pose_source, FakeJoints(), clock, CaptureSettings(**settings)), clock


def test_successful_capture(pose):
    capturer, clock = make_capturer(FakePoseSource.exact(pose), settle_time_s=0.5)
    start = clock.now()
    out = capturer.capture()
    assert isinstance(out, CapturedFrame)
    assert out.stamp >= start + 0.5  # image taken after the settle time
    assert np.allclose(out.T_world_cam, pose)
    assert out.joints == FakeJoints().joints


def test_no_image():
    clock = FakeClock()
    capturer, _ = make_capturer(FakePoseSource.exact(np.eye(4)), clock, FakeImageSource(clock, available=False))
    out = capturer.capture()
    assert isinstance(out, CaptureFailure) and "no image" in out.reason


def test_no_pose():
    capturer, _ = make_capturer(FakePoseSource(lambda stamp: None))
    out = capturer.capture()
    assert isinstance(out, CaptureFailure) and "no camera pose" in out.reason


def test_pose_too_far_from_image(pose):
    capturer, _ = make_capturer(FakePoseSource(lambda stamp: StampedPose(pose, stamp - 0.2)), max_sync_dt_s=0.05)
    out = capturer.capture()
    assert isinstance(out, CaptureFailure) and "ms from the image" in out.reason


def test_non_rigid_pose_rejected():
    bad = np.eye(4)
    bad[0, 0] = 2.0
    capturer, _ = make_capturer(FakePoseSource.exact(bad))
    assert isinstance(capturer.capture(), CaptureFailure)


def test_invalid_settings():
    with pytest.raises(ValueError):
        CaptureSettings(image_timeout_s=0)
