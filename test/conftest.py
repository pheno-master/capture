import sys
from pathlib import Path

import numpy as np
import pytest

# Make `plant_capture` and the `fakes` helper importable without installing.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.geometry import look_at  # noqa: E402
from src.core.model import Intrinsics  # noqa: E402


@pytest.fixture
def intrinsics() -> Intrinsics:
    return Intrinsics(fx=600.0, fy=600.0, cx=319.5, cy=239.5, width=640, height=480,
                      distortion=(-0.08, 0.01, 0.0, 0.0, 0.0))


@pytest.fixture
def pose() -> np.ndarray:
    return look_at(np.array([0.4, 0.1, 0.35]), np.array([0.0, 0.0, 0.1]))


@pytest.fixture
def example_config_path() -> Path:
    return ROOT / "config" / "scan_example.yaml"
