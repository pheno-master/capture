"""Image encoding behind a small interface, so the dataset writer does not
depend on OpenCV directly and tests can avoid touching the disk."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class ImageWriter(Protocol):
    def write(self, path: Path, image: np.ndarray) -> None:
        """Write a BGR uint8 (or single-channel) image losslessly."""


class Cv2ImageWriter:
    """Writes PNG files with OpenCV (expects BGR, OpenCV's own convention)."""

    def __init__(self, png_compression: int = 3) -> None:
        import cv2  # imported lazily: the rest of the core runs without OpenCV

        self._cv2 = cv2
        self._params = [cv2.IMWRITE_PNG_COMPRESSION, png_compression]

    def write(self, path: Path, image: np.ndarray) -> None:
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got {image.dtype}")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not self._cv2.imwrite(str(path), image, self._params):
            raise OSError(f"failed to write {path}")
