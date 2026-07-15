"""Model-agnostic interface for trained bike lane segmentation.

Everything downstream (tiling, width measurement, detect.py) depends only on
`Detection`/`Detector` here, never on a specific model's API -- that's the
swap point. A concrete adapter (e.g. `yolo_seg_detector.YoloSegDetector`)
implements `predict()`; a different segmentation model can be substituted by
writing a new adapter against this same interface, without touching
tiling.py, width.py, or detect.py.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    """A single detected instance."""

    mask: np.ndarray  # (H, W) bool, aligned to the input chip image
    score: float
    label: str


class Detector(Protocol):
    def predict(self, image: np.ndarray) -> list[Detection]: ...
