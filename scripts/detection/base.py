from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    """A single detected instance."""

    mask: np.ndarray
    score: float
    label: str


class Detector(Protocol):
    def predict(self, image: np.ndarray) -> list[Detection]: ...
