"""Measure a detected mask's physical width from its medial axis.

"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


@dataclass
class WidthStats:
    mean_m: float
    median_m: float
    min_m: float
    max_m: float
    n_samples: int


def measure_width_m(mask: np.ndarray, pixel_size_m: float) -> WidthStats | None:
    """Return width statistics in meters, or None if `mask` is empty."""
    if not mask.any():
        return None

    skeleton = skeletonize(mask)
    if not skeleton.any():
        # Mask too small/thin to skeletonize meaningfully; fall back to the mask itself.
        skeleton = mask

    distance_m = distance_transform_edt(mask) * pixel_size_m
    widths_m = distance_m[skeleton] * 2.0
    if widths_m.size == 0:
        return None

    return WidthStats(
        mean_m=float(widths_m.mean()),
        median_m=float(np.median(widths_m)),
        min_m=float(widths_m.min()),
        max_m=float(widths_m.max()),
        n_samples=int(widths_m.size),
    )
