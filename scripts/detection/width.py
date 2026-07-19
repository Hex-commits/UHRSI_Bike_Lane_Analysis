"""Measure a detected mask's physical width from its medial axis.

Skeletonize, then read the distance transform along the skeleton. This
suits a bike lane, whose traced mask is a narrow, near-solid ribbon, and it
is the only surface it is used for.

It does *not* generalize to a road, and that was measured rather than
assumed: a distance transform measures the distance to the nearest edge of
any kind, including the edges of interior holes, and a traced road is full
of them (parked cars, lane markings, tree shadow). On the representative
frame this returned a 4.0 px radius for a carriageway whose true maximum
inscribed radius was 27.3 px. A cross-track variant was written to work
around that and has since been removed with the rest of the mask-shape road
pipeline; road width now comes from detection/centerline_width.py, which
takes its direction from OSM geometry instead of from the mask.
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
