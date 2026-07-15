"""Chip a large raster tile into windows for model inference, and map results back."""

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


@dataclass
class Chip:
    image: np.ndarray  # (H, W, 3) uint8 RGB
    window: Window
    transform: Affine  # georeferenced transform for this chip, in the tile's CRS


def iter_chips(src: rasterio.DatasetReader, chip_size_px: int, overlap_px: int):
    """Yield `Chip`s covering `src`, each up to `chip_size_px` square with `overlap_px` overlap.

    Chips at the right/bottom edge of the raster are smaller than
    `chip_size_px` rather than padded. Overlapping detections across chip
    boundaries are not deduplicated (see README known limitations).
    """
    step = chip_size_px - overlap_px
    for y in range(0, src.height, step):
        for x in range(0, src.width, step):
            width = min(chip_size_px, src.width - x)
            height = min(chip_size_px, src.height - y)
            window = Window(x, y, width, height)
            rgb = src.read([1, 2, 3], window=window)
            image = np.transpose(rgb, (1, 2, 0))
            yield Chip(image=image, window=window, transform=src.window_transform(window))


def mask_to_polygon(mask: np.ndarray, transform: Affine) -> BaseGeometry | None:
    """Vectorize a chip-local boolean mask into a (multi)polygon in the tile's CRS."""
    geometries = [
        shape(geometry) for geometry, _ in shapes(mask.astype("uint8"), mask=mask, transform=transform)
    ]
    if not geometries:
        return None
    return unary_union(geometries)
