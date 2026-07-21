"""Assume the road surface from OSM instead of detecting it from imagery.

A fallback for scripts/detect_roads.py, enabled by config.USE_OSM_ROAD_FALLBACK,
for when the CNN road detector underperforms (see README's "Road detection").
Each OSM street centerline is buffered by half a default width for its highway
class (config.OSM_ROAD_DEFAULT_WIDTH_M) and rasterized onto the tile grid, in
place of the CNN surface mask. The result is drop-in: detection/centerline_width.py
measures width against it exactly as against a detected mask -- so a way's
reported width is the assumed default for its class, buffered symmetrically
about the centerline the ray-casting starts from, not anything read from
pixels. Classes absent from the table use OSM_ROAD_DEFAULT_WIDTH_FALLBACK_M.

This is the region-of-interest-as-measurement trade the main pipeline rejects
on purpose (README, "Road detection"): the number is the assumption, not a
measurement. It exists only as a coverage fallback where detection fails.
"""

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine

from pipeline.config import OSM_ROAD_DEFAULT_WIDTH_FALLBACK_M, OSM_ROAD_DEFAULT_WIDTH_M


def road_width_m(highway) -> float:
    """Full assumed carriageway width for an OSM `highway` class, in metres.

    Falls back to OSM_ROAD_DEFAULT_WIDTH_FALLBACK_M for an unlisted class or a
    missing/non-string tag (osmnx can hand back a list where a way carries
    several highway values).
    """
    if isinstance(highway, str):
        return OSM_ROAD_DEFAULT_WIDTH_M.get(highway, OSM_ROAD_DEFAULT_WIDTH_FALLBACK_M)
    return OSM_ROAD_DEFAULT_WIDTH_FALLBACK_M


def osm_road_surface(streets, transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    """Boolean road-surface mask: each street buffered by half its class width.

    `streets` is a GeoDataFrame with `geometry` and `highway` columns in the
    tile CRS. Buffering by half the full width centres the surface on the
    centerline, matching how detection/centerline_width.py casts its rays.
    Returns an all-False mask if nothing buffers to a positive area.
    """
    burned_shapes = []
    for street in streets.itertuples():
        half_width = road_width_m(getattr(street, "highway", None)) / 2.0
        if half_width <= 0:
            continue
        polygon = street.geometry.buffer(half_width)
        if not polygon.is_empty:
            burned_shapes.append((polygon, 1))

    if not burned_shapes:
        return np.zeros(shape, dtype=bool)
    return rasterize(burned_shapes, out_shape=shape, transform=transform, dtype=np.uint8).astype(bool)
