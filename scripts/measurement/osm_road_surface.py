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
