import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def buffer_by_category(
    gdf: gpd.GeoDataFrame, buffer_meters_by_category: dict[str, float]
) -> dict[str, BaseGeometry]:
    """Buffer and dissolve each category's geometries independently.

    `buffer_meters_by_category` maps a `category` column value (e.g.
    "bikelane", "street") to the buffer distance to apply for it.
    """
    buffered = {}
    for category, buffer_m in buffer_meters_by_category.items():
        subset = gdf.loc[gdf["category"] == category, "geometry"]
        if subset.empty:
            buffered[category] = GeometryCollection()
        else:
            buffered[category] = unary_union(subset.buffer(buffer_m))
    return buffered


def rasterize_mask(
    geometry: BaseGeometry, tile_transform: Affine, tile_shape: tuple[int, int]
) -> np.ndarray:
    """Rasterize a (multi)polygon onto a tile's grid.

    Returns a boolean array of `tile_shape`, True where a pixel falls inside
    the buffered geometry.
    """
    if geometry.is_empty:
        return np.zeros(tile_shape, dtype=bool)
    return rasterize(
        [(geometry, 1)],
        out_shape=tile_shape,
        transform=tile_transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
