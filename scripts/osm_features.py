"""Fetch bike lane and street geometry from OpenStreetMap for an area of interest."""

from pathlib import Path

import geopandas as gpd
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import box

from scripts.config import (
    BIKE_LANE_CYCLEWAY_VALUES,
    OSM_CACHE_PATH,
    OSM_CRS,
    OSM_TAGS,
    TILE_CRS,
)


def _classify(row) -> str:
    """Label a feature as 'bikelane' if it's dedicated/marked cycle infrastructure,
    else 'street' (any other queried road)."""
    if row.get("highway") == "cycleway":
        return "bikelane"
    if row.get("bicycle") == "designated":
        return "bikelane"
    for col in ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both"):
        value = row.get(col)
        if isinstance(value, str) and value in BIKE_LANE_CYCLEWAY_VALUES:
            return "bikelane"
    return "street"


def fetch_osm_features(
    bounds: tuple[float, float, float, float],
    bounds_crs: str = TILE_CRS,
    cache_path: Path = OSM_CACHE_PATH,
    force_refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Query OSM for bike lane and street geometries covering `bounds`.

    Returns a GeoDataFrame in TILE_CRS with a `category` column ("bikelane" or
    "street") and the raw `highway` tag, which the OSM road-surface fallback
    (detection/osm_road_surface.py) uses to look up a default width per class.
    Results are cached to `cache_path` since Overpass queries are slow and
    rate-limited; pass `force_refresh=True` to re-query. A cache written before
    `highway` was preserved is treated as stale and re-queried.
    """
    if cache_path.exists() and not force_refresh:
        cached = gpd.read_file(cache_path)
        if "highway" in cached.columns:
            return cached

    transformer = Transformer.from_crs(bounds_crs, OSM_CRS, always_xy=True)
    left, bottom, right, top = bounds
    west, south = transformer.transform(left, bottom)
    east, north = transformer.transform(right, top)
    aoi = box(west, south, east, north)

    gdf = ox.features_from_polygon(aoi, tags=OSM_TAGS)
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    gdf["category"] = gdf.apply(_classify, axis=1)
    if "highway" not in gdf.columns:
        gdf["highway"] = None
    gdf = gdf[["geometry", "category", "highway"]].to_crs(TILE_CRS)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache_path, driver="GPKG")
    return gdf
