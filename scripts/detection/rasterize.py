"""Rasterize detection results onto a raster grid, for visual inspection in a GIS/image viewer.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib import colormaps
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin


def rasterize_field(
    gdf: gpd.GeoDataFrame,
    field: str,
    bounds: tuple[float, float, float, float],
    pixel_size_m: float,
    out_path: Path,
    nodata: float = 0.0,
) -> Path:
    """Burn `gdf[field]` into a single-band float32 GeoTIFF covering `bounds`."""
    left, bottom, right, top = bounds
    width = round((right - left) / pixel_size_m)
    height = round((top - bottom) / pixel_size_m)
    transform = from_origin(left, top, pixel_size_m, pixel_size_m)

    shapes = [
        (geometry, value)
        for geometry, value in zip(gdf.geometry, gdf[field])
        if geometry is not None and not geometry.is_empty
    ]
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype="float32",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": gdf.crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(raster, 1)
        dst.set_band_description(1, field)

    return out_path


def render_preview_png(
    tif_path: Path, out_png_path: Path, cmap: str = "viridis", max_dimension_px: int = 1500
) -> Path:
    """Render a single-band GeoTIFF as a colorized PNG (nodata -> transparent).

    Downsamples to `max_dimension_px` on the long side (full-resolution
    mosaics are far larger than useful for a quick-look image) and stretches
    to the raster's own min/max, since raw values (e.g. 0.05-0.82 for
    scores) are too narrow a range to be visible unstretched.
    """
    with rasterio.open(tif_path) as src:
        scale = min(1.0, max_dimension_px / max(src.width, src.height))
        out_shape = (max(1, round(src.height * scale)), max(1, round(src.width * scale)))
        data = src.read(1, out_shape=out_shape, resampling=Resampling.average)
        nodata = src.nodata

    valid = data != nodata if nodata is not None else np.ones_like(data, dtype=bool)
    if valid.any():
        vmin, vmax = data[valid].min(), data[valid].max()
    else:
        vmin, vmax = 0.0, 1.0
    normalized = np.clip((data - vmin) / max(vmax - vmin, 1e-6), 0, 1)

    rgba = (colormaps[cmap](normalized) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0)  # transparent where nodata

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_png_path)
    return out_png_path
