"""Apply buffered bike lane / street masks to satellite imagery tiles.

Pixels outside either buffer are zeroed out, and a classification band is
appended recording which category (if any) each pixel belongs to.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from shapely.geometry.base import BaseGeometry

from scripts.config import BIKE_LANE_LABEL, NODATA_VALUE, STREET_LABEL
from scripts.mask import rasterize_mask

# The IDOP20 RGBI tiles are R, G, B, then near-infrared. GDAL's GeoTIFF
# driver otherwise defaults an untagged 4th band to Alpha, which makes GIS
# viewers render the masked-out areas as transparent instead of nodata.
_ORIGINAL_RGBI_COLORINTERP = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.undefined)


def filter_tile(
    tile_path: Path, buffered_by_category: dict[str, BaseGeometry], out_path: Path
) -> Path:
    """Mask a single imagery tile to its bike lane/street buffers and write it out.

    Output keeps the source's full resolution and pixel values losslessly
    (compress="deflate"); pixels outside both buffers are zeroed out. An
    extra classification band is appended (0=background, 1=bikelane,
    2=street), with bikelane taking priority where the two buffers overlap.
    """
    with rasterio.open(tile_path) as src:
        profile = src.profile.copy()
        data = src.read()
        band_count = src.count
        shape = (src.height, src.width)
        street_mask = rasterize_mask(buffered_by_category["street"], src.transform, shape)
        bikelane_mask = rasterize_mask(buffered_by_category["bikelane"], src.transform, shape)

    classification = np.zeros(shape, dtype=data.dtype)
    classification[street_mask] = STREET_LABEL
    classification[bikelane_mask] = BIKE_LANE_LABEL

    combined_mask = street_mask | bikelane_mask
    filtered_bands = np.where(combined_mask, data, NODATA_VALUE).astype(data.dtype)
    output = np.concatenate([filtered_bands, classification[np.newaxis, ...]], axis=0)

    profile.update(
        driver="GTiff",
        count=band_count + 1,
        nodata=NODATA_VALUE,
        compress="deflate",
        predictor=2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        # Color interpretation must be set before the first write: GDAL bakes
        # the TIFF ExtraSamples/alpha tag into the directory at that point.
        if band_count == 4:
            dst.colorinterp = _ORIGINAL_RGBI_COLORINTERP + (ColorInterp.undefined,)
        dst.write(output)
        dst.set_band_description(
            band_count + 1,
            "classification: 0=background, 1=bikelane, 2=street",
        )

    return out_path
