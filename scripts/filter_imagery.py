"""Apply buffered bike lane / street masks to satellite imagery tiles.

Pixels outside either buffer are zeroed out, shadowed pixels within the
buffers are brightness-normalized, and two extra bands are appended: one
classifying each pixel as bikelane/street, the other flagging whether it was
shadowed.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from shapely.geometry.base import BaseGeometry

from scripts.config import (
    APPLY_SHADOW_CORRECTION,
    BIKE_LANE_LABEL,
    NODATA_VALUE,
    NOT_SHADOW_LABEL,
    SHADOW_LABEL,
    STREET_LABEL,
)
from scripts.mask import rasterize_mask
from scripts.shadows import clean_shadow_mask, correct_shadows, detect_shadow_mask

# The IDOP20 RGBI tiles are R, G, B, then near-infrared. GDAL's GeoTIFF
# driver otherwise defaults an untagged 4th band to Alpha, which makes GIS
# viewers render the masked-out areas as transparent instead of nodata.
_ORIGINAL_RGBI_COLORINTERP = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.undefined)


def filter_tile(
    tile_path: Path, buffered_by_category: dict[str, BaseGeometry], out_path: Path
) -> Path:
    """Mask a single imagery tile to its bike lane/street buffers and write it out.

    Output keeps the source's full resolution (compress="deflate", lossless);
    pixels outside both buffers are zeroed out. If APPLY_SHADOW_CORRECTION is
    set, shadowed pixels within the buffers are brightness-normalized to
    match nearby sunlit pixels (see scripts/shadows.py), so retained pixel
    values are no longer guaranteed to be bit-identical to the source there.
    Two extra bands are appended: a classification band (0=background,
    1=bikelane, 2=street; bikelane takes priority where the buffers overlap)
    and a shadow band (0=not shadowed, 1=shadowed).
    """
    with rasterio.open(tile_path) as src:
        profile = src.profile.copy()
        data = src.read()
        band_count = src.count
        shape = (src.height, src.width)
        pixel_size_m = src.res[0]
        street_mask = rasterize_mask(buffered_by_category["street"], src.transform, shape)
        bikelane_mask = rasterize_mask(buffered_by_category["bikelane"], src.transform, shape)

    classification = np.zeros(shape, dtype=data.dtype)
    classification[street_mask] = STREET_LABEL
    classification[bikelane_mask] = BIKE_LANE_LABEL

    combined_mask = street_mask | bikelane_mask

    shadow_band = np.full(shape, NOT_SHADOW_LABEL, dtype=data.dtype)
    if APPLY_SHADOW_CORRECTION:
        shadow_mask = clean_shadow_mask(detect_shadow_mask(data[:3], combined_mask), pixel_size_m)
        data = correct_shadows(data, shadow_mask, combined_mask, pixel_size_m)
        shadow_band[shadow_mask] = SHADOW_LABEL

    filtered_bands = np.where(combined_mask, data, NODATA_VALUE).astype(data.dtype)
    output = np.concatenate(
        [filtered_bands, classification[np.newaxis, ...], shadow_band[np.newaxis, ...]], axis=0
    )

    profile.update(
        driver="GTiff",
        count=band_count + 2,
        nodata=NODATA_VALUE,
        compress="deflate",
        predictor=2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        # Color interpretation must be set before the first write: GDAL bakes
        # the TIFF ExtraSamples/alpha tag into the directory at that point.
        if band_count == 4:
            dst.colorinterp = _ORIGINAL_RGBI_COLORINTERP + (ColorInterp.undefined, ColorInterp.undefined)
        dst.write(output)
        dst.set_band_description(
            band_count + 1,
            "classification: 0=background, 1=bikelane, 2=street",
        )
        dst.set_band_description(
            band_count + 2,
            "shadow: 0=not shadowed, 1=shadowed",
        )

    return out_path
