"""Apply buffered bike lane / street masks to satellite imagery tiles.

Pixels outside either buffer are zeroed out. Shadowed pixels within the
buffers are handled per `SHADOW_HANDLING`: brightness-normalized
("correct"), dropped entirely ("cut", same as background), or left alone
("none"). If APPLY_RED_BOOST is set, reddish pixels (bike-lane paint) get a
saturation boost. Two extra bands are appended: one classifying each pixel
as bikelane/street, the other flagging whether it was shadowed.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from scipy.ndimage import binary_dilation
from shapely.geometry.base import BaseGeometry
from skimage.morphology import disk

from scripts.config import (
    APPLY_RED_BOOST,
    BIKE_LANE_LABEL,
    NODATA_VALUE,
    NOT_SHADOW_LABEL,
    SHADOW_CUT_MARGIN_M,
    SHADOW_HANDLING,
    SHADOW_LABEL,
    STREET_LABEL,
)
from scripts.mask import rasterize_mask
from scripts.redness import boost_red_saturation
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
    pixels outside both buffers are zeroed out. Shadowed pixels within the
    buffers are handled per `SHADOW_HANDLING`: "correct" brightness-normalizes
    them to match nearby sunlit pixels (see scripts/shadows.py), so retained
    pixel values are no longer guaranteed to be bit-identical to the source
    there; "cut" drops them entirely, same as background outside the buffer --
    and drops a further `SHADOW_CUT_MARGIN_M` beyond the detected mask too,
    since a real shadow's edge is a soft penumbra that the mask's hard
    threshold cuts through, so pixels just outside it can still be partially
    shadowed; "none" leaves shadowed pixels untouched. If APPLY_RED_BOOST is
    set, reddish pixels (bike-lane paint) within the retained buffer get a
    saturation boost afterwards (see scripts/redness.py). Two extra bands are
    appended: a classification band (0=background, 1=bikelane, 2=street;
    bikelane takes priority where the buffers overlap, reflecting whichever
    pixels were actually retained) and a shadow band (0=not shadowed,
    1=shadowed, reflecting detection only, not the cut margin; populated
    regardless of SHADOW_HANDLING, even where "cut" means those pixels end
    up as background/nodata in the other bands).
    """
    with rasterio.open(tile_path) as src:
        profile = src.profile.copy()
        data = src.read()
        band_count = src.count
        shape = (src.height, src.width)
        pixel_size_m = src.res[0]
        street_mask = rasterize_mask(buffered_by_category["street"], src.transform, shape)
        bikelane_mask = rasterize_mask(buffered_by_category["bikelane"], src.transform, shape)

    combined_mask = street_mask | bikelane_mask

    shadow_mask = clean_shadow_mask(detect_shadow_mask(data[:3], combined_mask), pixel_size_m)
    shadow_band = np.full(shape, NOT_SHADOW_LABEL, dtype=data.dtype)
    shadow_band[shadow_mask] = SHADOW_LABEL

    if SHADOW_HANDLING == "cut":
        margin_px = max(1, round(SHADOW_CUT_MARGIN_M / pixel_size_m))
        cut_mask = binary_dilation(shadow_mask, structure=disk(margin_px))
        street_mask = street_mask & ~cut_mask
        bikelane_mask = bikelane_mask & ~cut_mask
        combined_mask = combined_mask & ~cut_mask
    elif SHADOW_HANDLING == "correct":
        data = correct_shadows(data, shadow_mask, combined_mask, pixel_size_m)
    # "none": shadowed pixels are left untouched.

    classification = np.zeros(shape, dtype=data.dtype)
    classification[street_mask] = STREET_LABEL
    classification[bikelane_mask] = BIKE_LANE_LABEL

    if APPLY_RED_BOOST:
        data[:3] = boost_red_saturation(data[:3], combined_mask)

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
