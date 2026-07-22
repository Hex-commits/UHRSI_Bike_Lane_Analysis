"""Boost saturation of reddish pixels, so painted bike-lane paint stands out more.

Works in HSV: pixels close to red hue *and* already above a minimum saturation
get their saturation multiplied up (clamped to 1.0); everything else (asphalt,
vegetation, low-saturation noise) is left alone. Deliberately narrower than a
generic contrast stretch (tried and reverted) -- it only touches pixels that
already read as red, rather than stretching the whole tile's dynamic range.
Hue tolerance and saturation floor are calibrated against this repo's one
hand-annotated instance (paint at hue ~0.025-0.046, saturation ~0.11-0.30).
"""

import numpy as np
from skimage.color import hsv2rgb, rgb2hsv

RED_HUE_TOLERANCE = 0.08

MIN_SATURATION_FOR_BOOST = 0.05

SATURATION_BOOST = 1.8

# Rows converted per pass. The boost is purely per-pixel -- no neighbourhood,
# no tile-wide statistic -- so a row block gives bit-identical output to
# converting the whole tile at once, and bounds peak memory to something that
# does not depend on the tile's size. It has to: rgb2hsv promotes to float64,
# so a 10000x10000 chunk is 2.4 GB per intermediate and this function holds
# several at once, which OOM-killed a 16 GB machine outright.
ROWS_PER_BLOCK = 512


def boost_red_saturation(rgb: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    """Increase saturation of reddish pixels within `road_mask`.

    `rgb` is a (3, H, W) array (R, G, B). Pixels outside `road_mask`, or
    that aren't red enough to qualify, are returned unchanged.
    """
    info = np.iinfo(rgb.dtype)
    out = np.empty_like(rgb)

    for start in range(0, rgb.shape[1], ROWS_PER_BLOCK):
        stop = min(start + ROWS_PER_BLOCK, rgb.shape[1])
        block_mask = road_mask[start:stop]

        hsv = rgb2hsv(np.transpose(rgb[:, start:stop], (1, 2, 0)))
        hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        hue_distance_from_red = np.minimum(hue, 1.0 - hue)
        qualifies = (
            block_mask
            & (hue_distance_from_red <= RED_HUE_TOLERANCE)
            & (saturation >= MIN_SATURATION_FOR_BOOST)
        )

        boosted_saturation = saturation.copy()
        boosted_saturation[qualifies] = np.clip(saturation[qualifies] * SATURATION_BOOST, 0.0, 1.0)

        boosted_rgb = hsv2rgb(np.stack([hue, boosted_saturation, value], axis=-1))
        boosted_rgb = np.clip(boosted_rgb * info.max, info.min, info.max).astype(rgb.dtype)
        out[:, start:stop] = np.transpose(boosted_rgb, (2, 0, 1))

    return out
