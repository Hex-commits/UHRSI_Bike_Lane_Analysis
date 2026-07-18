"""Boost saturation of reddish pixels, so painted bike-lane paint stands out more.

Works in HSV: pixels whose hue is close to red *and* already have some
minimum saturation get their saturation multiplied up (clamped to 1.0);
everything else -- gray asphalt, vegetation, low-saturation noise where hue
is undefined/meaningless -- is left alone. This is deliberately narrower
than a generic contrast stretch (tried and reverted earlier): it only
touches pixels that already read as red, rather than stretching the whole
tile's dynamic range and risking new artifacts everywhere.

Hue tolerance and the saturation floor are calibrated against the one
hand-annotated instance in this repo: paint pixels there sit at hue
~0.025-0.046 (9-17 degrees) and saturation ~0.11-0.30, clearly within the
ranges used below.
"""

import numpy as np
from skimage.color import hsv2rgb, rgb2hsv

# Hue distance (circular, in [0, 0.5]) from pure red (hue=0) that still
# counts as the bike-lane paint's color. ~29 degrees, a generous margin
# around the calibrated 9-17 degree range.
RED_HUE_TOLERANCE = 0.08

# Below this saturation, hue is noisy/near-meaningless (close to gray) --
# excluded so boosting doesn't amplify color noise in neutral pavement.
MIN_SATURATION_FOR_BOOST = 0.05

# Multiplier applied to saturation for qualifying pixels, clamped to 1.0.
SATURATION_BOOST = 1.8


def boost_red_saturation(rgb: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    """Increase saturation of reddish pixels within `road_mask`.

    `rgb` is a (3, H, W) array (R, G, B). Pixels outside `road_mask`, or
    that aren't red enough to qualify, are returned unchanged.
    """
    info = np.iinfo(rgb.dtype)
    hwc = np.transpose(rgb, (1, 2, 0))
    hsv = rgb2hsv(hwc)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    hue_distance_from_red = np.minimum(hue, 1.0 - hue)
    qualifies = road_mask & (hue_distance_from_red <= RED_HUE_TOLERANCE) & (saturation >= MIN_SATURATION_FOR_BOOST)

    boosted_saturation = saturation.copy()
    boosted_saturation[qualifies] = np.clip(saturation[qualifies] * SATURATION_BOOST, 0.0, 1.0)

    boosted_rgb = hsv2rgb(np.stack([hue, boosted_saturation, value], axis=-1))
    boosted_rgb = np.clip(boosted_rgb * info.max, info.min, info.max).astype(rgb.dtype)
    return np.transpose(boosted_rgb, (2, 0, 1))
