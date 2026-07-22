import numpy as np
from skimage.color import hsv2rgb, rgb2hsv
from pipeline.config import (
    MIN_SATURATION_FOR_BOOST,
    REDNESS_ROWS_PER_BLOCK as ROWS_PER_BLOCK,
    RED_HUE_TOLERANCE,
    SATURATION_BOOST,
)






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
