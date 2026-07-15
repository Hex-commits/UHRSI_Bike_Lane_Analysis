"""Detect and normalize shadowed pixels within the road/bike-lane mask.

Shadow detection uses a blue-excess index, (B-R)/(B+R): cast shadows on
pavement are lit mainly by scattered blue skylight rather than direct sun,
so they read distinctly bluer than sunlit pavement of the same material. An
earlier version used Tsai's NSVDI (a saturation/value based index), but on
this imagery its saturation term turned out to be dominated by pavement
texture noise rather than shadow, which made detection wildly over-trigger;
verified against a hand-picked ground-truth shadow patch (RGB ~46/62/82,
blue-excess ~0.28) vs. adjacent sunlit pavement (RGB ~123/124/124,
blue-excess ~0.00), blue-excess cleanly separates the two.

Raw per-pixel detection is still noisy (lane markings, oil stains, and
compression artifacts can trip the same index), so the mask is cleaned up
morphologically before use: small gaps get filled (using a disk-shaped
structuring element, so the boundary comes out rounded rather than a
square-cornered staircase) and isolated specks below a minimum area are
dropped, since real cast shadows are spatially coherent blobs, not scattered
single pixels.

Correction brightens each shadowed pixel using a per-band offset derived
from *nearby* sunlit pixels only (a local window), not a single tile-wide
average — the road/bike-lane mask covers a mix of materials, so a single
global correction would rescale using statistics from unrelated materials
elsewhere in the tile. Each band is corrected independently (rather than
sharing one offset) because shadow isn't just dimmer, it's bluer:
neutralizing that color shift requires bringing the R/G/B channels back
into balance, not just brightening all three by the same amount.

A plain "match the local windowed average" correction (two earlier versions
of this module did exactly that, with the transition smoothed various ways)
turned out to systematically undercorrect: a window wide enough to be a
stable brightness estimate (15 m) averages over enough real variation that
its mean sits well below whatever specific pixel a shadow happens to be
touching at any given point, so even a "fully corrected" shadow interior
stayed visibly darker than its actual neighbor -- no amount of smoothing the
*transition* fixes a gap between two *regions*. Two smoothing attempts
confirmed this from opposite directions: Gaussian-blurring the gain field
across the boundary just spread that same regional gap into real sunlit
pixels (a bigger blur made the visible glow bigger, not smaller), and
tapering the gain to 1x purely inside the shadow left a dark under-corrected
rim right at the edge instead.

The fix used here is boundary-matched blending: near the shadow mask's own
edge, the correction target is the *actual value of the nearest sunlit
pixel* (via a Euclidean distance transform), not a windowed average, so the
corrected value is forced to meet its real neighbor almost exactly at the
crossing. Deeper inside the shadow (past `FEATHER_RADIUS_M`), the target
blends over to the windowed average instead, since nearest-single-pixel
matching gets noisy and unrepresentative far from the boundary. The
correction itself is additive (add an offset) rather than multiplicative
(scale by a gain), so a shadow pixel's own texture/noise relative to its
local shadow mean is preserved rather than amplified.
"""

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, label, uniform_filter
from skimage.morphology import disk

# How brightening is capped: a shadow pixel is never boosted more than this
# multiple of its original value, to avoid blowing out shadows that have
# little nearby sunlit reference (e.g. a fully shaded cul-de-sac).
MAX_GAIN = 3.0

# Minimum fraction of a local window that must be sunlit road/bike-lane
# pixels before that window is trusted as a brightness reference.
MIN_REFERENCE_DENSITY = 0.02

# Morphological cleanup of the raw shadow mask: gaps up to this size get
# filled, and isolated shadow regions smaller than this area get dropped.
CLOSING_RADIUS_M = 0.6
MIN_SHADOW_AREA_M2 = 1.5

# Distance in from the shadow mask's own boundary over which the correction
# target blends from "exact value of the nearest sunlit pixel" (right at the
# edge) to "windowed local average" (interior).
FEATHER_RADIUS_M = 2.0


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Return Otsu's threshold that best separates `values` into two classes."""
    hist, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    weight_bg = np.cumsum(hist).astype(np.float64)
    weight_fg = weight_bg[-1] - weight_bg

    running_sum = np.cumsum(hist * bin_centers)
    mean_bg = np.divide(running_sum, weight_bg, out=np.zeros_like(running_sum), where=weight_bg > 0)
    mean_fg = np.divide(
        running_sum[-1] - running_sum, weight_fg, out=np.zeros_like(running_sum), where=weight_fg > 0
    )

    between_class_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return bin_centers[np.argmax(between_class_var)]


def detect_shadow_mask(rgb: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    """Classify shadowed pixels within `road_mask` using blue-excess + Otsu thresholding.

    `rgb` is a (3, H, W) array (R, G, B). `road_mask` is a (H, W) boolean
    array marking the pixels to consider (e.g. the combined bike-lane/street
    mask); pixels outside it are never flagged as shadow. Returns a raw,
    unfiltered (H, W) boolean array — pass it through `clean_shadow_mask`
    before using it for correction or output.
    """
    r, b = rgb[0].astype(np.float32), rgb[2].astype(np.float32)
    blue_excess = (b - r) / (b + r + 1e-6)

    road_values = blue_excess[road_mask]
    if road_values.size == 0:
        return np.zeros(road_mask.shape, dtype=bool)

    threshold = _otsu_threshold(road_values)
    return road_mask & (blue_excess >= threshold)


def clean_shadow_mask(shadow_mask: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Fill small gaps and drop small isolated regions from a raw shadow mask."""
    closing_radius_px = max(1, round(CLOSING_RADIUS_M / pixel_size_m))
    closed = binary_closing(shadow_mask, structure=disk(closing_radius_px))

    labeled, num_labels = label(closed)
    if num_labels == 0:
        return closed

    min_area_px = MIN_SHADOW_AREA_M2 / (pixel_size_m**2)
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= min_area_px
    keep[0] = False
    return keep[labeled]


def _local_masked_mean(values: np.ndarray, mask: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (local mean of `values` over `mask`, local density of `mask`) per pixel."""
    mask_f = mask.astype(np.float32)
    density = uniform_filter(mask_f, size=window, mode="constant")
    local_sum = uniform_filter(values * mask_f, size=window, mode="constant")
    mean = np.divide(local_sum, density, out=np.zeros_like(local_sum), where=density > 0)
    return mean, density


def correct_shadows(
    data: np.ndarray,
    shadow_mask: np.ndarray,
    road_mask: np.ndarray,
    pixel_size_m: float,
    local_radius_m: float = 15.0,
) -> np.ndarray:
    """Brighten shadowed pixels using a local sunlit-pixel reference, per band.

    `data` is a (bands, H, W) array; every band is corrected independently
    using only pixels within `road_mask`, restricted to a `local_radius_m`
    neighborhood around each pixel for the windowed-average part of the
    target. Near the shadow mask's own boundary (within `FEATHER_RADIUS_M`),
    the target blends towards the exact value of the nearest sunlit pixel
    instead, so the correction meets its real neighbor almost exactly at the
    crossing (see module docstring). Pixels outside `road_mask`, or shadow
    pixels with too little nearby sunlit reference, are left unchanged.
    """
    window = 2 * max(1, round(local_radius_m / pixel_size_m)) + 1
    feather_radius_px = FEATHER_RADIUS_M / pixel_size_m
    sunlit_mask = road_mask & ~shadow_mask

    dist_to_boundary, nearest_idx = distance_transform_edt(shadow_mask, return_indices=True)
    taper = np.clip(dist_to_boundary / feather_radius_px, 0.0, 1.0)

    corrected = data.astype(np.float32)
    for band in range(data.shape[0]):
        band_data = corrected[band]

        sunlit_mean, sunlit_density = _local_masked_mean(band_data, sunlit_mask, window)
        shadow_mean, _ = _local_masked_mean(band_data, shadow_mask, window)
        nearest_sunlit_value = band_data[tuple(nearest_idx)]

        has_reference = shadow_mask & (sunlit_density >= MIN_REFERENCE_DENSITY) & (shadow_mean > 1)
        target = nearest_sunlit_value * (1.0 - taper) + sunlit_mean * taper
        max_offset = np.maximum(shadow_mean, 0.0) * (MAX_GAIN - 1.0)
        offset = np.clip(target - shadow_mean, 0.0, max_offset)

        band_data[has_reference] += offset[has_reference]

    info = np.iinfo(data.dtype)
    return np.clip(corrected, info.min, info.max).astype(data.dtype)
