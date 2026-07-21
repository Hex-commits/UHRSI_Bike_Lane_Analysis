"""Detect and normalize shadowed pixels within the road/bike-lane mask.

Detection uses a blue-excess index, (B-R)/(B+R): cast shadows on pavement are
lit mainly by scattered blue skylight, so they read distinctly bluer than
sunlit pavement of the same material (ground-truth shadow ~0.28 vs sunlit
~0.00). An earlier version used Tsai's NSVDI, but on this imagery its
saturation term was dominated by pavement texture noise and over-triggered.
Raw detection is still noisy (lane markings, oil stains, compression), so the
mask is cleaned morphologically: small gaps filled with a disk element
(rounded boundary, not a staircase), isolated specks below a minimum area
dropped, since real shadows are coherent blobs.

Correction brightens each shadowed pixel with a per-band offset from *nearby*
sunlit pixels (a local window), not a tile-wide average -- the mask spans a
mix of materials. Bands are corrected independently because shadow isn't just
dimmer, it's bluer, so the R/G/B channels must be rebalanced, not scaled
together.

A plain "match the local windowed average" correction (two earlier versions)
systematically undercorrected: a window wide enough to be stable (15 m)
averages over enough variation that its mean sits below whatever pixel the
shadow actually touches, so even a "fully corrected" interior stayed darker
than its neighbor -- no amount of smoothing the *transition* fixes a gap
between two *regions*. The fix is boundary-matched blending: near the mask
edge the target is the actual value of the nearest sunlit pixel (via distance
transform), so the correction meets its real neighbor at the crossing; deeper
in (past `FEATHER_RADIUS_M`) it blends to the windowed average, since
single-pixel matching gets noisy far from the boundary. The correction is
additive, not multiplicative, so a pixel's own texture is preserved not
amplified.

Two further bugs, only visible on large shadows (wider than `local_radius_m`,
e.g. a building's cast shadow across a street): pixels whose windowed sunlit
density fell below `MIN_REFERENCE_DENSITY` were left uncorrected instead of
falling back to the nearest-sunlit target, leaving a hard dark border where a
too-wide shadow met corrected pavement; and the "nearest sunlit pixel"
distance transform ran against *not-shadow* rather than *sunlit*, so near the
buffer edge the reference could be background/nodata outside the buffer.
"""

import numpy as np
from scipy.ndimage import binary_closing, distance_transform_edt, label, uniform_filter
from skimage.morphology import disk

MAX_GAIN = 3.0

MIN_REFERENCE_DENSITY = 0.02

CLOSING_RADIUS_M = 0.6
MIN_SHADOW_AREA_M2 = 1.5

FEATHER_RADIUS_M = 2.0

BLEED_RADIUS_M = 1.0


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

    `data` is a (bands, H, W) array; each band is corrected independently
    using only pixels within `road_mask`. Near the mask boundary (within
    `FEATHER_RADIUS_M`) the target blends towards the nearest sunlit pixel's
    exact value, so the correction meets its real neighbor at the crossing;
    deeper in it blends towards a `local_radius_m`-windowed average, since
    single-pixel matching gets noisy far from the boundary (see module
    docstring).

    Every shadow pixel is corrected however far it sits from a reference:
    where the windowed average lacks enough sunlit pixels to trust
    (`MIN_REFERENCE_DENSITY`, i.e. shadows wider than `local_radius_m`) the
    target falls back to the nearest sunlit pixel regardless of depth, rather
    than leaving a hard dark border where a too-wide shadow meets corrected
    pavement.

    The correction also bleeds a short distance (`BLEED_RADIUS_M`) past the
    mask into sunlit pixels, fading to zero, so it doesn't stop dead at the
    mask's hard Otsu edge -- which cuts through an optically soft penumbra
    that would otherwise read as a visible boundary.
    """
    window = 2 * max(1, round(local_radius_m / pixel_size_m)) + 1
    feather_radius_px = FEATHER_RADIUS_M / pixel_size_m
    bleed_radius_px = BLEED_RADIUS_M / pixel_size_m
    sunlit_mask = road_mask & ~shadow_mask

    dist_to_boundary, nearest_idx = distance_transform_edt(~sunlit_mask, return_indices=True)
    taper = np.clip(dist_to_boundary / feather_radius_px, 0.0, 1.0)

    dist_from_shadow, nearest_shadow_idx = distance_transform_edt(~shadow_mask, return_indices=True)
    outward_taper = np.clip(1.0 - dist_from_shadow / bleed_radius_px, 0.0, 1.0)
    bleed_mask = road_mask & ~shadow_mask & (dist_from_shadow <= bleed_radius_px)

    corrected = data.astype(np.float32)
    for band in range(data.shape[0]):
        band_data = corrected[band]

        sunlit_mean, sunlit_density = _local_masked_mean(band_data, sunlit_mask, window)
        shadow_mean, _ = _local_masked_mean(band_data, shadow_mask, window)
        nearest_sunlit_value = band_data[tuple(nearest_idx)]

        trusted_average = sunlit_density >= MIN_REFERENCE_DENSITY
        blended_target = nearest_sunlit_value * (1.0 - taper) + sunlit_mean * taper
        target = np.where(trusted_average, blended_target, nearest_sunlit_value)

        baseline = np.maximum(shadow_mean, 1.0)
        max_offset = baseline * (MAX_GAIN - 1.0)
        offset = np.clip(target - baseline, 0.0, max_offset)

        band_data[shadow_mask] += offset[shadow_mask]

        nearest_shadow_offset = offset[tuple(nearest_shadow_idx)]
        band_data[bleed_mask] += (nearest_shadow_offset * outward_taper)[bleed_mask]

    info = np.iinfo(data.dtype)
    return np.clip(corrected, info.min, info.max).astype(data.dtype)
