"""Precise bike-lane masks: CNN coarse localization, then classical color edge tracing.

Two detectors live here. `BikeLaneEdgeDetector` is the full pipeline: coarse
CNN region, pixel-precise color threshold inside it, per-component shape
regularization, directional bridging. `RoadEdgeDetector` is now only the
coarse CNN mask -- its color test was removed after being measured (it
discarded two thirds of its region of interest into 138 fragments, and every
cleanup step after moved the boundary a width gets measured from). The
asymmetry is deliberate: bike-lane paint has a strong color cue so a color
test localizes it, road surface does not. Road width is measured separately
from OSM centerlines (detection/centerline_width.py).

The CNN mask is only a region-of-interest: it is stamped in TEXTURE_WINDOW_PX
blocks (~4.4m at 0.2m/px), far wider than a real ~2m lane, so a width read
directly off it would measure the window grid. Inside the ROI the lane's true
edges are found by color threshold -- bike-lane paint is saturated red, and
that signal is precise to the pixel.

That raw color mask still isn't the lane's shape: it inherits local dropouts
and bulges, so `_regularize_band` rebuilds each connected component as a
constant-width band around a smoothed centerline (`_binned_centerline`: PCA
for the dominant axis, bin pixels along it, each bin's average position is a
centerline point, the median distance-to-edge is the radius). Binning is used
rather than skeletonization, which on this porous ~72%-recall mask produced a
branchy structure no spur-pruning could reduce to one clean line.

Regularizing per component can leave a lane split where the color signal
dropped out entirely (a parked car, deep shadow), so
`BikeLaneEdgeDetector.predict` bridges components whose endpoints are close
*and* aimed at each other (BRIDGE_* below); the direction check is what makes
generous distance safe.

Hue/saturation thresholds are NOT reused from redness.py -- its loose
enhancement thresholds recalled only ~55% of true path pixels here. The
values below were swept for ~72% recall against a real cycle-track crop while
keeping false positives against neighboring sidewalk low (~0.8%).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt, label, uniform_filter1d
from skimage.color import rgb2hsv
from skimage.draw import line as draw_line
from skimage.morphology import closing, disk, remove_small_objects

from pipeline.config import (
    COARSE_BRIDGE_M,
    COARSE_BRIDGE_ORIENTATIONS,
    INPUT_CHUNK_RES_M,
    scaled_area_px,
    scaled_px,
)
from scripts.detection.base import Detection
from scripts.detection.texture_detector import TextureEmbeddingDetector, bike_lane_detector, road_detector

# Colour thresholds are dimensionless -- a hue is a hue at any resolution --
# so they are stated outright. Every *_PX constant below is a ground distance
# in disguise and is stated at the 0.2 m/px it was swept at, then scaled to the
# chunk's actual resolution (see config.TUNED_AT_M).
EDGE_HUE_TOLERANCE = 0.15

EDGE_MIN_SATURATION = 0.07

# A ground distance, so it follows the chunk's resolution like the rest.
COARSE_BRIDGE_PX = max(0, round(COARSE_BRIDGE_M / INPUT_CHUNK_RES_M))

ROI_DILATION_PX = scaled_px(8)

CLOSING_RADIUS_PX = scaled_px(2)

MIN_COMPONENT_AREA_PX = scaled_area_px(15)

CENTERLINE_BIN_WIDTH_PX = scaled_px(3)

# Bins, not pixels: CENTERLINE_BIN_WIDTH_PX already carries the scale, so a
# minimum bin count spans the same ground however fine the imagery is.
MIN_CENTERLINE_BINS = 7

SMOOTHING_WIDTH_MULTIPLE = 3.0

BRIDGE_MAX_GAP_RADIUS_MULTIPLE = 4.0

BRIDGE_ALIGNMENT_COS_MIN = 0.82

BRIDGE_TANGENT_LOOKBACK_POINTS = 5


SHADOW_EXCLUSION_MARGIN_PX = scaled_px(5)

ROAD_MIN_COMPONENT_AREA_PX = scaled_area_px(200)

def _paint_mask(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Pixel-precise "is this bike-lane paint" mask within `roi`, by color alone."""
    hsv = rgb2hsv(image[..., :3])
    hue, saturation = hsv[..., 0], hsv[..., 1]
    hue_distance_from_red = np.minimum(hue, 1.0 - hue)
    is_red = (hue_distance_from_red <= EDGE_HUE_TOLERANCE) & (saturation >= EDGE_MIN_SATURATION)
    return roi & is_red


def _binned_centerline(mask: np.ndarray) -> list[tuple[float, float]] | None:
    """Ordered centerline points for `mask`, via PCA + binning along its own dominant axis.

    Each point is the average position of one bin's pixels, in order along
    that axis from one end of the component to the other. Returns None if
    the component doesn't span enough bins to be worth treating as a lane
    fragment (see MIN_CENTERLINE_BINS).
    """
    rows, cols = np.nonzero(mask)
    coords = np.stack([rows, cols], axis=1).astype(float)
    centroid = coords.mean(axis=0)
    centered = coords - centroid

    _, _, principal_axes = np.linalg.svd(centered, full_matrices=False)
    along_axis = principal_axes[0]
    perp_axis = np.array([-along_axis[1], along_axis[0]])

    positions_along = centered @ along_axis
    offsets_perp = centered @ perp_axis
    bin_index = np.floor(positions_along / CENTERLINE_BIN_WIDTH_PX).astype(int)

    unique_bins = np.unique(bin_index)
    if len(unique_bins) < MIN_CENTERLINE_BINS:
        return None

    points = []
    for current_bin in unique_bins:
        in_bin = bin_index == current_bin
        point = centroid + positions_along[in_bin].mean() * along_axis + offsets_perp[in_bin].mean() * perp_axis
        points.append((float(point[0]), float(point[1])))
    return points


@dataclass
class Segment:
    mask: np.ndarray
    points: list[tuple[int, int]]
    radius: float


def _regularize_band(mask: np.ndarray) -> Segment | None:
    """Rebuild `mask` as a constant-width band around its own binned-and-smoothed centerline.

    Returns None if `mask` doesn't have enough of a centerline to be worth
    regularizing (see MIN_CENTERLINE_BINS) -- residual noise, not a lane
    fragment.
    """
    points = _binned_centerline(mask)
    if points is None:
        return None

    distance = distance_transform_edt(mask)
    row_idx = np.clip(np.round([point[0] for point in points]).astype(int), 0, mask.shape[0] - 1)
    col_idx = np.clip(np.round([point[1] for point in points]).astype(int), 0, mask.shape[1] - 1)
    radius = float(np.median(distance[row_idx, col_idx]))

    window = max(3, round(SMOOTHING_WIDTH_MULTIPLE * radius))
    if window < len(points):
        smooth_rows = uniform_filter1d(row_idx.astype(float), size=window, mode="nearest")
        smooth_cols = uniform_filter1d(col_idx.astype(float), size=window, mode="nearest")
        row_idx = np.clip(np.round(smooth_rows).astype(int), 0, mask.shape[0] - 1)
        col_idx = np.clip(np.round(smooth_cols).astype(int), 0, mask.shape[1] - 1)

    smoothed_points = list(zip(row_idx.tolist(), col_idx.tolist()))
    centerline_mask = np.zeros(mask.shape, dtype=bool)
    centerline_mask[row_idx, col_idx] = True
    band = distance_transform_edt(~centerline_mask) <= radius
    return Segment(mask=band, points=smoothed_points, radius=radius)


def _endpoint_tangent(points: list[tuple[int, int]], at_start: bool) -> np.ndarray:
    """Unit vector pointing outward from the path at its start or end point."""
    lookback = min(BRIDGE_TANGENT_LOOKBACK_POINTS, len(points) - 1)
    if at_start:
        tip, inward = np.array(points[0], dtype=float), np.array(points[lookback], dtype=float)
    else:
        tip, inward = np.array(points[-1], dtype=float), np.array(points[-1 - lookback], dtype=float)
    vector = tip - inward
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _bridge(segment_a: Segment, segment_b: Segment, shape: tuple[int, int]) -> np.ndarray | None:
    """Return a bridge mask connecting `segment_a` and `segment_b` end to end, or None.

    Tries all four endpoint pairings (start/end of each); bridges the first
    pair that's both close enough and aimed at each other (see
    BRIDGE_MAX_GAP_RADIUS_MULTIPLE / BRIDGE_ALIGNMENT_COS_MIN).
    """
    max_gap = BRIDGE_MAX_GAP_RADIUS_MULTIPLE * min(segment_a.radius, segment_b.radius)
    for a_at_start in (True, False):
        endpoint_a = np.array(segment_a.points[0 if a_at_start else -1], dtype=float)
        tangent_a = _endpoint_tangent(segment_a.points, a_at_start)
        for b_at_start in (True, False):
            endpoint_b = np.array(segment_b.points[0 if b_at_start else -1], dtype=float)
            gap_vector = endpoint_b - endpoint_a
            gap = float(np.linalg.norm(gap_vector))
            if gap == 0 or gap > max_gap:
                continue
            bridge_direction = gap_vector / gap
            tangent_b = _endpoint_tangent(segment_b.points, b_at_start)
            if np.dot(bridge_direction, tangent_a) < BRIDGE_ALIGNMENT_COS_MIN:
                continue
            if np.dot(-bridge_direction, tangent_b) < BRIDGE_ALIGNMENT_COS_MIN:
                continue

            rows, cols = draw_line(*endpoint_a.round().astype(int), *endpoint_b.round().astype(int))
            line_mask = np.zeros(shape, dtype=bool)
            line_mask[rows, cols] = True
            radius = min(segment_a.radius, segment_b.radius)
            return distance_transform_edt(~line_mask) <= radius
    return None


def _line_element(length_px: int, degrees: float) -> np.ndarray:
    """A one-pixel-wide line of `length_px` at `degrees`, for morphology."""
    size = length_px | 1
    element = np.zeros((size, size), dtype=bool)
    centre = size // 2
    angle = np.deg2rad(degrees)
    offsets = np.arange(-centre, centre + 1)
    rows = np.rint(centre - offsets * np.sin(angle)).astype(int)
    cols = np.rint(centre + offsets * np.cos(angle)).astype(int)
    inside = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
    element[rows[inside], cols[inside]] = True
    return element


def connect_coarse(mask: np.ndarray, bridge_px: int = COARSE_BRIDGE_PX) -> np.ndarray:
    """Close gaps along linear structures in the coarse mask.

    The coarse scan drops out in bands along a continuous lane, because its
    window grid is anchored to the image while the lane runs at an angle to
    it: the window's paint fill oscillates, and windows that straddle the
    lane's edge are half asphalt and score as not-lane. The detector has
    already confirmed lane either side of such a gap, so bridging it adds no
    claim the scan did not make -- see COARSE_BRIDGE_M.

    Closed with a *line* element rather than a disk, swept over orientations
    and unioned: a line joins detections lying along it and leaves isolated
    blobs their own size, where a disk would grow every detection sideways and
    merge a lane into whatever borders it.
    """
    if bridge_px < 3 or not mask.any():
        return mask
    out = np.zeros_like(mask)
    for degrees in np.linspace(0, 180, COARSE_BRIDGE_ORIENTATIONS, endpoint=False):
        out |= binary_closing(mask, structure=_line_element(bridge_px, degrees))
    return out


def _traced_components(
    image: np.ndarray,
    coarse: list[Detection],
    surface_mask: Callable[[np.ndarray, np.ndarray], np.ndarray],
    min_component_area_px: int,
) -> list[np.ndarray]:
    """Find one surface inside a coarse detection's region, as connected components.

    Grow the coarse mask into a region of interest, keep the pixels inside
    it that pass `surface_mask`'s color test, close small gaps, drop specks.
    Only the bike-lane detector uses this now; `surface_mask` is kept as a
    parameter because it is the one part that knows what is being looked for.
    """
    # Bridge the scan-grid dropouts before growing the region of interest, so
    # a lane broken into bands by window alignment is traced as one run.
    roi = binary_dilation(connect_coarse(coarse[0].mask), iterations=ROI_DILATION_PX)
    mask = surface_mask(image, roi)
    if not mask.any():
        return []

    mask = closing(mask, disk(CLOSING_RADIUS_PX))
    mask = remove_small_objects(mask, max_size=min_component_area_px)
    if not mask.any():
        return []

    labeled, n_components = label(mask)
    return [labeled == component_id for component_id in range(1, n_components + 1)]


class BikeLaneEdgeDetector:
    """Detector (see detection/base.py) combining CNN coarse localization with
    classical color-based edge tracing for a mask precise enough to measure
    width from.
    """

    def __init__(self, coarse_detector: TextureEmbeddingDetector | None = None):
        self._coarse = coarse_detector or bike_lane_detector()

    def predict(self, image: np.ndarray, coarse: list[Detection] | None = None) -> list[Detection]:
        """`coarse` lets a caller that already ran the coarse scan pass it in
        directly, instead of paying for another CNN sliding-window scan here --
        by far the most expensive step in this pipeline.
        """
        if coarse is None:
            coarse = self._coarse.predict(image)
        if not coarse:
            return []

        segments = []
        for component in _traced_components(image, coarse, _paint_mask, MIN_COMPONENT_AREA_PX):
            segment = _regularize_band(component)
            if segment is not None:
                segments.append(segment)
        if not segments:
            return []

        combined = np.zeros(image.shape[:2], dtype=bool)
        for segment in segments:
            combined |= segment.mask
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                bridge_mask = _bridge(segments[i], segments[j], image.shape[:2])
                if bridge_mask is not None:
                    combined |= bridge_mask

        labeled_final, n_final = label(combined)
        return [
            Detection(mask=labeled_final == component_id, score=coarse[0].score, label="bikelane_edge")
            for component_id in range(1, n_final + 1)
        ]


class RoadEdgeDetector:
    """Detector (see detection/base.py) for road surface: the CNN mask, and nothing else.

    The colour test this used to run inside the coarse region was removed --
    on the representative frame it discarded two thirds of the region of
    interest into 138 fragments, and the closing and small-component filter it
    then needed each moved the boundary a width is measured from. The surface
    is now the thresholded CNN mask directly, at a stricter threshold
    (ROAD_SCORE_THRESHOLD) to compensate for no longer being refined
    downstream; the dilation and closing went with the colour test.

    This mask cannot locate an edge precisely -- stamped in TEXTURE_WINDOW_PX
    blocks (~4.4m), it answers "is there road here", not "where does it end".
    Width comes from detection/ribbon_fit.py, which fits edges against the
    continuous imagery.
    """

    def __init__(self, coarse_detector: TextureEmbeddingDetector | None = None):
        self._coarse = coarse_detector or road_detector()

    def predict(
        self, image: np.ndarray, coarse: list[Detection] | None = None, shadow: np.ndarray | None = None
    ) -> list[Detection]:
        """One Detection per connected component of the road surface."""
        mask = self.surface_mask(image, coarse, shadow)
        if not mask.any():
            return []
        score = coarse[0].score if coarse else 0.0
        labeled, n_components = label(mask)
        return [
            Detection(mask=labeled == component_id, score=score, label="road")
            for component_id in range(1, n_components + 1)
        ]

    def surface_mask(
        self, image: np.ndarray, coarse: list[Detection] | None = None, shadow: np.ndarray | None = None
    ) -> np.ndarray:
        """The road surface: the thresholded CNN mask, minus shadow, with specks dropped.

        `shadow` is the prefiltered tile's shadow band (band 6); where set,
        the surface is cut. In deep shadow this imagery carries almost no
        surface information -- shadowed carriageway and non-carriageway are
        indistinguishable to within noise, and a discriminant fit and scored
        on those very pixels still misclassifies 35% (vs 20% sunlit) -- so
        cutting it turns a silent error into an honest coverage gap. The cut
        extends SHADOW_EXCLUSION_MARGIN_PX past the mask, since a real shadow
        edge is a soft penumbra the Otsu threshold cuts a hard line through.
        The small-component filter runs last, since cutting shadow out
        mid-road is what leaves fragments too small to be meaningful.
        """
        if coarse is None:
            coarse = self._coarse.predict(image)
        if not coarse:
            return np.zeros(image.shape[:2], dtype=bool)

        mask = coarse[0].mask.copy()
        if shadow is not None:
            excluded = shadow.astype(bool)
            if SHADOW_EXCLUSION_MARGIN_PX > 0:
                excluded = binary_dilation(excluded, iterations=SHADOW_EXCLUSION_MARGIN_PX)
            mask &= ~excluded
        return remove_small_objects(mask, max_size=ROAD_MIN_COMPONENT_AREA_PX)
