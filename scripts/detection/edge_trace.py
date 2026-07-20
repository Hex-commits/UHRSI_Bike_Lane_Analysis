"""Precise bike-lane masks: CNN coarse localization, then classical color edge tracing.

Two detectors live here, and they are no longer variations on one another.

`BikeLaneEdgeDetector` is the full pipeline this module was built for:
coarse CNN region, pixel-precise color threshold inside it, per-component
shape regularization, directional bridging.

`RoadEdgeDetector` is now only the coarse CNN mask. Its color test and the
morphology serving it were removed after being measured: on the
representative frame the asphalt color test discarded two thirds of its own
region of interest (109,526 px -> 36,437 px) and left 138 fragments, and
every cleanup step after it moved the boundary that a width gets measured
from. What replaced it is a stricter CNN threshold and no per-pixel step at
all -- see `RoadEdgeDetector`. Road width is measured separately, from OSM
centerlines, in detection/centerline_width.py.

That asymmetry is deliberate and worth stating: bike-lane paint has a strong
color cue, so a color test genuinely localizes it. Road surface does not,
so a color test there was mostly discarding real road.

texture_detector.TextureEmbeddingDetector answers "is there bike-lane paint
somewhere in this window" -- useful for finding lanes, but its mask is
stamped in TEXTURE_WINDOW_PX blocks (22px = 4.4m at this imagery's 0.2m/px), well
over twice the width of a real lane (~10px/2m, see config.py's
TRAINING_CHIP_SIZE_PX comment). A width measurement taken directly from that
mask would measure the window grid, not the lane.

This module fixes that by treating the CNN mask as a region-of-interest
only, then finding the lane's true edges within it via plain color
thresholding: bike-lane paint is saturated red, and that signal is precise
down to the pixel rather than the window.

The color-threshold mask alone still isn't the lane's actual shape, though
-- it inherits every local dropout (shade dappling missing a patch) and
bulge (a fragment of similarly-colored ground bleeding in) pixel by pixel,
so its edges are noisy and uneven in a way a real lane's aren't: a lane
holds close to one width along its length rather than "bleeding out"
unevenly, and runs straight or gently curving rather than zigzagging.
`_regularize_band` fixes both, per connected component, via `_binned_centerline`:

1. find the component's own dominant direction (PCA on its pixel
   coordinates) and bin its pixels along that axis; each bin's *average*
   position is one centerline point. This assumes the component doesn't
   hairpin -- true for a bike lane, which a real one doesn't do.
2. take the *median* distance-to-edge sampled at those centerline points --
   the majority width, robust to the noisy fraction
3. smooth the (already-ordered, bin-index order) centerline with a moving
   average, then redraw the mask as a constant-radius band around it -- the
   shape a real lane would have, not the noisy one the raw color mask traced

Skeletonizing the raw mask (tried first) doesn't work here: this mask's
recall is only ~72% (see below), leaving it visibly porous, and
skeletonize() is very sensitive to exactly that kind of small-scale
boundary noise -- it turned a single lane into a highly branchy/loopy
structure that no amount of spur-pruning reduced to one clean line (dozens
of disconnected debris pixels, not one trunk), even after directly
morphologically smoothing the mask first. Binning avoids the problem
entirely by never computing a topological skeleton: noise gets averaged
into a bin's centroid rather than becoming a spurious branch.

Regularizing per component can still leave a lane split across several
pieces -- a stretch with zero color-threshold hits (a parked car, deep tree
shadow) has no pixels to bin at all, so it becomes a real gap between
components, not a shape defect `_binned_centerline` can smooth away. A real
lane doesn't have gaps just because something blocked the color signal over
part of it, so after regularizing each component separately,
`BikeLaneEdgeDetector.predict` re-connects them: any two components whose
nearest endpoints are close *and* both already heading straight at each
other (see BRIDGE_* below) get a straight bridge drawn between them at the
narrower segment's width, then components joined by a bridge are merged
back into one Detection. The direction check is what makes this safe to be
generous with distance -- it's what tells a lane continuing past an
occluding car apart from some unrelated red feature that just happens to
sit nearby.

The hue/saturation thresholds are NOT reused from redness.py: that module's
thresholds were deliberately loose (an enhancement pass wants to touch
anything even slightly red), which turned out to matter here -- an early
version of this module that did reuse them recalled only ~55% of true path
pixels sampled from a real cycle-track crop in
idop20rgbi_32_404_5757_1_nw_2025_bikelanes.tif (window x=4300,y=1330 --
tree-branch shadow dappling the path causes local hue/saturation dropout),
producing a speckled, gappy mask whose distance-transform width came out
noisy (0.40-5.26 m spread for one path, when the true width should vary
little along its length). The thresholds below were swept against that same
crop against a real adjacent-sidewalk sample (window x=4300,y=1330,
y=0:20,x=440:700) to find a looser hue tolerance that recovers much more of
the true path (recall ~0.72) while keeping the false-positive rate against
that neighboring surface low (~0.8%) -- see scratchpad calibrate_sweep.py in
that session for the swept table, not preserved in this repo.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label, uniform_filter1d
from skimage.color import rgb2hsv
from skimage.draw import line as draw_line
from skimage.morphology import closing, disk, remove_small_objects

from scripts.detection.base import Detection
from scripts.detection.texture_detector import TextureEmbeddingDetector, bike_lane_detector, road_detector

# Hue distance (circular, in [0, 0.5]) from pure red (hue=0) that still
# counts as lane/path paint -- looser than redness.py's RED_HUE_TOLERANCE
# (0.08) because that value, calibrated for an *enhancement* pass, missed
# too much real paint under partial shade to trace a clean edge from (see
# module docstring).
EDGE_HUE_TOLERANCE = 0.15

# Saturation floor for a pixel to count as paint. Slightly higher than
# redness.py's MIN_SATURATION_FOR_BOOST (0.05) -- paired with the wider hue
# tolerance above, this keeps the false-positive rate against a real
# adjacent sidewalk sample low (~0.8%) despite recalling much more of the
# true path than the boost thresholds did.
EDGE_MIN_SATURATION = 0.07

# How far (in pixels) to grow the CNN's coarse window-block mask before
# searching it for paint. The coarse mask is centered on the lane but wider
# than it (see module docstring); a real paint edge near the border of a
# scanned window could sit just outside the stamped block, so the ROI grows
# a bit past it. Cheap to over-grow -- color thresholding below still only
# keeps pixels that actually look like paint.
ROI_DILATION_PX = 8

# Gaps in the traced paint (worn patches, leaf litter, a shadow the color
# threshold missed) shouldn't split one lane into disconnected slivers --
# closed with a small radius before measuring width. Deliberately small: a
# bigger isotropic closing here was tried and rejected -- it fattened the
# *cross-track* width wherever two nearby blobs got dilated into each
# other, not just the gaps between them (mean width on a real test crop
# jumped from a plausible ~3-4m to an implausible ~5.5m). Real along-the-lane
# gaps are bridged directionally after regularization instead (see BRIDGE_*
# below).
CLOSING_RADIUS_PX = 2

# Isolated fleck of red (a leaf, a bit of noise) that's too small to be a
# real stretch of lane paint.
MIN_COMPONENT_AREA_PX = 15

# Bin size (px) along a component's own dominant axis for `_binned_centerline`.
# Small relative to a lane's own width so real curvature isn't flattened,
# large enough that each bin still averages over several noisy pixels.
CENTERLINE_BIN_WIDTH_PX = 3

# A component spanning fewer bins than this isn't a lane fragment to begin
# with -- it's residual blob/noise too short to have a meaningful centerline
# (a stray patch of similarly-colored ground, a car's reflection) -- and is
# dropped rather than turned into a fabricated band.
MIN_CENTERLINE_BINS = 7

# Binning already averages out a lot of noise, but bin-to-bin the centerline
# can still jitter a little; smoothed with a moving average (window tied to
# the component's own radius, same reasoning as the bin width above) so a
# real lane comes out running straight or gently curving, not in a zigzag.
SMOOTHING_WIDTH_MULTIPLE = 3.0

# Bridging two already-regularized segments end to end: max gap (as a
# multiple of the smaller segment's own radius) that's still worth
# connecting. Sized generously -- a real occlusion (a parked car sitting on
# the lane) can span several meters -- because BRIDGE_ALIGNMENT_COS_MIN
# below is what actually guards against connecting unrelated nearby
# features, not this distance cap.
BRIDGE_MAX_GAP_RADIUS_MULTIPLE = 4.0

# How closely the bridge direction (segment A's endpoint straight to
# segment B's endpoint) must match each segment's own direction of travel
# at that endpoint, as a cosine similarity -- both segments must be heading
# essentially straight at each other, not just nearby. cos(35 deg) =~ 0.82:
# generous enough for a lane's own gentle curvature, tight enough to reject
# a perpendicular feature (a crosswalk stripe) that happens to sit close by.
BRIDGE_ALIGNMENT_COS_MIN = 0.82

# How many points back from an endpoint to look when estimating that
# endpoint's direction of travel -- far enough that bin-level noise in the
# centerline doesn't dominate the estimate.
BRIDGE_TANGENT_LOOKBACK_POINTS = 5


# How far past the detected shadow mask the road surface is also cut, in
# pixels (5 px = 1.0 m at this imagery's 0.2 m/px, matching prefiltering's
# SHADOW_CUT_MARGIN_M). Same reasoning: a real shadow edge is a soft
# penumbra and the mask's Otsu threshold draws a hard line through it, so
# pixels just outside the mask still read as partially shadowed and are no
# more classifiable than the ones inside it.
SHADOW_EXCLUSION_MARGIN_PX = 5

# A road is an order of magnitude wider than a lane (~6-7 m of carriageway
# vs ~2 m), so the "too small to be real" floor scales with it: 200 px at
# 0.2 m/px is 8 m^2, about the footprint of a single car, below which a
# fragment isn't a stretch of carriageway.
ROAD_MIN_COMPONENT_AREA_PX = 200

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
    points: list[tuple[int, int]]  # ordered centerline, endpoint to endpoint
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
    parameter rather than inlined because it is the one part that knows what
    is being looked for, and that separation is what made it possible to
    measure the road color test's cost and then remove it.
    """
    roi = binary_dilation(coarse[0].mask, iterations=ROI_DILATION_PX)
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
        """`coarse` lets a caller that already ran (or otherwise obtained) the
        coarse scan pass it in directly, instead of paying for another run of
        the CNN sliding-window scan here -- it's by far the most expensive
        step in this pipeline.
        """
        if coarse is None:
            coarse = self._coarse.predict(image)
        if not coarse:
            return []

        # Regularize each component separately (see _regularize_band) --
        # components too small/blob-like to have a real centerline are
        # dropped rather than kept as noise.
        segments = []
        for component in _traced_components(image, coarse, _paint_mask, MIN_COMPONENT_AREA_PX):
            segment = _regularize_band(component)
            if segment is not None:
                segments.append(segment)
        if not segments:
            return []

        # A stretch with no color-threshold hits at all (a parked car, deep
        # shadow) becomes a real gap between components -- reconnect any two
        # that are both close and aimed straight at each other, then let
        # connected-component labeling on the combined result merge bridged
        # segments back into one Detection. See module docstring for why
        # direction, not just distance, gates this.
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

    The colour test this used to run inside the coarse region is gone. It
    was the single most destructive step in the chain -- on the
    representative frame it discarded two thirds of the region of interest
    (109,526 px -> 36,437 px) and shattered what survived into 138
    fragments, which then needed a morphological closing and a
    small-component filter to be usable at all. Every one of those steps
    moved the surface boundary, and the boundary is what a width is measured
    from.

    So the surface is now the thresholded CNN mask directly, at a stricter
    threshold to compensate for no longer being refined downstream (see
    ROAD_SCORE_THRESHOLD). The dilation went with the colour test, since it
    existed only to widen the region that test searched -- keeping it would
    now just inflate every road by 8 px on each side. The closing went too:
    the CNN mask is stamped in whole windows and is not speckled the way a
    per-pixel colour threshold is.

    What this mask cannot do is locate an edge precisely: it is stamped in
    TEXTURE_WINDOW_PX blocks, so its resolution is 4.4 m at this imagery's
    scale, and its score ramps over ~5 m across a real road edge rather than
    stepping at it. It answers "is there road here", not "where does it
    end". Width comes from detection/ribbon_fit.py, which fits the edges
    against the continuous imagery instead.
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

        `shadow` is the prefiltered tile's own shadow band (band 6), and
        where it is set the surface is cut. This is not a cosmetic filter.
        In deep shadow this imagery carries almost no surface information at
        all: shadowed carriageway and shadowed non-carriageway measure the
        same to within noise -- median hue distance from red 0.405 vs 0.405,
        median saturation 0.506 vs 0.519, mean RGB (44,60,81) vs (41,57,79)
        -- and fitting a discriminant on those pixels and scoring it on the
        very same pixels still misclassifies 35%, against 20% for the
        equivalent sunlit comparison. Whatever the detector marks as road
        inside shadow is therefore close to a coin flip, and cutting it
        turns a silent error into an honest coverage gap.

        The cut extends SHADOW_EXCLUSION_MARGIN_PX past the detected mask for
        the same reason `SHADOW_CUT_MARGIN_M` exists in prefiltering: a real
        shadow edge is a soft penumbra, and the mask's Otsu threshold draws
        a hard line through it, so pixels just outside still read as
        partially shadowed.

        The small-component filter runs last, since cutting shadow out of
        the middle of a run of road is exactly what leaves fragments too
        small to be meaningful.
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
