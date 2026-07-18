"""Precise bike-lane masks: CNN coarse localization, then classical color edge tracing.

texture_detector.TextureEmbeddingDetector answers "is there bike-lane paint
somewhere in this window" -- useful for finding lanes, but its mask is
stamped in WINDOW_PX blocks (22px = 4.4m at this imagery's 0.2m/px), well
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
`_regularize_band` fixes both, per connected component:

1. round off local bulges by morphologically opening the *mask itself* at
   close to its own radius, before deriving a centerline from it at all --
   skeletonizing the raw, un-opened mask directly tended to produce small
   loops/branch points at those bulges (not short dead-end spurs, genuine
   cycles), making the shape ambiguous to treat as one line
2. skeletonize the opened mask, then prune what dead-end spurs remain
3. take the *median* distance-to-edge along the pruned skeleton, evaluated
   against the original (unopened) mask so the opening step doesn't bias
   the width estimate itself -- the majority width, robust to the noisy
   fraction
4. order the pruned skeleton into a single walked line and smooth it with a
   moving average, then redraw the mask as a constant-radius band around
   that smoothed centerline -- the shape a real lane would have, not the
   noisy one the raw color mask traced

Trade-off worth knowing: the opening step (1) is what actually fixes local
bulges/width noise, but it also erodes away thin bridges connecting what
was previously one blobby-but-continuous traced region, so a lane can come
back as more, shorter disconnected segments than before regularization --
cleaner shape per segment, at the cost of more gaps between segments.

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
that neighboring surface low (~0.8%) -- see
scratchpad calibrate_sweep.py in that session for the swept table, not
preserved in this repo.
"""

import numpy as np
from scipy.ndimage import binary_dilation, convolve, distance_transform_edt, label, uniform_filter1d
from skimage.color import rgb2hsv
from skimage.morphology import closing, disk, opening, remove_small_objects, skeletonize

_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

from scripts.detection.base import Detection
from scripts.detection.texture_detector import TextureEmbeddingDetector

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
# closed with a small radius before measuring width.
CLOSING_RADIUS_PX = 2

# Isolated fleck of red (a leaf, a bit of noise) that's too small to be a
# real stretch of lane paint.
MIN_COMPONENT_AREA_PX = 15

# A component's skeleton spurs (from a local color-mask bulge, a fragment
# the closing step didn't fully absorb) are branches that peel off the main
# centerline and dead-end quickly -- real lane centerline doesn't do that.
# Pruned by iteratively eroding skeleton endpoints for a number of steps
# tied to the component's *own* raw median width rather than a fixed pixel
# count, so a wide lane and a narrow one both get spurs shorter than about
# their own width removed, without eating meaningfully into a genuine
# hundreds-of-pixels-long centerline.
SPUR_PRUNE_WIDTH_MULTIPLE = 1.5

# A component whose pruned skeleton doesn't clear this many pixels isn't a
# lane fragment to begin with -- it's residual blob/noise too short to have
# a meaningful centerline (a stray patch of similarly-colored ground, a
# car's reflection) -- and is dropped rather than turned into a fabricated
# band.
MIN_SKELETON_LENGTH_PX = 20

# Spur pruning alone still leaves a wiggly trunk -- the raw color mask's
# local bulges shift the skeleton side to side even where they're too small
# to have registered as a prunable side branch. Smoothing the ordered
# centerline with a moving average (window tied to the component's own
# median radius, same reasoning as spur pruning above) straightens that
# out; a real lane runs straight or gently curving, not in a zigzag.
SMOOTHING_WIDTH_MULTIPLE = 3.0


def _paint_mask(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Pixel-precise "is this bike-lane paint" mask within `roi`, by color alone."""
    hsv = rgb2hsv(image[..., :3])
    hue, saturation = hsv[..., 0], hsv[..., 1]
    hue_distance_from_red = np.minimum(hue, 1.0 - hue)
    is_red = (hue_distance_from_red <= EDGE_HUE_TOLERANCE) & (saturation >= EDGE_MIN_SATURATION)
    return roi & is_red


def _neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    return convolve(skeleton.astype(int), kernel, mode="constant")


def _prune_spurs(skeleton: np.ndarray, iterations: int) -> np.ndarray:
    """Erode skeleton endpoints for `iterations` steps, removing branches shorter than that."""
    pruned = skeleton.copy()
    for _ in range(iterations):
        endpoints = pruned & (_neighbor_count(pruned) == 1)
        if not endpoints.any():
            break
        pruned &= ~endpoints
    return pruned


def _order_skeleton_path(skeleton: np.ndarray) -> list[tuple[int, int]] | None:
    """Order a simple-path skeleton's pixels from one endpoint to the other.

    Returns None if `skeleton` isn't an unambiguous single line (a branch
    point, a loop) -- smoothing a centerline only makes sense when there's
    exactly one path to walk.
    """
    coords = set(zip(*np.nonzero(skeleton)))
    if not coords:
        return None

    neighbor_count = _neighbor_count(skeleton)
    if (neighbor_count[skeleton] > 2).any():
        return None

    endpoints = [rc for rc in coords if neighbor_count[rc] == 1]
    if len(endpoints) != 2:
        return None

    ordered = [endpoints[0]]
    visited = {endpoints[0]}
    current = endpoints[0]
    while len(visited) < len(coords):
        next_point = next(
            (
                (current[0] + dr, current[1] + dc)
                for dr, dc in _NEIGHBOR_OFFSETS
                if (current[0] + dr, current[1] + dc) in coords and (current[0] + dr, current[1] + dc) not in visited
            ),
            None,
        )
        if next_point is None:
            return None
        ordered.append(next_point)
        visited.add(next_point)
        current = next_point
    return ordered


def _smoothed_skeleton_mask(pruned_skeleton: np.ndarray, median_radius: float) -> np.ndarray:
    """Replace `pruned_skeleton` with a moving-average-smoothed version of its own centerline.

    Falls back to `pruned_skeleton` unchanged if it isn't a simple path to
    order (see `_order_skeleton_path`) or is too short relative to the
    smoothing window to smooth meaningfully.
    """
    ordered = _order_skeleton_path(pruned_skeleton)
    window = max(3, round(SMOOTHING_WIDTH_MULTIPLE * median_radius))
    if ordered is None or window >= len(ordered):
        return pruned_skeleton

    rows = np.array([point[0] for point in ordered], dtype=float)
    cols = np.array([point[1] for point in ordered], dtype=float)
    smooth_rows = uniform_filter1d(rows, size=window, mode="nearest")
    smooth_cols = uniform_filter1d(cols, size=window, mode="nearest")

    smoothed = np.zeros_like(pruned_skeleton)
    row_idx = np.clip(np.round(smooth_rows).astype(int), 0, pruned_skeleton.shape[0] - 1)
    col_idx = np.clip(np.round(smooth_cols).astype(int), 0, pruned_skeleton.shape[1] - 1)
    smoothed[row_idx, col_idx] = True
    return smoothed


def _regularize_band(mask: np.ndarray) -> np.ndarray | None:
    """Rebuild `mask` as a constant-width band around its own smoothed medial axis.

    Returns None if `mask` doesn't have enough of a centerline to be worth
    regularizing (see MIN_SKELETON_LENGTH_PX) -- residual noise, not a lane
    fragment.
    """
    raw_distance = distance_transform_edt(mask)
    raw_skeleton = skeletonize(mask)
    raw_widths = raw_distance[raw_skeleton]
    if raw_widths.size == 0:
        return None
    initial_radius = float(np.median(raw_widths))

    # Skeletonizing the raw color-threshold mask directly tends to produce
    # small loops/branch points at local bulges -- not the short dead-end
    # spurs _prune_spurs targets, but cycles that make the shape ambiguous
    # to walk as a single line at all. Opening the *mask* at close to its
    # own radius rounds those bulges off before a centerline is derived from
    # it, rather than trying to patch the resulting skeleton afterwards.
    opened = opening(mask, disk(max(1, round(initial_radius))))
    skeleton = skeletonize(opened) if opened.any() else raw_skeleton
    if not skeleton.any():
        skeleton = raw_skeleton

    prune_iterations = max(1, round(SPUR_PRUNE_WIDTH_MULTIPLE * initial_radius))
    pruned_skeleton = _prune_spurs(skeleton, prune_iterations)
    if pruned_skeleton.sum() < MIN_SKELETON_LENGTH_PX:
        return None

    # Radius is still measured against the original (unopened) mask, so the
    # opening step -- which can only shrink the shape -- doesn't bias the
    # width estimate itself, only the centerline's smoothness.
    median_radius = float(np.median(raw_distance[pruned_skeleton]))
    smoothed_skeleton = _smoothed_skeleton_mask(pruned_skeleton, median_radius)
    distance_from_skeleton = distance_transform_edt(~smoothed_skeleton)
    return distance_from_skeleton <= median_radius


class BikeLaneEdgeDetector:
    """Detector (see detection/base.py) combining CNN coarse localization with
    classical color-based edge tracing for a mask precise enough to measure
    width from.
    """

    def __init__(self, coarse_detector: TextureEmbeddingDetector | None = None):
        self._coarse = coarse_detector or TextureEmbeddingDetector()

    def predict(self, image: np.ndarray) -> list[Detection]:
        coarse = self._coarse.predict(image)
        if not coarse:
            return []

        roi = binary_dilation(coarse[0].mask, iterations=ROI_DILATION_PX)
        mask = _paint_mask(image, roi)
        if not mask.any():
            return []

        mask = closing(mask, disk(CLOSING_RADIUS_PX))
        mask = remove_small_objects(mask, max_size=MIN_COMPONENT_AREA_PX)
        if not mask.any():
            return []

        # Split into connected components so a stray fragment (a parked
        # car's paint that slipped through, a drain cover splitting the
        # path) gets its own Detection rather than being averaged into one
        # width measurement together with the real lane segment. Each
        # component's raw shape is then replaced with its regularized band
        # (see _regularize_band) -- components too small/blob-like to have a
        # real centerline are dropped rather than kept as noise.
        labeled, n_components = label(mask)
        detections = []
        for component_id in range(1, n_components + 1):
            regularized = _regularize_band(labeled == component_id)
            if regularized is None:
                continue
            detections.append(Detection(mask=regularized, score=coarse[0].score, label="bikelane_edge"))
        return detections
