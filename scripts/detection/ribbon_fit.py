"""Fit a road's two edges jointly along its whole length, against continuous imagery.

PARKED -- NOT WIRED INTO ANYTHING. Kept because the machinery is sound and
the failure is understood, not because it works. Nothing imports this.

Status, so this can be picked up without re-deriving it:

- The dynamic program, the smoothness coupling, the CNN reach bound and the
  unobserved-pixel handling all behave as intended.
- The *evidence function* is wrong, and that is what stops it. `_roadness`
  measures similarity to a reference sampled at the centerline, so
  similarity falls off monotonically as a ray moves outward, and the
  inside-vs-outside step therefore scores positive almost immediately. Every
  fit is biased narrow as a result: on a 1500x1500 test region it returned a
  median width of 5.35 m where these streets run about 9 m, with 13 of 43
  ways pinned at the minimum admissible half-width.
- Likely fixes, untested: anchor the reference somewhere other than the
  centerline (the CNN mask interior, say), or replace the difference of
  means with a scale-aware two-sample statistic so a shallow monotonic
  decline cannot outscore a real step.
- One hypothesis already tested and *disproven*: that the prefilter's buffer
  masking was the limiter. Running the fit against raw unmasked .jp2 imagery
  instead gave an identical median of 5.35 m. The buffer is not the problem.

Three real bugs were found and fixed while getting this far, all preserved
below: the inner comparison band reaching across the centerline into the
opposite side, the fit locking onto the prefilter's own mask boundary at
exactly 2x the buffer radius, and unmeasurable stretches silently collapsing
to the minimum offset instead of reporting nothing.

Every earlier width method in this project measured a *binary mask*: a
colour threshold turned pixels into road/not-road, morphology cleaned up the
result, and the width was read off whatever shape survived. That chain is
where the error was. On the representative frame the colour test alone
discarded two thirds of its input and broke the rest into 138 fragments, and
each cleanup step after it moved the boundary that the width is measured
from.

This module never binarises. It works directly on the imagery, and it fits
both edges of a road as a pair, along the entire way at once, rather than
finding them independently at each cross-section.

Why jointly, rather than per cross-section
------------------------------------------
Finding an edge independently at each sample does not work here, and that
was measured rather than assumed. Sampling luminance across one real
carriageway and looking for the strongest gradient produced seven candidate
edges, and the strongest of them (3.57) sat in the *middle* of the road,
not at either edge -- it was a lane marking. Cars and markings routinely
produce a stronger local gradient than the kerb does.

What separates a real edge from a lane marking is not its strength at one
point, it is consistency along the road: a kerb runs the length of the way
at a near-constant offset, a marking does not survive as a coherent
boundary. So the fit maximises edge evidence summed along the way, minus a
penalty for the offset changing between neighbouring samples, and solves it
exactly by dynamic programming over discretised offsets. A strong but
isolated response cannot win, because committing to it forces a large
offset jump on both sides.

What counts as edge evidence
----------------------------
Not a gradient. A gradient peaks at any local change, which is exactly the
failure above. Instead each candidate offset is scored as a *step*: how much
more road-like the imagery is inside that offset than just outside it (see
`_edge_evidence`). "Road-like" is similarity to a reference sampled from the
imagery near the centerline itself, over R, G, B and near-infrared together,
so no fixed colour is assumed and the reference adapts to each way.

That construction is what rejects a lane marking directly rather than by
smoothing it away: the marking is bright, but the surface beyond it is road
again, so the inside-vs-outside contrast across it is small. At the true
kerb the contrast stays large, because what lies beyond really is a
different surface.

Near-infrared carries real weight here. It is the one band the pipeline has
always computed and no detector has ever read, and it separates paving from
vegetation and verge far better than RGB does -- which is what a road edge
usually borders on.
"""

from dataclasses import dataclass

import numpy as np
from rasterio.transform import Affine
from shapely.geometry import LineString

# How far apart to fit cross-sections along a way. Finer than
# centerline_width's 5 m sampling: the smoothness penalty needs neighbouring
# samples close enough that a real edge genuinely varies little between them.
SAMPLE_INTERVAL_M = 2.0

# Largest half-width the fit will consider, and the step it discretises to.
# The step sets the resolution of the answer, so it is well below a pixel;
# the DP cost is linear in the number of offsets, so this is affordable.
MAX_HALF_WIDTH_M = 12.0
OFFSET_STEP_M = 0.1

# Half-widths below this are not considered at all -- a fit that collapsed
# onto the centerline would score well on any uniform surface.
MIN_HALF_WIDTH_M = 1.0

# Width of the band just outside a candidate edge that gets compared against
# the road interior. Wide enough to see past a kerb and a gutter into
# whatever the road actually borders, narrow enough not to reach across a
# footway into a hedge.
OUTER_BAND_M = 2.0

# How far in from a candidate edge to average the interior. Capped so a wide
# road is judged on the surface near its edge rather than diluted by the
# whole carriageway.
INNER_BAND_M = 2.5

# Penalty weight on the half-width changing between neighbouring samples,
# per meter of change. This is the parameter that makes the fit joint rather
# than pointwise: at zero it degenerates into the per-sample gradient search
# that the module docstring describes failing.
SMOOTHNESS_WEIGHT = 3.0

# Reference band sampled around the centerline to define "road-like" for
# this way. Kept narrow so it is carriageway even on the narrowest street.
REFERENCE_HALF_WIDTH_M = 1.0

# How far past the coarse CNN mask's own extent the fit may still place an
# edge. The mask is stamped in TEXTURE_WINDOW_PX blocks, so its boundary can
# legitimately fall up to most of a window short of the real edge; without
# some margin the fit would inherit the mask's 4.4 m quantisation, which is
# the entire thing this module exists to avoid.
REACH_MARGIN_M = 3.0

# Penalty applied to a candidate beyond the CNN mask's reach. Far larger
# than any real evidence value, so it is effectively a bound, but finite so
# that a sample where the mask is missing entirely still degrades to "no
# preference" rather than making the whole path unscoreable.
OUT_OF_REACH_PENALTY = 1000.0

# Fraction of a comparison band that must be real imagery for that band to
# count. Below this the band is mostly prefilter background and any step
# across it is an artefact of the mask, not a surface boundary.
MIN_OBSERVED_FRACTION = 0.6

# Smallest inside-vs-outside step that counts as having located an edge.
# Roadness is scaled in per-band standard deviations of the road surface
# itself, so this is "the surface outside differs by a quarter of the
# variation seen on the road" -- low, because it only has to separate a real
# boundary from flat evidence, not grade how good one is.
MIN_EDGE_EVIDENCE = 0.25


@dataclass
class RibbonFit:
    """A fitted road ribbon: per-sample half-widths either side of the centerline."""

    distance_along_m: np.ndarray
    left_m: np.ndarray
    right_m: np.ndarray
    evidence: np.ndarray  # per-sample edge evidence, summed over both sides
    measured: np.ndarray  # bool: an edge was actually found, on both sides

    @property
    def width_m(self) -> np.ndarray:
        """Fitted width per sample, NaN where no edge was found.

        NaN rather than a number, deliberately. Where the evidence is flat
        the dynamic program still has to return *something*, and what it
        returns is the smallest admissible offset -- so an unmeasurable
        stretch would otherwise be indistinguishable from a genuinely narrow
        road. Reporting nothing there is the honest outcome; the caller
        drops those samples.
        """
        return np.where(self.measured, self.left_m + self.right_m, np.nan)


def _sample_profiles(
    bands: np.ndarray,
    inverse_transform: Affine,
    line: LineString,
    distances_m: np.ndarray,
    offsets_m: np.ndarray,
) -> np.ndarray:
    """Sample `bands` on the grid of (sample point, perpendicular offset).

    Returns (n_samples, n_offsets, n_bands), NaN outside the raster. Offsets
    are signed: negative is one side of the centerline, positive the other.
    """
    n_bands, height, width = bands.shape
    profiles = np.full((len(distances_m), len(offsets_m), n_bands), np.nan, dtype=np.float32)

    for i, distance_m in enumerate(distances_m):
        point = line.interpolate(distance_m)
        before = line.interpolate(max(0.0, distance_m - SAMPLE_INTERVAL_M))
        after = line.interpolate(min(line.length, distance_m + SAMPLE_INTERVAL_M))
        tangent = np.array([after.x - before.x, after.y - before.y])
        norm = np.linalg.norm(tangent)
        if norm == 0:
            continue
        perpendicular = np.array([-tangent[1], tangent[0]]) / norm

        xs = point.x + perpendicular[0] * offsets_m
        ys = point.y + perpendicular[1] * offsets_m
        cols, rows = inverse_transform * (xs, ys)
        cols = np.round(cols).astype(int)
        rows = np.round(rows).astype(int)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        if inside.any():
            profiles[i, inside, :] = bands[:, rows[inside], cols[inside]].T
    return profiles


def _mark_unobserved(profiles: np.ndarray) -> np.ndarray:
    """NaN out pixels the prefilter masked away, so they are never read as surface.

    This is load-bearing, not hygiene. The prefiltered imagery is zeroed
    outside the OSM buffer, which puts a perfectly sharp, perfectly straight
    edge at exactly STREET_BUFFER_METERS from every centerline -- by far the
    strongest step anywhere in the scene, and an artefact of the pipeline
    rather than anything on the ground. Left in, the fit locks onto it and
    reports 2 x the buffer radius as the road width: measured on a test
    region, way after way came back at 12.00-12.10 m against a 6.0 m buffer.

    Treated as unobserved instead, a candidate whose outer band is all
    background yields no contrast either way, so the fit is neither drawn to
    the buffer edge nor pushed off it -- it simply has nothing to go on out
    there, which is the truth.
    """
    unobserved = (profiles == 0).all(axis=2)
    profiles = profiles.copy()
    profiles[unobserved] = np.nan
    return profiles


def _roadness(profiles: np.ndarray, offsets_m: np.ndarray) -> np.ndarray:
    """Per-sample, per-offset similarity to this way's own road surface.

    The reference is the median of the band values within
    REFERENCE_HALF_WIDTH_M of the centerline, taken over the whole way, so it
    adapts to each road's own surface rather than assuming a colour. Returns
    higher values for more road-like pixels.
    """
    near_centerline = np.abs(offsets_m) <= REFERENCE_HALF_WIDTH_M
    reference_pixels = profiles[:, near_centerline, :].reshape(-1, profiles.shape[2])
    reference_pixels = reference_pixels[~np.isnan(reference_pixels).any(axis=1)]
    if len(reference_pixels) == 0:
        return np.full(profiles.shape[:2], np.nan, dtype=np.float32)

    reference = np.median(reference_pixels, axis=0)
    # Scale each band by its own spread near the centerline, so a band that
    # is naturally noisy on road surface doesn't dominate the distance, and
    # near-infrared gets weighted on its real discriminating power rather
    # than its raw magnitude.
    spread = np.maximum(reference_pixels.std(axis=0), 1.0)
    distance = np.sqrt((((profiles - reference) / spread) ** 2).mean(axis=2))
    return -distance


def _edge_evidence(roadness: np.ndarray, offsets_m: np.ndarray, candidates_m: np.ndarray, side: int) -> np.ndarray:
    """Score every candidate half-width as an inside-vs-outside step in roadness.

    `side` is +1 or -1, selecting which half of the profile is being fitted.
    A candidate scores well when the surface stays road-like right up to it
    and stops being road-like just beyond -- which a lane marking fails,
    because the road resumes on its far side.
    """
    evidence = np.zeros((roadness.shape[0], len(candidates_m)), dtype=np.float32)
    signed_offsets = offsets_m * side

    for k, candidate_m in enumerate(candidates_m):
        # The inner band is clamped to this side of the centerline. Without
        # that, a candidate narrower than INNER_BAND_M reaches across the
        # centerline and averages in the *opposite* side's surface, which
        # made narrow candidates score on pixels they don't cover.
        inner = (signed_offsets > max(0.0, candidate_m - INNER_BAND_M)) & (signed_offsets <= candidate_m)
        outer = (signed_offsets > candidate_m) & (signed_offsets <= candidate_m + OUTER_BAND_M)
        if not inner.any() or not outer.any():
            continue
        with np.errstate(invalid="ignore"):
            inner_mean = np.nanmean(roadness[:, inner], axis=1)
            outer_mean = np.nanmean(roadness[:, outer], axis=1)
            # A band that is mostly unobserved cannot support a step either
            # way. Requiring most of both bands to be real is what keeps the
            # fit off the prefilter's own mask boundary (see
            # `_mark_unobserved`), where the outer band is background.
            inner_seen = (~np.isnan(roadness[:, inner])).mean(axis=1)
            outer_seen = (~np.isnan(roadness[:, outer])).mean(axis=1)
        step = np.nan_to_num(inner_mean - outer_mean, nan=0.0)
        observed = (inner_seen >= MIN_OBSERVED_FRACTION) & (outer_seen >= MIN_OBSERVED_FRACTION)
        evidence[:, k] = np.where(observed, step, 0.0)
    return evidence


def _solve_dp(evidence: np.ndarray, candidates_m: np.ndarray) -> np.ndarray:
    """Pick the half-width per sample maximising evidence minus a smoothness penalty.

    Exact, by dynamic programming over the discretised offsets -- the cost
    couples neighbouring samples only, so the global optimum is reachable in
    one forward pass and one backtrace. This is the step that makes an
    isolated strong response (a marking, a car) unable to win: taking it
    would force an offset jump on both sides of it.
    """
    n_samples, n_candidates = evidence.shape
    jump_cost = SMOOTHNESS_WEIGHT * np.abs(candidates_m[:, None] - candidates_m[None, :])

    total = np.zeros((n_samples, n_candidates), dtype=np.float32)
    backpointer = np.zeros((n_samples, n_candidates), dtype=np.int32)
    total[0] = evidence[0]
    for i in range(1, n_samples):
        transition = total[i - 1][:, None] - jump_cost
        backpointer[i] = np.argmax(transition, axis=0)
        total[i] = evidence[i] + transition[backpointer[i], np.arange(n_candidates)]

    path = np.zeros(n_samples, dtype=np.int32)
    path[-1] = int(np.argmax(total[-1]))
    for i in range(n_samples - 1, 0, -1):
        path[i - 1] = backpointer[i, path[i]]
    return candidates_m[path]


def _reach_limit_m(
    surface_profile: np.ndarray, offsets_m: np.ndarray, candidates_m: np.ndarray, side: int
) -> np.ndarray:
    """How far out the CNN road mask reaches, per sample, on one side.

    This is what stops the fit walking off the carriageway onto whatever
    paving adjoins it. The imagery alone cannot tell a carriageway from the
    bike lane, footway and car park it borders -- measured on the
    representative frame, the raw evidence on that side peaks at 11.9 m
    half-width, out at the far edge of the paving, because "road-like" never
    stops being true in between. The CNN discriminant does separate those
    surfaces, just coarsely, so it is used for exactly that: bounding how
    far the search may go, never for locating the edge within it.

    A margin past the mask's own extent is allowed, since the mask is
    stamped in whole scan windows and its edge can sit up to a window short
    of the real one.
    """
    signed_offsets = offsets_m * side
    limits = np.full(surface_profile.shape[0], candidates_m[-1], dtype=np.float32)
    for i in range(surface_profile.shape[0]):
        on_surface = surface_profile[i] & (signed_offsets > 0)
        if not on_surface.any():
            limits[i] = MIN_HALF_WIDTH_M
            continue
        limits[i] = float(signed_offsets[on_surface].max()) + REACH_MARGIN_M
    # Never below the smallest candidate: a sample whose limit fell under it
    # would have no admissible offset at all, and the DP would be choosing
    # among equally impossible options.
    return np.maximum(limits, candidates_m[0])


def fit_ribbon(
    line: LineString,
    bands: np.ndarray,
    transform: Affine,
    surface: np.ndarray | None = None,
    sample_interval_m: float = SAMPLE_INTERVAL_M,
) -> RibbonFit | None:
    """Fit both road edges along `line` against `bands` ((n_bands, H, W) imagery).

    `surface` is the coarse CNN road mask. It bounds how far the fit may
    reach on each side (see `_reach_limit_m`) and is not otherwise used --
    the edge itself is always located from the continuous imagery.

    Returns None if the way is too short to fit, or if no road-like
    reference could be sampled along it.
    """
    if line.length < 3 * sample_interval_m:
        return None

    distances_m = np.arange(0.0, line.length, sample_interval_m)
    offsets_m = np.arange(-MAX_HALF_WIDTH_M - OUTER_BAND_M, MAX_HALF_WIDTH_M + OUTER_BAND_M, OFFSET_STEP_M)
    profiles = _mark_unobserved(_sample_profiles(bands.astype(np.float32), ~transform, line, distances_m, offsets_m))

    roadness = _roadness(profiles, offsets_m)
    if np.isnan(roadness).all():
        return None

    surface_profile = None
    if surface is not None:
        sampled = _sample_profiles(surface[None].astype(np.float32), ~transform, line, distances_m, offsets_m)
        surface_profile = np.nan_to_num(sampled[..., 0], nan=0.0) > 0.5

    candidates_m = np.arange(MIN_HALF_WIDTH_M, MAX_HALF_WIDTH_M, OFFSET_STEP_M)
    solved, evidences = {}, {}
    for side in (-1, +1):
        evidence = _edge_evidence(roadness, offsets_m, candidates_m, side)
        if surface_profile is not None:
            limits = _reach_limit_m(surface_profile, offsets_m, candidates_m, side)
            # A large finite penalty rather than -inf: the DP compares sums
            # across samples, and -inf would poison a whole path rather than
            # just making one offset unattractive at one sample.
            evidence = np.where(candidates_m[None, :] <= limits[:, None], evidence, evidence - OUT_OF_REACH_PENALTY)
        solved[side] = _solve_dp(evidence, candidates_m)
        evidences[side] = evidence

    left_m, right_m = solved[-1], solved[+1]
    evidence = np.array(
        [
            evidences[-1][i, np.argmin(np.abs(candidates_m - left_m[i]))]
            + evidences[+1][i, np.argmin(np.abs(candidates_m - right_m[i]))]
            for i in range(len(distances_m))
        ]
    )
    # A sample counts as measured only if both sides produced a real step.
    # Where a side's evidence is flat, the DP's answer there is an artefact
    # of having to choose, not a located edge.
    measured = np.array(
        [
            evidences[-1][i].max() >= MIN_EDGE_EVIDENCE and evidences[+1][i].max() >= MIN_EDGE_EVIDENCE
            for i in range(len(distances_m))
        ]
    )
    return RibbonFit(
        distance_along_m=distances_m, left_m=left_m, right_m=right_m, evidence=evidence, measured=measured
    )
