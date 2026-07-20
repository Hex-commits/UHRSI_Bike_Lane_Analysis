"""Fit a road's two edges jointly along its whole length, against continuous imagery.

PARKED -- NOT WIRED INTO ANYTHING. Kept because the machinery is sound and
the failure is understood, not because it works. Nothing imports this.

Status, so this can be picked up without re-deriving it: the dynamic program,
smoothness coupling, CNN reach bound and unobserved-pixel handling all behave
as intended. The *evidence function* is what stops it -- `_roadness` measures
similarity to a reference sampled at the centerline, so similarity falls off
monotonically outward and the inside-vs-outside step scores positive almost
immediately, biasing every fit narrow (5.35 m median where these streets run
~9 m, 13 of 43 ways pinned at the minimum half-width). Likely fixes, untested:
anchor the reference off the centerline (the CNN mask interior), or replace
the difference of means with a scale-aware two-sample statistic. Already
disproven: that the prefilter's buffer masking was the limiter -- against raw
unmasked .jp2 imagery the median was identical (5.35 m).

Three real bugs were found and fixed on the way, all preserved below: the
inner comparison band reaching across the centerline, the fit locking onto
the prefilter's mask boundary at 2x the buffer radius, and unmeasurable
stretches silently collapsing to the minimum offset instead of reporting
nothing.

Unlike every earlier width method here, this never binarises (a colour
threshold + morphology chain is where their error was -- the colour test
alone discarded two thirds of its input into 138 fragments, each cleanup step
moving the boundary). It works directly on the imagery and fits both edges as
a pair along the entire way at once.

Why jointly: finding an edge independently per cross-section fails here.
Sampling luminance across one carriageway, the strongest gradient sat in the
*middle* of the road -- a lane marking; cars and markings routinely beat the
kerb's local gradient. What separates a real edge from a marking is
consistency along the road, so the fit maximises edge evidence summed along
the way minus a penalty for the offset changing between samples, solved
exactly by dynamic programming over discretised offsets -- an isolated strong
response can't win, since taking it forces a large offset jump on both sides.

Edge evidence is a *step*, not a gradient (which peaks at any change): how
much more road-like the imagery is inside a candidate offset than just
outside it (`_edge_evidence`). "Road-like" is similarity to a reference
sampled near the centerline over R, G, B and near-infrared, so no fixed
colour is assumed. This rejects a lane marking directly -- the road resumes
on its far side, so the contrast across it is small, while at a real kerb it
stays large. Near-infrared carries real weight: the one band no detector ever
read, it separates paving from the vegetation and verge a road edge borders.
"""

from dataclasses import dataclass

import numpy as np
from rasterio.transform import Affine
from shapely.geometry import LineString

SAMPLE_INTERVAL_M = 2.0

MAX_HALF_WIDTH_M = 12.0
OFFSET_STEP_M = 0.1

MIN_HALF_WIDTH_M = 1.0

OUTER_BAND_M = 2.0

INNER_BAND_M = 2.5

SMOOTHNESS_WEIGHT = 3.0

REFERENCE_HALF_WIDTH_M = 1.0

REACH_MARGIN_M = 3.0

OUT_OF_REACH_PENALTY = 1000.0

MIN_OBSERVED_FRACTION = 0.6

MIN_EDGE_EVIDENCE = 0.25


@dataclass
class RibbonFit:
    """A fitted road ribbon: per-sample half-widths either side of the centerline."""

    distance_along_m: np.ndarray
    left_m: np.ndarray
    right_m: np.ndarray
    evidence: np.ndarray
    measured: np.ndarray

    @property
    def width_m(self) -> np.ndarray:
        """Fitted width per sample, NaN where no edge was found.

        NaN, not a number: where evidence is flat the DP still returns the
        smallest admissible offset, so an unmeasurable stretch would otherwise
        be indistinguishable from a genuinely narrow road. The caller drops
        those samples.
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

    Load-bearing, not hygiene. The prefiltered imagery is zeroed outside the
    OSM buffer, putting a perfectly sharp edge at exactly STREET_BUFFER_METERS
    from every centerline -- the strongest step in the scene, a pipeline
    artefact. Left in, the fit locks onto it and reports 2x the buffer radius
    as the width (12.00-12.10 m against a 6.0 m buffer on a test region).
    Treated as unobserved, an all-background outer band yields no contrast
    either way, so the fit has nothing to go on out there -- which is the truth.
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
        inner = (signed_offsets > max(0.0, candidate_m - INNER_BAND_M)) & (signed_offsets <= candidate_m)
        outer = (signed_offsets > candidate_m) & (signed_offsets <= candidate_m + OUTER_BAND_M)
        if not inner.any() or not outer.any():
            continue
        with np.errstate(invalid="ignore"):
            inner_mean = np.nanmean(roadness[:, inner], axis=1)
            outer_mean = np.nanmean(roadness[:, outer], axis=1)
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

    This stops the fit walking off the carriageway onto adjoining paving. The
    imagery alone cannot tell a carriageway from the bike lane, footway and
    car park it borders -- raw evidence peaks at 11.9 m half-width out at the
    far edge of the paving, because "road-like" never stops being true in
    between. The CNN discriminant separates those surfaces coarsely, so it is
    used only to bound how far the search may go, never to locate the edge. A
    margin past the mask's extent is allowed, since the mask is stamped in
    whole scan windows and can sit a window short of the real edge.
    """
    signed_offsets = offsets_m * side
    limits = np.full(surface_profile.shape[0], candidates_m[-1], dtype=np.float32)
    for i in range(surface_profile.shape[0]):
        on_surface = surface_profile[i] & (signed_offsets > 0)
        if not on_surface.any():
            limits[i] = MIN_HALF_WIDTH_M
            continue
        limits[i] = float(signed_offsets[on_surface].max()) + REACH_MARGIN_M
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
    measured = np.array(
        [
            evidences[-1][i].max() >= MIN_EDGE_EVIDENCE and evidences[+1][i].max() >= MIN_EDGE_EVIDENCE
            for i in range(len(distances_m))
        ]
    )
    return RibbonFit(
        distance_along_m=distances_m, left_m=left_m, right_m=right_m, evidence=evidence, measured=measured
    )
