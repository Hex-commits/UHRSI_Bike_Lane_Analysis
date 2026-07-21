from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks

SAMPLE_STEP_M = 0.05
SMOOTH_SIGMA_M = 0.15

MIN_RUN_M = 0.0

EDGE_PROMINENCE_SIGMA = 2.5

ILLUMINATION_RATIO = 25.0

NDVI_VEGETATION = 0.15
SHADOW_RUN_FRACTION = 0.2

REDNESS_PAINT = 0.05
BRIGHTNESS_MARKING = 165.0

MARKING_MAX_WIDTH_M = 0.5
MARKING_SMOOTH_M = 0.05
MARKING_BASELINE_M = 1.0
MARKING_MIN_EXCESS = 18.0

ASPHALT = "asphalt"
SHADOW_UNKNOWN = "unknown (shadow)"


@dataclass
class Edge:
    """One located boundary, and what kind of boundary it is."""

    distance_m: float
    material: float
    illumination: float

    @property
    def is_illumination(self) -> bool:
        return self.illumination > ILLUMINATION_RATIO * max(self.material, 1e-9)


@dataclass
class Run:
    """A stretch of one material between two edges."""

    start_m: float
    end_m: float
    label: str
    ndvi: float
    redness: float
    brightness: float
    shadow_fraction: float

    @property
    def width_m(self) -> float:
        return self.end_m - self.start_m

    def contains(self, distance_m: float) -> bool:
        return self.start_m <= distance_m <= self.end_m


@dataclass
class CrossSection:
    """One measured cross-section, in the profile's own coordinate."""

    distance_m: np.ndarray
    bands: np.ndarray
    edges: list[Edge]
    runs: list[Run] = field(default_factory=list)
    shadow_fraction: float = 0.0
    shadow: np.ndarray | None = None
    lane: np.ndarray | None = None  # the detected lane mask, sampled along this profile

    @property
    def material_edges(self) -> list[Edge]:
        return [e for e in self.edges if not e.is_illumination]

    def lane_edge_m(self, lane_centre_m: float) -> float | None:
        """Where the detected lane begins, in profile coordinates.

        Read from the lane mask itself rather than from the spectral
        segmentation. `run_at(...).start_m` was the obvious candidate and is
        wrong: where road and lane are the same asphalt with no resolvable
        edge between them, segmentation returns one run spanning both, whose
        `start_m` is the profile's start rather than the lane's near edge --
        which silently collapsed every separated cycle track to a 0 m gap.

        Takes the contiguous block of lane mask containing `lane_centre_m`,
        not the first block along the profile: a cross-section 12 m long can
        clip a different lane fragment on its way out, and the block the
        lane's own centreline sits in is the one being measured.

        Returns None when the mask is absent or the centreline does not land
        on it -- the caller then has no lane edge to measure from.
        """
        if self.lane is None or not self.lane.any():
            return None
        index = int(np.argmin(np.abs(self.distance_m - lane_centre_m)))
        if not self.lane[index]:
            return None
        start = index
        while start > 0 and self.lane[start - 1]:
            start -= 1
        return float(self.distance_m[start])

    def run_at(self, distance_m: float) -> Run | None:
        for run in self.runs:
            if run.contains(distance_m):
                return run
        return None

    def shadow_fraction_between(self, start_m: float, end_m: float) -> float:
        """Shadow fraction over just the [start_m, end_m] span of this section.

        Lets a caller judge only the stretch it actually reads from pixels --
        e.g. the OSM-fallback gap reads the lane side but not the road behind
        it, so shadow over the road should not count against the measurement.
        Falls back to the whole-section fraction if no per-sample shadow was
        sampled, and returns 0 for an empty span.
        """
        if self.shadow is None:
            return self.shadow_fraction
        window = (self.distance_m >= start_m) & (self.distance_m <= end_m)
        return float(self.shadow[window].mean()) if window.any() else 0.0


def features(bands: np.ndarray) -> dict[str, np.ndarray]:
    """Brightness plus the illumination-invariant ratios."""
    r, g, b, nir = bands.astype(np.float32)
    eps = 1e-6
    total = r + g + b + eps
    return {
        "brightness": total / 3,
        "r_chroma": r / total,
        "g_chroma": g / total,
        "b_chroma": b / total,
        "ndvi": (nir - r) / (nir + r + eps),
        "redness": (r - (g + b) / 2) / total,
    }


def sample(bands, inverse_transform, origin_xy, direction, start_m, end_m,
           extra: dict | None = None):
    """Bilinear-sample every band along `direction` from `start_m` to `end_m`.

    Sampling at SAMPLE_STEP_M rather than pixel centres is what makes the
    subpixel fit meaningful: a gradient peak is resolved by several samples
    instead of falling between two.
    """
    distance = np.arange(start_m, end_m + SAMPLE_STEP_M, SAMPLE_STEP_M)
    xs = origin_xy[0] + direction[0] * distance
    ys = origin_xy[1] + direction[1] * distance
    cols, rows = inverse_transform * (xs, ys)
    coords = np.vstack([rows, cols])
    profile = np.stack([map_coordinates(band, coords, order=1, mode="nearest")
                        for band in bands.astype(np.float32)])
    sampled = {name: map_coordinates(array.astype(np.float32), coords, order=0, mode="nearest")
               for name, array in (extra or {}).items()}
    return distance, profile, sampled


def _gradient_magnitude(stack: np.ndarray) -> np.ndarray:
    sigma = SMOOTH_SIGMA_M / SAMPLE_STEP_M
    smoothed = np.stack([gaussian_filter1d(c, sigma) for c in np.atleast_2d(stack)])
    return np.linalg.norm(np.gradient(smoothed, SAMPLE_STEP_M, axis=1), axis=0)


def detect_edges(distance: np.ndarray, profile: np.ndarray) -> list[Edge]:
    """Locate material boundaries, subpixel, tagging illumination boundaries."""
    f = features(profile)
    material = _gradient_magnitude(
        np.vstack([np.stack([f["r_chroma"], f["g_chroma"], f["b_chroma"]]) * 3.0,
                   f["ndvi"][None] * 1.5])
    )
    illumination = _gradient_magnitude(f["brightness"][None] / 255.0)

    spread = np.percentile(material, 75) - np.percentile(material, 25)
    prominence = max(np.median(material) + EDGE_PROMINENCE_SIGMA * spread, 0.02)
    peaks, _ = find_peaks(material, prominence=prominence,
                          distance=max(1, int(MIN_RUN_M / SAMPLE_STEP_M)))

    edges = []
    for peak in peaks:
        if 0 < peak < len(material) - 1:
            y0, y1, y2 = material[peak - 1], material[peak], material[peak + 1]
            denominator = y0 - 2 * y1 + y2
            offset = (np.clip(0.5 * (y0 - y2) / denominator, -1, 1)
                      if abs(denominator) > 1e-9 else 0.0)
        else:
            offset = 0.0
        window = slice(max(0, peak - 4), peak + 5)
        edges.append(Edge(distance_m=float(distance[peak] + offset * SAMPLE_STEP_M),
                          material=float(material[peak]),
                          illumination=float(illumination[window].max())))
    return edges


def detect_markings(distance: np.ndarray, profile: np.ndarray) -> list[tuple[float, float]]:
    """Locate narrow bright painted lines, returned as (start_m, end_m) spans.

    A marking is a brightness spike, not a step: bright, flanked by darker
    asphalt on *both* sides, only a few decimetres wide. Detected on lightly
    smoothed brightness minus a broad local baseline (so a narrow line stands
    proud regardless of the asphalt's absolute brightness) and width-limited so
    a car or manhole doesn't qualify.
    """
    brightness = features(profile)["brightness"]
    spike = gaussian_filter1d(brightness, MARKING_SMOOTH_M / SAMPLE_STEP_M)
    baseline = gaussian_filter1d(brightness, MARKING_BASELINE_M / SAMPLE_STEP_M)
    excess = spike - baseline

    max_width = max(2, int(MARKING_MAX_WIDTH_M / SAMPLE_STEP_M))
    peaks, props = find_peaks(excess, prominence=MARKING_MIN_EXCESS, width=(None, max_width))
    return [(float(distance[0] + lo * SAMPLE_STEP_M),
             float(distance[0] + hi * SAMPLE_STEP_M))
            for lo, hi in zip(props["left_ips"], props["right_ips"])]


def _marking_edges(markings: list[tuple[float, float]]) -> list[Edge]:
    """The two flanks of each marking, as material edges (illumination 0)."""
    edges = []
    for lo, hi in markings:
        edges.append(Edge(distance_m=lo, material=1.0, illumination=0.0))
        edges.append(Edge(distance_m=hi, material=1.0, illumination=0.0))
    return edges


def segment(distance, profile, edges, shadow=None, markings=None) -> list[Run]:
    """Split at material edges and label each stretch."""
    f = features(profile)
    shadow = np.zeros(distance.shape, dtype=bool) if shadow is None else shadow.astype(bool)
    markings = markings or []

    cuts = sorted(int(round((e.distance_m - distance[0]) / SAMPLE_STEP_M))
                  for e in edges if not e.is_illumination)
    bounds = [0, *cuts, len(distance) - 1]

    runs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a or (b - a) * SAMPLE_STEP_M < MIN_RUN_M:
            continue
        ndvi = float(f["ndvi"][a:b].mean())
        redness = float(f["redness"][a:b].mean())
        brightness = float(f["brightness"][a:b].mean())
        shadow_fraction = float(shadow[a:b].mean())

        mid = 0.5 * (distance[a] + distance[b])
        in_marking = any(lo - 1e-6 <= mid <= hi + 1e-6 for lo, hi in markings)

        if shadow_fraction >= SHADOW_RUN_FRACTION:
            label = SHADOW_UNKNOWN
        elif in_marking:
            label = "bright/marking"
        elif ndvi > NDVI_VEGETATION:
            label = "vegetation"
        elif redness > REDNESS_PAINT:
            label = "red paint"
        elif brightness > BRIGHTNESS_MARKING:
            label = "bright/marking"
        else:
            label = ASPHALT

        runs.append(Run(start_m=float(distance[a]), end_m=float(distance[b]), label=label,
                        ndvi=ndvi, redness=redness, brightness=brightness,
                        shadow_fraction=shadow_fraction))
    return runs


def measure(bands, inverse_transform, origin_xy, direction, start_m, end_m,
            shadow_mask=None, lane_mask=None) -> CrossSection:
    """Extract, find edges in, and segment one cross-section."""
    extra = {}
    if shadow_mask is not None:
        extra["shadow"] = shadow_mask
    if lane_mask is not None:
        extra["lane"] = lane_mask
    distance, profile, sampled = sample(bands, inverse_transform, origin_xy, direction,
                                        start_m, end_m, extra)
    shadow = sampled.get("shadow")
    lane = sampled.get("lane")
    markings = detect_markings(distance, profile)
    edges = detect_edges(distance, profile) + _marking_edges(markings)
    edges.sort(key=lambda e: e.distance_m)
    runs = segment(distance, profile, edges, shadow, markings)
    return CrossSection(distance_m=distance, bands=profile, edges=edges, runs=runs,
                        shadow_fraction=float(shadow.mean()) if shadow is not None else 0.0,
                        shadow=shadow,
                        lane=lane.astype(bool) if lane is not None else None)


def edge_precision_m(sections: list[CrossSection], max_offset_m: float = 0.5) -> dict:
    """How much a shared edge wanders between neighbouring cross-sections.

    Consecutive sections a couple of metres apart cut the *same* boundary, so
    spread in where it's reported is measurement noise -- the measured version
    of the "subpixel" claim. Conditional on matching: an edge that fails to
    reappear within `max_offset_m` in the neighbour is excluded, so this is the
    precision of edges that survive. Read it as optimistic.
    """
    offsets = []
    for previous, current in zip(sections[:-1], sections[1:]):
        for edge in current.material_edges:
            candidates = [abs(edge.distance_m - other.distance_m)
                          for other in previous.material_edges]
            if candidates and min(candidates) < max_offset_m:
                offsets.append(min(candidates))
    if not offsets:
        return {"n_pairs": 0}
    offsets = np.array(offsets)
    return {"n_pairs": int(offsets.size),
            "median_m": float(np.median(offsets)),
            "p90_m": float(np.percentile(offsets, 90)),
            "gsd_ratio": float(np.median(offsets) / 0.2)}
