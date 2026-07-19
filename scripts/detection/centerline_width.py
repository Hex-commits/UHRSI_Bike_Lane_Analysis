"""Measure a surface's width by casting rays perpendicular to a known centerline.

The methods in width.py both infer a direction from the mask itself -- the
medial axis, or PCA on the pixel coordinates. Both work on a single isolated
stretch of surface and both break down at tile scale, because traced
pavement connects into networks: a T-junction has no dominant axis, so PCA
on it returns something diagonal and the "width" that follows is measured
across the junction rather than across either road. Measured on one 300x300 m
test region, that produced a 28 m road (a T-junction) and a 55 m road (a
parking lot).

Here the direction comes from OSM's road centerlines instead, which the
pipeline already fetches and caches (`scripts/osm_features.py`). For each
sampled point along a way, the local tangent gives a perpendicular, and the
width is how far the traced surface actually extends along that
perpendicular in each direction. That fixes three things at once:

- junctions stop mattering, because each OSM way is measured as its own unit
  regardless of what it touches
- surfaces with no centerline are never measured at all, so pavement the
  coarse texture discriminant wrongly picked up does not become a road
  unless OSM says a road is there
- results are keyed to OSM ways, so a width can be joined back to that way's
  own tags

This is *not* the rejected idea of using the OSM buffer as a region of
interest. The buffer is not the measurement: the ray stops where the traced
asphalt stops. The buffer does still bound how far a ray can possibly get,
since the prefiltered imagery is masked to it, so `buffer_limited` records
any sample that ran into that edge rather than into a real surface
boundary -- a clipped measurement stays visible instead of silently
reading as a narrow road.
"""

from dataclasses import dataclass

import numpy as np
from rasterio.transform import Affine
from shapely.geometry import LineString, MultiLineString

# How far apart to sample along a way. 5 m is short enough to catch a real
# change in width along a street and long enough that one tile's road
# network stays a few thousand samples rather than a few hundred thousand.
SAMPLE_INTERVAL_M = 5.0

# How far a ray may travel from the centerline before giving up. 15 m is
# already wider than any half-carriageway in this imagery, so a ray that
# reaches it is following something other than the road it started on --
# across a junction, or out into an adjoining car park.
MAX_HALF_WIDTH_M = 15.0

# A ray crosses gaps up to this long and keeps going. The traced surface is
# interrupted by lane markings, cars and tree shadow, and a ray that stopped
# at the first one would measure the distance to the nearest painted line
# rather than the road. Sized to pass a lane marking or a car's width but
# not a whole sidewalk.
GAP_TOLERANCE_M = 1.5

# Step along the ray, as a fraction of a pixel. Below 1 px so a ray crossing
# diagonally can't skip past a thin feature between samples.
RAY_STEP_PX = 0.5

# Distance either side of a sample point used to estimate the local tangent.
# Wide enough not to be dominated by the vertex spacing of an OSM way, which
# can be very fine on a curve.
TANGENT_HALF_SPAN_M = 2.5


@dataclass
class WidthSample:
    """One cross-section measured perpendicular to a centerline."""

    distance_along_m: float
    width_m: float
    buffer_limited: bool  # a ray stopped at masked-out background, not a real edge
    unbounded: bool  # a ray hit the cap without finding an edge -- width is a lower bound


@dataclass
class CenterlineWidth:
    """Aggregated width for one centerline.

    Read `median_m` together with `unbounded_fraction`: where rays did not
    find an edge, the width is a lower bound reported at the ray cap rather
    than an observed measurement.
    """

    median_m: float
    mean_m: float
    min_m: float
    max_m: float
    n_samples: int
    buffer_limited_fraction: float
    unbounded_fraction: float


def _iter_lines(geometry):
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def _ray_extent_m(
    mask: np.ndarray,
    inverse_transform: Affine,
    origin_xy: tuple[float, float],
    direction: np.ndarray,
    pixel_size_m: float,
) -> tuple[float, bool, bool]:
    """Walk from `origin_xy` along `direction` until the surface ends.

    Returns (distance travelled in meters, hit the masked-out background,
    never found an edge at all). The last of those qualifies the
    measurement: a ray that runs the full MAX_HALF_WIDTH_M with surface the
    whole way returns the cap, not an observed edge, and a caller needs to
    be able to tell those apart from real ones.
    """
    height, width = mask.shape
    step_m = RAY_STEP_PX * pixel_size_m
    max_steps = int(MAX_HALF_WIDTH_M / step_m)
    gap_tolerance_steps = int(GAP_TOLERANCE_M / step_m)

    last_on_surface = 0.0
    gap_steps = 0
    for step in range(1, max_steps + 1):
        distance_m = step * step_m
        x = origin_xy[0] + direction[0] * distance_m
        y = origin_xy[1] + direction[1] * distance_m
        col, row = inverse_transform * (x, y)
        col, row = int(col), int(row)
        if not (0 <= row < height and 0 <= col < width):
            # Ran off the raster itself; treat like hitting the buffer edge,
            # since either way the surface was not observed to end.
            return last_on_surface, True, False
        if mask[row, col]:
            last_on_surface = distance_m
            gap_steps = 0
        else:
            gap_steps += 1
            if gap_steps > gap_tolerance_steps:
                return last_on_surface, False, False
    return last_on_surface, False, True


def measure_along_centerline(
    line: LineString,
    mask: np.ndarray,
    transform: Affine,
    pixel_size_m: float,
    sample_interval_m: float = SAMPLE_INTERVAL_M,
) -> list[WidthSample]:
    """Sample `line` and measure the traced surface's perpendicular extent at each point.

    Samples where the centerline itself doesn't land on the surface are
    dropped rather than recorded as zero width: they mean the road wasn't
    detected there at all (deep shadow, or an OSM way with no imagery under
    it), which is a coverage gap, not a road of width zero.
    """
    inverse_transform = ~transform
    height, width = mask.shape
    samples = []

    for distance_m in np.arange(0.0, line.length, sample_interval_m):
        point = line.interpolate(distance_m)
        col, row = inverse_transform * (point.x, point.y)
        col, row = int(col), int(row)
        if not (0 <= row < height and 0 <= col < width) or not mask[row, col]:
            continue

        before = line.interpolate(max(0.0, distance_m - TANGENT_HALF_SPAN_M))
        after = line.interpolate(min(line.length, distance_m + TANGENT_HALF_SPAN_M))
        tangent = np.array([after.x - before.x, after.y - before.y])
        norm = np.linalg.norm(tangent)
        if norm == 0:
            continue
        tangent /= norm
        perpendicular = np.array([-tangent[1], tangent[0]])

        origin = (point.x, point.y)
        left_m, left_clipped, left_open = _ray_extent_m(mask, inverse_transform, origin, perpendicular, pixel_size_m)
        right_m, right_clipped, right_open = _ray_extent_m(mask, inverse_transform, origin, -perpendicular, pixel_size_m)
        samples.append(
            WidthSample(
                distance_along_m=float(distance_m),
                width_m=float(left_m + right_m + pixel_size_m),  # + the centerline pixel itself
                buffer_limited=bool(left_clipped or right_clipped),
                unbounded=bool(left_open or right_open),
            )
        )
    return samples


def aggregate(samples: list[WidthSample]) -> CenterlineWidth | None:
    """Summarize samples for one way. Median, not mean, is the headline figure.

    A way that runs through a junction picks up a few samples whose rays
    escape down the crossing road before the gap tolerance stops them, and
    those are large enough to move a mean noticeably while leaving a median
    alone.
    """
    if not samples:
        return None
    widths = np.array([sample.width_m for sample in samples])
    return CenterlineWidth(
        median_m=float(np.median(widths)),
        mean_m=float(widths.mean()),
        min_m=float(widths.min()),
        max_m=float(widths.max()),
        n_samples=len(samples),
        buffer_limited_fraction=float(np.mean([sample.buffer_limited for sample in samples])),
        unbounded_fraction=float(np.mean([sample.unbounded for sample in samples])),
    )
