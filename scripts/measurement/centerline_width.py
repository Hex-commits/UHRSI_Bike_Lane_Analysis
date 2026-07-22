from dataclasses import dataclass

import numpy as np
from rasterio.transform import Affine
from shapely.geometry import LineString, MultiLineString
from pipeline.config import (
    GAP_TOLERANCE_M,
    MAX_HALF_WIDTH_M,
    RAY_STEP_PX,
    WIDTH_SAMPLE_INTERVAL_M as SAMPLE_INTERVAL_M,
    WIDTH_TANGENT_HALF_SPAN_M as TANGENT_HALF_SPAN_M,
)







@dataclass
class WidthSample:
    """One cross-section measured perpendicular to a centerline."""

    distance_along_m: float
    width_m: float
    buffer_limited: bool
    unbounded: bool


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
                width_m=float(left_m + right_m + pixel_size_m),
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
