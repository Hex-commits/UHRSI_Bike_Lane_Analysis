"""Dissolve a detected surface mask into one coherent network polygon layer.

The road surface comes out of the detector as a raster mask, and it is
fragmented for reasons that have nothing to do with where road actually is:
it is stamped in whole scan windows, so its edges follow the scan grid; and
shadowed pixels are cut from it entirely, because the detector cannot
classify them (see `RoadEdgeDetector.surface_mask`). A road crossed by a
building's shadow therefore arrives as two pieces with a hole between them.

This module closes that geometrically: vectorize the mask, grow every
polygon by `buffer_m`, union the lot, then shrink by the same amount. That
is a morphological closing done in vector space, and it has exactly the
properties a closing has -- gaps narrower than twice `buffer_m` are filled,
polygons closer than twice `buffer_m` are merged into one, and the outer
boundary is otherwise left where it was.

Two consequences worth being explicit about, since both are inherent to the
operation rather than to this implementation:

- **It partially undoes the shadow cut.** A shadow gap narrower than
  2 x `buffer_m` gets filled straight back in. That is not a bug -- bridging
  a road under a shadow is usually what you want from a *network* geometry
  -- but the filled area is inferred, not detected, and `dissolve_surface`
  reports how much of it there is so the trade is visible rather than
  silent.
- **It does not restore a true outline.** Closing fills concavities; it
  cannot recover a boundary the mask never had. The result is a cleaner
  network, not a more accurate one, and it must not be used to measure
  width -- that still comes from detection/centerline_width.py.

Note what closing does *not* do: remove anything. An isolated false-positive
blob is grown and shrunk back to itself, and survives untouched. Discarding
outliers takes the opposite operation, an opening -- shrink first, grow back
-- which erases whatever is too thin to survive the shrink. That is the
optional `open_m` stage, off by default because it is destructive: it
deletes detected surface, and at a radius anywhere near `buffer_m` it
deletes roads rather than noise. The two compose as close-then-open: join
what belongs together, then drop what is too thin to be road.

**The opening does not, however, remove this data's false positives, and it
was measured rather than assumed.** On a full tile at `open_m` = 1.0 it took
off 0.7% of the area while fragmenting the network from 179 polygons to 224
-- it severs narrow chokepoints, undoing connectivity the closing had just
built, and deletes almost nothing. The reason is that the false positives
here are car parks, forecourts and driveways: large *compact* blobs, not
thin slivers, and an opening only erases thin things. No radius that spares
a 5 m carriageway will touch a 20 m car park. It is kept because it is the
right tool when the noise really is speckle, but it is not the fix for this
mask, and the default is off accordingly.

Shape filters were tried against the same problem and dropped -- see "Road
detection > Off-street surface" in README.md for the measurements. Neither a
min-rotated-rectangle elongation ratio (post-dissolve) nor an effective
width / compactness test (pre-dissolve) separates road from not-road on this
data at all.
"""

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.ops import unary_union

# Distance to grow and then shrink, in meters. At 3 m this closes gaps up to
# 6 m across -- wide enough to bridge a shadow band or a scan-window notch,
# narrow enough not to weld a road to the car park beside it.
DISSOLVE_BUFFER_M = 3.0

# Polygons smaller than this are dropped before dissolving. At 0.2 m/px a
# single scan window is 484 px = 19 m^2, so this keeps whole windows and
# discards the slivers left where a window was clipped.
MIN_POLYGON_AREA_M2 = 15.0

# Vertex tolerance for simplifying the staircase boundary a block-stamped
# mask produces. Buffering a raster-traced polygon is expensive in direct
# proportion to its vertex count, and those vertices encode the scan grid
# rather than anything real. Below the mask's own 4.4 m resolution, so it
# cannot remove real detail the mask contained.
SIMPLIFY_TOLERANCE_M = 0.5

# Radius for the optional opening stage (`open_m`), which shrinks and then
# grows -- the reverse of the closing, and the operation that actually
# removes things rather than joining them. An opening at radius r erases
# anything narrower than 2r, so this cannot be set anywhere near
# DISSOLVE_BUFFER_M without deleting real roads: at 3 m it would erase every
# feature under 6 m across, which is most residential streets in this
# imagery. At 1 m it clears speckle and slivers up to 2 m wide and leaves
# even a narrow carriageway intact.
OPEN_BUFFER_M = 1.0


@dataclass
class DissolveResult:
    """Dissolved network, with the accounting needed to judge it."""

    geometry: gpd.GeoSeries
    polygons_before: int
    polygons_after: int
    area_before_m2: float
    area_after_m2: float
    # Only set when the opening stage ran, so its cost can be read separately
    # from the closing's -- the two move area in opposite directions and a
    # single net figure hides both.
    open_m: float | None = None
    polygons_after_close: int | None = None
    area_after_close_m2: float | None = None

    @property
    def area_added_m2(self) -> float:
        """Net area this inferred -- gaps filled, not surface detected."""
        return self.area_after_m2 - self.area_before_m2

    @property
    def area_removed_by_open_m2(self) -> float:
        """Area the opening took back off. Zero when it didn't run."""
        if self.area_after_close_m2 is None:
            return 0.0
        return self.area_after_close_m2 - self.area_after_m2

    @property
    def polygon_delta_from_open(self) -> int:
        """Change in polygon count across the opening. Zero when it didn't run.

        Deliberately signed, and not named "dropped": an opening both
        deletes polygons and *severs* them, and on this data the severing
        dominates, so the count usually goes up. A single "removed" figure
        would report that as the opposite of what it is.
        """
        if self.polygons_after_close is None:
            return 0
        return self.polygons_after - self.polygons_after_close


def _parts(geometry) -> list:
    return list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]


def dissolve_surface(
    mask: np.ndarray,
    transform: Affine,
    crs,
    buffer_m: float = DISSOLVE_BUFFER_M,
    open_m: float | None = None,
) -> DissolveResult | None:
    """Vectorize `mask`, then close it by buffering out, unioning and buffering back.

    `open_m` is off by default. Pass a radius (see OPEN_BUFFER_M) to follow
    the closing with an opening -- shrink by `open_m`, grow back -- which
    erases anything narrower than twice it. That is the stage that removes
    false positives; the closing alone cannot, since an isolated blob comes
    through a close unchanged. It is optional because it is destructive in a
    way the closing is not: it deletes detected surface, and a radius set
    too high deletes real roads rather than just speckle.

    Returns None if the mask is empty or nothing survives the minimum-area
    filter.
    """
    polygons = [
        shape(geometry)
        for geometry, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform)
        if value == 1
    ]
    polygons = [p for p in polygons if p.area >= MIN_POLYGON_AREA_M2]
    if not polygons:
        return None

    area_before = float(sum(p.area for p in polygons))
    simplified = [p.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True) for p in polygons]

    # Grow, union, shrink. The union has to happen while everything is
    # grown -- that is what turns two polygons separated by a gap into one
    # polygon, and shrinking afterwards is what stops the result being 3 m
    # fatter than the road.
    closed = unary_union([p.buffer(buffer_m) for p in simplified]).buffer(-buffer_m)
    if closed.is_empty:
        return None

    parts = _parts(closed)
    if not open_m:
        return DissolveResult(
            geometry=gpd.GeoSeries(parts, crs=crs),
            polygons_before=len(polygons),
            polygons_after=len(parts),
            area_before_m2=area_before,
            area_after_m2=float(closed.area),
        )

    # Opening: shrink, then grow back. Anything thinner than 2 x `open_m`
    # vanishes at the shrink and has nothing to grow back from, so slivers
    # and speckle go while the body of a road survives. Applied after the
    # closing rather than before deliberately -- opening first would delete
    # the small fragments the closing exists to reconnect.
    opened = closed.buffer(-open_m).buffer(open_m)
    if opened.is_empty:
        return None

    # The area filter runs again here: an opening can sever one polygon into
    # several, and the offcuts are exactly the outliers this stage is for.
    kept = [p for p in _parts(opened) if p.area >= MIN_POLYGON_AREA_M2]
    if not kept:
        return None

    return DissolveResult(
        geometry=gpd.GeoSeries(kept, crs=crs),
        polygons_before=len(polygons),
        polygons_after=len(kept),
        area_before_m2=area_before,
        area_after_m2=float(sum(p.area for p in kept)),
        open_m=open_m,
        polygons_after_close=len(parts),
        area_after_close_m2=float(closed.area),
    )
