"""Dissolve a detected surface mask into one coherent network polygon layer.

The surface arrives from the detector as a raster mask fragmented for reasons
unrelated to where road is: it is stamped in whole scan windows, and shadowed
pixels are cut entirely (see `RoadEdgeDetector.surface_mask`), so a road under
a building's shadow arrives as two pieces with a hole between them.

This module closes that geometrically -- vectorize, grow every polygon by
`buffer_m`, union, shrink back -- a morphological closing in vector space:
gaps and separations narrower than 2x `buffer_m` are filled/merged, the outer
boundary is otherwise left where it was. Two inherent consequences: it
partially undoes the shadow cut (a narrow shadow gap fills back in -- usually
what you want from a *network*, but inferred not detected, so
`dissolve_surface` reports how much area it added); and it does not restore a
true outline (closing fills concavities but cannot recover a boundary the mask
never had), so the result must not be used to measure width -- that comes from
detection/centerline_width.py.

Closing does not *remove* anything: an isolated false-positive blob survives
untouched. Removing outliers takes the opposite operation, an opening (shrink
then grow), which erases anything too thin to survive the shrink -- the
optional `open_m` stage, off by default because it is destructive and near
`buffer_m` deletes roads not noise. But the opening does not remove this
data's false positives either, and that was measured: at `open_m`=1.0 it took
off 0.7% of a tile's area while fragmenting 179 polygons into 224 -- the false
positives here are car parks and forecourts, large *compact* blobs, and an
opening only erases thin things. It is kept for genuine speckle, not this.

Shape filters were also tried and dropped (see "Road detection > Off-street
surface" in README.md): neither a min-rotated-rectangle elongation ratio nor
an effective-width/compactness test separates road from not-road on this data.
"""

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape
from shapely.ops import unary_union

DISSOLVE_BUFFER_M = 3.0

MIN_POLYGON_AREA_M2 = 15.0

SIMPLIFY_TOLERANCE_M = 0.5

OPEN_BUFFER_M = 1.0


@dataclass
class DissolveResult:
    """Dissolved network, with the accounting needed to judge it."""

    geometry: gpd.GeoSeries
    polygons_before: int
    polygons_after: int
    area_before_m2: float
    area_after_m2: float
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

        Signed, and not named "dropped": an opening both deletes polygons and
        *severs* them, and on this data severing dominates, so the count
        usually goes up -- a single "removed" figure would invert that.
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
    the close with an opening -- shrink by `open_m`, grow back -- erasing
    anything narrower than twice it; that is the destructive stage that can
    remove false positives (and, too high, real roads), which the close alone
    cannot. Returns None if the mask is empty or nothing survives the
    minimum-area filter.
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

    opened = closed.buffer(-open_m).buffer(open_m)
    if opened.is_empty:
        return None

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
