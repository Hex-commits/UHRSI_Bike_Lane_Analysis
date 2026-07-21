"""Detect bike-lane centrelines from the imagery, for measure_bikelane_gap.

The gap tool takes *road* locations from OSM, but bike-lane locations must come
from the satellite imagery, never OSM: a lane OSM has not mapped, or has placed
wrongly, would otherwise be invisible or measured against the wrong geometry.

The lanes come from `BikeLaneEdgeDetector` (detection/edge_trace.py) -- the CNN
coarse region plus classical colour edge tracing -- not the trained YOLO-seg
model, whose recall on this project's sparse "sample" annotations is too low to
find the real tracks (it misses the validated cycle track outright). The edge
tracer already builds each lane as a constant-width band around a smoothed,
bridged centreline, so here each detection is just reduced back to that
centreline (`_binned_centerline`) for the gap tool to cut cross-sections along.

Run on the *prefiltered* tile (data/output/*.tif): the detector keys on the
red-boosted paint. Only the lane *location* comes from here; every gap distance
is still measured from raw pixels in measure_bikelane_gap.

Cost note: this runs the coarse CNN scan over the requested extent. Over a
whole 5000x5000 tile that is ~20 min (see detect_roads / README); over a bounded
window it is seconds. Pass a window for iteration.
"""

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import label
from shapely.geometry import LineString

from scripts.config import MIN_LANE_COMPONENT_PX, TILE_CRS
from scripts.detection.edge_trace import BikeLaneEdgeDetector, _binned_centerline
from scripts.detection.texture_detector import bike_lane_detector


def lane_centerlines_from_mask(
    mask_path,
    window: Window | None = None,
    min_component_px: int = MIN_LANE_COMPONENT_PX,
) -> gpd.GeoDataFrame:
    """Centrelines read from a cached full-tile lane detection raster.

    Same output contract as `detect_lane_centerlines` -- LineStrings in
    TILE_CRS, one per lane -- but from a raster already on disk rather than
    from a scan run here and now. That matters twice over: the cached masks
    are markedly more complete than what an in-process trace of the same
    ground produces, and reading one costs seconds where the scan costs
    ~20 min a tile, which is the difference between a whole-tile gap run
    being routine and being an overnight job.

    Each connected component is one lane fragment, reduced to a centreline by
    the same `_binned_centerline` the live tracer uses, so nothing downstream
    can tell the two sources apart.
    """
    with rasterio.open(mask_path) as src:
        mask = np.asarray(src.read(1, window=window)) > 0.5
        transform = src.window_transform(window) if window is not None else src.transform

    labelled, count = label(mask)
    lines = []
    for index in range(1, count + 1):
        component = labelled == index
        if component.sum() < min_component_px:
            continue
        points = _binned_centerline(component)
        if points is None:
            continue
        lines.append(LineString([transform * (col, row) for row, col in points]))
    return gpd.GeoDataFrame({"geometry": lines}, geometry="geometry", crs=TILE_CRS)


def detect_lane_centerlines(
    tile_path,
    window: Window | None = None,
    coarse_detector=None,
    progress=None,
) -> gpd.GeoDataFrame:
    """Bike-lane centrelines detected in a prefiltered `tile_path`.

    Returns a GeoDataFrame of LineStrings in TILE_CRS, one per detected lane
    (components too short to reduce to a centreline are dropped). `window`
    limits detection to that extent; `coarse_detector` is accepted so a caller
    can reuse a loaded model. The coarse scan is run here rather than inside
    `BikeLaneEdgeDetector` so `progress` can report it -- over a whole tile it
    is ~20 min, and a silent wait that long is indistinguishable from a hang.
    """
    with rasterio.open(tile_path) as src:
        image = np.transpose(src.read([1, 2, 3], window=window), (1, 2, 0))
        transform = src.window_transform(window) if window is not None else src.transform

    coarse_detector = coarse_detector or bike_lane_detector()
    coarse = coarse_detector.predict(image, progress=progress)
    detector = BikeLaneEdgeDetector(coarse_detector=coarse_detector)
    lines = []
    for detection in detector.predict(image, coarse=coarse):
        points = _binned_centerline(detection.mask)
        if points is None:
            continue
        lines.append(LineString([transform * (col, row) for row, col in points]))
    return gpd.GeoDataFrame({"geometry": lines}, geometry="geometry", crs=TILE_CRS)
