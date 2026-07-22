import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import label
from shapely.geometry import LineString

from pipeline.config import MIN_LANE_COMPONENT_PX, TILE_CRS
from scripts.detection.edge_trace import BikeLaneEdgeDetector, _binned_centerline
from scripts.detection.texture_detector import bike_lane_detector


def load_lane_mask(mask_path, window: Window | None = None) -> np.ndarray:
    """The cached lane detection as a boolean array on the tile grid.

    The gap measurement needs the mask itself, not just the centrelines
    derived from it: the lane's near edge is read off this directly, because
    the spectral segmentation cannot resolve a lane/road boundary where both
    are the same asphalt.
    """
    with rasterio.open(mask_path) as src:
        return np.asarray(src.read(1, window=window)) > 0.5


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


def detect_lanes(
    tile_path,
    window: Window | None = None,
    coarse_detector=None,
    progress=None,
) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Bike-lane centrelines *and* their mask, detected in a prefiltered tile.

    Returns `(centrelines, mask)`: LineStrings in TILE_CRS, one per detected
    lane (components too short to reduce to a centreline are dropped), plus the
    union of the detections' own masks on the tile grid -- the same thing
    `load_lane_mask` reads back from a cached detection raster.

    Both, not just the centrelines, because the gap measurement needs both and
    they must describe the same lanes. Under USE_OSM_ROAD_FALLBACK the lane's
    near edge is read from the mask (`CrossSection.lane_edge_m`), since where
    lane and road are the same asphalt the spectral segmentation merges them
    into one run. Returning centrelines alone left every caller on this path
    passing `lane_mask=None`, and every cross-section was then skipped as
    "unresolved" -- a tile with no cached mask silently measured zero gaps.

    `window` limits detection to that extent; `coarse_detector` is accepted so
    a caller can reuse a loaded model. The coarse scan is run here rather than
    inside `BikeLaneEdgeDetector` so `progress` can report it -- over a whole
    tile it is ~20 min, and a silent wait that long is indistinguishable from
    a hang.
    """
    with rasterio.open(tile_path) as src:
        image = np.transpose(src.read([1, 2, 3], window=window), (1, 2, 0))
        transform = src.window_transform(window) if window is not None else src.transform

    coarse_detector = coarse_detector or bike_lane_detector()
    coarse = coarse_detector.predict(image, progress=progress)
    detector = BikeLaneEdgeDetector(coarse_detector=coarse_detector)
    lines = []
    mask = np.zeros(image.shape[:2], dtype=bool)
    for detection in detector.predict(image, coarse=coarse):
        points = _binned_centerline(detection.mask)
        if points is None:
            continue
        mask |= detection.mask
        lines.append(LineString([transform * (col, row) for row, col in points]))
    return gpd.GeoDataFrame({"geometry": lines}, geometry="geometry", crs=TILE_CRS), mask


def detect_lane_centerlines(
    tile_path,
    window: Window | None = None,
    coarse_detector=None,
    progress=None,
) -> gpd.GeoDataFrame:
    """`detect_lanes`, centrelines only -- for callers with no use for the mask."""
    lanes, _mask = detect_lanes(tile_path, window, coarse_detector, progress)
    return lanes
