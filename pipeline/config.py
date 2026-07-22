from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CHUNK_PATH = PROJECT_ROOT / "data" / "input" / "dop10rgbi_32_406_5758_1_nw_2026.jp2"

INPUT_TILES_DIR = PROJECT_ROOT / "data" / "input" / "idop_kacheln"
OSM_CACHE_PATH = PROJECT_ROOT / "data" / "osm" / "osm_features.gpkg"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


def raw_tile_path(stem: str) -> Path:
    """The raw imagery a pipeline tile stem was read from.

    The configured chunk wins; anything else is looked up in the tile archive,
    so a stem from an earlier run (the diagnostics report pins one) still
    resolves after INPUT_CHUNK_PATH moves on.
    """
    if stem == INPUT_CHUNK_PATH.stem:
        return INPUT_CHUNK_PATH
    return INPUT_TILES_DIR / f"{stem}.jp2"


TUNED_AT_M = 0.2


def pixel_size_m(path: Path, default: float = TUNED_AT_M) -> float:
    """`path`'s ground resolution, or `default` if it cannot be read.

    Falls back rather than raising: config is imported by every entry point,
    including ones that never touch the chunk, and a missing file should not
    stop `--help` from running.
    """
    try:
        import rasterio

        with rasterio.open(path) as src:
            return float(src.res[0])
    except Exception:
        return default


INPUT_CHUNK_RES_M = pixel_size_m(INPUT_CHUNK_PATH)

PX_SCALE = TUNED_AT_M / INPUT_CHUNK_RES_M


def scaled_px(pixels_at_tuned_res: float) -> int:
    """A tuned *length* in pixels, restated for the chunk's resolution."""
    return max(1, round(pixels_at_tuned_res * PX_SCALE))


def scaled_area_px(pixels_at_tuned_res: float) -> int:
    """A tuned *area* in pixels, restated for the chunk's resolution."""
    return max(1, round(pixels_at_tuned_res * PX_SCALE**2))


TEXTURES_DIR = PROJECT_ROOT / "data" / "input" / "textures"

TILE_CRS = "EPSG:25832"
OSM_CRS = "EPSG:4326"

BIKE_LANE_HIGHWAY_VALUES = ["cycleway"]

BIKE_LANE_CYCLEWAY_VALUES = {
    "lane",
    "track",
    "shared_lane",
    "opposite_lane",
    "opposite_track",
    "share_busway",
    "opposite_share_busway",
}

STREET_HIGHWAY_VALUES = [
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
]

OSM_TAGS = {
    "highway": BIKE_LANE_HIGHWAY_VALUES + STREET_HIGHWAY_VALUES,
    "cycleway": True,
    "cycleway:left": True,
    "cycleway:right": True,
    "cycleway:both": True,
    "bicycle": ["designated"],
}

BIKE_LANE_BUFFER_METERS = 4.5
STREET_BUFFER_METERS = 6.0

BACKGROUND_LABEL = 0
BIKE_LANE_LABEL = 1
STREET_LABEL = 2

NODATA_VALUE = BACKGROUND_LABEL

SHADOW_HANDLING = "none"

SHADOW_CUT_MARGIN_M = 1.0

APPLY_RED_BOOST = True

NOT_SHADOW_LABEL = 0
SHADOW_LABEL = 1


DETECTION_INPUT_DIR = OUTPUT_DIR

ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "input" / "annotated_bike_lanes"

TRAINING_DIR = PROJECT_ROOT / "data" / "training"
TRAINING_CHIP_SIZE_PX = 640
TRAINING_CHIP_OVERLAP_PX = 64
TRAINING_VAL_FRACTION = 0.2

YOLO_SEG_BASE_CHECKPOINT = "yolo11n-seg.pt"
YOLO_SEG_TRAINED_WEIGHTS_PATH = PROJECT_ROOT / "runs" / "segment" / "train" / "weights" / "best.pt"

DETECTION_OUTPUT_PATH = PROJECT_ROOT / "data" / "detections" / "bikelanes.gpkg"


TEXTURE_WINDOW_M = 2.2
TEXTURE_WINDOW_PX = max(1, round(TEXTURE_WINDOW_M / INPUT_CHUNK_RES_M))

BIKE_LANE_TEXTURE_LABELS = ("bikelane", ("negative",))
ROAD_TEXTURE_LABELS = ("road", ("bikelane", "negative"))

TEXTURE_STRIDE_PX = max(1, TEXTURE_WINDOW_PX // 2)

COARSE_BRIDGE_M = 4.0

COARSE_BRIDGE_ORIENTATIONS = 12


USE_OSM_ROAD_FALLBACK = True

OSM_ROAD_DEFAULT_WIDTH_M = {
    "primary": 12.0,
    "primary_link": 7.0,
    "secondary": 10.0,
    "secondary_link": 6.5,
    "tertiary": 8.0,
    "tertiary_link": 6.0,
    "unclassified": 6.0,
    "residential": 5.5,
    "living_street": 4.5,
    "service": 3.5,
}

OSM_ROAD_DEFAULT_WIDTH_FALLBACK_M = 6.0


PIPELINE_TILE_STEMS = [
    INPUT_CHUNK_PATH.stem,
]

GAP_SECTION_INTERVAL_M = 2.0
GAP_TANGENT_HALF_SPAN_M = 2.5

GAP_MAX_LANE_TO_ROAD_M = 20.0

GAP_MAX_ROAD_ANGLE_DEG = 35.0

GAP_EXCLUDED_ROAD_CLASSES = {"service"}

GAP_BEHIND_ROAD_M = 3.0
GAP_BEYOND_LANE_M = 4.0

GAP_SHADOW_EDGE_MARGIN_M = 3.0

GAP_SHADOW_CORRIDOR_M = 14.0

GAP_MAP_BREAKS_M = [0.0, 0.50, 1.00, 2.00, 4.00]

GAP_MAP_SHOW_ROAD = False

GAP_OUTPUT_PATH = PROJECT_ROOT / "data" / "detections" / "bikelane_gap.gpkg"
GAP_MAP_PATH = PROJECT_ROOT / "data" / "detections" / "bikelane_gap_map.png"

USE_CACHED_BIKELANE_MASK = True

BIKELANE_MASK_PATHS = {
    "idop20rgbi_32_404_5757_1_nw_2025":
        PROJECT_ROOT / "data" / "detections" / "bikelane_detections" /
        "bikelane_edge_mask_404_5757_hires.tif",
    "idop20rgbi_32_404_5758_1_nw_2025":
        PROJECT_ROOT / "data" / "detections" / "bikelane_detections" /
        "bikelane_edge_mask_404_5758_hires.tif",
}

MIN_LANE_COMPONENT_PX = scaled_area_px(200)


EDGE_HUE_TOLERANCE = 0.15

EDGE_MIN_SATURATION = 0.07

COARSE_BRIDGE_PX = max(0, round(COARSE_BRIDGE_M / INPUT_CHUNK_RES_M))

ROI_DILATION_PX = scaled_px(8)

CLOSING_RADIUS_PX = scaled_px(2)

MIN_COMPONENT_AREA_PX = scaled_area_px(15)

CENTERLINE_BIN_WIDTH_PX = scaled_px(3)

MIN_CENTERLINE_BINS = 7

SMOOTHING_WIDTH_MULTIPLE = 3.0

BRIDGE_MAX_GAP_RADIUS_MULTIPLE = 4.0

BRIDGE_ALIGNMENT_COS_MIN = 0.82

BRIDGE_TANGENT_LOOKBACK_POINTS = 5

SHADOW_EXCLUSION_MARGIN_PX = scaled_px(5)

ROAD_MIN_COMPONENT_AREA_PX = scaled_area_px(200)

EMBED_BATCH_SIZE = 64

SCORE_THRESHOLD = -0.10

ROAD_SCORE_THRESHOLD = 0.18

INPUT_SIZE_PX = 256

SEGMENT_COLORS = [
    (0, 255, 255),
    (255, 0, 255),
    (255, 220, 0),
    (0, 255, 0),
    (255, 120, 0),
    (120, 160, 255),
]

WIDTH_SAMPLE_INTERVAL_M = 5.0

MAX_HALF_WIDTH_M = 15.0

GAP_TOLERANCE_M = 1.5

RAY_STEP_PX = 0.5

WIDTH_TANGENT_HALF_SPAN_M = 2.5

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

SHADOW_BAND = 6

PREVIEW_MAX_PX = 2500

MIN_SAMPLES_PER_WAY = 3

WIDTH_COLOR_MIN_M = 4.0

WIDTH_COLOR_MAX_M = 20.0

CENTERLINE_WIDTH_PX = 5

DISSOLVE_BUFFER_M = 3.0

MIN_POLYGON_AREA_M2 = 15.0

SIMPLIFY_TOLERANCE_M = 0.5

OPEN_BUFFER_M = 1.0

RED_HUE_TOLERANCE = 0.08

MIN_SATURATION_FOR_BOOST = 0.05

SATURATION_BOOST = 1.8

REDNESS_ROWS_PER_BLOCK = 512

SHADOW_MAX_GAIN = 3.0

MIN_REFERENCE_DENSITY = 0.02

SHADOW_CLOSING_RADIUS_M = 0.6

MIN_SHADOW_AREA_M2 = 1.5

FEATHER_RADIUS_M = 2.0

BLEED_RADIUS_M = 1.0

REPORT_TILE_STEM = PIPELINE_TILE_STEMS[0]

REPORT_BOUNDS = (
    406537.8249676815,
    5758617.074135774,
    406712.8249676815,
    5758792.074135774,
)

REPORT_FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

REPORT_PATH = PROJECT_ROOT / "docs" / "pipeline_report.md"

CONNECTION_PREVIEW_PATH = PROJECT_ROOT / "connection_preview.png"

CONNECTION_PREVIEW_BRIDGE_M = (2.0, COARSE_BRIDGE_M)
