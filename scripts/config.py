"""Central configuration for the bike lane prefiltering pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_TILES_DIR = PROJECT_ROOT / "data" / "input" / "idop_kacheln"
OSM_CACHE_PATH = PROJECT_ROOT / "data" / "osm" / "osm_features.gpkg"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

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

DETECTION_CHIP_SIZE_PX = TRAINING_CHIP_SIZE_PX
DETECTION_CHIP_OVERLAP_PX = TRAINING_CHIP_OVERLAP_PX
DETECTION_CONFIDENCE_THRESHOLD = 0.25

DETECTION_OUTPUT_PATH = PROJECT_ROOT / "data" / "detections" / "bikelanes.gpkg"

DETECTION_RASTER_FIELDS = ["score", "width_mean_m"]


TEXTURE_WINDOW_PX = 22

BIKE_LANE_TEXTURE_LABELS = ("bikelane", ("negative",))
ROAD_TEXTURE_LABELS = ("road", ("bikelane", "negative"))

TEXTURE_STRIDE_PX = 11
