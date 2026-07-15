"""Central configuration for the bike lane prefiltering pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_TILES_DIR = PROJECT_ROOT / "data" / "input" / "idop_kacheln"
OSM_CACHE_PATH = PROJECT_ROOT / "data" / "osm" / "osm_features.gpkg"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# The IDOP20 tiles are delivered as ETRS89 / UTM zone 32N.
TILE_CRS = "EPSG:25832"
OSM_CRS = "EPSG:4326"

# highway=cycleway is dedicated bike infrastructure with its own geometry.
BIKE_LANE_HIGHWAY_VALUES = ["cycleway"]

# cycleway(:left/:right/:both) values that mean a lane/track is painted or
# raised along the road itself (geometry is the road's centerline).
BIKE_LANE_CYCLEWAY_VALUES = {
    "lane",
    "track",
    "shared_lane",
    "opposite_lane",
    "opposite_track",
    "share_busway",
    "opposite_share_busway",
}

# General road classes bikes are legally allowed to ride on in mixed
# traffic. Excludes motorway/trunk (Autobahn/Kraftfahrstrasse), where
# cycling is prohibited in Germany.
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

# Buffer applied around each feature's centerline, in meters. Streets get a
# wider buffer since they cover a full carriageway rather than a single lane.
BIKE_LANE_BUFFER_METERS = 6.0
STREET_BUFFER_METERS = 8.0

# Integer labels written to the classification band.
BACKGROUND_LABEL = 0
BIKE_LANE_LABEL = 1
STREET_LABEL = 2

# Value written to pixels outside the buffered mask (also doubles as the
# background label in the classification band).
NODATA_VALUE = BACKGROUND_LABEL

# Whether to detect and brightness-normalize shadowed pixels within the
# road/bike-lane mask before writing output tiles.
APPLY_SHADOW_CORRECTION = True

# Integer labels written to the shadow band.
NOT_SHADOW_LABEL = 0
SHADOW_LABEL = 1

# --- Detection: trained YOLO-seg model (see scripts/detection/) ---

# Runs on the prefiltered data/output/ tiles (bands 1-3 = RGB), not the raw
# input tiles -- matches what the CVAT annotations were drawn on.
DETECTION_INPUT_DIR = OUTPUT_DIR

# CVAT export(s) in "Ultralytics YOLO segmentation 1.0" format: each
# subdirectory is one export task, containing data.yaml + labels/train/*.txt.
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "input" / "annotated_bike_lanes"

TRAINING_DIR = PROJECT_ROOT / "data" / "training"
# 640px matches ultralytics' default imgsz, so training chips aren't
# rescaled/shrunk further -- bike lanes are only ~10px wide at 0.2m/px, so
# extra downscaling would make them very hard to learn.
TRAINING_CHIP_SIZE_PX = 640
TRAINING_CHIP_OVERLAP_PX = 64
TRAINING_VAL_FRACTION = 0.2

YOLO_SEG_BASE_CHECKPOINT = "yolo11n-seg.pt"
YOLO_SEG_TRAINED_WEIGHTS_PATH = PROJECT_ROOT / "runs" / "segment" / "train" / "weights" / "best.pt"

# Matches TRAINING_CHIP_SIZE_PX: YOLO models perform best near their
# trained imgsz, so inference chips use the same size the model was
# fine-tuned on rather than an independently-chosen value.
DETECTION_CHIP_SIZE_PX = TRAINING_CHIP_SIZE_PX
DETECTION_CHIP_OVERLAP_PX = TRAINING_CHIP_OVERLAP_PX
DETECTION_CONFIDENCE_THRESHOLD = 0.25

DETECTION_OUTPUT_PATH = PROJECT_ROOT / "data" / "detections" / "bikelanes.gpkg"

# Fields burned into a GeoTIFF each (data/detections/bikelanes_<field>.tif),
# for visual inspection without GIS software / overlay against data/output/.
DETECTION_RASTER_FIELDS = ["score", "width_mean_m"]
