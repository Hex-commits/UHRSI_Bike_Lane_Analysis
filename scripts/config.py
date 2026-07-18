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
# Narrowed from 6.0/8.0 -- the wider buffer was generous enough to routinely
# bleed onto adjacent building rooftops in dense blocks (see Known
# limitations), and still comfortably covers a real lane (~1.5-3 m) or
# carriageway (~5-7 m) with margin at these values.
BIKE_LANE_BUFFER_METERS = 4.5
STREET_BUFFER_METERS = 6.0

# Integer labels written to the classification band.
BACKGROUND_LABEL = 0
BIKE_LANE_LABEL = 1
STREET_LABEL = 2

# Value written to pixels outside the buffered mask (also doubles as the
# background label in the classification band).
NODATA_VALUE = BACKGROUND_LABEL

# How to handle shadowed pixels within the road/bike-lane mask:
#   "correct" -- brightness-normalize them to match nearby sunlit pixels
#                (see scripts/shadows.py)
#   "cut"     -- drop them entirely (zeroed to NODATA_VALUE, same as
#                background outside the buffer), rather than attempt
#                correction -- for imagery where shadow correction still
#                distorts too much of the tile to be usable
#   "none"    -- leave shadowed pixels untouched (still detected, so the
#                shadow band is populated, but nothing about the imagery
#                itself is modified or removed)
SHADOW_HANDLING = "none"

# When SHADOW_HANDLING is "cut", pixels within this margin of the detected
# shadow mask are cut too, not just the mask itself. Real shadow edges are
# soft (penumbra); Otsu's threshold draws a hard line through that gradient,
# so pixels just outside the detected mask can still be partially shadowed.
# Cutting only exactly at the mask boundary would keep those, leaving a
# sharp edge between "cut" and "retained but still slightly shadowed".
SHADOW_CUT_MARGIN_M = 1.0

# Whether to boost the saturation of reddish pixels within the road/bike-lane
# mask, so painted bike-lane paint stands out more from gray asphalt (see
# scripts/redness.py). Unlike a generic contrast stretch, this only touches
# pixels that already read as red -- gray asphalt, vegetation, etc. are
# unaffected.
APPLY_RED_BOOST = True

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
