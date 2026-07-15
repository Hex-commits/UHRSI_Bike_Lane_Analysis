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
