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

# Where scripts/detect.py writes the bike lanes it detected. (The YOLO-seg
# inference chip size, overlap and confidence threshold that used to live
# here went with that pipeline -- see "Retired" in README.)
DETECTION_OUTPUT_PATH = PROJECT_ROOT / "data" / "detections" / "bikelanes.gpkg"


TEXTURE_WINDOW_PX = 22

BIKE_LANE_TEXTURE_LABELS = ("bikelane", ("negative",))
ROAD_TEXTURE_LABELS = ("road", ("bikelane", "negative"))

TEXTURE_STRIDE_PX = 11


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


# --- Bike-lane gap: the pipeline's final product (scripts/detect.py) ---

# Tiles the pipeline runs over. Each is read twice, from two different
# versions of itself: lane *detection* runs on the prefiltered output
# (data/output/), because the edge tracer keys on red-boosted paint, while
# every gap *distance* is measured on the raw input tile, whose pixels no
# buffer mask has touched. Measuring on the prefiltered tile would put an
# artificial edge exactly where a lane's outer boundary sits.
PIPELINE_TILE_STEMS = [
    "idop20rgbi_32_404_5757_1_nw_2025",
]

# Spacing of cross-sections along a detected lane, and the span either side
# of a sample used to estimate the lane's local direction.
GAP_SECTION_INTERVAL_M = 2.0
GAP_TANGENT_HALF_SPAN_M = 2.5

# A lane further than this from any street centerline is not "alongside a
# road" -- it is a park path or a separate route, and its distance to the
# nearest carriageway is not a meaningful safety measure.
GAP_MAX_LANE_TO_ROAD_M = 20.0

# The reference road must run roughly *alongside* the lane. Without this the
# nearest way wins outright, and the nearest way is very often a driveway or
# parking aisle the lane simply crosses: measured on tile 404_5757, 47% of
# references were `service` ways and 27% sat beyond 60 degrees to the lane.
# The distance to a road you cross is not a separation from it.
GAP_MAX_ROAD_ANGLE_DEG = 35.0

# Highway classes never used as the road reference. A cycle track running
# beside a tertiary road should be measured against that road, not against
# the service driveway that happens to pass closer.
GAP_EXCLUDED_ROAD_CLASSES = {"service"}

# How far a cross-section reaches past each end: back behind the road
# centerline so the carriageway run is bounded on both sides, and past the
# lane so the lane's own far edge falls inside the profile.
GAP_BEHIND_ROAD_M = 3.0
GAP_BEYOND_LANE_M = 4.0

# Shadow correction leaves a residual false edge at the shadow boundary --
# the peak material gradient there only drops ~36%. Past ~3 m the residual
# b-chroma deviation falls to +0.004, so cross-sections centred nearer than
# this are dropped rather than trusted.
GAP_SHADOW_EDGE_MARGIN_M = 3.0

# Corridor half-width that gives shadow detection a single population to
# threshold. Handing detect_shadow_mask every non-vegetated pixel puts
# rooftops in with pavement, and Otsu then splits roof-from-road instead of
# shadow-from-sunlit, flagging whole buildings as shadow.
GAP_SHADOW_CORRIDOR_M = 14.0

# Colour breaks for the gap map, in metres. Quantised rather than linear:
# the distribution is heavily skewed towards small gaps, so a linear ramp
# spends most of its range on a few wide outliers and renders everything
# under a metre as one indistinguishable pale blue.
#
# Five bands, not more, and that is a legibility limit rather than a
# statistical one. The map is read at poster distance, where each class has
# to be told from its neighbours by colour alone; a one-hue ramp bright
# enough to sit on dark imagery spans about 0.47 of OKLCH lightness, so five
# steps clear the 0.06 per-step separation floor and eight (the earlier
# breaks, down to 0.10 m) came out at 0.047 -- a smooth gradient rather than
# a scale. Sub-metre detail below 0.5 m therefore lives in the GeoPackage's
# `gap_m` and `composition` fields, not in the map's colours.
GAP_MAP_BREAKS_M = [0.0, 0.50, 1.00, 2.00, 4.00]

# Whether the gap map draws the road surface it measured against. Off for
# poster use: under USE_OSM_ROAD_FALLBACK the road is a class-width buffer,
# an assumption rather than a detection, and drawn as a solid band it is the
# largest, most confident-looking object on the map. Hiding it leaves the
# lane and the measured ribbon, which are the parts read off pixels.
GAP_MAP_SHOW_ROAD = False

GAP_OUTPUT_PATH = PROJECT_ROOT / "data" / "detections" / "bikelane_gap.gpkg"
GAP_MAP_PATH = PROJECT_ROOT / "data" / "detections" / "bikelane_gap_map.png"

# Where bike-lane locations come from.
#
# True: read the cached full-tile detection raster below. These masks are the
# best lane detection this project has produced -- on tile 404_5757 the mask
# holds 163 components against the ~26 a fresh in-process trace finds over
# the same ground, because it was produced by a tuned full-tile run rather
# than re-derived per call. Reading it also costs seconds instead of the
# ~20 min a coarse CNN scan takes, which is what makes a whole-tile gap run
# practical at all.
#
# False: re-trace lanes in-process with detection/bikelane_centerlines.py.
# Correct but slower and sparser; use it for a tile with no cached mask.
USE_CACHED_BIKELANE_MASK = True

# Cached detection rasters, per tile stem. Any tile absent here falls back to
# in-process tracing regardless of the flag above.
BIKELANE_MASK_PATHS = {
    "idop20rgbi_32_404_5757_1_nw_2025":
        PROJECT_ROOT / "data" / "detections" / "bikelane_detections" /
        "bikelane_edge_mask_404_5757_hires.tif",
    "idop20rgbi_32_404_5758_1_nw_2025":
        PROJECT_ROOT / "data" / "detections" / "bikelane_detections" /
        "bikelane_edge_mask_404_5758_hires.tif",
}

# Connected components smaller than this are dropped before reducing a mask
# to centerlines. At 0.2 m/px, 200 px is 8 m^2 -- below a real lane fragment,
# above the speckle the edge tracer leaves behind.
MIN_LANE_COMPONENT_PX = 200
