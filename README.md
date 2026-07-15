# UHRSI Bike Lane Analysis

Prefilters high-resolution aerial orthophoto tiles of Münster, Germany down
to just the bike lane / street network, for use as an ML training set. Bike
lane and street geometry comes from OpenStreetMap; the imagery is masked to
a buffered area around that geometry, shadows within the mask are
brightness-normalized, and a classification band records whether each
retained pixel is a bike lane or a general street.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.14 (pinned in
`.python-version`; `uv` will install it automatically if needed).

```bash
# Install dependencies into a local .venv
uv sync
```

### Input data

The source imagery is too large for this repo. Download the IDOP20 RGBI
tiles for the area of interest from the [NRW
Geoportal](https://www.geoportal.nrw) and place the `.jp2` files under:

```
data/input/idop_kacheln/
```

## Running the pipeline

```bash
uv run python main.py
```

This loads the input tiles, fetches/caches the OSM bike lane and street
geometry, masks and shadow-corrects each tile, and writes one output
GeoTIFF per input tile to `data/output/`.

Re-running is safe and idempotent: OSM results are cached to
`data/osm/osm_features.gpkg`, so subsequent runs skip the (slow,
rate-limited) Overpass query unless that cache is deleted or
`force_refresh=True` is passed to `fetch_osm_features`.

### Inspecting output tiles

```bash
# Inspect every tile in data/output/
uv run python -m scripts.inspect

# Inspect specific files
uv run python -m scripts.inspect data/output/idop20rgbi_32_405_5758_1_nw_2025_bikelanes.tif
```

Prints each tile's CRS, dimensions/resolution, and a per-band breakdown
(dtype, color interpretation, nodata, description).

## Project layout

```
main.py           # thin orchestrator: calls into scripts/ in sequence
scripts/           # all pipeline logic lives here as importable modules
data/
  input/           # raw IDOP20 .jp2 tiles (not tracked in git)
  osm/             # cached OSM query results (not tracked in git)
  output/          # filtered/corrected output GeoTIFFs (not tracked in git)
```

## Scripts

- **`scripts/config.py`** — central configuration: paths, CRS (EPSG:25832),
  which OSM tags count as a bike lane vs. a street, buffer distances,
  classification band labels, and the shadow-correction toggle.

- **`scripts/tiles.py`** — discovers input `.jp2` tile paths and computes
  their combined bounding box, used to scope the OSM query to the actual
  area covered by the imagery.

- **`scripts/osm_features.py`** — queries OpenStreetMap (via `osmnx`) for
  bike lane and street geometry within the tiles' bounding box, classifies
  each feature as `"bikelane"` (dedicated cycleways, or roads with a mapped
  cycle lane/track) or `"street"` (general road classes bikes may legally
  use in mixed traffic), and caches the result to a GeoPackage.

- **`scripts/mask.py`** — buffers each category's geometry by its
  configured distance and dissolves it into a single polygon per category;
  rasterizes a polygon onto a tile's pixel grid to produce a boolean mask.

- **`scripts/shadows.py`** — detects shadowed pixels within the road mask
  using an HSV-based shadow index (Tsai's NSVDI), then brightens them using
  a locally-windowed sunlit-pixel reference (not a single tile-wide
  average), so the correction doesn't mix statistics across unrelated
  materials elsewhere in the tile.

- **`scripts/filter_imagery.py`** — combines the above: rasterizes both
  category masks for a tile, applies shadow correction, zeroes out every
  pixel outside the combined mask, appends a classification band
  (`0`=background, `1`=bikelane, `2`=street), and writes the result as a
  lossless GeoTIFF.

- **`scripts/inspect.py`** — standalone diagnostic CLI; prints an output
  tile's dimensions and per-band metadata to the terminal.

## Output format

Each output GeoTIFF has 5 bands: the original Red/Green/Blue/Infrared bands
from the source tile (pixels outside the mask zeroed out), plus a 5th
classification band (`0`=background, `1`=bikelane, `2`=street).

## Known limitations

- **OSM coverage gaps**: bike lane mapping in OSM is sometimes incomplete —
  a lane can just stop because it isn't mapped, not because it physically
  ends. Including general streets (not just dedicated bike lanes) in the
  mask mitigates this, since bikes are legally allowed to use most streets
  in mixed traffic, and the classification band still records which is
  which.
- **Street buffer bleeds onto adjacent buildings**: in dense blocks with
  narrow streets, the buffered street polygon can overlap adjacent building
  rooftops, since OSM road centerlines sometimes run close to building
  footprints. Not yet fixed — a likely fix is subtracting OSM building
  footprints from the buffer before rasterizing.
- **Shadow correction is a statistical approximation**, not physical
  de-shadowing: it normalizes brightness using nearby sunlit pixels, but
  can't recover detail lost in very deep, low-contrast shadow interiors,
  and has no sun-angle/DSM data to model illumination directly.
