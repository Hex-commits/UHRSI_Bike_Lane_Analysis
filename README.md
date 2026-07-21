# UHRSI Bike Lane Analysis

Filters Münster aerial imagery down to the bike lane / street network, for ML training data prep.

- bike lane + street geometry from OpenStreetMap
- imagery masked to a buffer around that geometry
- shadows detected, then brightness-corrected, cut entirely, or left untouched (configurable)
- reddish (bike-lane paint) pixels get a saturation boost, so they stand out more against gray asphalt
- classification + shadow bands appended to the output

**[docs/pipeline_report.md](docs/pipeline_report.md)** — a figure per pipeline stage (raw imagery → OSM mask → shadow detection → red boost → prefiltered output → CNN scan → edge tracing/regularization/bridging → width measurement → the same detection pipeline applied to road surface) on one fixed worked example, for reference in writeups. Regenerate after any pipeline change with `uv run python -m scripts.generate_pipeline_report` (takes well under a minute — it's scoped to one small region, not a full tile).

## Setup

Needs `uv` and Python 3.14 (pinned in `.python-version`, uv installs it automatically).

```bash
uv sync
```

### Input data

Imagery is too large for the repo. Download IDOP20 RGBI tiles from the [NRW
Geoportal](https://www.geoportal.nrw), drop the `.jp2` files in:

```
data/input/idop_kacheln/
```

## Pre-processing

```bash
uv run python preprocessing.py
```

- loads tiles, fetches/caches OSM geometry, masks + corrects each tile, writes one GeoTIFF per input tile to `data/output/`
- safe to re-run: OSM results are cached (`data/osm/osm_features.gpkg`), so it skips the Overpass query unless you delete the cache or pass `force_refresh=True`

### Inspecting output

```bash
uv run python -m scripts.inspect                     # all tiles in data/output/
uv run python -m scripts.inspect data/output/foo.tif  # specific file
```

Prints CRS, dimensions, and per-band metadata.

## Scripts

- `config.py` — paths, CRS, OSM tag rules, buffer sizes, band labels, toggles
