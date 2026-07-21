# UHRSI Bike Lane Analysis

Filters Münster aerial imagery down to the bike lane / street network, for ML training data prep.

- bike lane + street geometry from OpenStreetMap
- imagery masked to a buffer around that geometry
- shadows detected, then brightness-corrected, cut entirely, or left untouched (configurable)
- reddish (bike-lane paint) pixels get a saturation boost, so they stand out more against gray asphalt
- classification + shadow bands appended to the output

**[docs/pipeline_report.md](docs/pipeline_report.md)** — a figure per pipeline stage (raw imagery → OSM mask → shadow detection → red boost → prefiltered output → CNN scan → edge tracing/regularization/bridging → road surface → road-to-bike-lane gap) on one fixed worked example, for reference in writeups. Regenerate after any pipeline change with `uv run python -m scripts.diagnostics.generate_pipeline_report` (takes well under a minute — it's scoped to one small region, not a full tile).

## Running the pipeline

Two stages, in order. Everything tunable lives in `pipeline/config.py`.

```bash
uv sync                                  # 0. install (once)
uv run python -m pipeline.preprocessing  # 1. raw tiles  → data/output/*.tif
uv run python -m pipeline.detect         # 2. prefiltered → the gap map
```

| # | Stage | Command | Reads | Writes | Time |
|---|---|---|---|---|---|
| 1 | **Pre-process** | `uv run python -m pipeline.preprocessing` | `data/input/idop_kacheln/*.jp2` | `data/output/<tile>_bikelanes.tif` | minutes |
| 2 | **Detect + measure the gap** | `uv run python -m pipeline.detect` | raw tiles + cached lane mask + OSM | `data/detections/bikelane_gap.gpkg`, `bikelane_gap_map.png`, `bikelanes.gpkg` | ~1 min/tile |

**Stage 2 is the deliverable**: `bikelane_gap_map.png` is the bike lane network coloured by its distance to the road beside it. Bike-lane geometry comes from the imagery (never OSM, so a lane OSM never mapped is still measured); road position comes from OSM. Scope it to a window while iterating — `col row width height` in pixels:

```bash
uv run python -m pipeline.detect 1600 1600 1600 1600   # 320 m window, seconds
```

Windowed runs write window-suffixed filenames, so they never overwrite the whole-tile result.

### Optional tools

Not part of the two-stage run — use them to inspect, diagnose, or measure something else.

**Inspect a prefiltered tile** — CRS, dimensions and per-band metadata.

```sh
uv run python -m scripts.diagnostics.inspect                      # every tile in data/output/
uv run python -m scripts.diagnostics.inspect data/output/foo.tif  # one file
```

**Rebuild the pipeline report** — regenerates `docs/pipeline_report.md` and its figures from one fixed example region. Run after any pipeline change.

```sh
uv run python -m scripts.diagnostics.generate_pipeline_report
```

**Check the texture detector's reference crops** — pairwise similarity between the crops in `data/input/textures/`. This is what catches a reference crop that no longer discriminates.

```sh
uv run python -m scripts.diagnostics.texture_analysis
```

**Scan one region with the texture detector** — writes a score heatmap and traced-mask figure for the given `x y width height` window.

```sh
uv run python -m scripts.diagnostics.texture_analysis data/output/foo.tif 80 1990 870 580
```

**Measure road widths per OSM way** — a separate product from the gap; writes a GeoPackage and a width-coloured map.

```sh
uv run python -m scripts.measurement.detect_roads data/output/foo.tif        # whole tile
uv run python -m scripts.measurement.detect_roads data/output/foo.tif 22     # coarser scan, ~4x faster
```

**Train the YOLO-seg model** — *deprecated*. The inference side was retired, so this produces weights nothing loads. Kept only so the annotation-to-dataset step stays reproducible; it is not the detector the pipeline uses.

```sh
uv run python train.py
```

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
uv run python -m pipeline.preprocessing
```

- loads tiles, fetches/caches OSM geometry, masks + corrects each tile, writes one GeoTIFF per input tile to `data/output/`
- safe to re-run: OSM results are cached (`data/osm/osm_features.gpkg`), so it skips the Overpass query unless you delete the cache or pass `force_refresh=True`

## Scripts

- `pipeline/config.py` — paths, CRS, OSM tag rules, buffer sizes, band labels, toggles
