# UHRSI Bike Lane Analysis

Filters Münster aerial imagery down to the bike lane / street network, for ML training data prep.

- bike lane + street geometry from OpenStreetMap
- imagery masked to a buffer around that geometry
- shadows detected and brightness-corrected
- classification + shadow bands appended to the output

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

## Running

```bash
uv run python main.py
```

- loads tiles, fetches/caches OSM geometry, masks + corrects each tile, writes one GeoTIFF per input tile to `data/output/`
- safe to re-run: OSM results are cached (`data/osm/osm_features.gpkg`), so it skips the Overpass query unless you delete the cache or pass `force_refresh=True`

### Inspecting output

```bash
uv run python -m scripts.inspect                     # all tiles in data/output/
uv run python -m scripts.inspect data/output/foo.tif  # specific file
```

Prints CRS, dimensions, and per-band metadata.

## Layout

```
main.py     # orchestrator, calls into scripts/
scripts/    # pipeline logic, one module per step
data/
  input/    # raw .jp2 tiles (not in git)
  osm/      # cached OSM query results (not in git)
  output/   # final GeoTIFFs (not in git)
```

## Scripts

- `config.py` — paths, CRS, OSM tag rules, buffer sizes, band labels, toggles
- `tiles.py` — finds input tiles, computes their combined bounding box
- `osm_features.py` — queries OSM for bike lanes/streets, classifies each feature, caches to GeoPackage
- `mask.py` — buffers + dissolves geometry per category, rasterizes onto a tile's grid
- `shadows.py` — detects shadow via a blue-excess index, cleans up the mask, brightens shadowed pixels using a local sunlit reference per band
- `filter_imagery.py` — ties it together: masks, corrects shadows, appends classification + shadow bands, writes the GeoTIFF
- `inspect.py` — CLI to get a tile's dimensions and band info

## Shadow correction

- **detect** — blue-excess index `(B-R)/(B+R)`: shadows are lit by scattered blue skylight rather than direct sun, so they read measurably bluer than sunlit pavement. Threshold picked automatically per tile via [Otsu's method](https://en.wikipedia.org/wiki/Otsu%27s_method).
- **clean up** — morphological closing fills small gaps, then blobs under 1.5 m² get dropped. Real cast shadows are big coherent patches, not scattered specks (lane markings, oil stains trip the same index).
- **correct** — each shadow pixel is brightened using the mean of nearby (15 m radius) sunlit pixels in the same band, capped at 3x gain. Done per band rather than one shared gain, since shadow shifts color, not just brightness.

Tried and dropped: Tsai's NSVDI (a saturation/value shadow index from the remote-sensing literature) — its saturation term turned out to track pavement texture noise more than actual shadow in this imagery.

### Correction steps

- window size comes from `local_radius_m` (15 m default), converted to pixels using the tile's resolution
- road pixels split into "shadow" and "sunlit" using the cleaned mask
- per band, independently:
  - local mean of sunlit pixels in a sliding window
  - local mean of shadow pixels in the same window
  - gain = sunlit mean / shadow mean, clamped to 1x–3x
  - pixels with too little sunlit reference nearby (<2% of the window) are left uncorrected rather than guessed
  - shadow pixel value × its gain
- result clipped back to 0–255 and cast back to the original dtype

## Output format

6 bands per GeoTIFF:

1. R
2. G
3. B
4. Infrared (source pixels, masked to 0 outside the buffer)
5. classification — `0` background, `1` bikelane, `2` street
6. shadow — `0` not shadowed, `1` shadowed

## Known limitations

- **OSM gaps** — bike lane mapping stops abruptly where OSM data is incomplete, not where the lane physically ends. Including streets alongside dedicated lanes covers most of this.
- **Buffer bleeds onto buildings** — in dense blocks, narrow streets mean the buffer overlaps adjacent rooftops. Not fixed yet; likely fix is subtracting OSM building footprints from the buffer.
- **Shadow correction is statistical, not physical** — brightens using nearby sunlit pixels, doesn't reconstruct real detail in deep shadow, no sun-angle/DSM data involved. Mainly, because we don't have the data for it.
