# UHRSI Bike Lane Analysis

Filters Münster aerial imagery down to the bike lane / street network, for ML training data prep.

- bike lane + street geometry from OpenStreetMap
- imagery masked to a buffer around that geometry
- shadows detected, then either brightness-corrected or cut entirely (configurable)
- classification + shadow bands appended to the output

## Research Paper

We found a research paper that classified, whether a road had a bikelane or not, however it did do that with a static bounding box and YOLO.

That is not exactly feasible for us, because we ideally want to trace a certain part of the road.

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

## Layout

```
preprocessing.py  # orchestrator for the pre-processing phase (see above)
train.py          # orchestrator for training the segmentation model (see below)
detect.py         # orchestrator for the detection phase (see below)
scripts/          # pipeline logic, one module per step
  detection/      # dataset export, trained detector, width measurement
data/
  input/
    idop_kacheln/          # raw .jp2 tiles (not in git)
    annotated_bike_lanes/  # CVAT-exported YOLO-seg annotations (in git -- small, hand-labeled)
  osm/            # cached OSM query results (not in git)
  output/         # final GeoTIFFs (not in git)
  training/       # chipped YOLO dataset, generated from annotated_bike_lanes/ (not in git)
  detections/     # detection GeoPackage + raster/PNG previews (not in git)
runs/             # ultralytics training output, incl. trained weights (not in git)
```

## Scripts

- `config.py` — paths, CRS, OSM tag rules, buffer sizes, band labels, toggles
- `tiles.py` — finds input tiles, computes their combined bounding box
- `osm_features.py` — queries OSM for bike lanes/streets, classifies each feature, caches to GeoPackage
- `mask.py` — buffers + dissolves geometry per category, rasterizes onto a tile's grid
- `shadows.py` — detects shadow via a blue-excess index, cleans up the mask, brightens shadowed pixels using a local sunlit reference per band
- `filter_imagery.py` — ties it together: masks, handles shadows per `SHADOW_HANDLING` (correct or cut), appends classification + shadow bands, writes the GeoTIFF
- `inspect.py` — CLI to get a tile's dimensions and band info
- `detection/base.py` — the model swap point: `Detector` protocol, `Detection` type. Nothing else in the detection code depends on a specific model.
- `detection/dataset.py` — converts CVAT YOLO-seg exports (drawn on the full 5000x5000 tiles) into a chipped, trainable YOLO dataset
- `detection/yolo_seg_detector.py` — current adapter: loads a fine-tuned YOLO-seg checkpoint, runs inference on a chip
- `detection/tiling.py` — chips a tile into windows for inference, maps masks back to tile pixel/geo coordinates
- `detection/width.py` — skeletonize + distance transform → width stats, model-agnostic (works on any mask)
- `detection/rasterize.py` — burns a numeric field (score, width_mean_m, ...) from the detections into a GeoTIFF aligned to the source tile grid, plus a colorized PNG preview (raw single-band GeoTIFFs render as flat grayscale in a plain image viewer without one)

## Shadow handling

`SHADOW_HANDLING` in `config.py` picks one of two strategies for shadowed pixels within the road/bike-lane buffer:

- `"correct"` (default) — brightness-normalize them to match nearby sunlit pixels, described below.
- `"cut"` — drop them entirely: zeroed to `NODATA_VALUE` and reclassified as background, exactly like pixels outside the buffer. For imagery where correction still distorts too much of the tile to be usable, this trades shadowed coverage away entirely rather than keeping a degraded version of it. Also cuts a further `SHADOW_CUT_MARGIN_M` (1 m default) beyond the detected shadow mask, since a real shadow's edge is a soft penumbra that the mask's hard Otsu threshold cuts straight through — cutting only exactly at the mask would still retain a ring of partially-shadowed pixels just outside it, leaving a hard edge anyway.

Either way, the shadow band (see Output format) records which pixels were shadowed, even under `"cut"` where those pixels end up as background/nodata in every other band; it reflects detection only, not the cut margin.

### Correction (`"correct"`)

- **detect** — blue-excess index `(B-R)/(B+R)`: shadows are lit by scattered blue skylight rather than direct sun, so they read measurably bluer than sunlit pavement. Threshold picked automatically per tile via [Otsu's method](https://en.wikipedia.org/wiki/Otsu%27s_method).
- **clean up** — morphological closing (disk-shaped structuring element, so the boundary comes out rounded rather than square-cornered) fills small gaps, then blobs under 1.5 m² get dropped. Real cast shadows are big coherent patches, not scattered specks (lane markings, oil stains trip the same index).
- **correct** — each shadow pixel is brightened by an additive, per-band offset. Deep inside a shadow the offset is the gap between the mean of nearby (15 m radius) sunlit pixels and the local shadow mean; near the shadow mask's own boundary (within 2 m) it instead blends towards matching the *exact value of the nearest sunlit pixel*, so the corrected value meets its real neighbor almost exactly at the crossing rather than a regional average. Done per band rather than one shared offset, since shadow shifts color, not just brightness.
- **bleed** — the correction also fades a short distance (`BLEED_RADIUS_M`, 1 m default) past the mask into nominally-sunlit pixels, rather than stopping dead exactly at it: each pixel in that margin gets a fraction of its nearest shadow pixel's own offset, scaled down by distance to zero at the margin's outer edge. The mask's Otsu threshold draws a hard line through what's optically a soft penumbra, so pixels just outside it can still read slightly off from genuinely untouched pavement — even with the boundary-matching above, that shows up as a visible edge over a short stretch rather than a one-pixel discontinuity.

Tried and dropped: Tsai's NSVDI (a saturation/value shadow index from the remote-sensing literature) — its saturation term turned out to track pavement texture noise more than actual shadow in this imagery. Also tried two ways of smoothing a plain windowed-average correction instead of boundary-matching it: Gaussian-blurring the gain field across the boundary in both directions (spread real brightening into never-shadowed pixels — a wider blur just made the resulting glow bigger, not smaller), and tapering gain to 1x purely inside the shadow (left a dark under-corrected rim at the edge instead). Both missed the actual problem: a windowed average is a *regional* estimate, so even "fully corrected" pixels don't match whatever specific neighbor they're touching — no amount of smoothing the transition fixes a gap between two regions.

Also found and fixed two bugs, both specific to shadows wider than the 15 m reference window (e.g. a building's cast shadow spanning a whole street): pixels with too little sunlit reference nearby used to be left completely uncorrected instead of falling back to the nearest-sunlit-pixel target, keeping their real unmodified edge — a hard dark border wherever a large shadow met corrected pavement; and the "nearest sunlit pixel" lookup was finding the nearest *non-shadow* pixel rather than the nearest *sunlit* one, which for a shadow near the buffer's own edge could resolve to background/nodata just outside the buffer, silently corrupting correction for any pixel that picked it as a reference.

### Correction steps

- window size comes from `local_radius_m` (15 m default), converted to pixels using the tile's resolution
- road pixels split into "shadow" and "sunlit" using the cleaned mask
- per band, independently:
  - local mean of sunlit pixels in a sliding window, local mean of shadow pixels in the same window
  - nearest *sunlit* pixel's exact value, via a Euclidean distance transform from the sunlit mask (not just "not shadow" — see bugs above)
  - target = nearest-neighbor value near the mask boundary, blending to the windowed sunlit mean over the innermost `FEATHER_RADIUS_M` (2 m) of the shadow region — but only where that windowed mean has enough real sunlit pixels nearby to be trustworthy; otherwise the nearest-neighbor value is used regardless of depth, so every shadow pixel gets corrected however far it sits from a sunlit reference
  - offset = target − local shadow mean, clamped to [0, shadow mean × 2] (equivalent to the old 3x gain cap)
  - pixel value + its offset
  - nearest *shadow* pixel's own offset, via a Euclidean distance transform from the shadow mask, scaled by `1 − distance/BLEED_RADIUS_M` (clamped to 0–1) and added to sunlit pixels within `BLEED_RADIUS_M` of the mask
- result clipped back to 0–255 and cast back to the original dtype

## Output format

6 bands per GeoTIFF:

1. R
2. G
3. B
4. Infrared (source pixels, masked to 0 outside the buffer)
5. classification — `0` background, `1` bikelane, `2` street
6. shadow — `0` not shadowed, `1` shadowed

## Detection

Two phases, two scripts. Both operate on the prefiltered `data/output/*.tif` tiles (RGB bands only) — matches what the CVAT annotations were drawn on.

**We tried zero-shot first (OWLv2, then YOLO-World) and dropped it.** Neither has ever seen top-down aerial orthophoto imagery, only ground-level natural photos, and it showed on real test chips: text prompts scored ~0.02–0.09 confidence (noise); image-exemplar/large-checkpoint attempts scored confidently but matched whole chips or unrelated features (rooftops, tree canopy), not lanes. A domain gap zero-shot prompting doesn't bridge. See git history if you want the details — this README now only covers the current approach.

**Also tried a classical (non-ML) color+shape detector and dropped it.** Otsu-adaptive red/white color splits plus shape filtering (elongated, constant-width) found real lane paint in isolation, but at full-tile scale it couldn't reliably separate lane paint from other similarly-colored surfaces in this imagery -- terracotta rooftops (near-identical redness/hue to the paint) and bare dirt/leaf-litter ground under winter trees both cleared the same color thresholds. Shape and border filtering cut down but didn't eliminate either confound. See git history if you want the details.

### Training

```bash
uv run python train.py
```

- `scripts/detection/dataset.py` chips the annotated tiles from `data/input/annotated_bike_lanes/` (CVAT export, "Ultralytics YOLO segmentation 1.0" format) into 640px training images + labels, clipping polygons at chip boundaries, and writes `data/training/` (standard YOLO-seg dataset layout)
- only chips containing at least one instance are kept — a chip with zero instances is **not** treated as a negative example, since the annotated tile isn't necessarily exhaustively labeled end-to-end yet (a truly unannotated bike lane elsewhere in the tile would otherwise become a false negative)
- fine-tunes `yolo11n-seg.pt` on that dataset, writes `runs/segment/train/weights/best.pt`

### Inference

```bash
uv run python detect.py
```

Loads the trained checkpoint (`scripts/detection/yolo_seg_detector.py`), runs it chip-by-chip over all `data/output/*.tif` tiles, measures width, and writes the same outputs as before: `data/detections/bikelanes.gpkg` (`width_mean_m` / `width_median_m` / `width_min_m` / `width_max_m` / `score` per instance) plus `bikelanes_<field>.tif` / `.png` previews for `score` and `width_mean_m`.

One API gotcha worth knowing if you touch `yolo_seg_detector.py`: `results.masks.data` from ultralytics is at the model's internal letterboxed/padded resolution, not necessarily the input chip's — verified live with a non-square test chip (1000x873 in, `masks.data` came back 640x576). `results.masks.xy` is reliably rescaled to the original input pixel coordinates regardless of internal padding, so masks are rasterized from that instead.

### Current results: real training, but far too little data

The training data is 6 hand-labeled "Red Bike Lane" instances in a single annotated tile — chipping only yields 2 training images / 1 validation image. That's nowhere near enough to learn anything general; running the fine-tuned model across all 6 tiles at conf≥0.25 produces exactly 4 detections, and checking each one against the source imagery: one lands on plausible pavement, one lands entirely on masked-out background (no imagery at all), one on a plain gray parking lot with no visible red marking, one on a rooftop. Widths reported (1.8–2.8m) are physically plausible, which is a good sign the geometry pipeline itself is sound, but the detections it's measuring aren't trustworthy yet.

This is a data-quantity problem, not a pipeline bug: unlike the zero-shot attempts (wrong architecture/domain for the task), this is the right approach, just needs more annotated tiles before the results mean anything. Annotate more tiles in CVAT, export in the same format into a new subdirectory under `data/input/annotated_bike_lanes/`, and re-run `train.py` — the dataset export already aggregates across every export subdirectory found there.

## Known limitations

- **OSM gaps** — bike lane mapping stops abruptly where OSM data is incomplete, not where the lane physically ends. Including streets alongside dedicated lanes covers most of this.
- **Buffer bleeds onto buildings** — in dense blocks, narrow streets mean the buffer overlaps adjacent rooftops. Not fixed yet; likely fix is subtracting OSM building footprints from the buffer.
- **Shadow correction is statistical, not physical** — brightens using nearby sunlit pixels, doesn't reconstruct real detail in deep shadow, no sun-angle/DSM data involved. Mainly, because we don't have the data for it.
- **YOLO only sees RGB** — `detection/tiling.py` and `detection/dataset.py` both read bands `[1, 2, 3]` only, so the infrared band (and the classification/shadow bands) are computed but never reach the model. CoCo-Weights are being used, they are expecting 3 bands and not more. Adding infrared would mean a 4-channel first conv layer (losing direct compatibility with the pretrained RGB checkpoint's weights for that layer) and a custom dataset loader, since chips are currently exported as plain 3-channel PNGs. Not attempted yet.
