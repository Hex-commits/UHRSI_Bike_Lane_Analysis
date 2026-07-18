# UHRSI Bike Lane Analysis

Filters Münster aerial imagery down to the bike lane / street network, for ML training data prep.

- bike lane + street geometry from OpenStreetMap
- imagery masked to a buffer around that geometry
- shadows detected, then brightness-corrected, cut entirely, or left untouched (configurable)
- reddish (bike-lane paint) pixels get a saturation boost, so they stand out more against gray asphalt
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
    textures/              # texture-embedding reference crops, one subfolder per label (in git -- small)
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
- `redness.py` — boosts saturation of reddish (bike-lane paint) pixels within the road/bike-lane mask
- `filter_imagery.py` — ties it together: masks, handles shadows per `SHADOW_HANDLING` (correct, cut, or none), boosts red pixels if `APPLY_RED_BOOST`, appends classification + shadow bands, writes the GeoTIFF
- `inspect.py` — CLI to get a tile's dimensions and band info
- `texture_embedding.py` — frozen pretrained-CNN feature extractor + discriminant-score classification (see Texture-embedding detector below)
- `texture_analysis.py` — CLI diagnostics for the above: reference-set similarity report, and scan-a-region visualization
- `detection/base.py` — the model swap point: `Detector` protocol, `Detection` type. Nothing else in the detection code depends on a specific model.
- `detection/dataset.py` — converts CVAT YOLO-seg exports (drawn on the full 5000x5000 tiles) into a chipped, trainable YOLO dataset
- `detection/yolo_seg_detector.py` — current adapter: loads a fine-tuned YOLO-seg checkpoint, runs inference on a chip
- `detection/texture_detector.py` — `Detector` adapter for texture_embedding.py: sliding-window scan + discriminant scoring
- `detection/edge_trace.py` — `Detector` adapter combining the above coarse scan with classical color-based edge tracing, for a mask precise enough to measure width from (see Width measurement below)
- `detection/tiling.py` — chips a tile into windows for inference, maps masks back to tile pixel/geo coordinates
- `detection/width.py` — skeletonize + distance transform → width stats, model-agnostic (works on any mask)
- `detection/rasterize.py` — burns a numeric field (score, width_mean_m, ...) from the detections into a GeoTIFF aligned to the source tile grid, plus a colorized PNG preview (raw single-band GeoTIFFs render as flat grayscale in a plain image viewer without one)

## Shadow handling

`SHADOW_HANDLING` in `config.py` picks one of three strategies for shadowed pixels within the road/bike-lane buffer:

- `"correct"` — brightness-normalize them to match nearby sunlit pixels, described below.
- `"cut"` — drop them entirely: zeroed to `NODATA_VALUE` and reclassified as background, exactly like pixels outside the buffer. For imagery where correction still distorts too much of the tile to be usable, this trades shadowed coverage away entirely rather than keeping a degraded version of it. Also cuts a further `SHADOW_CUT_MARGIN_M` (1 m default) beyond the detected shadow mask, since a real shadow's edge is a soft penumbra that the mask's hard Otsu threshold cuts straight through — cutting only exactly at the mask would still retain a ring of partially-shadowed pixels just outside it, leaving a hard edge anyway.
- `"none"` (default) — leave shadowed pixels exactly as they are in the source imagery. No brightness correction, nothing cut. Both `"correct"` and `"cut"` turned out to still leave visible artifacts (a correction that doesn't quite match, or literal holes in the imagery); `"none"` is the fallback of doing nothing to them at all.

Either way, the shadow band (see Output format) records which pixels were shadowed, even under `"cut"` where those pixels end up as background/nodata in every other band, or `"none"` where the imagery itself is unmodified; it reflects detection only, not the cut margin.

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

## Red boost

`APPLY_RED_BOOST` in `config.py` (on by default) boosts the saturation of reddish pixels within the road/bike-lane buffer, so painted bike-lane paint stands out more from gray asphalt — see `scripts/redness.py`.

- convert to HSV; pixels within `RED_HUE_TOLERANCE` (~29°) of pure red *and* above `MIN_SATURATION_FOR_BOOST` (0.05) qualify — the saturation floor keeps hue-is-noisy near-gray pixels from getting pulled in
- qualifying pixels: saturation × `SATURATION_BOOST` (1.8), clamped to 1.0; hue and value untouched
- convert back to RGB

Deliberately narrower than a generic contrast stretch (tried earlier and reverted for introducing artifacts elsewhere in the tile): this only touches pixels that already read as red, so gray asphalt and vegetation are unaffected. Hue/saturation ranges are calibrated against the one hand-annotated instance in this repo (paint pixels: hue ~9-17°, saturation ~0.11-0.30). Not selective for *lane* paint specifically — any sufficiently red object in the buffer (a parked red car, for instance) gets boosted too.

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

## Texture-embedding detector

A third approach, alongside YOLO and the dropped classical color+shape detector: a frozen, pretrained-on-aerial-imagery CNN used purely for embedding similarity, no training at all. See `scripts/texture_embedding.py`, `scripts/detection/texture_detector.py`, `scripts/texture_analysis.py`.

- **backbone** — TorchGeo's Swin V2-B, `NAIP_RGB_SI_SATLAS` weights (SatlasPretrain, pretrained on NAIP aerial RGB imagery). Picked deliberately over TorchGeo's other pretrained options (Sentinel-2 at 10 m/px, Landsat at 30 m/px) since NAIP is the closest available domain match to this project's own aerial orthophoto imagery — the others would see an entire bike lane as at most a sub-pixel smudge. Classifier head replaced with `Identity()`, keeping the 1024-dim pooled embedding.
- **reference examples** — `data/input/textures/<label>/*.png`, one subfolder per class. Currently: `bikelane` (2 hand-picked lane-paint crops) and `negative` (road, sidewalk, and 4 rooftops — rooftops specifically needed several examples spanning different roof colors; one wasn't enough to generalize).
- **classification: `discriminant_score`, not raw similarity** — plain nearest-neighbor cosine similarity against individual references doesn't work here: pairwise similarity between *any* two reference crops stays high (0.75–0.95) almost regardless of class, so a different-class pair can score higher than a same-class pair (`texture_analysis.py`'s report catches this directly). `discriminant_score` instead projects a candidate embedding onto the direction between the two classes' mean embeddings — still zero training, just arithmetic on frozen embeddings, but isolates the component that actually separates the classes rather than whatever's common to all pavement-like patches. See `texture_embedding.py`'s module docstring for the numbers that motivated this.
- **orthophoto caveat** — this imagery is a *standard* orthophoto (terrain-corrected only, not a *true* ortho built from a full surface model), so tall objects like buildings show relief displacement: a rooftop's visible pixels are offset from the building's true OSM footprint, confirmed by overlaying OSM building polygons directly on the imagery. That's why rooftop confusion was fixed with more reference *pixel* examples rather than by subtracting building footprints from the mask — footprint subtraction would remove pixels at the building's true ground position while the displaced, visible rooftop pixels stayed untouched.

### Analysis / diagnostics

```bash
uv run python -m scripts.texture_analysis                                            # pairwise reference-embedding similarity report
uv run python -m scripts.texture_analysis <tile.tif> <x> <y> <width> <height>        # scan + visualize one region
uv run python -m scripts.texture_analysis edges <tile.tif> <x> <y> <width> <height>  # coarse scan + edge trace + width
```

The second form runs the actual sliding-window detector (`TextureEmbeddingDetector`, 22px window / 11px stride by default) over the given pixel window and saves `texture_scan_result.png`: raw RGB, continuous discriminant-score heatmap, and the thresholded detection mask, side by side. Slow — ~90s for an ~870x580 region on this machine's MPS backend (the backbone runs at ~28ms/image regardless of batching) — not something to run over a whole tile or all 6 tiles in one sitting.

The third form (`edges`) additionally runs `BikeLaneEdgeDetector` (see Width measurement below) and saves `texture_edge_trace_result.png`: RGB, the coarse window-block mask, and the pixel-precise traced mask, side by side — plus prints width statistics per traced segment to stdout.

### Not yet done

- Not wired into `detect.py` as a selectable option (still hardcodes `YoloSegDetector`).
- Never run across all 6 tiles in one go — a single full 5000x5000 tile takes on the order of 20-30 minutes at the default (highest-resolution) window/stride settings, so batch-processing all 6 wasn't attempted; validated on bounded regions, individual crops, and one full tile.

`SCORE_THRESHOLD` (0.10, in `detection/texture_detector.py`) is calibrated from this session's validation crops, not guessed: every genuine lane-paint crop tested scored +0.16 to +0.25, every clean negative scored ≤ -0.10, and the one edge case (a partially-shadowed rooftop) scored +0.042 — still below the lowest lane score. 0.10 sits strictly between the highest validated negative and the lowest validated positive score, so it fixes that edge case as a side effect too. (The original default, 0.0, was too loose — the continuous heatmap cleanly traced real lane paint, but the thresholded mask also picked up a lot of plain street, since much of it still scored weakly positive.)

### Width measurement (`detection/edge_trace.py`)

`TextureEmbeddingDetector`'s mask isn't precise enough to measure width from directly: it's stamped in 22px (4.4m) window blocks, more than twice the width of a real lane (~10px/2m — see `TRAINING_CHIP_SIZE_PX` comment in `config.py`), so its cross-track shape is the scan window's footprint, not the lane's. `BikeLaneEdgeDetector` treats that coarse mask purely as a region-of-interest, then finds the true edges within it by classical HSV color thresholding — a signal precise down to the pixel — and feeds the result to `detection/width.py`'s `measure_width_m`.

The color thresholds (`EDGE_HUE_TOLERANCE=0.15`, `EDGE_MIN_SATURATION=0.07`) are deliberately *not* reused from `redness.py`: that module's thresholds are calibrated for an enhancement pass (better to touch too much than too little) and, tried here first, recalled only ~55% of true path pixels in a real cycle-track crop — tree-branch shadow dappling causes local hue/saturation dropout — producing a speckled mask whose width came out noisy (0.40–5.26m spread along one path). The values above were sourced by sampling real path vs. real adjacent-sidewalk pixels from that same crop and sweeping for a looser hue tolerance that recovers much more of the true path (~72% recall) while keeping the false-positive rate against the neighboring surface low (~0.8%).

One real width measurement isn't one pixel-blob: the traced mask is split into connected components (`scipy.ndimage.label`) and each gets its own `Detection`, so a fragment that breaks off the main path (a drain cover, a gap the color threshold missed) doesn't get averaged into the same statistic as the real lane segment.

**Shape regularization.** The color-threshold mask alone still isn't a lane's actual shape — it inherits every local dropout (shade dappling) and bulge (a similarly-colored fragment bleeding in) pixel by pixel, so its raw edges are noisy and its raw centerline zigzags, when a real lane holds close to one width and runs straight or gently curving. `_regularize_band` fixes both, per component, via `_binned_centerline`: PCA on the component's own pixel coordinates finds its dominant direction, its pixels are binned along that axis, and each bin's *average* position becomes one ordered centerline point. Skeletonizing was tried first and rejected: this mask's ~72% recall (see above) leaves it visibly porous, and `skimage.morphology.skeletonize` is extremely sensitive to exactly that kind of small-scale boundary noise — it turned a single lane into a highly branchy/loopy structure that no amount of spur-pruning reduced to one clean line, even after directly smoothing the mask first. Binning sidesteps the problem entirely: noise gets averaged into a bin's centroid rather than becoming a spurious skeleton branch. The **median** distance-to-edge sampled at the centerline points becomes the segment's radius, the centerline is smoothed with a moving average, and the mask is redrawn as a constant-radius band around it.

**Bridging.** A stretch with zero color-threshold hits at all (a parked car, deep tree shadow) has no pixels to bin — a real gap between components, not a shape defect regularization can smooth away, even though a real lane doesn't actually stop there. So after regularizing each component separately, `BikeLaneEdgeDetector.predict` reconnects them: any two components whose nearest endpoints are both close *and* already heading straight at each other (`BRIDGE_MAX_GAP_RADIUS_MULTIPLE=4.0`, `BRIDGE_ALIGNMENT_COS_MIN=0.82` ≈ 35°) get a straight bridge drawn between them at the narrower segment's width; components joined by a bridge merge back into one `Detection`. The direction check is what makes it safe to be generous with distance — it's what tells a lane continuing past an occluding car apart from an unrelated red feature that just happens to sit nearby.

Tested on a real cycle-track crop: before any of this, a single segment's width came out at a noisy 0.40–6.66m spread, and the raw traced mask was visibly blobby and fragmented into 3–5 disconnected pieces (a parked car, tree shadow gaps). After regularization + bridging: **one continuous segment**, 677 width samples along it, mean 2.81m / median 2.80m with a tight 2.04–3.60m spread — matching a plausible raised cycle track, visibly a single smooth constant-width band running the crop's full length in the output PNG, including straight through where two parked cars occlude the paint. A couple of small (<350px) unbridged fragments remained near the cars — short/isolated enough that they didn't meet the alignment or distance bar, and no further tuning was attempted on that.

## Known limitations

- **OSM gaps** — bike lane mapping stops abruptly where OSM data is incomplete, not where the lane physically ends. Including streets alongside dedicated lanes covers most of this.
- **Buffer bleeds onto buildings** — in dense blocks, narrow streets mean the buffer overlaps adjacent rooftops. `BIKE_LANE_BUFFER_METERS`/`STREET_BUFFER_METERS` were narrowed (6.0/8.0 → 4.5/6.0) to reduce this, but it's a mitigation, not a fix — still happens in tight blocks. The actual fix would be subtracting OSM building footprints from the buffer.
- **Shadow correction is statistical, not physical** — brightens using nearby sunlit pixels, doesn't reconstruct real detail in deep shadow, no sun-angle/DSM data involved. Mainly, because we don't have the data for it.
- **YOLO only sees RGB** — `detection/tiling.py` and `detection/dataset.py` both read bands `[1, 2, 3]` only, so the infrared band (and the classification/shadow bands) are computed but never reach the model. CoCo-Weights are being used, they are expecting 3 bands and not more. Adding infrared would mean a 4-channel first conv layer (losing direct compatibility with the pretrained RGB checkpoint's weights for that layer) and a custom dataset loader, since chips are currently exported as plain 3-channel PNGs. Not attempted yet.
