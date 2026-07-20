# UHRSI Bike Lane Analysis

Filters Münster aerial imagery down to the bike lane / street network, for ML training data prep.

- bike lane + street geometry from OpenStreetMap
- imagery masked to a buffer around that geometry
- shadows detected, then brightness-corrected, cut entirely, or left untouched (configurable)
- reddish (bike-lane paint) pixels get a saturation boost, so they stand out more against gray asphalt
- classification + shadow bands appended to the output

**[docs/pipeline_report.md](docs/pipeline_report.md)** — a figure per pipeline stage (raw imagery → OSM mask → shadow detection → red boost → prefiltered output → CNN scan → edge tracing/regularization/bridging → width measurement → the same detection pipeline applied to road surface) on one fixed worked example, for reference in writeups. Regenerate after any pipeline change with `uv run python -m scripts.generate_pipeline_report` (takes well under a minute — it's scoped to one small region, not a full tile).

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
  detections/     # detection GeoPackage + raster/PNG previews, incl. per-tile road runs (not in git)
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
- `generate_pipeline_report.py` — regenerates `docs/pipeline_report.md`, a visual walkthrough of every stage on one fixed example region (see below)
- `detection/base.py` — the model swap point: `Detector` protocol, `Detection` type. Nothing else in the detection code depends on a specific model.
- `detection/dataset.py` — converts CVAT YOLO-seg exports (drawn on the full 5000x5000 tiles) into a chipped, trainable YOLO dataset
- `detection/yolo_seg_detector.py` — current adapter: loads a fine-tuned YOLO-seg checkpoint, runs inference on a chip
- `detection/texture_detector.py` — `Detector` adapter for texture_embedding.py: sliding-window scan + discriminant scoring. One class, two configurations — `bike_lane_detector()` and `road_detector()` differ only in which reference labels they discriminate between and their thresholds
- `detection/edge_trace.py` — `BikeLaneEdgeDetector`, the full colour-trace pipeline for lane paint (see Width measurement below), and `RoadEdgeDetector`, which is now only the thresholded CNN mask — its colour test was measured and removed (see Road detection below)
- `detection/tiling.py` — chips a tile into windows for inference, maps masks back to tile pixel/geo coordinates
- `detection/width.py` — `measure_width_m`: skeletonize + distance transform. Bike lanes only; it does not generalize to roads (see Road detection below)
- `detection/centerline_width.py` — width measured by casting rays perpendicular to a known OSM centerline, the only method that survives tile scale (see Road detection below)
- `detect_roads.py` — orchestrator for a whole-tile road run: coarse CNN scan → per-OSM-way width, writing a GeoPackage, a width-coloured map and the cached surface mask
- `detection/cross_section.py` — measures a road profile in 1-D, at the imagery's own 0.2 m resolution: subpixel edge location on illumination-invariant features, plus a separate narrow-bright-line detector for painted markings the material gradient cannot see (see Bike-lane gap below)
- `measure_bikelane_gap.py` — orchestrator for the gap between carriageway and bike lane: cuts a cross-section from road centerline to lane, reads both edges from pixels, writes a GeoPackage and a lane-coloured map (see Bike-lane gap below)
- `detection/ribbon_fit.py` — **parked, imported by nothing.** Joint two-edge fit against continuous imagery; the machinery works, the evidence function is biased narrow (see Road detection below)
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

**We also tried a classical (non-ML) color+shape detector and dropped it.** Otsu-adaptive red/white color splits plus shape filtering (elongated, constant-width) found real lane paint in isolation, but at full-tile scale it couldn't reliably separate lane paint from other similarly-colored surfaces in this imagery -- terracotta rooftops (near-identical redness/hue to the paint) and bare dirt/leaf-litter ground under winter trees both cleared the same color thresholds. Shape and border filtering cut down but didn't eliminate either confound. See git history if you want the details.

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
- **reference examples** — `data/input/textures/<label>/*.png`, one subfolder per class. Currently: `bikelane` (2 hand-picked lane-paint crops), `negative` (road, sidewalk, and 4 rooftops — rooftops specifically needed several examples spanning different roof colors; one wasn't enough to generalize), and `road` (7 sunlit carriageway crops, added for the road detector — see Road detection). Which folders count as positive vs. negative is per-detector, set by `BIKE_LANE_TEXTURE_LABELS` / `ROAD_TEXTURE_LABELS` in `config.py`, since road is a negative when looking for lane paint and the positive when looking for road.
- **classification: `discriminant_score`, not raw similarity** — plain nearest-neighbor cosine similarity against individual references doesn't work here: pairwise similarity between *any* two reference crops stays high (0.75–0.95) almost regardless of class, so a different-class pair can score higher than a same-class pair (`texture_analysis.py`'s report catches this directly). `discriminant_score` instead projects a candidate embedding onto the direction between the two classes' mean embeddings — still zero training, just arithmetic on frozen embeddings, but isolates the component that actually separates the classes rather than whatever's common to all pavement-like patches. See `texture_embedding.py`'s module docstring for the numbers that motivated this.
- **orthophoto caveat** — this imagery is a *standard* orthophoto (terrain-corrected only, not a *true* ortho built from a full surface model), so tall objects like buildings show relief displacement: a rooftop's visible pixels are offset from the building's true OSM footprint, confirmed by overlaying OSM building polygons directly on the imagery. That's why rooftop confusion was fixed with more reference *pixel* examples rather than by subtracting building footprints from the mask — footprint subtraction would remove pixels at the building's true ground position while the displaced, visible rooftop pixels stayed untouched.

### Analysis / diagnostics

```bash
uv run python -m scripts.texture_analysis                                                      # pairwise reference-embedding similarity report
uv run python -m scripts.texture_analysis <tile.tif> <x> <y> <width> <height> [stride_px]      # scan + visualize one region
uv run python -m scripts.texture_analysis edges <tile.tif> <x> <y> <width> <height> [stride_px] # coarse scan + edge trace + width
uv run python -m scripts.texture_analysis road <tile.tif> <x> <y> <width> <height> [stride_px]  # the same, for road surface
```

The second form runs the actual sliding-window detector (`TextureEmbeddingDetector`, `TEXTURE_WINDOW_PX`/`TEXTURE_STRIDE_PX` in `config.py`, 22px/11px by default) over the given pixel window and saves `texture_scan_result.png`: raw RGB, continuous discriminant-score heatmap, and the thresholded detection mask, side by side. Slow — ~90s for an ~870x580 region on this machine's MPS backend (the backbone runs at ~28ms/image regardless of batching) — not something to run over a whole tile or all 6 tiles in one sitting.

The third form (`edges`) additionally runs `BikeLaneEdgeDetector` (see Width measurement below) and saves `texture_edge_trace_result.png`: RGB, the coarse window-block mask, and the pixel-precise traced mask, side by side — plus prints width statistics per traced segment to stdout.

The optional `stride_px` overrides `TEXTURE_STRIDE_PX` for that one run — smaller means a finer-resolution score map/heatmap (more overlapping sample points instead of the same score block-stamped over a wider area), at roughly `(11 / stride_px)²` the compute cost, since window count grows quadratically as stride shrinks. Only practical scoped to a bounded cutout like this CLI takes, not a whole tile — `TEXTURE_STRIDE_PX`'s default (11) already takes ~23 minutes over a full 5000x5000 tile; `stride_px=4` (~7.6x the cost) took ~3 minutes over a ~750x180 cutout and produced consistent results with the default (main segment mean width 2.71m vs. 2.81m) — a sharper score map, not a different answer, on the region tested so far.

### Not yet done

- Not wired into `detect.py` as a selectable option (still hardcodes `YoloSegDetector`) — this applies to the road detector too, which is currently reachable only via `scripts.texture_analysis road` and the pipeline report.
- Never run across all 6 tiles in one go — a single full 5000x5000 tile takes on the order of 20-30 minutes at the default (highest-resolution) window/stride settings, so batch-processing all 6 wasn't attempted; validated on bounded regions, individual crops, and one full tile.

`SCORE_THRESHOLD` (0.10, in `detection/texture_detector.py`) is calibrated from this session's validation crops, not guessed: every genuine lane-paint crop tested scored +0.16 to +0.25, every clean negative scored ≤ -0.10, and the one edge case (a partially-shadowed rooftop) scored +0.042 — still below the lowest lane score. 0.10 sits strictly between the highest validated negative and the lowest validated positive score, so it fixes that edge case as a side effect too. (The original default, 0.0, was too loose — the continuous heatmap cleanly traced real lane paint, but the thresholded mask also picked up a lot of plain street, since much of it still scored weakly positive.)

### Width measurement (`detection/edge_trace.py`)

`TextureEmbeddingDetector`'s mask isn't precise enough to measure width from directly: it's stamped in 22px (4.4m) window blocks, more than twice the width of a real lane (~10px/2m — see `TRAINING_CHIP_SIZE_PX` comment in `config.py`), so its cross-track shape is the scan window's footprint, not the lane's. `BikeLaneEdgeDetector` treats that coarse mask purely as a region-of-interest, then finds the true edges within it by classical HSV color thresholding — a signal precise down to the pixel — and feeds the result to `detection/width.py`'s `measure_width_m`.

The color thresholds (`EDGE_HUE_TOLERANCE=0.15`, `EDGE_MIN_SATURATION=0.07`) are deliberately *not* reused from `redness.py`: that module's thresholds are calibrated for an enhancement pass (better to touch too much than too little) and, tried here first, recalled only ~55% of true path pixels in a real cycle-track crop — tree-branch shadow dappling causes local hue/saturation dropout — producing a speckled mask whose width came out noisy (0.40–5.26m spread along one path). The values above were sourced by sampling real path vs. real adjacent-sidewalk pixels from that same crop and sweeping for a looser hue tolerance that recovers much more of the true path (~72% recall) while keeping the false-positive rate against the neighboring surface low (~0.8%).

One real width measurement isn't one pixel-blob: the traced mask is split into connected components (`scipy.ndimage.label`) and each gets its own `Detection`, so a fragment that breaks off the main path (a drain cover, a gap the color threshold missed) doesn't get averaged into the same statistic as the real lane segment.

**Shape regularization.** The color-threshold mask alone still isn't a lane's actual shape — it inherits every local dropout (shade dappling) and bulge (a similarly-colored fragment bleeding in) pixel by pixel, so its raw edges are noisy and its raw centerline zigzags, when a real lane holds close to one width and runs straight or gently curving. `_regularize_band` fixes both, per component, via `_binned_centerline`: PCA on the component's own pixel coordinates finds its dominant direction, its pixels are binned along that axis, and each bin's *average* position becomes one ordered centerline point. Skeletonizing was tried first and rejected: this mask's ~72% recall (see above) leaves it visibly porous, and `skimage.morphology.skeletonize` is extremely sensitive to exactly that kind of small-scale boundary noise — it turned a single lane into a highly branchy/loopy structure that no amount of spur-pruning reduced to one clean line, even after directly smoothing the mask first. Binning sidesteps the problem entirely: noise gets averaged into a bin's centroid rather than becoming a spurious skeleton branch. The **median** distance-to-edge sampled at the centerline points becomes the segment's radius, the centerline is smoothed with a moving average, and the mask is redrawn as a constant-radius band around it.

**Bridging.** A stretch with zero color-threshold hits at all (a parked car, deep tree shadow) has no pixels to bin — a real gap between components, not a shape defect regularization can smooth away, even though a real lane doesn't actually stop there. So after regularizing each component separately, `BikeLaneEdgeDetector.predict` reconnects them: any two components whose nearest endpoints are both close *and* already heading straight at each other (`BRIDGE_MAX_GAP_RADIUS_MULTIPLE=4.0`, `BRIDGE_ALIGNMENT_COS_MIN=0.82` ≈ 35°) get a straight bridge drawn between them at the narrower segment's width; components joined by a bridge merge back into one `Detection`. The direction check is what makes it safe to be generous with distance — it's what tells a lane continuing past an occluding car apart from an unrelated red feature that just happens to sit nearby.

Tested on a real cycle-track crop: before any of this, a single segment's width came out at a noisy 0.40–6.66m spread, and the raw traced mask was visibly blobby and fragmented into 3–5 disconnected pieces (a parked car, tree shadow gaps). After regularization + bridging: **one continuous segment**, 677 width samples along it, mean 2.81m / median 2.80m with a tight 2.04–3.60m spread — matching a plausible raised cycle track, visibly a single smooth constant-width band running the crop's full length in the output PNG, including straight through where two parked cars occlude the paint. A couple of small (<350px) unbridged fragments remained near the cars — short/isolated enough that they didn't meet the alignment or distance bar, and no further tuning was attempted on that.

## Road detection

Road detection is **not** the bike-lane pipeline with the labels swapped. It was built that way first, and the pixel-precise half was then removed after being measured. What remains is deliberately much smaller:

- **`road_detector()`** — the texture CNN at `ROAD_SCORE_THRESHOLD` (0.18), producing a mask stamped in 22 px scan-window blocks. This is the whole of the surface detection.
- **`detection/centerline_width.py`** — width measured by casting rays perpendicular to OSM road centerlines, against that mask.

**Why the colour test was removed.** An asphalt colour test (`hue_distance_from_red >= 0.25 & saturation <= 0.25`) used to refine the coarse mask, followed by a morphological closing and a small-component filter. Accounting for each step on the representative frame:

| step | pixels | change | components |
|---|---|---|---|
| coarse CNN mask | 84,579 | | 4 |
| `binary_dilation`, 8 px | 109,526 | +24,947 | 1 |
| asphalt colour test | 36,437 | **−73,089** | 138 |
| `closing(disk(2))` | 37,877 | +1,440 | 54 |
| drop components ≤200 px | 36,961 | −916 | 6 |

The colour test discarded two thirds of its own region of interest and shattered the rest into 138 fragments; the morphology after it existed only to make that usable again. Every one of those steps moves the boundary a width is measured from. Bike-lane paint has a strong colour cue so a colour test genuinely localizes it — road surface does not, and there the test was mostly discarding real road.

`ROAD_SCORE_THRESHOLD` was raised 0.14 → 0.18 to compensate. 0.14 was Youden's J optimum, but J weighs recall and false positives equally and they are not equally costly here: with nothing refining the mask downstream, anything wrongly included goes straight into the surface. 0.18 cuts the false-positive rate 33.7% → 15.1% for 72.7% → 48.1% recall. A missing road is a visible coverage gap; an invented one is not.

Everything below was measured on the same representative frame the rest of this document uses (tile 404_5757, x=4300 y=1330 w=750 h=180), scored against the prefiltered tile's own classification band as ground truth — carriageway (class 2) as positive, bike-lane buffer (class 1) as negative.

**The width measurement works; the coarse localization is much weaker than the bike-lane equivalent.** That weakness is a property of the signal, not of the implementation: bike-lane paint has a strong color cue, road has only texture, and almost everything inside the prefiltered buffer is pavement of some kind.

- **Coarse discriminant** — carriageway pixels score a median +0.177 against +0.114 for the bike-lane buffer: overlapping distributions, not separated ones. Sweeping the threshold, Youden's J peaks at 0.14 (72.7% of carriageway retained, 33.7% of bike-lane buffer wrongly retained). J = 0.39, where the bike-lane discriminant reaches 0.73 at its own shipped threshold on the same frame. So the coarse road mask covers about two thirds of this frame, including the parking area and sidewalk, and is only ever used to bound a region of interest — never as a decision on its own.
- **Asphalt trace** — 82% recall at 32% false-positive rate on sunlit pixels. The FPR is far worse than the bike-lane tracer's ~0.8%, partly genuinely (gray is a weak signal) and partly by measurement: the "negative" class here is the bike-lane *buffer*, which in this frame overlaps real carriageway and covers an asphalt cycle path, so some of what counts against it is asphalt correctly identified as asphalt.

**Shadow is cut from the road surface entirely** (`SHADOW_EXCLUSION_MARGIN_PX`, 5 px = 1 m past the detected mask, matching prefiltering's `SHADOW_CUT_MARGIN_M`). The shadow band is already computed and stored as band 6, so this costs nothing to apply. On the representative frame it removes 20% of the detected road surface; across the full tile, 13%.

**Deep shadow is not recoverable, and is excluded deliberately rather than missed.** Shadowed carriageway and shadowed non-carriageway are statistically indistinguishable in this imagery: median hue distance from red 0.405 vs 0.405, median saturation 0.506 vs 0.519, mean RGB (44,60,81) vs (41,57,79). Embedding them and fitting a discriminant on the same pixels it is then scored on — an optimistic upper bound — still misclassifies 35%, against 20% for the equivalent sunlit comparison. Shadowed pavement is lit by blue skylight and so reads strongly saturated, which means the saturation ceiling rejects it as a side effect; that is left as-is. A darker or looser branch added to recover the shadowed carriageway would admit the shadowed sidewalk beside it in equal measure, fabricating coverage rather than measuring it. This is also why the `road` reference folder contains only sunlit crops: including shadowed ones as positives, with no shadowed negatives to balance them, would have made brightness the dominant axis of the discriminant.

### Width methods tried and dropped

Three ways of measuring road width from the *shape of a mask* were built and discarded before the centerline method below. The findings are kept because each one rules something out:

- **Medial axis** (`measure_width_m`, still used for bike lanes). A distance transform measures the distance to the nearest edge of any kind, including the edges of interior holes, and a traced road is full of them. On the largest carriageway component it returned a 4.0 px radius where the true maximum inscribed radius was 27.3 px. `binary_fill_holes` was tried as a cheap fix and did essentially nothing (1.60 m → 1.74 m): the interruptions are open indentations connected to the exterior, not enclosed islands.
- **Constant-width band** (`_regularize_band` + bridging, the bike-lane back half). Produced 1.1–4.7 m on a carriageway plainly 12 m across — a road reduced to a thin line. A lane holds one width along its length; a road does not.
- **PCA cross-track extent, split by width run.** This one worked on a single isolated frame (carriageway 12.03 m over 119 cross-sections) and failed completely at tile scale, because pavement connects into networks: a T-junction has no dominant axis, so PCA returns something diagonal and the width is measured across the junction. On one 300×300 m region it reported a 28 m road (a junction) and a 55 m road (a car park).

**Intersecting the coarse mask with the OSM street class was also tried and rejected.** It removes the car-park false positives, but `STREET_BUFFER_METERS` is 6.0, so the buffer is 12 m wide and *clips the carriageway* — measured widths dropped to 7.57–8.59 m. The number reported would be the buffer's width rather than anything measured from the imagery. For a detector whose output is a width, constraining the region by an assumed width defeats the purpose.

**A ribbon fit is parked, not abandoned** (`detection/ribbon_fit.py`, wired into nothing). It fits both edges jointly along a whole way by dynamic programming against continuous imagery, never binarising — the right shape of answer, and it fixed three real bugs on the way. Its evidence function is biased narrow (median 5.35 m where these streets run ~9 m) because "road-like" is defined by similarity to a reference sampled at the centerline, so similarity declines monotonically outward. The module docstring records the state and the two candidate fixes. One hypothesis was tested and **disproven**: that the prefilter's buffer masking was the limiter — running against raw unmasked `.jp2` imagery gave an identical 5.35 m.

### Tile scale needs the centerline, not the mask's own shape

Everything above measures a road's width from the shape of the traced mask itself. That holds on one hand-picked frame and **stops holding as soon as the extent contains a junction**, which at tile scale is immediately.

Every method above infers a direction from the mask itself — a medial axis, or PCA on the pixel coordinates. A T-junction has no dominant axis, so PCA returns something diagonal and the width that follows is measured across the junction rather than across either road. Run on one 300×300 m region, the mask-shape method reported a 28 m road (a T-junction) and a 55 m road (a car park).

`detection/centerline_width.py` takes the direction from OSM's road centerlines instead, which the pipeline already fetches and caches. For each point sampled along a way, the local tangent gives a perpendicular, and the width is how far the traced asphalt actually extends along it in each direction. Three things fall out of that at once:

- junctions stop mattering, because each OSM way is measured as its own unit regardless of what it touches
- surfaces with no centerline are never measured, which is what finally excludes the parking lots the coarse texture discriminant cannot reject
- results are keyed to OSM ways, so a width joins straight back to that way's tags

**This is not the buffer-as-ROI idea rejected above.** The buffer is not the measurement — the ray stops where the traced asphalt stops. The buffer only bounds how far a ray can possibly travel, since the prefiltered imagery is masked to it, and `buffer_limited_fraction` records any sample that ran into that edge so a clipped measurement stays visible rather than silently reading as a narrow road.

**Two flags do the honest work here, and both are load-bearing:**

- `buffer_limited_fraction` — the sample hit masked-out background, so the road may be wider than reported.
- `unbounded_fraction` — the ray traveled the full 15 m cap without ever finding an edge, so the width for that sample is a lower bound reported at the cap rather than an observed measurement. Both are per-way statistics on the measurement itself; neither is used to classify a way as anything.

### Result on a full tile

One complete 5000×5000 tile (`404_5757`, 1 km²), default stride, 23 minutes for the coarse scan:

| | |
|---|---|
| coarse road mask | 13% of the tile |
| traced asphalt surface | 1,864,726 px (7% of the tile) |
| OSM street ways crossing the tile | 173 |
Run twice, before and after the colour test was removed. The comparison is the useful part:

| | thr 0.14 + colour test | **thr 0.18, CNN only** |
|---|---|---|
| coarse mask | 13% of tile | **8%** |
| traced surface | 1,864,726 px | **2,108,062 px** |
| ways with ≥3 cross-sections (of 173) | 113 | **101** |
| median width | 9.20 m | **11.10 m** |
| p10–p90 | 6.42–14.54 m | **6.00–17.80 m** |
| samples stopped by the buffer edge | 1.3% | **0.9%** |
| mean `unbounded_fraction` | — | **16.9%** |

Two things worth reading carefully.

**The mask got stricter and the surface got bigger** — 13% → 8% coarse, but 1.86M → 2.11M px of surface. That is the colour test's cost stated in one line: it was discarding more than the stricter threshold ever did.

**The widths got worse, not better.** They are systematically ~2 m wider, with the upper tail moving much further (p90 14.54 → 17.80 m), and 11.10 m is too wide for a median German urban street where 9.20 m was already plausible. The mechanism is the quantisation this run accepted: the CNN mask is stamped in 22 px blocks, so its boundary overshoots the true road edge by up to half a block on each side, and a ray casting against it stops late. Neither figure is validated against ground truth, so the honest claim is directional rather than absolute — but the bias has a known sign, and it is up.

Coverage fell at each step (113 → 101 → 91 ways), which is both changes behaving as intended: the stricter threshold claims less, and the shadow cut refuses to guess. Cutting shadow removed 13% of the tile's road surface and pulled the median width down 1 m, which is consistent with shadowed blocks having been inflating it.

Buffer clipping is negligible in both runs, so `STREET_BUFFER_METERS` is not meaningfully constraining the measurement at these widths.

**The 35% of ways with no measurement is the real coverage limit**, and it is mostly the shadow problem: a way with fewer than 3 measurable cross-sections is one whose road surface was not traced under it, and deep shadow is where the asphalt test cannot work at all (see above). This is a coverage gap rather than a wrong number — those ways are absent from the output, not present with a bad width.

### Running it on a tile

```bash
uv run python -m scripts.detect_roads data/output/foo.tif             # whole tile, ~35 min
uv run python -m scripts.detect_roads data/output/foo.tif 22          # coarser scan, ~4x faster
uv run python -m scripts.detect_roads data/output/foo.tif 11 0 0 1500 1500   # one window
```

Writes three things to `data/detections/`:

- `<tile>_roads.gpkg` — one row per OSM way: `width_median_m` / `mean` / `min` / `max`, `n_samples`, `buffer_limited_fraction`, `unbounded_fraction`
- `<tile>_roads_width.png` — every measured way drawn over the imagery, colored by median width (4 m blue → 20 m red). This is the one to look at: a road reading 20 m stands out against its neighbors at a glance, where a table of 100 rows hides it
- `<tile>_roads_surface.png` and `.npz` — the traced surface as an overlay, and as a mask. The scan is the expensive part by far, so keeping it means a different overlay or a re-run of the width measurement doesn't mean paying for it again

### Off-street surface, and why it is not filtered out

About **9% of the traced surface area is not road**: car parks, forecourts and driveways that the coarse discriminant cannot tell from carriageway on texture alone. Measured by buffering OSM street centerlines by 8 m — 362 of 444 raw polygons (61,685 m²) fall mostly inside it, 67 (6,731 m²) mostly outside. Three ways of cleaning it up were tried and all three dropped:

- **Geometric dissolve** — vectorize, buffer out 3 m, union, buffer back in 3 m. A morphological closing: it merges fragments split by the scan grid or by the shadow cut, taking 444 polygons to 179. **Dropped because it overruns bike lanes.** Closing is *extensive* — it can only add area — and it fills every concavity finer than 6 m, so the road network ends up ~1–2 m fatter per side. A bike lane alongside the carriageway is only 1.5–3 m wide, so the fattened road surface swallows exactly the separation this project exists to measure. It also made false positives worse rather than better: scattered detections across a car park weld into one solid 17,000 m² polygon that reads as far more confident than the evidence behind it.
- **Morphological opening** — shrink then grow, the reverse operation. At 1.0 m it removes 0.7% of the area while *fragmenting* the network, because it severs narrow chokepoints. An opening only erases *thin* things, and these false positives are large compact blobs. No radius that spares a 5 m carriageway touches a 20 m car park.
- **Shape filters** — effective width (`2·area/perimeter`) and compactness do not separate the two classes at all, because raw polygons are blocky scan-window fragments of median area 46 m² whose shape encodes the scan grid rather than the surface. The distributions are identical to within noise (width p10/median/p90: 1.92/2.75/4.52 on-street vs 1.85/2.64/4.35 off-street), and a width cut at 8 m removes 0% of off-street area. A min-rotated-rectangle elongation ratio fares no better: a branching road network has a square-ish bounding rectangle, so median elongation is 1.5 for both classes.

The OSM-proximity test used to measure the 9% would itself work as a filter, but it would make the surface unable to contain any road OSM does not already know about. Deferred rather than adopted: the root cause is the scan-grid stamping and the weak coarse discriminant, and every one of these filters is a downstream symptom of it.

This deliberately does **not** reuse `detect.py`'s chip loop. `iter_chips` cuts a tile into 640 px squares and runs the detector on each independently, which is fine for a model that decides per chip. 640 px is 128 m — short enough that chip boundaries would slice most roads into pieces, each measured as if it were a separate road. The coarse scan and the trace both run against the full extent in one pass; only the sliding scan window is chunked, by batch, inside the detector. The cost of that is time, not memory — a full tile's masks are a few hundred MB, while the scan is on the order of half an hour at the default stride, scaling with 1/stride².

### Remaining known problem

The coarse discriminant is still too weak to tell carriageway from other large paved surfaces on texture alone. The centerline method routes around that for *measurement* — nothing without an OSM centerline under it is ever measured — but the underlying discriminant is unchanged, and the traced surface mask itself still covers pavement that is not road (~9% of its area, quantified under "Off-street surface" above). Anything that consumes the mask rather than the per-way widths inherits that.

The widths themselves are also only as good as the traced surface they are measured against, and that surface comes from a colour threshold plus morphological cleanup. Where a way's `unbounded_fraction` is high, the ray never found an edge and the width is a lower bound, not a measurement.

## Bike-lane gap (`measure_bikelane_gap.py`)

The deliverable: for every point along a bike lane, how far is it from the carriageway beside it, in metres, shown on a map.

```bash
uv run python -m scripts.measure_bikelane_gap                    # whole tile, ~70 s
uv run python -m scripts.measure_bikelane_gap 1600 1600 1600 1600  # one window: col row w h
```

Writes `data/detections/bikelane_gap.gpkg` (one LineString per cross-section, road point → lane point, with `gap_m`, `composition`, `shadow_fraction`, `reliable`) and `bikelane_gap_map.png` (the lane drawn over the imagery, coloured by gap).

### Why 1-D, not the mask

The 2-D route cannot do this, and the reason is resolution, not effort. The gap is 1.5–3 m; the coarse CNN mask is stamped in 22 px windows, so its edge is quantised to 4.4 m blocks (see "Road detection" — the mask "answers *is there road here*, not *where does it end*"). A 4.4 m ruler cannot measure a 2 m feature, and nothing geometric repairs a boundary that never carried the information — a buffer/union dissolve, a morphological opening and two shape filters were all tried against the road mask and all failed.

The imagery is 0.2 m/px, so the information is in the pixels. `detection/cross_section.py` cuts a 1-D profile straight from them at a stated budget — 0.05 m sampling, 0.15 m smoothing, ~0.30 m two-edge separation limit, subpixel edge location by parabola fit. Measured edge precision on this tile is **0.08 m** (0.4× the pixel), against the mask's 4.4 m — the first ruler fine enough for the job. Edges are found on chromaticity + NDVI so a cast shadow (which scales all bands together) does not register as a material boundary.

### OSM is a scaffold, nothing more

OSM street/lane centerlines say *where* to cut a cross-section and which way to face it — the one thing this imagery cannot reliably supply, since the satellite-derived bike-lane mask's two densest regions on this tile are a shadow edge across a car park and a rooftop. Every measured number is read off pixels; no edge, width or gap comes from OSM geometry. Runs on the **raw** tiles, not the prefiltered ones: those are masked to an OSM buffer, which would put an artificial edge exactly where a lane's outer boundary sits.

### Painted markings are a separate detector

White paint on grey asphalt is almost purely a *brightness* change, and brightness is deliberately excluded from the edge detector (that is how shadows read). So the lane-edge line — the very thing marking a road/lane boundary — is invisible to it. `detect_markings` finds these separately as narrow bright spikes, told apart from a shadow step by shape: a shadow is a monotonic step, a marking is a symmetric spike flanked by darker asphalt on both sides, a few decimetres wide. On this tile it recovers 63 lane lines the gradient missed and sharpens edge precision from 0.09 to 0.08 m.

### Reading the result

A gap of **0 m is a measurement, not a blank**: `MIN_RUN_M = 0` formalises that two surfaces which touch are separated by exactly nothing. Where one continuous asphalt run spans road centerline and lane centerline with no boundary between them — no material change, and now that markings are detected, no paint line either — there is no separating strip, so the gap is 0. These are labelled `contiguous`; a boundary found with zero strip between is `abutting`. On this tile **88% of measured lanes have no separating strip** (median gap 0.00 m), which is the real picture of the district: most cycling infrastructure is painted onto or flush with the carriageway. The coloured 0.25–2 m stretches are where a verge, buffer or paved strip actually separates them.

Two honest limits:

- **`contiguous` means "no separating strip *detected*"**, not a metrology-grade zero — a paint line fainter than `MARKING_MIN_EXCESS` could still be missed and land here as 0. The `composition` field preserves the distinction (`contiguous` = no boundary at all, vs `abutting` = boundary found, zero strip) so it can be audited.
- **Grey on the map is shadow only** (~2% of the tile). Shadow correction leaves a residual false edge at the shadow boundary, so cross-sections within 3 m of one are dropped rather than trusted; `reliable=False` marks them.

The small composition classes are not yet trustworthy at these counts: `vegetation` at a 9 m "gap" is a park strip or a mis-paired road, not a lane separator, and `red paint` is suspect on imagery where red clay roof tiles dominate the redness index and the lanes are not red-painted. Only `contiguous`, `abutting`, `bright/marking` and `asphalt` have the counts to mean anything.

## Known limitations

- **OSM gaps** — bike lane mapping stops abruptly where OSM data is incomplete, not where the lane physically ends. Including streets alongside dedicated lanes covers most of this.
- **Buffer bleeds onto buildings** — in dense blocks, narrow streets mean the buffer overlaps adjacent rooftops. `BIKE_LANE_BUFFER_METERS`/`STREET_BUFFER_METERS` were narrowed (6.0/8.0 → 4.5/6.0) to reduce this, but it's a mitigation, not a fix — still happens in tight blocks. The actual fix would be subtracting OSM building footprints from the buffer.
- **Shadow correction is statistical, not physical** — brightens using nearby sunlit pixels, doesn't reconstruct real detail in deep shadow, no sun-angle/DSM data involved. Mainly, because we don't have the data for it.
- **Gap `contiguous` cannot certify a true zero** — the bike-lane gap reads 0 m wherever no separating strip is detected between carriageway and lane. That is correct for a lane painted on or flush with the road, but a paint line fainter than `MARKING_MIN_EXCESS` or a low-contrast material change would be missed and also land at 0. The measurement says "nothing separates them that the imagery can see", not "they physically touch"; ground-truth cross-sections would be needed to put an error bar on it.
- **YOLO only sees RGB** — `detection/tiling.py` and `detection/dataset.py` both read bands `[1, 2, 3]` only, so the infrared band (and the classification/shadow bands) are computed but never reach the model. CoCo-Weights are being used, they are expecting 3 bands and not more. Adding infrared would mean a 4-channel first conv layer (losing direct compatibility with the pretrained RGB checkpoint's weights for that layer) and a custom dataset loader, since chips are currently exported as plain 3-channel PNGs. Not attempted yet.
