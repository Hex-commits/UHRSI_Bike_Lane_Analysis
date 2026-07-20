"""Generate a visual, stage-by-stage walkthrough of the pipeline for writeups.

Runs one fixed, validated example region (tile 404_5757, the window used
throughout development) through every stage -- raw imagery, OSM masking,
shadow detection, red boost, prefiltered output, CNN texture scan, edge
tracing/regularization/bridging -- and writes docs/pipeline_report.md with one
figure per stage plus a width-measurement table. Scoped to a small region
rather than a full tile so it takes under a minute (a full-tile CNN scan takes
20+ minutes), making "regenerate after every change" practical:

    uv run python -m scripts.generate_pipeline_report
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from shapely.geometry import box

from scripts.config import INPUT_TILES_DIR, OUTPUT_DIR, TILE_CRS
from scripts.detection.texture_detector import bike_lane_detector, road_detector
from scripts.detection.width import measure_width_m
from scripts.measure_bikelane_gap import load_chunk, measure_gaps, prepare_shadow, render_map
from scripts.osm_features import fetch_osm_features
from scripts.texture_analysis import visualize_edge_trace, visualize_scan

TILE_STEM = "idop20rgbi_32_404_5757_1_nw_2025"
RAW_TILE_PATH = INPUT_TILES_DIR / f"{TILE_STEM}.jp2"
OUTPUT_TILE_PATH = OUTPUT_DIR / f"{TILE_STEM}_bikelanes.tif"
WINDOW = Window(4300, 1330, 750, 180)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
REPORT_PATH = PROJECT_ROOT / "docs" / "pipeline_report.md"


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.55) -> np.ndarray:
    out = image.astype(np.float32)
    out[mask] = out[mask] * (1 - alpha) + np.array(color) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _side_by_side(left: np.ndarray, right: np.ndarray, path: Path) -> None:
    combined = Image.new("RGB", (left.shape[1] + right.shape[1] + 10, left.shape[0]), "white")
    combined.paste(Image.fromarray(left), (0, 0))
    combined.paste(Image.fromarray(right), (left.shape[1] + 10, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(path)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    return result.stdout.strip() or "unknown"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(RAW_TILE_PATH) as src:
        raw_rgb = np.transpose(src.read([1, 2, 3], window=WINDOW), (1, 2, 0))

    with rasterio.open(OUTPUT_TILE_PATH) as src:
        prefiltered_rgb = np.transpose(src.read([1, 2, 3], window=WINDOW), (1, 2, 0))
        classification = src.read(5, window=WINDOW)
        shadow = src.read(6, window=WINDOW)
        pixel_size_m = src.res[0]

    _save_png(raw_rgb, FIGURES_DIR / "01_raw.png")

    osm_overlay = _overlay(raw_rgb, classification == 1, (255, 0, 0))
    osm_overlay = _overlay(osm_overlay, classification == 2, (0, 128, 255))
    _save_png(osm_overlay, FIGURES_DIR / "02_osm_mask.png")

    shadow_overlay = _overlay(raw_rgb, shadow == 1, (255, 200, 0))
    _save_png(shadow_overlay, FIGURES_DIR / "03_shadow_mask.png")

    _side_by_side(raw_rgb, prefiltered_rgb, FIGURES_DIR / "04_red_boost.png")

    _save_png(prefiltered_rgb, FIGURES_DIR / "05_prefiltered.png")

    coarse_detector = bike_lane_detector()
    visualize_scan(OUTPUT_TILE_PATH, WINDOW, FIGURES_DIR / "06_cnn_scan.png", detector=coarse_detector)

    edge_detections = visualize_edge_trace(
        OUTPUT_TILE_PATH, WINDOW, FIGURES_DIR / "07_edge_trace.png", coarse_detector=coarse_detector
    )

    width_rows = _width_rows(edge_detections, pixel_size_m)

    road_coarse = road_detector()
    road_detections = visualize_edge_trace(
        OUTPUT_TILE_PATH, WINDOW, FIGURES_DIR / "09_road_trace.png", coarse_detector=road_coarse, surface="road"
    )
    road_surface_px = int(sum(d.mask.sum() for d in road_detections))

    gap_summary = _bikelane_gap(FIGURES_DIR / "10_bikelane_gap.png")

    _write_report(width_rows, road_surface_px, len(road_detections), gap_summary)
    print(f"Wrote {REPORT_PATH} and {len(list(FIGURES_DIR.glob('*.png')))} figures in {FIGURES_DIR}")


def _bikelane_gap(figure_path: Path) -> dict:
    """Run the 1-D gap measurement on the report window and render its map.

    Reuses `scripts.measure_bikelane_gap` end to end -- the same code the
    standalone tool runs -- so this figure can never drift from the tool.
    Scoped to the report's fixed WINDOW like every other stage; it happens to
    hold 9 OSM bike-lane ways alongside 10 streets, enough for a worked
    example.
    """
    bands, transform, bounds, pixel_size_m = load_chunk(WINDOW)
    osm = fetch_osm_features(bounds)
    clip = box(*bounds)
    streets = osm[osm.category == "street"].clip(clip)
    lanes = osm[osm.category == "bikelane"].clip(clip)

    corrected, shadow, near_edge = prepare_shadow(bands, transform, bounds, pixel_size_m, streets)
    records, _sections, _skipped = measure_gaps(corrected, transform, bounds, shadow,
                                                near_edge, streets, lanes)
    frame = gpd.GeoDataFrame(records, crs=TILE_CRS)

    aspect = WINDOW.width / WINDOW.height
    render_map(bands, transform, frame, lanes, figure_path, pixel_size_m,
               figsize=(13, 13 / aspect + 3))

    reliable = frame[frame.reliable]
    gaps = reliable.gap_m.to_numpy()
    composition = {k: int(v) for k, v in reliable.composition.value_counts().items()}
    return {
        "measured": len(reliable),
        "total": len(frame),
        "in_shadow": int((~frame.reliable).sum()),
        "median_gap_m": float(np.median(gaps)) if len(gaps) else float("nan"),
        "no_strip_pct": float((gaps == 0).mean()) if len(gaps) else 0.0,
        "composition": composition,
    }


def _width_rows(detections: list, pixel_size_m: float) -> list[tuple[int, int, "object"]]:
    rows = []
    for i, detection in enumerate(sorted(detections, key=lambda d: -d.mask.sum())):
        stats = measure_width_m(detection.mask, pixel_size_m)
        if stats:
            rows.append((i, int(detection.mask.sum()), stats))
    return rows


def _write_report(width_rows: list[tuple[int, int, "object"]], road_surface_px: int,
                  road_components: int, gap: dict) -> None:
    width_table_rows = "\n".join(
        f"| {i} | {px:,} | {stats.mean_m:.2f} | {stats.median_m:.2f} | {stats.min_m:.2f} | {stats.max_m:.2f} | {stats.n_samples} |"
        for i, px, stats in width_rows
    )
    gap_composition = ", ".join(f"{count} {kind}" for kind, count in
                                sorted(gap["composition"].items(), key=lambda kv: -kv[1]))
    content = f"""# Pipeline walkthrough: one worked example

*Auto-generated by `scripts/generate_pipeline_report.py`. Do not edit by hand; regenerate after any
pipeline change with:*

```bash
uv run python -m scripts.generate_pipeline_report
```

*Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} from commit `{_git_commit()}`.*

Every stage below runs on the same fixed example region:

- **Tile:** `{TILE_STEM}`
- **Pixel window:** x={WINDOW.col_off:.0f}, y={WINDOW.row_off:.0f}, w={WINDOW.width:.0f}, h={WINDOW.height:.0f}
- **Choice of region:** the area used throughout this project's development to validate each stage
- **Purpose of this document:** a visual trail through the pipeline for reference in writeups; see
  `README.md` for the full technical writeup

## 1. Raw input imagery

- **Content:** unmodified source tile crop (`data/input/idop_kacheln/{TILE_STEM}.jp2`)

![raw input](figures/01_raw.png)

## 2. OSM road/bike-lane buffer mask

- **Geometry source:** OSM features queried via `osmnx` and buffered per `BIKE_LANE_BUFFER_METERS` /
  `STREET_BUFFER_METERS` (`scripts/osm_features.py`, `scripts/mask.py`), overlaid on the raw imagery
- **Red overlay:** dedicated bike-lane buffer
- **Blue overlay:** general street buffer
- **Effect on prefiltering:** survival of only those pixels inside one of these buffers

![osm mask](figures/02_osm_mask.png)

## 3. Shadow detection

- **Method:** blue-excess index, Otsu threshold, and morphological cleanup (`scripts/shadows.py`),
  overlaid in yellow
- **Current config:** `SHADOW_HANDLING="none"`, hence recording of the detected shadow (see "Output
  format" in `README.md`) without any modification of the imagery itself
- **Role of this figure:** completeness of what the pipeline detects, rather than a stage currently
  acted on

![shadow mask](figures/03_shadow_mask.png)

## 4. Red-saturation boost

- **Left:** raw imagery
- **Right:** result of `scripts/redness.py`'s saturation boost on reddish (bike-lane paint) pixels
  within the buffer mask

![red boost before/after](figures/04_red_boost.png)

## 5. Prefiltered output

- **Content:** actual output of the prefiltering stage (`data/output/*.tif`, RGB bands)
- **Downstream use:** input to all detection stages, in place of the raw source tile

![prefiltered](figures/05_prefiltered.png)

## 6. Texture-embedding CNN scan

- **Detector:** `TextureEmbeddingDetector`, a frozen Swin V2-B backbone with `discriminant_score`
  classification (see the "Texture-embedding detector" section of `README.md`)
- **Operation:** sliding-window scan of the prefiltered crop
- **Left:** RGB
- **Middle:** continuous discriminant-score heatmap (red = bikelane-side, blue = negative-side)
- **Right:** thresholded detection mask at window-block resolution, not yet precise enough for width
  measurement

![cnn scan](figures/06_cnn_scan.png)

## 7. Edge tracing, shape regularization, and bridging

- **Detector:** `BikeLaneEdgeDetector` (`scripts/detection/edge_trace.py`)
- **Processing steps:** classical color thresholding within the CNN's coarse region, PCA-binned
  centerline extraction, constant-width band reconstruction, and directional bridging across gaps
  (parked cars, shadow)
- **Left:** RGB
- **Middle:** coarse CNN mask, for reference only; its shape is the scan window's footprint, not the
  lane's
- **Right:** final pixel-precise, regularized, bridged mask

![edge trace](figures/07_edge_trace.png)

## 8. Width measurement

- **Method:** per-segment width statistics via skeletonization and distance transform
  (`scripts/detection/width.py`)
- **Input:** the final regularized mask from step 7

| segment | px | mean (m) | median (m) | min (m) | max (m) | n samples |
|---|---|---|---|---|---|---|
{width_table_rows}

## 9. Road surface

- **Detector:** `road_detector()` + `RoadEdgeDetector` -- the CNN discriminant at
  `ROAD_SCORE_THRESHOLD` (0.18) and nothing else
- **Left:** RGB
- **Middle:** the raw coarse mask, before shadow is cut
- **Right:** the road surface -- the same mask with shadowed pixels removed. The difference is the
  scatter of blocks across the dark road on the left, which the coarse detector claimed and
  which cannot be verified either way

![road surface](figures/09_road_trace.png)

**Shadow is cut, not kept.** In deep shadow this imagery carries almost no surface information:
shadowed road and shadowed non-road measure the same to within noise (median hue
distance from red 0.405 vs 0.405, saturation 0.506 vs 0.519), and a discriminant fitted and scored
on the very same pixels still misclassifies 35%. Anything marked road there is close to a coin
flip, so it is removed along with a 5 px penumbra margin. On this frame that cut 20% of the road
surface; across the whole tile, 13%. That converts a silent error into a visible coverage gap.

Note also the stair-stepped boundary. The mask is stamped in whole `TEXTURE_WINDOW_PX` scan windows,
so its edges follow the scan grid rather than any kerb. That is the 4.4 m quantisation, visible
directly, and it is why widths measured against this surface come out biased wide.

**Surface found on this frame:** {road_surface_px:,} px across {road_components} component(s).

**No width table here, deliberately.** This mask is stamped in `TEXTURE_WINDOW_PX` blocks, so its
resolution is 4.4 m, and its score ramps over ~5 m across a real road edge rather than stepping at
it. It answers "is there road here", not "where does it end", and any width read off its shape would
be measuring the scan grid. Road widths are measured from OSM centerlines by
`scripts.detect_roads` -- see "Road detection" in `README.md`.

The colour test and morphology that used to sit here were removed after being measured: the colour
test discarded two thirds of its own region of interest and left 138 fragments, and every cleanup
step after it moved the boundary a width would be taken from.

## 10. road-to-bike-lane gap

- **Orchestrator:** `scripts.measure_bikelane_gap`, measuring in 1-D directly on the **raw** tile (not
  the prefiltered output above), at the imagery's own 0.2 m resolution -- see "Bike-lane gap" in
  `README.md`
- **Why not the mask:** the deliverable is a 1.5-3 m gap, and the coarse mask on the left of step 9 is
  quantised to 4.4 m blocks. A cross-section cut from the pixels locates each edge subpixel (measured
  precision ~0.08 m on this tile) where the mask cannot resolve the feature at all
- **OSM as scaffold only:** street/lane centerlines say *where* to cut and which way to face; every
  edge, width and gap is read off pixels
- **Colour:** each lane segment coloured by the gap to the road; **0 m** (light blue) is a
  measured result -- the lane is flush with or painted on the road, with no separating strip -- not a
  blank. Grey is shadow, the only genuinely unmeasurable case

![bikelane gap](figures/10_bikelane_gap.png)

**On this frame:** {gap["measured"]} of {gap["total"]} cross-sections measured, {gap["in_shadow"]} in
shadow; median gap {gap["median_gap_m"]:.2f} m, and {gap["no_strip_pct"]:.0%} with no separating strip
at all ({gap_composition}). That most lanes read 0 m is the real picture of the district: most cycling
infrastructure here is painted onto or flush with the road. A gap only opens up where a verge,
buffer or paved strip physically separates the two -- those are the coloured stretches.

**`contiguous` is "no separating strip detected", not a certified zero.** A paint line fainter than
`MARKING_MIN_EXCESS`, or a low-contrast material change, would be missed and also land at 0 m. The
`composition` field keeps the distinction (`contiguous` = no boundary found, vs `abutting` = boundary
found with zero strip) so it can be audited; putting an error bar on it would need ground-truth
cross-sections.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content)


if __name__ == "__main__":
    main()
