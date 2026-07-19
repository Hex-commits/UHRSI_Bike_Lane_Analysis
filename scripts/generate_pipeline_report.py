"""Generate a visual, stage-by-stage walkthrough of the pipeline for writeups.

Runs one fixed, already-validated example region (tile 404_5757, the pixel
window used throughout this project's development to test each stage) through
every stage of the pipeline -- raw imagery, OSM masking, shadow detection, red
boost, the prefiltered output, the CNN texture scan, and edge tracing/shape
regularization/bridging -- and writes docs/pipeline_report.md with one figure
per stage plus a width-measurement table.

Deliberately scoped to a small region rather than a full tile: a full-tile
CNN scan takes 20+ minutes (or hours at a finer stride), which would make
"regenerate after every change" impractical. This takes well under a minute.
Re-run after any pipeline change to keep the report current:

    uv run python -m scripts.generate_pipeline_report
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

from scripts.config import INPUT_TILES_DIR, OUTPUT_DIR
from scripts.detection.texture_detector import bike_lane_detector, road_detector
from scripts.detection.width import measure_width_m
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

    # 1. raw input imagery
    _save_png(raw_rgb, FIGURES_DIR / "01_raw.png")

    # 2. OSM road/bike-lane buffer mask
    osm_overlay = _overlay(raw_rgb, classification == 1, (255, 0, 0))
    osm_overlay = _overlay(osm_overlay, classification == 2, (0, 128, 255))
    _save_png(osm_overlay, FIGURES_DIR / "02_osm_mask.png")

    # 3. shadow detection
    shadow_overlay = _overlay(raw_rgb, shadow == 1, (255, 200, 0))
    _save_png(shadow_overlay, FIGURES_DIR / "03_shadow_mask.png")

    # 4. red-saturation boost, before/after
    _side_by_side(raw_rgb, prefiltered_rgb, FIGURES_DIR / "04_red_boost.png")

    # 5. prefiltered output -- what detection actually runs on
    _save_png(prefiltered_rgb, FIGURES_DIR / "05_prefiltered.png")

    # 6. texture-embedding CNN coarse scan
    coarse_detector = bike_lane_detector()
    visualize_scan(OUTPUT_TILE_PATH, WINDOW, FIGURES_DIR / "06_cnn_scan.png", detector=coarse_detector)

    # 7. edge tracing + shape regularization + bridging (reuses the scan above)
    edge_detections = visualize_edge_trace(
        OUTPUT_TILE_PATH, WINDOW, FIGURES_DIR / "07_edge_trace.png", coarse_detector=coarse_detector
    )

    # 8. width measurement, per final segment
    width_rows = _width_rows(edge_detections, pixel_size_m)

    # 9. the same two stages again for road surface, not bike-lane paint
    road_coarse = road_detector()
    road_detections = visualize_edge_trace(
        OUTPUT_TILE_PATH, WINDOW, FIGURES_DIR / "09_road_trace.png", coarse_detector=road_coarse, surface="road"
    )
    road_surface_px = int(sum(d.mask.sum() for d in road_detections))

    _write_report(width_rows, road_surface_px, len(road_detections))
    print(f"Wrote {REPORT_PATH} and {len(list(FIGURES_DIR.glob('*.png')))} figures in {FIGURES_DIR}")


def _width_rows(detections: list, pixel_size_m: float) -> list[tuple[int, int, "object"]]:
    rows = []
    for i, detection in enumerate(sorted(detections, key=lambda d: -d.mask.sum())):
        stats = measure_width_m(detection.mask, pixel_size_m)
        if stats:
            rows.append((i, int(detection.mask.sum()), stats))
    return rows


def _write_report(width_rows: list[tuple[int, int, "object"]], road_surface_px: int, road_components: int) -> None:
    width_table_rows = "\n".join(
        f"| {i} | {px:,} | {stats.mean_m:.2f} | {stats.median_m:.2f} | {stats.min_m:.2f} | {stats.max_m:.2f} | {stats.n_samples} |"
        for i, px, stats in width_rows
    )
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
- **Middle / right:** *identical* on this frame, and that is the point. The road pipeline no longer
  has a pixel-precise stage: the middle panel is the coarse mask, the right one is the same mask with
  isolated specks dropped, and here nothing was small enough to drop -- 57,596 px in both

![road surface](figures/09_road_trace.png)

Note the stair-stepped boundary. The mask is stamped in whole `TEXTURE_WINDOW_PX` scan windows, so
its edges follow the scan grid rather than any kerb. That is the 4.4 m quantisation, visible
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
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content)


if __name__ == "__main__":
    main()
