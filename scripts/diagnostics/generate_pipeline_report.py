"""Generate a visual, stage-by-stage walkthrough of the pipeline for writeups.

Runs one fixed, validated example region (tile 404_5757, the window used
throughout development) through every stage -- raw imagery, OSM masking,
shadow detection, red boost, prefiltered output, CNN texture scan, edge
tracing/regularization/bridging -- and writes docs/pipeline_report.md with one
figure per stage, ending with the road-to-bike-lane gap. Scoped to a small region
rather than a full tile so it takes under a minute (a full-tile CNN scan takes
20+ minutes), making "regenerate after every change" practical:

    uv run python -m scripts.diagnostics.generate_pipeline_report
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from scipy.ndimage import label
from shapely.geometry import box

from pipeline.config import (
    PROJECT_ROOT,
    BIKELANE_MASK_PATHS,
    INPUT_TILES_DIR,
    OUTPUT_DIR,
    TILE_CRS,
    USE_CACHED_BIKELANE_MASK,
    USE_OSM_ROAD_FALLBACK,
)
from scripts.detection.bikelane_centerlines import (
    detect_lane_centerlines,
    lane_centerlines_from_mask,
    load_lane_mask,
)
from scripts.measurement.osm_road_surface import osm_road_surface
from scripts.detection.texture_detector import bike_lane_detector, road_detector
from scripts.measurement.measure_bikelane_gap import load_chunk, measure_gaps, prepare_shadow, render_map
from scripts.preprocessing.osm_features import fetch_osm_features
from scripts.diagnostics.texture_analysis import stack_panels, visualize_edge_trace, visualize_scan

TILE_STEM = "idop20rgbi_32_404_5757_1_nw_2025"
RAW_TILE_PATH = INPUT_TILES_DIR / f"{TILE_STEM}.jp2"
OUTPUT_TILE_PATH = OUTPUT_DIR / f"{TILE_STEM}_bikelanes.tif"
WINDOW = Window(4300, 1330, 750, 180)

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
REPORT_PATH = PROJECT_ROOT / "docs" / "pipeline_report.md"


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.55) -> np.ndarray:
    out = image.astype(np.float32)
    out[mask] = out[mask] * (1 - alpha) + np.array(color) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


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

    _save_png(raw_rgb, FIGURES_DIR / "01_raw.png")

    osm_overlay = _overlay(raw_rgb, classification == 1, (255, 0, 0))
    osm_overlay = _overlay(osm_overlay, classification == 2, (0, 128, 255))
    _save_png(osm_overlay, FIGURES_DIR / "02_osm_mask.png")

    shadow_overlay = _overlay(raw_rgb, shadow == 1, (255, 200, 0))
    _save_png(shadow_overlay, FIGURES_DIR / "03_shadow_mask.png")

    stack_panels([raw_rgb, prefiltered_rgb], FIGURES_DIR / "04_red_boost.png")

    _save_png(prefiltered_rgb, FIGURES_DIR / "05_prefiltered.png")

    coarse_detector = bike_lane_detector()
    visualize_scan(OUTPUT_TILE_PATH, WINDOW, FIGURES_DIR / "06_cnn_scan.png", detector=coarse_detector)

    visualize_edge_trace(
        OUTPUT_TILE_PATH, WINDOW, FIGURES_DIR / "07_edge_trace.png", coarse_detector=coarse_detector
    )

    road_surface_px, road_components, road_is_osm = _road_figure(FIGURES_DIR / "09_road_trace.png")

    gap_summary = _bikelane_gap(FIGURES_DIR / "10_bikelane_gap.png")

    _write_report(road_surface_px, road_components, gap_summary, road_is_osm)
    print(f"Wrote {REPORT_PATH} and {len(list(FIGURES_DIR.glob('*.png')))} figures in {FIGURES_DIR}")


def _road_figure(figure_path: Path) -> tuple[int, int, bool]:
    """Section 8's road surface, honouring `USE_OSM_ROAD_FALLBACK`.

    With the flag set, the CNN scan is skipped and the surface is the assumed
    OSM class-width buffers (the same source `detect_roads` uses); the figure
    becomes RGB above that surface. Otherwise it is the CNN road trace.
    Returns (surface px, buffer/component count, whether OSM was used).
    """
    if not USE_OSM_ROAD_FALLBACK:
        road_coarse = road_detector()
        detections = visualize_edge_trace(
            OUTPUT_TILE_PATH, WINDOW, figure_path, coarse_detector=road_coarse, surface="road"
        )
        return int(sum(d.mask.sum() for d in detections)), len(detections), False

    with rasterio.open(OUTPUT_TILE_PATH) as src:
        rgb = np.transpose(src.read([1, 2, 3], window=WINDOW), (1, 2, 0))
        transform = src.window_transform(WINDOW)
        bounds = src.window_bounds(WINDOW)
    streets = fetch_osm_features(bounds)
    streets = streets[streets.category == "street"].clip(box(*bounds))
    surface = osm_road_surface(streets, transform, rgb.shape[:2])
    stack_panels([rgb, _overlay(rgb, surface, (0, 128, 255))], figure_path)
    _, n_components = label(surface)
    return int(surface.sum()), int(n_components), True


def _bikelane_gap(figure_path: Path) -> dict:
    """Run the 1-D gap measurement on the report window and render its map.

    Reuses `scripts.measurement.measure_bikelane_gap` end to end -- the same code the
    standalone tool runs -- so this figure can never drift from the tool.
    Scoped to the report's fixed WINDOW like every other stage: roads come from
    OSM, but the bike lanes are detected from the imagery (not OSM), the same
    cycle track step 7 traces.
    """
    bands, transform, bounds, pixel_size_m = load_chunk(WINDOW)
    osm = fetch_osm_features(bounds)
    clip = box(*bounds)
    streets = osm[osm.category == "street"].clip(clip)
    cached_mask = BIKELANE_MASK_PATHS.get(TILE_STEM)
    if USE_CACHED_BIKELANE_MASK and cached_mask and cached_mask.exists():
        lanes = lane_centerlines_from_mask(cached_mask, WINDOW)
        lane_mask = load_lane_mask(cached_mask, WINDOW)
    else:
        lanes = detect_lane_centerlines(OUTPUT_TILE_PATH, WINDOW)
        lane_mask = None

    corrected, shadow, near_edge = prepare_shadow(bands, transform, bounds, pixel_size_m, streets)
    records, _sections, _skipped = measure_gaps(corrected, transform, bounds, shadow,
                                                near_edge, streets, lanes, lane_mask)
    frame = gpd.GeoDataFrame(records, crs=TILE_CRS)

    render_map(bands, transform, frame, lanes, figure_path, pixel_size_m,
               streets=streets, lane_mask=lane_mask, bare=True)

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


def _section_9(road_surface_px: int, road_components: int, road_is_osm: bool) -> str:
    """Section 8's markdown, matching whichever road source `_road_figure` used."""
    if road_is_osm:
        return f"""## 8. Road surface (OSM-width fallback)

- **Source:** `USE_OSM_ROAD_FALLBACK` is on, so the CNN road detector is skipped entirely. Each OSM
  street is buffered to half a default width for its `highway` class (`OSM_ROAD_DEFAULT_WIDTH_M`,
  `scripts/detection/osm_road_surface.py`) and rasterised as the road surface -- see "OSM-width
  fallback" under "Road detection" in `README.md`.
- **Top:** RGB
- **Bottom:** the assumed road surface (blue), a class-width band centred on each OSM centerline

![road surface](figures/09_road_trace.png)

**No width is measured here.** The surface is exactly the class-width buffer, so a width measured
against it would only echo the assumption back -- so under this flag `scripts.measurement.detect_roads` skips
width measurement entirely and writes just the surface, no width map or GeoPackage. This is the
region-of-interest-as-measurement trade the CNN path avoids on purpose; the fallback is kept only for
coverage, when a detected surface is worse than a sensible per-class guess. The per-class widths in
`OSM_ROAD_DEFAULT_WIDTH_M` are the one thing to tune for a new area.

**Surface assumed on this frame:** {road_surface_px:,} px across {road_components} street buffer(s)."""

    return f"""## 8. Road surface

- **Detector:** `road_detector()` + `RoadEdgeDetector` -- the CNN discriminant at
  `ROAD_SCORE_THRESHOLD` (0.18) and nothing else
- **Top:** RGB
- **Middle:** the raw coarse mask, before shadow is cut
- **Bottom:** the road surface -- the same mask with shadowed pixels removed. The difference is the
  scatter of blocks across the dark road in the middle panel, which the coarse detector claimed and
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
`scripts.measurement.detect_roads` -- see "Road detection" in `README.md`.

The colour test and morphology that used to sit here were removed after being measured: the colour
test discarded two thirds of its own region of interest and left 138 fragments, and every cleanup
step after it moved the boundary a width would be taken from."""


def _gap_bullets(road_is_osm: bool) -> str:
    """Section 9's intro bullets, reflecting where the road edge comes from."""
    if road_is_osm:
        return """- **Orchestrator:** `scripts.measurement.measure_bikelane_gap`, measuring in 1-D directly on the **raw** tile, at
  the imagery's own 0.2 m resolution -- see "Bike-lane gap" in `README.md`
- **Bike lanes from imagery, not OSM:** lane centrelines are detected by the colour edge tracer
  (`detection/bikelane_centerlines.py`, the same trace as step 7), so a lane OSM never mapped is still
  measured and one it misplaced is not measured in the wrong spot; only the *road* comes from OSM
- **Road edge from OSM (`USE_OSM_ROAD_FALLBACK`):** the road edge is taken from the road's
  highway-class width, at half-width along the cross-section, *not* from pixels. The lane edge and the
  separating strip between are still measured from the imagery, so the gap reads as the distance from
  the *assumed* road edge to the *measured* lane
- **Reading the figure:** orange is the assumed road surface, green the detected bike lane -- both
  flat, since they are identities, not magnitudes. The ribbon between them is the gap, and it alone
  carries the blue scale inset at bottom left; **0 m** (light blue) means the assumed road reaches the
  lane, with no strip between. Every cross-section is drawn: the road edge comes from OSM, which
  shadow cannot obscure, so nothing is withheld as unmeasurable"""

    return """- **Orchestrator:** `scripts.measurement.measure_bikelane_gap`, measuring in 1-D directly on the **raw** tile (not
  the prefiltered output above), at the imagery's own 0.2 m resolution -- see "Bike-lane gap" in
  `README.md`
- **Why not the mask:** the deliverable is a 1.5-3 m gap, and the coarse mask in step 8 is
  quantised to 4.4 m blocks. A cross-section cut from the pixels locates each edge subpixel (measured
  precision ~0.08 m on this tile) where the mask cannot resolve the feature at all
- **Sources:** bike-lane centrelines are detected from the imagery
  (`detection/bikelane_centerlines.py`, the same trace as step 7); OSM supplies only the *road*
  centerline -- where to cut and which way to face. Every edge, width and gap is read off pixels
- **Reading the figure:** orange is the road surface, green the detected bike lane -- both flat, since
  they are identities, not magnitudes. The ribbon between them is the gap, and it alone carries the
  blue scale inset at bottom left; **0 m** (light blue) is a measured result -- the lane is flush with
  or painted on the road, with no separating strip -- not a blank"""


def _write_report(road_surface_px: int, road_components: int,
                  gap: dict, road_is_osm: bool) -> None:
    gap_composition = ", ".join(f"{count} {kind}" for kind, count in
                                sorted(gap["composition"].items(), key=lambda kv: -kv[1]))
    section_9 = _section_9(road_surface_px, road_components, road_is_osm)
    gap_bullets = _gap_bullets(road_is_osm)
    content = f"""# Pipeline walkthrough: one worked example

*Auto-generated by `scripts/generate_pipeline_report.py`. Do not edit by hand; regenerate after any
pipeline change with:*

```bash
uv run python -m scripts.diagnostics.generate_pipeline_report
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

- **Top:** raw imagery
- **Bottom:** result of `scripts/redness.py`'s saturation boost on reddish (bike-lane paint) pixels
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
- **Top:** RGB
- **Middle:** continuous discriminant-score heatmap (red = bikelane-side, blue = negative-side)
- **Bottom:** thresholded detection mask at window-block resolution, not yet precise enough for width
  measurement

![cnn scan](figures/06_cnn_scan.png)

## 7. Edge tracing, shape regularization, and bridging

- **Detector:** `BikeLaneEdgeDetector` (`scripts/detection/edge_trace.py`)
- **Processing steps:** classical color thresholding within the CNN's coarse region, PCA-binned
  centerline extraction, constant-width band reconstruction, and directional bridging across gaps
  (parked cars, shadow)
- **Top:** RGB
- **Middle:** coarse CNN mask, for reference only; its shape is the scan window's footprint, not the
  lane's
- **Bottom:** final pixel-precise, regularized, bridged mask

![edge trace](figures/07_edge_trace.png)

{section_9}

## 9. road-to-bike-lane gap

{gap_bullets}

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
