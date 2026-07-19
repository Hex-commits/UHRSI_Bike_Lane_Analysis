"""Run the road detector over a whole tile (or one window of it) and write results.

    uv run python -m scripts.detect_roads data/output/foo.tif             # whole tile
    uv run python -m scripts.detect_roads data/output/foo.tif 22          # whole tile, coarser scan
    uv run python -m scripts.detect_roads data/output/foo.tif 11 0 0 1500 1500   # one window

Writes a GeoPackage of per-OSM-way road widths, a width-colored map of them
over the imagery, and the traced surface as both an overlay and a mask.

Width is measured along OSM centerlines rather than from the traced mask's
own shape -- see detection/centerline_width.py for why nothing derived from
the mask's shape survives a junction.

Why this doesn't reuse detect.py's chip loop: `iter_chips` cuts the tile
into 640 px squares and runs the detector on each independently, which is
fine for a model that decides per chip, but wrong here. A 640 px chip is
128 m, so chip boundaries would cut both the traced surface and the OSM ways
crossing them, and a way clipped into pieces would be measured several times
over from too few samples each. So the coarse CNN scan, the trace and the
measurement all run against the full extent in one pass; only the sliding
scan window is chunked, by batch, inside the detector.

The cost of that is time, not memory: the scan is the expensive stage by far
(a 5000x5000 tile took 23 min at the default stride, scaling with
1/stride^2), while a full tile's worth of masks is a few hundred MB. The
surface mask is cached alongside the results so that re-rendering an overlay
or re-running the measurement doesn't mean paying for the scan again.
"""

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib import colormaps
from PIL import Image, ImageDraw
from rasterio.windows import Window
from shapely.geometry import box

from scripts.config import TEXTURE_STRIDE_PX, TILE_CRS
from scripts.detection.centerline_width import _iter_lines, aggregate, measure_along_centerline
from scripts.detection.edge_trace import RoadEdgeDetector
from scripts.detection.texture_detector import road_detector
from scripts.osm_features import fetch_osm_features
from scripts.texture_analysis import SEGMENT_COLORS

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "detections"

# Long side of the overlay PNG. A 5000x5000 overlay at full resolution is a
# ~75 MB image that no viewer opens comfortably, and roads are wide enough
# to stay legible when downsampled this far.
PREVIEW_MAX_PX = 2500

# A way needs at least this many measurable cross-sections to be reported.
# At SAMPLE_INTERVAL_M that's 15 m of road actually detected under it --
# below that the median is being taken over too few samples to mean much,
# and it's usually a way that only clips the corner of the extent.
MIN_SAMPLES_PER_WAY = 3

# Width-to-color range for the map rendered by `render_width_map`. Spans
# roughly a narrow residential street to a wide multi-lane corridor, so
# ordinary streets land mid-scale and outliers read as outliers.
WIDTH_COLOR_MIN_M = 4.0
WIDTH_COLOR_MAX_M = 20.0

CENTERLINE_WIDTH_PX = 5


def _progress(done: int, total: int, started: float) -> None:
    if done % (64 * 40) and done != total:
        return
    elapsed = time.time() - started
    rate = done / max(elapsed, 1e-6)
    remaining = (total - done) / max(rate, 1e-6)
    print(f"  scanned {done:,}/{total:,} windows ({done / total:.0%}) -- {remaining / 60:.1f} min left", flush=True)


def detect_roads(tile_path: Path, window: Window | None = None, stride_px: int = TEXTURE_STRIDE_PX):
    """Measure road width along every OSM street way crossing `tile_path` (or `window`).

    Returns (records, overlay_image, surface_mask). `records` are dicts
    ready for a GeoDataFrame, one per way with a measurable width.
    """
    with rasterio.open(tile_path) as src:
        image = np.transpose(src.read([1, 2, 3], window=window), (1, 2, 0))
        transform = src.window_transform(window) if window is not None else src.transform
        bounds = src.window_bounds(window) if window is not None else src.bounds
        pixel_size_m = src.res[0]

    print(f"{tile_path.name}: {image.shape[1]}x{image.shape[0]} px at {pixel_size_m} m/px, stride {stride_px}")

    coarse_detector = road_detector(stride_px=stride_px)
    started = time.time()
    coarse = coarse_detector.predict(image, progress=lambda d, t: _progress(d, t, started))
    if not coarse:
        print("  coarse scan found nothing")
        return [], image, np.zeros(image.shape[:2], dtype=bool)
    print(f"  coarse scan took {(time.time() - started) / 60:.1f} min; "
          f"mask covers {coarse[0].mask.mean():.0%} of frame")

    surface = RoadEdgeDetector(coarse_detector=coarse_detector).surface_mask(image, coarse=coarse)
    print(f"  traced road surface: {surface.sum():,} px ({surface.mean():.0%} of frame)")

    # Width comes from OSM centerlines rather than the mask's own shape --
    # see detection/centerline_width.py for why that's necessary at tile
    # scale. Only streets: bike lanes are a separate surface with their own
    # detector, and measuring them against the asphalt mask would report the
    # road they run alongside.
    osm = fetch_osm_features(bounds)
    streets = osm[osm.category == "street"].clip(box(*bounds))
    print(f"  {len(streets)} OSM street way(s) crossing this extent")

    records = []
    for way in streets.itertuples():
        samples = []
        for line in _iter_lines(way.geometry):
            samples.extend(measure_along_centerline(line, surface, transform, pixel_size_m))
        width = aggregate(samples)
        if width is None or width.n_samples < MIN_SAMPLES_PER_WAY:
            continue
        records.append(
            {
                "geometry": way.geometry,
                "tile": tile_path.name,
                "width_median_m": width.median_m,
                "width_mean_m": width.mean_m,
                "width_min_m": width.min_m,
                "width_max_m": width.max_m,
                "n_samples": width.n_samples,
                "buffer_limited_fraction": width.buffer_limited_fraction,
                "unbounded_fraction": width.unbounded_fraction,
            }
        )
    records.sort(key=lambda r: -r["n_samples"])
    print(f"  {len(records)} way(s) with at least {MIN_SAMPLES_PER_WAY} measurable cross-sections")

    overlay = image.astype(np.float32)
    color = np.array(SEGMENT_COLORS[0], dtype=np.float32)
    overlay[surface] = overlay[surface] * 0.45 + color * 0.55
    return records, np.clip(overlay, 0, 255).astype(np.uint8), surface


def render_width_map(
    tile_path: Path,
    records: list[dict],
    out_path: Path,
    window: Window | None = None,
) -> Path:
    """Draw each measured way over the imagery, colored by its median width.

    Deliberately drawn over plain imagery, with the traced surface *not*
    tinted underneath. Tinting it was tried and reverted: the tint reads as
    another color in the same blue-cyan range the width scale uses at its
    low end, so the network washes out to one hue and the widths stop being
    readable -- which is the only thing this figure exists to show. Coverage
    is what `<tile>_roads_surface.png` is for; keeping the two figures to
    one question each keeps both legible.
    """
    with rasterio.open(tile_path) as src:
        image = np.transpose(src.read([1, 2, 3], window=window), (1, 2, 0))
        transform = src.window_transform(window) if window is not None else src.transform
    inverse_transform = ~transform

    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    colormap = colormaps["turbo"]

    for record in records:
        normalized = float(np.clip((record["width_median_m"] - WIDTH_COLOR_MIN_M) /
                                   (WIDTH_COLOR_MAX_M - WIDTH_COLOR_MIN_M), 0, 1))
        color = tuple(int(channel * 255) for channel in colormap(normalized)[:3])
        for line in _iter_lines(record["geometry"]):
            points = [inverse_transform * (x, y) for x, y in line.coords]
            draw.line(points, fill=color, width=CENTERLINE_WIDTH_PX)

    scale = min(1.0, PREVIEW_MAX_PX / max(canvas.size))
    if scale < 1.0:
        canvas = canvas.resize((round(canvas.width * scale), round(canvas.height * scale)), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    tile_path = Path(args[0])
    stride_px = int(args[1]) if len(args) > 1 else TEXTURE_STRIDE_PX
    window = Window(*(int(v) for v in args[2:6])) if len(args) >= 6 else None

    records, overlay, surface = detect_roads(tile_path, window, stride_px)

    # Windowed runs get the window in their filenames. Without this a quick
    # test over a corner of a tile silently overwrites the outputs of a
    # full-tile run of the same tile -- which costs half an hour to redo.
    stem = tile_path.stem
    if window is not None:
        stem += f"_x{window.col_off:.0f}y{window.row_off:.0f}w{window.width:.0f}h{window.height:.0f}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = OUTPUT_DIR / f"{stem}_roads_surface.png"
    preview = Image.fromarray(overlay)
    scale = min(1.0, PREVIEW_MAX_PX / max(preview.size))
    if scale < 1.0:
        preview = preview.resize((round(preview.width * scale), round(preview.height * scale)), Image.LANCZOS)
    preview.save(preview_path)
    print(f"Wrote {preview_path}")

    # The scan is the expensive part by far, so keep its result -- a
    # different overlay or a re-run of the width measurement shouldn't mean
    # paying for it again.
    surface_path = OUTPUT_DIR / f"{stem}_roads_surface.npz"
    np.savez_compressed(surface_path, surface=surface)
    print(f"Wrote {surface_path}")

    if not records:
        print("No road segments found.")
        return

    width_map_path = render_width_map(
        tile_path, records, OUTPUT_DIR / f"{stem}_roads_width.png", window
    )
    print(f"Wrote {width_map_path}")

    gpkg_path = OUTPUT_DIR / f"{stem}_roads.gpkg"
    gpd.GeoDataFrame(records, crs=TILE_CRS).to_file(gpkg_path, driver="GPKG")
    print(f"Wrote {len(records)} segments to {gpkg_path}")

    widths = np.array([r["width_median_m"] for r in records])
    clipped = np.mean([r["buffer_limited_fraction"] for r in records])
    print(f"\nWidth across {len(records)} ways: median {np.median(widths):.2f} m, "
          f"10th-90th percentile {np.percentile(widths, 10):.2f}-{np.percentile(widths, 90):.2f} m")
    print(f"Samples stopped by the prefilter's buffer edge rather than a real surface edge: {clipped:.1%}")
    print(f"\n{'way':>4}  {'median m':>9}  {'mean m':>8}  {'min m':>7}  {'max m':>7}  "
          f"{'samples':>8}  {'clipped':>8}  {'unbnd':>6}")
    for i, record in enumerate(records[:20]):
        print(f"{i:4d}  {record['width_median_m']:9.2f}  {record['width_mean_m']:8.2f}  "
              f"{record['width_min_m']:7.2f}  {record['width_max_m']:7.2f}  "
              f"{record['n_samples']:8d}  {record['buffer_limited_fraction']:7.0%}  "
              f"{record['unbounded_fraction']:5.0%}")


if __name__ == "__main__":
    main()
