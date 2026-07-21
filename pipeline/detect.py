"""The detection pipeline, end to end. Its final product is the bike-lane gap.

    uv run python -m pipeline.detect                       # every PIPELINE_TILE_STEMS tile
    uv run python -m pipeline.detect 1600 1600 1600 1600   # one window: col row w h

This replaces the YOLO-seg pipeline that used to live in `detect.py` at the
repository root. That one chipped each tile, ran a fine-tuned YOLO-seg model
and wrote polygons plus per-lane widths; it was retired because its recall on
this project's sparse annotations is too low to find the real tracks (it
misses the validated cycle track outright), and because lane *width* read off
a detected mask's own shape was never a number worth reporting. Lanes now come
from the cached edge-mask detection instead, and the question this answers is
the project's actual question:

    how far is each bike lane from the road beside it, in metres?

Two stages, and the split between their data sources is deliberate:

1. **Bike lanes, from the imagery.** `detection/bikelane_centerlines.py`
   traces lanes on the *prefiltered* tile, where `redness.py` has boosted the
   paint. Lane geometry must never come from OSM: a lane OSM has not mapped,
   or has placed wrongly, would otherwise be invisible or measured against
   the wrong line. This is the expensive stage -- a coarse CNN scan over a
   whole tile is ~20 min; pass a window to iterate in seconds.

2. **The gap, from raw pixels.** `measure_bikelane_gap.py` cuts a
   cross-section from each OSM road centerline out through the detected lane
   and reads both edges subpixel off the *raw* tile. OSM supplies only where
   the road is -- with `USE_OSM_ROAD_FALLBACK` it also supplies the road's
   assumed width, which is an assumption rather than a measurement (see
   detection/osm_road_surface.py).

Everything tunable lives in `config.py` under "Bike-lane gap".
"""

import sys
import time

import geopandas as gpd
from rasterio.windows import Window

from pipeline.config import (
    BIKELANE_MASK_PATHS,
    DETECTION_OUTPUT_PATH,
    GAP_MAP_PATH,
    GAP_OUTPUT_PATH,
    INPUT_TILES_DIR,
    OUTPUT_DIR,
    PIPELINE_TILE_STEMS,
    TILE_CRS,
    USE_CACHED_BIKELANE_MASK,
    USE_OSM_ROAD_FALLBACK,
)
from scripts.detection.bikelane_centerlines import (
    detect_lane_centerlines,
    lane_centerlines_from_mask,
    load_lane_mask,
)
from scripts.measurement.measure_bikelane_gap import (
    load_chunk,
    measure_gaps,
    prepare_shadow,
    render_map,
)
from scripts.preprocessing.osm_features import fetch_osm_features
from shapely.geometry import box


def _scan_progress(done: int, total: int, started: float) -> None:
    if done % 2560 and done != total:
        return
    elapsed = time.time() - started
    remaining = (total - done) / max(done / max(elapsed, 1e-6), 1e-6)
    print(f"    scanned {done:,}/{total:,} windows ({done / total:.0%}) "
          f"-- {remaining / 60:.1f} min left", flush=True)


def run_tile(stem: str, window: Window | None):
    """Detect lanes in one tile, then measure each one's gap to the road."""
    raw_tile = INPUT_TILES_DIR / f"{stem}.jp2"
    prefiltered_tile = OUTPUT_DIR / f"{stem}_bikelanes.tif"

    bands, transform, bounds, pixel_size_m = load_chunk_for(raw_tile, window)
    print(f"{stem}: {bands.shape[2]}x{bands.shape[1]} px at {pixel_size_m} m/px")

    # 1. bike lanes, from the imagery
    started = time.time()
    cached_mask = BIKELANE_MASK_PATHS.get(stem)
    if USE_CACHED_BIKELANE_MASK and cached_mask and cached_mask.exists():
        print(f"  [1/2] reading cached lane detection {cached_mask.name}", flush=True)
        lanes = lane_centerlines_from_mask(cached_mask, window)
        lane_mask = load_lane_mask(cached_mask, window)
        print(f"        {len(lanes)} lane(s) in {time.time() - started:.1f} s")
    else:
        lane_mask = None
        print("  [1/2] tracing bike lanes from the prefiltered imagery "
              "(no cached mask; this runs a coarse CNN scan)...", flush=True)
        lanes = detect_lane_centerlines(
            prefiltered_tile, window,
            progress=lambda d, t: _scan_progress(d, t, started),
        )
        print(f"        {len(lanes)} lane(s) traced in {(time.time() - started) / 60:.1f} min")
    if lanes.empty:
        print("        no lanes detected; nothing to measure")
        return None, lanes

    # 2. the gap, from raw pixels, against OSM road centerlines
    print("  [2/2] measuring the gap to the road...", flush=True)
    osm = fetch_osm_features(bounds)
    streets = osm[osm.category == "street"].clip(box(*bounds))
    print(f"        {len(streets)} OSM street way(s)"
          + ("; road edge = OSM class width (USE_OSM_ROAD_FALLBACK)"
             if USE_OSM_ROAD_FALLBACK else "; road edge measured from pixels"))

    corrected, shadow, near_edge = prepare_shadow(bands, transform, bounds,
                                                  pixel_size_m, streets)
    records, sections, skipped = measure_gaps(corrected, transform, bounds, shadow,
                                              near_edge, streets, lanes, lane_mask)
    print(f"        {len(records)} cross-sections measured "
          f"({skipped['far']} lane too far from a road, "
          f"{skipped['shadow']} at a shadow edge, {skipped['unresolved']} unresolved)")
    if not records:
        return None, lanes

    frame = gpd.GeoDataFrame(records, crs=TILE_CRS)
    frame["tile"] = stem
    return (frame, bands, transform, pixel_size_m), lanes


def load_chunk_for(raw_tile, window):
    """`load_chunk`, but for an arbitrary tile rather than the configured one."""
    import rasterio
    with rasterio.open(raw_tile) as src:
        bands = src.read([1, 2, 3, 4], window=window)
        transform = src.window_transform(window) if window else src.transform
        bounds = src.window_bounds(window) if window else src.bounds
        pixel_size_m = src.res[0]
    return bands, transform, bounds, pixel_size_m


def main() -> None:
    args = sys.argv[1:]
    window = Window(*(int(v) for v in args[:4])) if len(args) >= 4 else None
    suffix = ""
    if window is not None:
        suffix = (f"_c{window.col_off:.0f}r{window.row_off:.0f}"
                  f"w{window.width:.0f}h{window.height:.0f}")

    started = time.time()
    frames, all_lanes, last = [], [], None
    for stem in PIPELINE_TILE_STEMS:
        result, lanes = run_tile(stem, window)
        all_lanes.append(lanes)
        if result is not None:
            frame, bands, transform, pixel_size_m = result
            frames.append(frame)
            last = (bands, transform, frame, lanes, pixel_size_m)

    if not frames:
        print("\nno gaps measured")
        return

    gaps = gpd.GeoDataFrame(gpd.pd.concat(frames, ignore_index=True), crs=TILE_CRS)

    # The detected lanes are a product in their own right, and give
    # DETECTION_OUTPUT_PATH the meaning it was declared with.
    lane_frame = gpd.GeoDataFrame(gpd.pd.concat(all_lanes, ignore_index=True), crs=TILE_CRS)
    if not lane_frame.empty:
        lane_path = DETECTION_OUTPUT_PATH.with_name(
            f"{DETECTION_OUTPUT_PATH.stem}{suffix}.gpkg")
        lane_path.parent.mkdir(parents=True, exist_ok=True)
        lane_frame.to_file(lane_path, driver="GPKG")
        print(f"\nWrote {len(lane_frame)} detected lane(s) to {lane_path}")

    gap_path = GAP_OUTPUT_PATH.with_name(f"{GAP_OUTPUT_PATH.stem}{suffix}.gpkg")
    gaps.drop(columns=["lane_point"]).to_file(gap_path, driver="GPKG")
    print(f"Wrote {len(gaps)} gap measurements to {gap_path}")

    bands, transform, frame, lanes, pixel_size_m = last
    map_path = render_map(bands, transform, frame, lanes,
                          GAP_MAP_PATH.with_name(f"{GAP_MAP_PATH.stem}{suffix}.png"),
                          pixel_size_m)
    print(f"Wrote {map_path}")

    # Reported over every cross-section, matching the map. The `reliable`
    # and `shadow_fraction` columns are still written to the GeoPackage for
    # anyone who wants to filter on them, but a shadowed stretch is no longer
    # withheld: with USE_OSM_ROAD_FALLBACK the road edge comes from the OSM
    # class width, which shadow cannot obscure.
    if len(gaps):
        values = gaps.gap_m.to_numpy()
        print("\nFINAL RESULT -- road-to-bike-lane gap")
        print(f"  {len(gaps)} cross-sections along {gaps.lane_id.nunique()} lane(s)")
        print(f"  median {gpd.pd.Series(values).median():.2f} m; "
              f"{(values == 0).mean():.0%} with no separating strip")
        for kind, count in gaps.composition.value_counts().items():
            median = gaps[gaps.composition == kind].gap_m.median()
            print(f"    {kind:18s} {count:5d}  median {median:.2f} m")
    print(f"\ntook {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
