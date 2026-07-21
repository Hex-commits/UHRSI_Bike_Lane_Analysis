import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import nearest_points

from pipeline.config import (
    GAP_BEHIND_ROAD_M,
    GAP_BEYOND_LANE_M,
    GAP_MAP_BREAKS_M,
    GAP_MAP_PATH,
    GAP_MAX_LANE_TO_ROAD_M,
    GAP_OUTPUT_PATH,
    GAP_SECTION_INTERVAL_M,
    GAP_SHADOW_CORRIDOR_M,
    GAP_SHADOW_EDGE_MARGIN_M,
    GAP_TANGENT_HALF_SPAN_M,
    INPUT_TILES_DIR,
    OUTPUT_DIR as PREFILTERED_DIR,
    PIPELINE_TILE_STEMS,
    PROJECT_ROOT,
    TILE_CRS,
    USE_OSM_ROAD_FALLBACK,
)
from scripts.detection.bikelane_centerlines import detect_lane_centerlines
from scripts.measurement.cross_section import (
    ASPHALT,
    SHADOW_UNKNOWN,
    edge_precision_m,
    features,
    measure,
)
from scripts.measurement.osm_road_surface import road_width_m
from scripts.preprocessing.osm_features import fetch_osm_features
from scripts.preprocessing.shadows import clean_shadow_mask, correct_shadows, detect_shadow_mask

_STEM = PIPELINE_TILE_STEMS[0]
RAW_TILE = INPUT_TILES_DIR / f"{_STEM}.jp2"
PREFILTERED_TILE = PREFILTERED_DIR / f"{_STEM}_bikelanes.tif"
OUTPUT_DIR = GAP_OUTPUT_PATH.parent

# Every tunable lives in config.py; these are module-local aliases so the
# code below reads in metres rather than in config lookups.
SECTION_INTERVAL_M = GAP_SECTION_INTERVAL_M
TANGENT_HALF_SPAN_M = GAP_TANGENT_HALF_SPAN_M
MAX_LANE_TO_ROAD_M = GAP_MAX_LANE_TO_ROAD_M
BEHIND_ROAD_M = GAP_BEHIND_ROAD_M
BEYOND_LANE_M = GAP_BEYOND_LANE_M
SHADOW_EDGE_MARGIN_M = GAP_SHADOW_EDGE_MARGIN_M
SHADOW_CORRIDOR_M = GAP_SHADOW_CORRIDOR_M


def _iter_lines(geometry):
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def load_chunk(window):
    with rasterio.open(RAW_TILE) as src:
        bands = src.read([1, 2, 3, 4], window=window)
        transform = src.window_transform(window) if window else src.transform
        bounds = src.window_bounds(window) if window else src.bounds
        pixel_size_m = src.res[0]
    return bands, transform, bounds, pixel_size_m


def prepare_shadow(bands, transform, bounds, pixel_size_m, streets):
    """Detect shadow on the road corridor, correct it, and mask its edges."""
    shape = bands.shape[1:]
    if streets.empty:
        ground = features(bands)["ndvi"] < 0.15
    else:
        corridor = streets.union_all().buffer(SHADOW_CORRIDOR_M)
        burned = rasterize([(corridor, 1)], out_shape=shape, transform=transform,
                           dtype=np.uint8).astype(bool)
        ground = burned & (features(bands)["ndvi"] < 0.15)

    shadow = clean_shadow_mask(detect_shadow_mask(bands[:3], ground), pixel_size_m)
    corrected = correct_shadows(bands, shadow, ground, pixel_size_m)
    if not shadow.any():
        return corrected, shadow, np.zeros(shape, dtype=bool)

    distance_to_boundary = np.where(shadow,
                                    distance_transform_edt(shadow),
                                    distance_transform_edt(~shadow))
    return corrected, shadow, distance_to_boundary <= SHADOW_EDGE_MARGIN_M / pixel_size_m


def measure_gaps(bands, transform, bounds, shadow, near_edge, streets, lanes):
    """One record per cross-section along every bike lane."""
    inverse = ~transform
    height, width = shadow.shape
    street_geoms = list(streets.geometry)
    street_highways = list(streets["highway"]) if "highway" in streets.columns else [None] * len(street_geoms)
    if not street_geoms:
        return [], [], {"far": 0, "shadow": 0, "off": 0, "unresolved": 0}
    street_tree = STRtree(street_geoms)
    records, sections, skipped = [], [], {"far": 0, "shadow": 0, "off": 0, "unresolved": 0}

    for lane_index, lane in enumerate(lanes.itertuples()):
        for part_index, line in enumerate(_iter_lines(lane.geometry)):
            lane_id = f"{lane_index}:{part_index}"
            for offset in np.arange(0.0, line.length, SECTION_INTERVAL_M):
                lane_point = line.interpolate(offset)
                nearest_idx = int(street_tree.nearest(lane_point))
                nearest_street = street_geoms[nearest_idx]
                road_point, _ = nearest_points(nearest_street, lane_point)
                separation = road_point.distance(lane_point)
                if not 0.5 < separation <= MAX_LANE_TO_ROAD_M:
                    skipped["far"] += 1
                    continue

                col, row = inverse * (road_point.x, road_point.y)
                if not (0 <= int(row) < height and 0 <= int(col) < width):
                    skipped["off"] += 1
                    continue
                if USE_OSM_ROAD_FALLBACK:
                    lcol, lrow = inverse * (lane_point.x, lane_point.y)
                    near_shadow = (0 <= int(lrow) < height and 0 <= int(lcol) < width
                                   and near_edge[int(lrow), int(lcol)])
                else:
                    near_shadow = bool(near_edge[int(row), int(col)])
                if near_shadow:
                    skipped["shadow"] += 1
                    continue

                direction = np.array([lane_point.x - road_point.x,
                                      lane_point.y - road_point.y]) / separation
                section = measure(bands, inverse, (road_point.x, road_point.y), direction,
                                  -BEHIND_ROAD_M, separation + BEYOND_LANE_M,
                                  shadow_mask=shadow)
                sections.append(section)

                lane_run = section.run_at(separation)
                if lane_run is None:
                    skipped["unresolved"] += 1
                    continue

                if USE_OSM_ROAD_FALLBACK:
                    road_edge_m = road_width_m(street_highways[nearest_idx]) / 2.0
                    lane_edge_m = lane_run.start_m
                    gap_m = max(lane_edge_m - road_edge_m, 0.0)
                    between = [r for r in section.runs
                               if r.end_m > road_edge_m and r.start_m < lane_edge_m and r is not lane_run]
                    composition = ("contiguous" if gap_m <= 0.0 else
                                   "abutting" if not between else
                                   SHADOW_UNKNOWN if any(r.label == SHADOW_UNKNOWN for r in between)
                                   else max(between, key=lambda r: r.width_m).label)
                    lane_side_shadow = section.shadow_fraction_between(min(road_edge_m, separation), separation)
                    records.append({
                        "geometry": LineString([road_point, lane_point]),
                        "lane_point": Point(lane_point.x, lane_point.y),
                        "gap_m": float(gap_m),
                        "composition": composition,
                        "road_edge_m": float(road_edge_m),
                        "lane_edge_m": float(lane_edge_m),
                        "osm_separation_m": float(separation),
                        "shadow_fraction": float(lane_side_shadow),
                        "reliable": composition != SHADOW_UNKNOWN and lane_side_shadow < 0.2,
                        "lane_id": lane_id,
                        "offset_m": float(offset),
                    })
                    continue

                carriageway = section.run_at(0.0)
                if carriageway is None or carriageway.label != ASPHALT:
                    skipped["unresolved"] += 1
                    continue

                if carriageway is lane_run:
                    records.append({
                        "geometry": LineString([road_point, lane_point]),
                        "lane_point": Point(lane_point.x, lane_point.y),
                        "gap_m": 0.0,
                        "composition": "contiguous",
                        "road_edge_m": np.nan,
                        "lane_edge_m": np.nan,
                        "osm_separation_m": float(separation),
                        "shadow_fraction": float(section.shadow_fraction),
                        "reliable": True,
                        "lane_id": lane_id,
                        "offset_m": float(offset),
                    })
                    continue

                gap_m = max(lane_run.start_m - carriageway.end_m, 0.0)
                between = [r for r in section.runs
                           if r.start_m >= carriageway.end_m and r.end_m <= lane_run.start_m]
                composition = ("abutting" if not between else
                               SHADOW_UNKNOWN if any(r.label == SHADOW_UNKNOWN for r in between)
                               else max(between, key=lambda r: r.width_m).label)

                records.append({
                    "geometry": LineString([road_point, lane_point]),
                    "lane_point": Point(lane_point.x, lane_point.y),
                    "gap_m": float(gap_m),
                    "composition": composition,
                    "road_edge_m": float(carriageway.end_m),
                    "lane_edge_m": float(lane_run.start_m),
                    "osm_separation_m": float(separation),
                    "shadow_fraction": float(section.shadow_fraction),
                    "reliable": composition != SHADOW_UNKNOWN and section.shadow_fraction < 0.2,
                    "lane_id": lane_id,
                    "offset_m": float(offset),
                })
    return records, sections, skipped


def render_map(bands, transform, frame, lanes, out_path, pixel_size_m, figsize=(13, 13)):
    """Draw every measured lane point over the imagery, coloured by gap.

    Gap width is a magnitude, so a single-hue sequential ramp. "Undetermined"
    is the absence of a measurement, not a small gap, so it takes a neutral
    off-ramp colour drawn underneath, never reading as "0 m".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"
    ramp = LinearSegmentedColormap.from_list("gap", ["#9ec5ef", "#2a78d6", "#0d2f56"])

    inverse = ~transform
    preview_scale = min(1.0, 2200 / max(bands.shape[1], bands.shape[2]))
    rgb = np.clip(np.transpose(bands[:3], (1, 2, 0)) / 255.0, 0, 1)
    if preview_scale < 1.0:
        step = int(round(1 / preview_scale))
        rgb = rgb[::step, ::step]
    else:
        step = 1

    grey = rgb.mean(axis=2, keepdims=True)
    rgb = np.clip((rgb * 0.35 + grey * 0.65) * 0.62 + 0.10, 0, 1)

    fig, ax = plt.subplots(figsize=figsize, facecolor=surface)
    ax.imshow(rgb)

    reliable = frame[frame.reliable]
    breaks = list(GAP_MAP_BREAKS_M)
    binned = ListedColormap([ramp(i / (len(breaks) - 1)) for i in range(len(breaks))])
    norm = BoundaryNorm(breaks, binned.N, extend="max")

    # Every cross-section is drawn with its measured gap. There is no
    # "unmeasurable" class on this map: with USE_OSM_ROAD_FALLBACK the road
    # edge is the OSM class-width buffer, which shadow cannot obscure, so a
    # shadowed stretch still yields a gap. The `reliable` and
    # `shadow_fraction` columns survive in the GeoPackage for anyone who
    # wants to filter on them -- the information is kept, just not used to
    # blank out the map.
    measured = []
    for lane_id, group in frame.groupby("lane_id"):
        ordered = group.sort_values("offset_m")
        points = np.array([[p.x, p.y] for p in ordered.lane_point])
        cols, rows = inverse * (points[:, 0], points[:, 1])
        xy = np.column_stack([cols / step, rows / step])
        offsets = ordered.offset_m.to_numpy()
        gaps = ordered.gap_m.to_numpy()

        for i in range(len(xy) - 1):
            # Don't bridge a break in sampling with one long segment.
            if offsets[i + 1] - offsets[i] > SECTION_INTERVAL_M * 1.6:
                continue
            if not (np.isfinite(gaps[i]) and np.isfinite(gaps[i + 1])):
                continue
            measured.append(([xy[i], xy[i + 1]], 0.5 * (gaps[i] + gaps[i + 1])))

    if measured:
        ax.add_collection(LineCollection([seg for seg, _ in measured], colors="#12100d",
                                         linewidths=5.6, zorder=3, capstyle="round", alpha=0.55))
    if measured:
        segments, values = zip(*measured)
        collection = LineCollection(list(segments), cmap=binned, norm=norm,
                                    linewidths=4.2, zorder=5, capstyle="round")
        collection.set_array(np.array(values))
        ax.add_collection(collection)
        bar = fig.colorbar(collection, ax=ax, fraction=0.032, pad=0.02,
                           spacing="uniform", ticks=breaks)
        bar.ax.set_yticklabels([f"{b:g}" for b in breaks])
        bar.set_label("gap between road and bike lane (m)", color=muted, fontsize=10)
        bar.ax.tick_params(colors=muted, labelsize=9)
        bar.outline.set_edgecolor("#d8d7d2")

    legend = ax.legend(handles=[
        Line2D([], [], color=ramp(0.0), lw=4.0,
               label="0 m — bike lane flush with the road"),
    ], frameon=True, fontsize=9.5, labelcolor=ink, loc="lower left",
        borderpad=0.7, handlelength=2.6)
    legend.get_frame().set_facecolor(surface)
    legend.get_frame().set_edgecolor("#d8d7d2")
    legend.set_zorder(6)

    span_m = bands.shape[2] * pixel_size_m
    ax.set_title(f"Road-to-bike-lane gap\n"
                 f"{len(measured)} lane segments · {span_m:.0f} m across",
                 color=ink, fontsize=13, loc="left", weight="bold")
    ax.axis("off")
    fig.savefig(out_path, dpi=140, facecolor=surface, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = sys.argv[1:]
    window = Window(*(int(v) for v in args[:4])) if len(args) >= 4 else None

    started = time.time()
    bands, transform, bounds, pixel_size_m = load_chunk(window)
    print(f"{RAW_TILE.name} {'whole tile' if window is None else f'window {args[:4]}'} "
          f"-- {bands.shape[2]}x{bands.shape[1]} px at {pixel_size_m} m/px")

    osm = fetch_osm_features(bounds)
    clip = box(*bounds)
    streets = osm[osm.category == "street"].clip(clip)
    lanes = detect_lane_centerlines(PREFILTERED_TILE, window)
    print(f"roads: {len(streets)} OSM street way(s); "
          f"bike lanes: {len(lanes)} detected from imagery (not OSM)")
    if USE_OSM_ROAD_FALLBACK:
        print("road edge: OSM highway-class width (USE_OSM_ROAD_FALLBACK), not pixels")

    corrected, shadow, near_edge = prepare_shadow(bands, transform, bounds,
                                                  pixel_size_m, streets)
    print(f"shadow: {shadow.mean():.1%} of frame, "
          f"{near_edge.mean():.1%} within {SHADOW_EDGE_MARGIN_M:.0f} m of an edge")

    records, sections, skipped = measure_gaps(corrected, transform, bounds, shadow,
                                              near_edge, streets, lanes)
    print(f"\n{len(records)} gaps measured from {len(sections)} cross-sections")
    print(f"  skipped: {skipped['far']} lane too far from a road, "
          f"{skipped['shadow']} at a shadow edge, {skipped['unresolved']} unresolved, "
          f"{skipped['off']} outside frame")
    if not records:
        print("nothing measured")
        return

    precision = edge_precision_m(sections)
    if precision.get("n_pairs"):
        print(f"\nedge precision (neighbouring sections, n={precision['n_pairs']}): "
              f"median {precision['median_m']:.3f} m "
              f"({precision['gsd_ratio']:.2f}x the {pixel_size_m:.1f} m GSD)")

    frame = gpd.GeoDataFrame(records, crs=TILE_CRS)
    # Reported over every cross-section, matching the map. `reliable` and
    # `shadow_fraction` are still written to the GeoPackage to filter on, but
    # nothing is withheld for shadow: the road edge comes from the OSM class
    # width, which shadow cannot obscure.
    print(f"\n{len(frame)} gaps measured")
    gaps = frame.gap_m.to_numpy()
    print(f"  gap: median {np.median(gaps):.2f} m, "
          f"10-90pct {np.percentile(gaps, 10):.2f}-{np.percentile(gaps, 90):.2f} m")
    print(f"  no separating strip (contiguous or abutting): {(gaps == 0).mean():.0%}")
    print("\n  what separates them:")
    for kind, count in frame.composition.value_counts().items():
        subset = frame[frame.composition == kind].gap_m
        print(f"    {kind:18s} {count:5d}  median {subset.median():.2f} m")

    # A windowed run writes window-suffixed filenames. Without this, a quick
    # 140 m test overwrites the canonical whole-tile map and GeoPackage with
    # a sliver of the district, and the only sign is that the "final result"
    # silently got smaller -- which is exactly what happened once.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if window is not None:
        suffix = (f"_c{window.col_off:.0f}r{window.row_off:.0f}"
                  f"w{window.width:.0f}h{window.height:.0f}")
    out = GAP_OUTPUT_PATH.with_name(f"{GAP_OUTPUT_PATH.stem}{suffix}.gpkg")
    frame.drop(columns=["lane_point"]).to_file(out, driver="GPKG")
    print(f"\nWrote {len(frame)} measurements to {out}")

    map_path = render_map(bands, transform, frame, lanes,
                          GAP_MAP_PATH.with_name(f"{GAP_MAP_PATH.stem}{suffix}.png"),
                          pixel_size_m)
    print(f"Wrote {map_path}")
    print(f"took {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
