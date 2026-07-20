"""Measure the gap between the carriageway and the bike lane beside it.

    uv run python -m scripts.measure_bikelane_gap                    # whole tile
    uv run python -m scripts.measure_bikelane_gap 1600 1600 1600 1600  # col row w h

For each point along a bike lane, a cross-section is cut from the road
centerline out through the lane; the road surface ends somewhere and the lane
surface begins somewhere, and the distance between is the gap. Both boundaries
are located from pixels, subpixel, on illumination-invariant features (see
detection/cross_section.py for the resolution budget). A gap of 0 is a real
result -- the lane abuts the carriageway with no strip between. What separates
them is reported alongside the width, since a 2 m grass verge and a 2 m painted
buffer are identical in metres but different as infrastructure.

OSM is a scaffold only: it says where a lane and road are -- the one thing this
imagery can't reliably tell us (the satellite mask's two densest regions here
are a shadow edge across a car park and a rooftop) -- but every distance is
read off pixels, never OSM geometry. Runs on the *raw* tiles: the prefiltered
ones are masked to an OSM buffer, putting an artificial edge exactly where a
lane's outer boundary would be.
"""

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

from scripts.config import PROJECT_ROOT, TILE_CRS
from scripts.detection.cross_section import (
    ASPHALT,
    SHADOW_UNKNOWN,
    edge_precision_m,
    features,
    measure,
)
from scripts.osm_features import fetch_osm_features
from scripts.shadows import clean_shadow_mask, correct_shadows, detect_shadow_mask

RAW_TILE = (PROJECT_ROOT / "data" / "input" / "idop_kacheln" /
            "idop20rgbi_32_404_5757_1_nw_2025.jp2")
OUTPUT_DIR = PROJECT_ROOT / "data" / "detections"

SECTION_INTERVAL_M = 2.0
TANGENT_HALF_SPAN_M = 2.5

MAX_LANE_TO_ROAD_M = 20.0

BEHIND_ROAD_M = 3.0
BEYOND_LANE_M = 4.0

SHADOW_EDGE_MARGIN_M = 3.0

SHADOW_CORRIDOR_M = 14.0


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
    if not street_geoms:
        return [], [], {"far": 0, "shadow": 0, "off": 0, "unresolved": 0}
    street_tree = STRtree(street_geoms)
    records, sections, skipped = [], [], {"far": 0, "shadow": 0, "off": 0, "unresolved": 0}

    for lane_index, lane in enumerate(lanes.itertuples()):
        for part_index, line in enumerate(_iter_lines(lane.geometry)):
            lane_id = f"{lane_index}:{part_index}"
            for offset in np.arange(0.0, line.length, SECTION_INTERVAL_M):
                lane_point = line.interpolate(offset)
                nearest_street = street_geoms[street_tree.nearest(lane_point)]
                road_point, _ = nearest_points(nearest_street, lane_point)
                separation = road_point.distance(lane_point)
                if not 0.5 < separation <= MAX_LANE_TO_ROAD_M:
                    skipped["far"] += 1
                    continue

                col, row = inverse * (road_point.x, road_point.y)
                if not (0 <= int(row) < height and 0 <= int(col) < width):
                    skipped["off"] += 1
                    continue
                if near_edge[int(row), int(col)]:
                    skipped["shadow"] += 1
                    continue

                direction = np.array([lane_point.x - road_point.x,
                                      lane_point.y - road_point.y]) / separation
                section = measure(bands, inverse, (road_point.x, road_point.y), direction,
                                  -BEHIND_ROAD_M, separation + BEYOND_LANE_M,
                                  shadow_mask=shadow)
                sections.append(section)

                carriageway = section.run_at(0.0)
                lane_run = section.run_at(separation)
                if carriageway is None or lane_run is None or carriageway.label != ASPHALT:
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
    undetermined_color = "#6f6d68"

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
    breaks = [0.0, 0.10, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00]
    binned = ListedColormap([ramp(i / (len(breaks) - 1)) for i in range(len(breaks))])
    norm = BoundaryNorm(breaks, binned.N, extend="max")

    measured, unmeasured = [], []
    for lane_id, group in frame.groupby("lane_id"):
        ordered = group.sort_values("offset_m")
        points = np.array([[p.x, p.y] for p in ordered.lane_point])
        cols, rows = inverse * (points[:, 0], points[:, 1])
        xy = np.column_stack([cols / step, rows / step])
        offsets = ordered.offset_m.to_numpy()
        gaps = ordered.gap_m.to_numpy()
        ok = ordered.reliable.to_numpy()

        for i in range(len(xy) - 1):
            if offsets[i + 1] - offsets[i] > SECTION_INTERVAL_M * 1.6:
                continue
            segment = [xy[i], xy[i + 1]]
            if ok[i] and ok[i + 1]:
                measured.append((segment, 0.5 * (gaps[i] + gaps[i + 1])))
            else:
                unmeasured.append(segment)

    casing = unmeasured + [seg for seg, _ in measured]
    if casing:
        ax.add_collection(LineCollection(casing, colors="#12100d", linewidths=5.6,
                                         zorder=3, capstyle="round", alpha=0.55))
    if unmeasured:
        ax.add_collection(LineCollection(unmeasured, colors=undetermined_color,
                                         linewidths=3.0, zorder=4, capstyle="round",
                                         linestyles=(0, (1.6, 1.6))))
    if measured:
        segments, values = zip(*measured)
        collection = LineCollection(list(segments), cmap=binned, norm=norm,
                                    linewidths=4.2, zorder=5, capstyle="round")
        collection.set_array(np.array(values))
        ax.add_collection(collection)
        bar = fig.colorbar(collection, ax=ax, fraction=0.032, pad=0.02,
                           spacing="uniform", ticks=breaks)
        bar.ax.set_yticklabels([f"{b:g}" for b in breaks])
        bar.set_label("gap between carriageway and bike lane (m)", color=muted, fontsize=10)
        bar.ax.tick_params(colors=muted, labelsize=9)
        bar.outline.set_edgecolor("#d8d7d2")

    legend = ax.legend(handles=[
        Line2D([], [], color=ramp(0.0), lw=4.0,
               label="0 m — bike lane flush with the carriageway"),
        Line2D([], [], color=undetermined_color, lw=3.0, ls=(0, (1.6, 1.6)),
               label=f"not measurable — in shadow ({len(unmeasured)} segments)"),
    ], frameon=True, fontsize=9.5, labelcolor=ink, loc="lower left",
        borderpad=0.7, handlelength=2.6)
    legend.get_frame().set_facecolor(surface)
    legend.get_frame().set_edgecolor("#d8d7d2")
    legend.set_zorder(6)

    span_m = bands.shape[2] * pixel_size_m
    ax.set_title(f"Carriageway-to-bike-lane gap, measured from pixels\n"
                 f"{len(measured)} lane segments measured · {len(unmeasured)} in shadow · "
                 f"{span_m:.0f} m across",
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
    lanes = osm[osm.category == "bikelane"].clip(clip)
    print(f"scaffold: {len(streets)} street way(s), {len(lanes)} bike lane way(s)")

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
    unreliable = frame[~frame.reliable]
    print(f"\nnot measurable (in shadow): {len(unreliable)} of {len(frame)} "
          f"({len(unreliable) / len(frame):.0%})")

    reliable = frame[frame.reliable]
    print(f"\n{len(reliable)} of {len(frame)} gaps measured "
          f"({len(reliable) / len(frame):.0%})")
    if len(reliable):
        gaps = reliable.gap_m.to_numpy()
        print(f"  gap: median {np.median(gaps):.2f} m, "
              f"10-90pct {np.percentile(gaps, 10):.2f}-{np.percentile(gaps, 90):.2f} m")
        print(f"  no separating strip (contiguous or abutting): {(gaps == 0).mean():.0%}")
        print("\n  what separates them:")
        for kind, count in reliable.composition.value_counts().items():
            subset = reliable[reliable.composition == kind].gap_m
            print(f"    {kind:18s} {count:5d}  median {subset.median():.2f} m")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "bikelane_gap.gpkg"
    frame.drop(columns=["lane_point"]).to_file(out, driver="GPKG")
    print(f"\nWrote {len(frame)} measurements to {out}")

    map_path = render_map(bands, transform, frame, lanes,
                          OUTPUT_DIR / "bikelane_gap_map.png", pixel_size_m)
    print(f"Wrote {map_path}")
    print(f"took {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
