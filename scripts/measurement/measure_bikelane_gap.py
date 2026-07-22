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
    GAP_MAP_SHOW_ROAD,
    GAP_MAX_LANE_TO_ROAD_M,
    GAP_MAX_ROAD_ANGLE_DEG,
    GAP_EXCLUDED_ROAD_CLASSES,
    GAP_OUTPUT_PATH,
    GAP_SECTION_INTERVAL_M,
    GAP_SHADOW_CORRIDOR_M,
    GAP_SHADOW_EDGE_MARGIN_M,
    GAP_TANGENT_HALF_SPAN_M,
    BIKELANE_MASK_PATHS,
    USE_CACHED_BIKELANE_MASK,
    OUTPUT_DIR as PREFILTERED_DIR,
    PIPELINE_TILE_STEMS,
    PROJECT_ROOT,
    TILE_CRS,
    USE_OSM_ROAD_FALLBACK,
    raw_tile_path,
)
from scripts.detection.bikelane_centerlines import (
    detect_lanes,
    lane_centerlines_from_mask,
    load_lane_mask,
)
from scripts.measurement.cross_section import (
    ASPHALT,
    SHADOW_UNKNOWN,
    edge_precision_m,
    features,
    measure,
)
from scripts.measurement.osm_road_surface import osm_road_surface, road_width_m
from scripts.preprocessing.osm_features import fetch_osm_features
from scripts.preprocessing.shadows import clean_shadow_mask, correct_shadows, detect_shadow_mask

_STEM = PIPELINE_TILE_STEMS[0]
RAW_TILE = raw_tile_path(_STEM)
PREFILTERED_TILE = PREFILTERED_DIR / f"{_STEM}_bikelanes.tif"
OUTPUT_DIR = GAP_OUTPUT_PATH.parent

# The two things the maps draw flat, module-level so every figure that shows a
# detected lane uses the same colour. The diagnostics edge-trace panel imports
# LANE_COLOR from here rather than keeping its own copy, so the report's
# figures cannot drift apart from the gap map's.
ROAD_COLOR, LANE_COLOR = "#eda100", "#008300"

# Every tunable lives in config.py; these are module-local aliases so the
# code below reads in metres rather than in config lookups.
SECTION_INTERVAL_M = GAP_SECTION_INTERVAL_M
TANGENT_HALF_SPAN_M = GAP_TANGENT_HALF_SPAN_M
MAX_LANE_TO_ROAD_M = GAP_MAX_LANE_TO_ROAD_M
MAX_ROAD_ANGLE_DEG = GAP_MAX_ROAD_ANGLE_DEG
BEHIND_ROAD_M = GAP_BEHIND_ROAD_M
BEYOND_LANE_M = GAP_BEYOND_LANE_M
SHADOW_EDGE_MARGIN_M = GAP_SHADOW_EDGE_MARGIN_M
SHADOW_CORRIDOR_M = GAP_SHADOW_CORRIDOR_M


def _tangent(geometry, distance_along: float):
    """Unit direction of `geometry` at `distance_along`, or None if degenerate."""
    before = geometry.interpolate(max(0.0, distance_along - TANGENT_HALF_SPAN_M))
    after = geometry.interpolate(min(geometry.length, distance_along + TANGENT_HALF_SPAN_M))
    vector = np.array([after.x - before.x, after.y - before.y])
    norm = np.linalg.norm(vector)
    return vector / norm if norm else None


def _reference_road(tree, geoms, highways, lane_point, lane_tangent):
    """Nearest street that actually runs *alongside* the lane here.

    Not simply the nearest street. The nearest street is very often a
    driveway or parking aisle the lane crosses, and the distance to a road
    you cross is not a separation from it. Candidates must be within
    MAX_LANE_TO_ROAD_M, not an excluded class, and within
    MAX_ROAD_ANGLE_DEG of the lane's own direction; the closest survivor
    wins. Returns (index, road_point, separation) or None.
    """
    best = None
    for index in tree.query(lane_point.buffer(MAX_LANE_TO_ROAD_M)):
        index = int(index)
        highway = highways[index]
        if isinstance(highway, str) and highway in GAP_EXCLUDED_ROAD_CLASSES:
            continue
        street = geoms[index]
        road_point, _ = nearest_points(street, lane_point)
        separation = road_point.distance(lane_point)
        if not 0.5 < separation <= MAX_LANE_TO_ROAD_M:
            continue
        street_tangent = _tangent(street, street.project(lane_point))
        if lane_tangent is None or street_tangent is None:
            continue
        angle = np.degrees(np.arccos(min(1.0, abs(float(np.dot(lane_tangent, street_tangent))))))
        if angle > MAX_ROAD_ANGLE_DEG:
            continue
        if best is None or separation < best[2]:
            best = (index, road_point, separation)
    return best


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


def measure_gaps(bands, transform, bounds, shadow, near_edge, streets, lanes, lane_mask=None):
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
                reference = _reference_road(street_tree, street_geoms, street_highways,
                                            lane_point, _tangent(line, offset))
                if reference is None:
                    skipped["far"] += 1
                    continue
                nearest_idx, road_point, separation = reference

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
                                  shadow_mask=shadow, lane_mask=lane_mask)
                sections.append(section)

                lane_run = section.run_at(separation)
                if lane_run is None:
                    skipped["unresolved"] += 1
                    continue

                if USE_OSM_ROAD_FALLBACK:
                    road_edge_m = road_width_m(street_highways[nearest_idx]) / 2.0
                    # The lane's near edge comes from the detected lane mask,
                    # not from `lane_run.start_m`. Where road and lane are the
                    # same asphalt, segmentation merges them into one run whose
                    # start is the profile's start, which collapsed every
                    # separated cycle track to a spurious 0 m gap.
                    mask_edge_m = section.lane_edge_m(separation)
                    if mask_edge_m is None:
                        skipped["unresolved"] += 1
                        continue
                    lane_edge_m = mask_edge_m
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


def render_map(bands, transform, frame, lanes, out_path, pixel_size_m, figsize=(13, 13),
               streets=None, lane_mask=None, bare=False):
    """Draw the two things being compared, and colour the space between them.

    The figure has to answer "the gap between *what* and *what*", so the road
    and the bike lane are each drawn as a flat, uniform shape -- they are
    identities, not magnitudes, and colouring them by anything would compete
    with the one quantity that is a magnitude. The measured gap is the ribbon
    between them, and it alone carries the sequential ramp.

    An earlier version coloured the *lane* by its gap, which put the number on
    the wrong object: the reader saw a coloured lane and no indication of what
    it was measured against.

    `GAP_MAP_SHOW_ROAD` can drop the road half of that pairing. It is off by
    default while the road comes from an OSM class width: an assumption drawn
    as a solid band is the most confident-looking object on the map, and the
    ribbon's inner edge already shows where the road was taken to end.

    `bare` drops the title, legend and surrounding white margin, and moves the
    gap scale onto the map as an inset -- the form the pipeline report wants,
    where the surrounding prose already says what road and lane are and the
    figure sits in a stack of edge-to-edge image panels. `figsize`'s height is
    then derived from the imagery so nothing is letterboxed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch

    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"
    # The gap ramp is blue, so road and lane are deliberately non-blue: no
    # step of a sequential ramp should be confusable with a categorical fill.
    #
    # The ramp runs dark->light because it sits on *dark* imagery: the anchor
    # of a sequential scale flips with its surface, and the old ramp's deep
    # end (#0d2f56) disappeared into the asphalt it was drawn over, taking
    # the widest gaps -- the ones worth seeing -- with it. These five steps
    # ascend in lightness by ~0.095 OKLCH each, above the 0.06 floor where
    # neighbouring classes stop being tellable apart, and the darkest still
    # clears the dimmed basemap at 2.9:1.
    RAMP_STEPS = ["#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
    ramp = LinearSegmentedColormap.from_list("gap", RAMP_STEPS)

    inverse = ~transform
    preview_scale = min(1.0, 2200 / max(bands.shape[1], bands.shape[2]))
    rgb = np.clip(np.transpose(bands[:3], (1, 2, 0)) / 255.0, 0, 1)
    step = int(round(1 / preview_scale)) if preview_scale < 1.0 else 1
    if step > 1:
        rgb = rgb[::step, ::step]

    # The imagery is context, not data: desaturated and dimmed hard, so the
    # ramp has room to be bright above it. Every step of the scale has to
    # clear whatever it lands on, and asphalt is the brightest thing under
    # the ribbon.
    grey = rgb.mean(axis=2, keepdims=True)
    rgb = np.clip((rgb * 0.30 + grey * 0.70) * 0.34 + 0.04, 0, 1)

    if bare:
        figsize = (figsize[0], figsize[0] * rgb.shape[0] / rgb.shape[1])
    fig, ax = plt.subplots(figsize=figsize, facecolor=surface)
    if bare:
        ax.set_position([0, 0, 1, 1])
    ax.imshow(rgb)

    # --- the two things being compared, each one flat colour ---------------
    if GAP_MAP_SHOW_ROAD and streets is not None and not streets.empty:
        road = osm_road_surface(streets, transform, bands.shape[1:])[::step, ::step]
        ax.imshow(np.ma.masked_where(~road, road), cmap=ListedColormap([ROAD_COLOR]),
                  alpha=0.55, zorder=2, interpolation="nearest")
    if lane_mask is not None and lane_mask.any():
        lane = np.asarray(lane_mask)[::step, ::step]
        ax.imshow(np.ma.masked_where(~lane, lane), cmap=ListedColormap([LANE_COLOR]),
                  alpha=0.85, zorder=3, interpolation="nearest")

    # --- the gap itself, carrying the scale --------------------------------
    breaks = list(GAP_MAP_BREAKS_M)
    binned = ListedColormap([ramp(i / (len(breaks) - 1)) for i in range(len(breaks))])
    norm = BoundaryNorm(breaks, binned.N, extend="max")

    def edge_points(row):
        """(road-edge, lane-edge) in preview pixels, along this cross-section."""
        road_pt = np.array(row.geometry.coords[0])
        lane_pt = np.array(row.geometry.coords[-1])
        span = np.linalg.norm(lane_pt - road_pt)
        if span == 0 or not np.isfinite(row.gap_m):
            return None
        unit = (lane_pt - road_pt) / span
        out = []
        for along in (row.road_edge_m, row.lane_edge_m):
            x, y = road_pt + unit * along
            col, r = inverse * (x, y)
            out.append((col / step, r / step))
        return out

    quads, values = [], []
    for _lane_id, group in frame.groupby("lane_id"):
        ordered = group.sort_values("offset_m")
        rows = list(ordered.itertuples())
        offsets = ordered.offset_m.to_numpy()
        anchors = np.array([g.coords[0] for g in ordered.geometry])
        for i in range(len(rows) - 1):
            # Don't bridge a break in sampling with one long quad.
            if offsets[i + 1] - offsets[i] > SECTION_INTERVAL_M * 1.6:
                continue
            # Nor connect two cross-sections anchored to different roads --
            # that quad spans the space between two streets, not a gap, and
            # renders as a twist across the figure.
            if np.linalg.norm(anchors[i + 1] - anchors[i]) > SECTION_INTERVAL_M * 2.5:
                continue
            a, b = edge_points(rows[i]), edge_points(rows[i + 1])
            if a is None or b is None:
                continue
            # road-edge_i -> lane-edge_i -> lane-edge_i+1 -> road-edge_i+1
            quads.append([a[0], a[1], b[1], b[0]])
            values.append(0.5 * (rows[i].gap_m + rows[i + 1].gap_m))

    if quads:
        band = PolyCollection(quads, cmap=binned, norm=norm, zorder=4,
                              edgecolors="face", linewidths=0.4)
        band.set_array(np.array(values))
        ax.add_collection(band)
        if bare:
            # On the map, not beside it: a bar in the corner, its labels
            # stroked in dark so they hold up over both imagery and ramp.
            cax = ax.inset_axes([0.015, 0.19, 0.30, 0.055], zorder=7)
            bar = fig.colorbar(band, cax=cax, orientation="horizontal",
                               spacing="uniform", ticks=breaks)
            bar.ax.set_xticklabels([f"{b:g}" for b in breaks])
            bar.set_label("gap (m)", color=surface, fontsize=13, labelpad=4)
            bar.ax.tick_params(colors=surface, labelsize=12, length=2, pad=3)
            bar.outline.set_edgecolor(surface)
            stroke = [path_effects.withStroke(linewidth=2.2, foreground="#0b0b0b")]
            for text in [*bar.ax.get_xticklabels(), bar.ax.xaxis.label]:
                text.set_path_effects(stroke)
        else:
            bar = fig.colorbar(band, ax=ax, fraction=0.032, pad=0.02,
                               spacing="uniform", ticks=breaks)
            bar.ax.set_yticklabels([f"{b:g}" for b in breaks])
            bar.set_label("gap between road and bike lane (m)", color=muted, fontsize=10)
            bar.ax.tick_params(colors=muted, labelsize=9)
            bar.outline.set_edgecolor("#d8d7d2")

    if not bare:
        handles = [Patch(facecolor=LANE_COLOR, alpha=0.85, label="bike lane (detected from imagery)")]
        if GAP_MAP_SHOW_ROAD:
            handles.insert(0, Patch(facecolor=ROAD_COLOR, alpha=0.55,
                                    label="road (OSM centreline, assumed class width)"))
        legend = ax.legend(handles=handles, frameon=True, fontsize=9.5,
                           labelcolor=ink, loc="lower left", borderpad=0.7)
        legend.get_frame().set_facecolor(surface)
        legend.get_frame().set_edgecolor("#d8d7d2")
        legend.set_zorder(6)

        span_m = bands.shape[2] * pixel_size_m
        ax.set_title(f"Road-to-bike-lane gap\n"
                     f"{len(quads)} measured spans · {span_m:.0f} m across",
                     color=ink, fontsize=13, loc="left", weight="bold")

    ax.axis("off")
    fig.savefig(out_path, dpi=140, facecolor=surface,
                **({"pad_inches": 0} if bare else {"bbox_inches": "tight"}))
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
    # Same lane source as the pipeline: the cached detection where there is
    # one, an in-process trace otherwise.
    cached_mask = BIKELANE_MASK_PATHS.get(_STEM)
    if USE_CACHED_BIKELANE_MASK and cached_mask and cached_mask.exists():
        lanes = lane_centerlines_from_mask(cached_mask, window)
        lane_mask = load_lane_mask(cached_mask, window)
    else:
        lanes, lane_mask = detect_lanes(PREFILTERED_TILE, window)
    print(f"roads: {len(streets)} OSM street way(s); "
          f"bike lanes: {len(lanes)} detected from imagery (not OSM)")
    if USE_OSM_ROAD_FALLBACK:
        print("road edge: OSM highway-class width (USE_OSM_ROAD_FALLBACK), not pixels")

    corrected, shadow, near_edge = prepare_shadow(bands, transform, bounds,
                                                  pixel_size_m, streets)
    print(f"shadow: {shadow.mean():.1%} of frame, "
          f"{near_edge.mean():.1%} within {SHADOW_EDGE_MARGIN_M:.0f} m of an edge")

    records, sections, skipped = measure_gaps(corrected, transform, bounds, shadow,
                                              near_edge, streets, lanes, lane_mask)
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
                          pixel_size_m, streets=streets, lane_mask=lane_mask)
    print(f"Wrote {map_path}")
    print(f"took {time.time() - started:.1f} s")


if __name__ == "__main__":
    main()
