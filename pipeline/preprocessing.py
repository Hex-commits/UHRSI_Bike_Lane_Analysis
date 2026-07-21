from pipeline.config import BIKE_LANE_BUFFER_METERS, OUTPUT_DIR, STREET_BUFFER_METERS
from scripts.preprocessing.filter_imagery import filter_tile
from scripts.preprocessing.mask import buffer_by_category
from scripts.preprocessing.osm_features import fetch_osm_features
from scripts.preprocessing.tiles import find_tile_paths, union_bounds


def main():
    tile_paths = find_tile_paths()
    print(f"Found {len(tile_paths)} imagery tiles")

    bounds = union_bounds(tile_paths)
    features = fetch_osm_features(bounds)
    counts = features["category"].value_counts()
    print(f"Loaded {len(features)} OSM features ({dict(counts)})")

    buffered = buffer_by_category(
        features,
        {"bikelane": BIKE_LANE_BUFFER_METERS, "street": STREET_BUFFER_METERS},
    )

    for tile_path in tile_paths:
        out_path = OUTPUT_DIR / f"{tile_path.stem}_bikelanes.tif"
        filter_tile(tile_path, buffered, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
