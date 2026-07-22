from pipeline.config import (
    BIKE_LANE_BUFFER_METERS,
    INPUT_CHUNK_PATH,
    OUTPUT_DIR,
    STREET_BUFFER_METERS,
)
from scripts.preprocessing.filter_imagery import filter_tile
from scripts.preprocessing.mask import buffer_by_category
from scripts.preprocessing.osm_features import fetch_osm_features
from scripts.preprocessing.tiles import union_bounds


def main():
    # Just the configured chunk, not every tile in the archive: the detect
    # stage runs on INPUT_CHUNK_PATH alone, and prefiltering the other five
    # tiles cost minutes to produce output nothing downstream reads.
    tile_paths = [INPUT_CHUNK_PATH]
    print(f"Prefiltering {INPUT_CHUNK_PATH.name}")

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
