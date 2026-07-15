import geopandas as gpd
import rasterio

from scripts.config import (
    DETECTION_CHIP_OVERLAP_PX,
    DETECTION_CHIP_SIZE_PX,
    DETECTION_INPUT_DIR,
    DETECTION_OUTPUT_PATH,
    DETECTION_RASTER_FIELDS,
    TILE_CRS,
)
from scripts.detection.rasterize import rasterize_field, render_preview_png
from scripts.detection.tiling import iter_chips, mask_to_polygon
from scripts.detection.width import measure_width_m
from scripts.detection.yolo_seg_detector import YoloSegDetector
from scripts.tiles import find_tile_paths, union_bounds


def main():
    detector = YoloSegDetector()  # the only place a concrete model is referenced

    tile_paths = find_tile_paths(DETECTION_INPUT_DIR, pattern="*.tif")
    print(f"Found {len(tile_paths)} tiles in {DETECTION_INPUT_DIR}")

    records = []
    for tile_path in tile_paths:
        with rasterio.open(tile_path) as src:
            pixel_size_m = src.res[0]
            for chip in iter_chips(src, DETECTION_CHIP_SIZE_PX, DETECTION_CHIP_OVERLAP_PX):
                for detection in detector.predict(chip.image):
                    width = measure_width_m(detection.mask, pixel_size_m)
                    geometry = mask_to_polygon(detection.mask, chip.transform)
                    if width is None or geometry is None:
                        continue
                    records.append(
                        {
                            "geometry": geometry,
                            "tile": tile_path.name,
                            "label": detection.label,
                            "score": detection.score,
                            "width_mean_m": width.mean_m,
                            "width_median_m": width.median_m,
                            "width_min_m": width.min_m,
                            "width_max_m": width.max_m,
                        }
                    )
        print(f"{tile_path.name}: {len(records)} cumulative detections")

    DETECTION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        print("No detections found.")
        return

    gdf = gpd.GeoDataFrame(records, crs=TILE_CRS)
    gdf.to_file(DETECTION_OUTPUT_PATH, driver="GPKG")
    print(f"Wrote {len(records)} detections to {DETECTION_OUTPUT_PATH}")

    bounds = union_bounds(tile_paths)
    for field in DETECTION_RASTER_FIELDS:
        tif_path = DETECTION_OUTPUT_PATH.with_name(f"{DETECTION_OUTPUT_PATH.stem}_{field}.tif")
        rasterize_field(gdf, field, bounds, pixel_size_m, tif_path)
        png_path = render_preview_png(tif_path, tif_path.with_suffix(".png"))
        print(f"Wrote {tif_path} and {png_path}")


if __name__ == "__main__":
    main()
