"""Print raster dimensions and band metadata to the terminal."""

import sys
from pathlib import Path

import rasterio


def describe_tile(path: Path) -> None:
    with rasterio.open(path) as src:
        print(path.name)
        print(f"  CRS:        {src.crs}")
        print(f"  Dimensions: {src.width} x {src.height} px  ({src.res[0]} m/px)")
        print(f"  Bands:      {src.count}")
        for i in range(1, src.count + 1):
            desc = src.descriptions[i - 1] or "-"
            print(
                f"    Band {i}: dtype={src.dtypes[i - 1]:<6} "
                f"colorinterp={src.colorinterp[i - 1].name:<10} "
                f"nodata={src.nodatavals[i - 1]}  {desc}"
            )


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        from pipeline.config import OUTPUT_DIR

        paths = sorted(OUTPUT_DIR.glob("*.tif"))

    for path in paths:
        describe_tile(path)
        print()


if __name__ == "__main__":
    main()
