from pathlib import Path

import rasterio

from pipeline.config import INPUT_TILES_DIR


def find_tile_paths(tiles_dir: Path = INPUT_TILES_DIR, pattern: str = "*.jp2") -> list[Path]:
    """Return all imagery tile paths in `tiles_dir` matching `pattern`, sorted for reproducibility."""
    return sorted(tiles_dir.glob(pattern))


def tile_bounds(tile_path: Path) -> tuple[float, float, float, float]:
    """Return (left, bottom, right, top) bounds of a tile in its native CRS."""
    with rasterio.open(tile_path) as src:
        return tuple(src.bounds)


def union_bounds(tile_paths: list[Path]) -> tuple[float, float, float, float]:
    """Return the combined (left, bottom, right, top) bounds covering all tiles."""
    lefts, bottoms, rights, tops = zip(*(tile_bounds(p) for p in tile_paths))
    return min(lefts), min(bottoms), max(rights), max(tops)
