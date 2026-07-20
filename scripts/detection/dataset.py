"""Convert CVAT-exported YOLO-seg annotations into a chipped training dataset.

Annotations are drawn on full tiles, but YOLO trains on smaller images (640px
chips, matching `ultralytics`' default `imgsz` so chips aren't shrunk further
-- bike lanes are only ~10px wide here). This chips each annotated tile, clips
each polygon to the chip it falls in (a straddling polygon becomes multiple
pieces), and writes a standard Ultralytics YOLO-seg dataset:

    data/training/
      images/train/*.png, images/val/*.png
      labels/train/*.txt, labels/val/*.txt
      dataset.yaml

Only chips with at least one instance are kept. Empty chips are NOT treated as
background negatives: this is "sample" annotation data, not exhaustively
labeled, so an empty chip might just be an unannotated bike lane. Revisit once
coverage is closer to exhaustive.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import yaml
from PIL import Image
from shapely.geometry import Polygon, box


@dataclass
class _Instance:
    class_id: int
    polygon: Polygon


def _load_task(task_dir: Path) -> tuple[dict[int, str], dict[str, list[_Instance]]]:
    """Parse one CVAT export folder into (class names, {image stem: instances})."""
    with open(task_dir / "data.yaml") as f:
        class_names = yaml.safe_load(f)["names"]

    instances_by_stem: dict[str, list[_Instance]] = {}
    for label_path in sorted((task_dir / "labels" / "train").glob("*.txt")):
        stem = label_path.stem
        instances = []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            if len(coords) < 6:
                continue
            xy_norm = list(zip(coords[0::2], coords[1::2]))
            instances.append((class_id, xy_norm))
        instances_by_stem[stem] = instances

    return class_names, instances_by_stem


def _denormalize(xy_norm: list[tuple[float, float]], width: int, height: int) -> Polygon:
    return Polygon([(x * width, y * height) for x, y in xy_norm])


def _clip_to_chip(polygon: Polygon, chip_box: Polygon) -> list[Polygon]:
    """Intersect `polygon` with `chip_box`, returning zero or more clipped pieces."""
    clipped = polygon.intersection(chip_box)
    if clipped.is_empty:
        return []
    geoms = list(clipped.geoms) if clipped.geom_type == "GeometryCollection" else [clipped]
    polygons = []
    for geom in geoms:
        if geom.geom_type == "Polygon" and geom.area > 0:
            polygons.append(geom)
        elif geom.geom_type == "MultiPolygon":
            polygons.extend(g for g in geom.geoms if g.area > 0)
    return polygons


def export_dataset(
    annotations_dir: Path,
    output_dir: Path,
    source_tiles_dir: Path,
    chip_size_px: int,
    chip_overlap_px: int,
    val_fraction: float,
) -> dict[int, str]:
    """Build the chipped YOLO dataset. Returns the class-id -> name mapping used."""
    task_dirs = sorted(d for d in annotations_dir.iterdir() if d.is_dir())
    if not task_dirs:
        raise FileNotFoundError(f"No annotation task exports found under {annotations_dir}")

    class_names: dict[int, str] = {}
    instances_by_stem: dict[str, list[tuple[int, list[tuple[float, float]]]]] = {}
    for task_dir in task_dirs:
        task_class_names, task_instances = _load_task(task_dir)
        class_names.update(task_class_names)
        for stem, instances in task_instances.items():
            instances_by_stem.setdefault(stem, []).extend(instances)

    chips_written: list[tuple[Path, str]] = []
    step = chip_size_px - chip_overlap_px

    for stem, raw_instances in instances_by_stem.items():
        tile_path = source_tiles_dir / f"{stem}.tif"
        if not tile_path.exists():
            print(f"Skipping {stem}: no matching tile at {tile_path}")
            continue

        with rasterio.open(tile_path) as src:
            width, height = src.width, src.height
            instances = [
                _Instance(class_id, _denormalize(xy_norm, width, height))
                for class_id, xy_norm in raw_instances
            ]

            for y in range(0, height, step):
                for x in range(0, width, step):
                    chip_w = min(chip_size_px, width - x)
                    chip_h = min(chip_size_px, height - y)
                    chip_geom = box(x, y, x + chip_w, y + chip_h)

                    label_lines = []
                    for instance in instances:
                        for piece in _clip_to_chip(instance.polygon, chip_geom):
                            local_xy = [((px - x) / chip_w, (py - y) / chip_h) for px, py in piece.exterior.coords]
                            coords_str = " ".join(f"{v:.6f}" for xy in local_xy for v in xy)
                            label_lines.append(f"{instance.class_id} {coords_str}")

                    if not label_lines:
                        continue

                    rgb = src.read([1, 2, 3], window=rasterio.windows.Window(x, y, chip_w, chip_h))
                    image = np.transpose(rgb, (1, 2, 0))
                    chip_name = f"{stem}_{x}_{y}"
                    chips_written.append((chip_name, image, "\n".join(label_lines)))

    if not chips_written:
        raise RuntimeError("No chips contained any annotated instance -- nothing to export")

    chips_written.sort(key=lambda c: c[0])
    for images_dir in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dir / images_dir).mkdir(parents=True, exist_ok=True)

    n_val = max(1, round(len(chips_written) * val_fraction)) if len(chips_written) > 1 else 0
    n_val = min(n_val, len(chips_written) - 1) if n_val else 0
    val_start = len(chips_written) - n_val

    for i, (chip_name, image, label_text) in enumerate(chips_written):
        split = "val" if i >= val_start else "train"
        Image.fromarray(image).save(output_dir / "images" / split / f"{chip_name}.png")
        (output_dir / "labels" / split / f"{chip_name}.txt").write_text(label_text + "\n")

    dataset_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val" if n_val else "images/train",
        "names": class_names,
    }
    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    print(f"Exported {len(chips_written)} chips ({len(chips_written) - n_val} train, {n_val} val)")
    return class_names
