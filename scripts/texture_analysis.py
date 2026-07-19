"""Diagnostic tools for the texture-embedding detector (see texture_embedding.py, detection/texture_detector.py).

Not used by the production pipeline -- these are standalone sanity checks,
run by hand while developing/tuning the reference set or the detector, not
part of any automated flow. Two things live here:

- `print_report` -- pairwise cosine similarity between every reference
  embedding, flagging same-class vs. different-class overlap. Run whenever
  reference images under TEXTURES_DIR change:

      uv run python -m scripts.texture_analysis

  This is exactly the ad-hoc check that caught the reference set's original
  problem: bikelane vs. road (different class) came out *more* similar
  (0.87) than bikelane vs. bikelane (same class, 0.85) -- overlap that
  `discriminant_score` was written to route around, but that's still worth
  checking directly rather than only noticing it indirectly through
  misclassified real crops.

- `visualize_scan` -- runs the sliding-window detector over one bounded
  region and saves a 3-panel PNG (RGB | continuous score heatmap |
  thresholded mask). Slow (~90s for an ~870x580 region on this machine's
  MPS backend) -- meant for eyeballing the detector against a specific
  region you already have some expectation for, not routine use:

      uv run python -m scripts.texture_analysis data/output/foo.tif 80 1990 870 580

- `visualize_edge_trace` -- runs the coarse CNN detector, then
  detection/edge_trace.py's classical color-based edge tracer, over one
  bounded region; saves a 3-panel PNG (RGB | coarse window-block mask |
  traced pixel-precise mask) and prints width statistics from the traced
  mask via detection/width.py. The coarse mask alone is not precise enough
  to measure width from (its shape is the scan window's footprint, not the
  lane's -- see edge_trace.py's module docstring); this is the step that
  makes width measurement meaningful:

      uv run python -m scripts.texture_analysis edges data/output/foo.tif 80 1990 870 580
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

from scripts.config import TEXTURE_STRIDE_PX, TEXTURES_DIR
from scripts.detection.base import Detection
from scripts.detection.edge_trace import BikeLaneEdgeDetector
from scripts.detection.texture_detector import TextureEmbeddingDetector
from scripts.detection.width import measure_width_m
from scripts.texture_embedding import cosine_similarity, load_references


def pairwise_similarities(references: dict[str, dict[str, "object"]]) -> list[tuple[str, str, float]]:
    """Return (name_a, name_b, similarity) for every pair of reference embeddings.

    Names are "<label>/<stem>"; sorted by similarity, descending.
    """
    flat = [
        (f"{label}/{stem}", embedding) for label, embeddings in references.items() for stem, embedding in embeddings.items()
    ]
    results = []
    for i, (name_a, embedding_a) in enumerate(flat):
        for name_b, embedding_b in flat[i + 1 :]:
            results.append((name_a, name_b, cosine_similarity(embedding_a, embedding_b)))
    return sorted(results, key=lambda row: -row[2])


def print_report(textures_dir=TEXTURES_DIR) -> None:
    references = load_references(textures_dir)
    pairs = pairwise_similarities(references)

    print(f"{'reference A':28s} {'reference B':28s} similarity  class")
    same_class_sims, different_class_sims = [], []
    for name_a, name_b, similarity in pairs:
        same_class = name_a.split("/")[0] == name_b.split("/")[0]
        (same_class_sims if same_class else different_class_sims).append(similarity)
        tag = "same class" if same_class else "different class"
        print(f"{name_a:28s} {name_b:28s} {similarity:.4f}      {tag}")

    print()
    if not same_class_sims or not different_class_sims:
        print("Need at least two labels, each with 2+ references, to check separation.")
        return

    min_same, max_different = min(same_class_sims), max(different_class_sims)
    if min_same > max_different:
        print(f"Clean separation: least-similar same-class pair ({min_same:.4f}) still beats "
              f"most-similar different-class pair ({max_different:.4f}).")
    else:
        print(f"WARNING: overlap -- most-similar different-class pair ({max_different:.4f}) beats "
              f"least-similar same-class pair ({min_same:.4f}). Raw nearest-neighbor similarity will "
              f"misclassify some inputs; rely on discriminant_score, not cosine_similarity, for "
              f"classification.")


def visualize_scan(
    tile_path: Path,
    window: Window,
    output_path: Path,
    detector: TextureEmbeddingDetector | None = None,
) -> None:
    """Scan `window` of `tile_path` and save a 3-panel PNG: RGB | score heatmap | thresholded mask.

    Heatmap uses a diverging colormap centered on the discriminant midpoint
    (red = bikelane-side, blue = negative-side); unscanned/empty pixels are
    black in both the heatmap and the mask panels.
    """
    detector = detector or TextureEmbeddingDetector()
    with rasterio.open(tile_path) as src:
        rgb = src.read([1, 2, 3], window=window)
    image = np.transpose(rgb, (1, 2, 0))

    score_map = detector.score_map(image)
    detections = detector.predict(image)

    scanned = ~np.isnan(score_map)
    normalized = np.clip((score_map + 0.3) / 0.6, 0, 1)  # roughly maps [-0.3, 0.3] -> [0, 1]
    heatmap = (matplotlib.colormaps["RdBu_r"](normalized)[..., :3] * 255).astype(np.uint8)
    heatmap[~scanned] = 0

    mask = detections[0].mask if detections else np.zeros(image.shape[:2], dtype=bool)
    overlay = image.astype(np.float32)
    overlay[mask] = overlay[mask] * 0.5 + np.array([0.0, 255.0, 0.0]) * 0.5
    overlay = overlay.astype(np.uint8)

    combined = Image.new("RGB", (image.shape[1] * 3 + 20, image.shape[0]), "white")
    combined.paste(Image.fromarray(image), (0, 0))
    combined.paste(Image.fromarray(heatmap), (image.shape[1] + 10, 0))
    combined.paste(Image.fromarray(overlay), (image.shape[1] * 2 + 20, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)

    mean_score = f"{detections[0].score:.4f}" if detections else "n/a"
    print(f"Wrote {output_path}  ({len(detections)} detection(s), mean score {mean_score})")


def visualize_edge_trace(
    tile_path: Path,
    window: Window,
    output_path: Path,
    coarse_detector: TextureEmbeddingDetector | None = None,
) -> list[Detection]:
    """Scan `window` of `tile_path` and save a 3-panel PNG: RGB | coarse CNN mask | traced mask.

    Also prints width statistics (detection/width.py) measured from the
    traced mask -- the coarse mask's shape is the scan window's footprint,
    not the lane's, so only the traced mask is meaningful to measure.
    Returns the traced-mask detections, so a caller that also wants width
    stats (e.g. generate_pipeline_report.py) doesn't have to re-run the scan.
    """
    coarse_detector = coarse_detector or TextureEmbeddingDetector()
    edge_detector = BikeLaneEdgeDetector(coarse_detector=coarse_detector)
    with rasterio.open(tile_path) as src:
        rgb = src.read([1, 2, 3], window=window)
        pixel_size_m = src.res[0]
    image = np.transpose(rgb, (1, 2, 0))

    coarse_detections = coarse_detector.predict(image)
    coarse_mask = coarse_detections[0].mask if coarse_detections else np.zeros(image.shape[:2], dtype=bool)

    edge_detections = edge_detector.predict(image, coarse=coarse_detections)
    traced_mask = np.zeros(image.shape[:2], dtype=bool)
    for detection in edge_detections:
        traced_mask |= detection.mask

    def overlay(mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
        blended = image.astype(np.float32)
        blended[mask] = blended[mask] * 0.5 + np.array(color) * 0.5
        return blended.astype(np.uint8)

    coarse_panel = overlay(coarse_mask, (0.0, 255.0, 0.0))
    traced_panel = overlay(traced_mask, (0.0, 255.0, 255.0))

    combined = Image.new("RGB", (image.shape[1] * 3 + 20, image.shape[0]), "white")
    combined.paste(Image.fromarray(image), (0, 0))
    combined.paste(Image.fromarray(coarse_panel), (image.shape[1] + 10, 0))
    combined.paste(Image.fromarray(traced_panel), (image.shape[1] * 2 + 20, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)

    print(f"Wrote {output_path}  (coarse px {coarse_mask.sum()}, traced px {traced_mask.sum()})")
    print(f"{len(edge_detections)} traced segment(s):")
    for i, detection in enumerate(sorted(edge_detections, key=lambda d: -d.mask.sum())):
        stats = measure_width_m(detection.mask, pixel_size_m)
        stats_str = (
            f"mean={stats.mean_m:.2f}m median={stats.median_m:.2f}m "
            f"min={stats.min_m:.2f}m max={stats.max_m:.2f}m n={stats.n_samples}"
            if stats
            else "n/a"
        )
        print(f"  segment {i} ({detection.mask.sum()} px): {stats_str}")
    return edge_detections


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print_report()
        return

    if args[0] == "edges":
        args = args[1:]
        mode = "edges"
    else:
        mode = "scan"

    if len(args) not in (5, 6):
        print("Usage:")
        print("  uv run python -m scripts.texture_analysis                                          # pairwise reference report")
        print("  uv run python -m scripts.texture_analysis <tile.tif> <x> <y> <w> <h> [stride_px]        # scan + visualize a region")
        print("  uv run python -m scripts.texture_analysis edges <tile.tif> <x> <y> <w> <h> [stride_px]  # coarse + edge-trace + width")
        print()
        print(f"  stride_px overrides TEXTURE_STRIDE_PX (config.py, default {TEXTURE_STRIDE_PX}) for this run")
        print("  only -- smaller means a finer-resolution scan (more overlapping sample points), at")
        print("  roughly (default/stride_px)^2 the compute cost. Only practical on a bounded cutout like")
        print("  this, not a whole tile -- see TEXTURE_STRIDE_PX's comment in config.py for the full-tile cost.")
        sys.exit(1)

    tile_path, x, y, width, height = args[:5]
    stride_px = int(args[5]) if len(args) == 6 else TEXTURE_STRIDE_PX
    window = Window(int(x), int(y), int(width), int(height))
    detector = TextureEmbeddingDetector(stride_px=stride_px)
    if mode == "edges":
        visualize_edge_trace(Path(tile_path), window, Path("texture_edge_trace_result.png"), coarse_detector=detector)
    else:
        visualize_scan(Path(tile_path), window, Path("texture_scan_result.png"), detector=detector)


if __name__ == "__main__":
    main()
