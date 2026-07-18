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
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

from scripts.config import TEXTURES_DIR
from scripts.detection.texture_detector import TextureEmbeddingDetector
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


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print_report()
        return

    if len(args) != 5:
        print("Usage:")
        print("  uv run python -m scripts.texture_analysis                          # pairwise reference report")
        print("  uv run python -m scripts.texture_analysis <tile.tif> <x> <y> <w> <h>  # scan + visualize a region")
        sys.exit(1)

    tile_path, x, y, width, height = args
    visualize_scan(
        Path(tile_path),
        Window(int(x), int(y), int(width), int(height)),
        Path("texture_scan_result.png"),
    )


if __name__ == "__main__":
    main()
