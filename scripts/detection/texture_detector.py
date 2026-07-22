from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from pipeline.config import (
    BIKE_LANE_TEXTURE_LABELS,
    ROAD_TEXTURE_LABELS,
    TEXTURE_STRIDE_PX,
    TEXTURE_WINDOW_PX,
    TEXTURES_DIR,
)
from scripts.detection.base import Detection
from scripts.detection.texture_embedding import discriminant_direction, embed_batch, load_references
from pipeline.config import (
    EMBED_BATCH_SIZE as BATCH_SIZE,
    ROAD_SCORE_THRESHOLD,
    SCORE_THRESHOLD,
)





class TextureEmbeddingDetector:
    """Detector (see detection/base.py) using texture_embedding's discriminant score."""

    def __init__(
        self,
        positive_label: str = BIKE_LANE_TEXTURE_LABELS[0],
        negative_labels: Sequence[str] = BIKE_LANE_TEXTURE_LABELS[1],
        textures_dir: Path = TEXTURES_DIR,
        window_px: int = TEXTURE_WINDOW_PX,
        stride_px: int = TEXTURE_STRIDE_PX,
        threshold: float = SCORE_THRESHOLD,
    ):
        references = load_references(textures_dir)
        self._direction, self._midpoint = discriminant_direction(references, positive_label, negative_labels)
        self._label = positive_label
        self._window_px = window_px
        self._stride_px = stride_px
        self._threshold = threshold

    def _scan(
        self, image: np.ndarray, progress: Callable[[int, int], None] | None = None
    ) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Slide the window across `image`; return (top-left positions, scores) for non-empty windows.

        `progress` is called with (windows scanned so far, total) after each
        batch -- a full-tile scan runs for tens of minutes, long enough that
        a caller wants to be able to show it's still moving.
        """
        height, width = image.shape[:2]
        window = self._window_px
        positions = [
            (y, x)
            for y in range(0, height - window + 1, self._stride_px)
            for x in range(0, width - window + 1, self._stride_px)
            if image[y : y + window, x : x + window].any()
        ]
        if not positions:
            return positions, np.empty(0, dtype=np.float32)

        scores = np.empty(len(positions), dtype=np.float32)
        direction_norm = np.linalg.norm(self._direction) + 1e-8
        for start in range(0, len(positions), BATCH_SIZE):
            batch = positions[start : start + BATCH_SIZE]
            windows = [image[y : y + window, x : x + window] for y, x in batch]
            embeddings = embed_batch(windows)
            unit_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            scores[start : start + len(batch)] = (unit_embeddings - self._midpoint) @ self._direction / direction_norm
            if progress is not None:
                progress(min(start + BATCH_SIZE, len(positions)), len(positions))
        return positions, scores

    def predict(self, image: np.ndarray, progress: Callable[[int, int], None] | None = None) -> list[Detection]:
        positions, scores = self._scan(image, progress)
        hits = scores > self._threshold
        if not hits.any():
            return []

        window = self._window_px
        mask = np.zeros(image.shape[:2], dtype=bool)
        for (y, x), hit in zip(positions, hits):
            if hit:
                mask[y : y + window, x : x + window] = True
        return [Detection(mask=mask, score=float(scores[hits].mean()), label=f"{self._label}_texture")]

    def score_map(self, image: np.ndarray) -> np.ndarray:
        """Return a per-pixel discriminant-score raster (NaN where unscanned) -- for visualization."""
        positions, scores = self._scan(image)
        window = self._window_px
        score_sum = np.zeros(image.shape[:2], dtype=np.float32)
        score_count = np.zeros(image.shape[:2], dtype=np.float32)
        for (y, x), score in zip(positions, scores):
            score_sum[y : y + window, x : x + window] += score
            score_count[y : y + window, x : x + window] += 1
        return np.divide(score_sum, score_count, out=np.full_like(score_sum, np.nan), where=score_count > 0)


def bike_lane_detector(**kwargs) -> TextureEmbeddingDetector:
    """Coarse detector for bike-lane paint (the original configuration)."""
    positive, negatives = BIKE_LANE_TEXTURE_LABELS
    return TextureEmbeddingDetector(positive, negatives, threshold=SCORE_THRESHOLD, **kwargs)


def road_detector(**kwargs) -> TextureEmbeddingDetector:
    """Coarse detector for road surface -- same scan, road-vs-rest discriminant."""
    positive, negatives = ROAD_TEXTURE_LABELS
    return TextureEmbeddingDetector(positive, negatives, threshold=ROAD_SCORE_THRESHOLD, **kwargs)
