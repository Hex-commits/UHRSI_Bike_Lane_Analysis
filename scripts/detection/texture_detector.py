"""Sliding-window bike-lane texture detector: frozen CNN embeddings, no training.

Scans an image with a small window (matching the reference crop size),
scores each window via texture_embedding.discriminant_score, and keeps the
ones on the "bikelane" side of the discriminant. Windows are batched through
the frozen backbone (~28 ms/image on this machine's MPS backend regardless
of batch size -- batching didn't reduce per-image cost for this model, but
still avoids redundant Python overhead).

This is not fast: a full 5000x5000 tile has far more windows than makes
sense to scan in one sitting on this hardware (a single 640x640 chip alone
means ~3000 windows at 50% stride). Meant for scanning a bounded
region-of-interest, not batch-processing whole tiles end to end.
"""

from pathlib import Path

import numpy as np

from scripts.config import TEXTURE_STRIDE_PX, TEXTURE_WINDOW_PX, TEXTURES_DIR
from scripts.detection.base import Detection
from scripts.texture_embedding import discriminant_direction, embed_batch, load_references

BATCH_SIZE = 64

# Calibrated from this session's validation crops, not guessed: every genuine
# lane-paint crop tested scored +0.16 to +0.25; every clean negative scored
# <= -0.10; the one edge case (a partially-shadowed rooftop) scored +0.042,
# still below the lowest lane score. 0.10 sits strictly between the highest
# validated negative and the lowest validated positive, so it also happens
# to fix that edge case as a side effect. The old default (0.0) let a lot of
# plain street through, since much of it still scored weakly positive.
SCORE_THRESHOLD = 0.10


class TextureEmbeddingDetector:
    """Detector (see detection/base.py) using texture_embedding's discriminant score."""

    def __init__(
        self,
        textures_dir: Path = TEXTURES_DIR,
        window_px: int = TEXTURE_WINDOW_PX,
        stride_px: int = TEXTURE_STRIDE_PX,
        threshold: float = SCORE_THRESHOLD,
    ):
        references = load_references(textures_dir)
        self._direction, self._midpoint = discriminant_direction(references)
        self._window_px = window_px
        self._stride_px = stride_px
        self._threshold = threshold

    def _scan(self, image: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Slide the window across `image`; return (top-left positions, scores) for non-empty windows."""
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
        return positions, scores

    def predict(self, image: np.ndarray) -> list[Detection]:
        positions, scores = self._scan(image)
        hits = scores > self._threshold
        if not hits.any():
            return []

        window = self._window_px
        mask = np.zeros(image.shape[:2], dtype=bool)
        for (y, x), hit in zip(positions, hits):
            if hit:
                mask[y : y + window, x : x + window] = True
        return [Detection(mask=mask, score=float(scores[hits].mean()), label="bikelane_texture")]

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
