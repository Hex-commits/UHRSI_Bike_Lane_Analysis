import numpy as np
from skimage.draw import polygon as sk_polygon
from ultralytics import YOLO

from scripts.config import DETECTION_CONFIDENCE_THRESHOLD, YOLO_SEG_TRAINED_WEIGHTS_PATH
from scripts.detection.base import Detection


def _polygon_to_mask(xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    rows, cols = sk_polygon(xy[:, 1], xy[:, 0], shape=shape)
    mask[rows, cols] = True
    return mask


class YoloSegDetector:
    def __init__(
        self,
        weights_path=YOLO_SEG_TRAINED_WEIGHTS_PATH,
        confidence: float = DETECTION_CONFIDENCE_THRESHOLD,
    ):
        self._model = YOLO(str(weights_path))
        self._confidence = confidence

    def predict(self, image: np.ndarray) -> list[Detection]:
        results = self._model.predict(image, conf=self._confidence, verbose=False)[0]
        if results.masks is None:
            return []

        shape = image.shape[:2]
        detections = []
        for xy, score, cls_id in zip(
            results.masks.xy, results.boxes.conf.tolist(), results.boxes.cls.tolist()
        ):
            if xy.shape[0] < 3:
                continue
            mask = _polygon_to_mask(xy, shape)
            if not mask.any():
                continue
            detections.append(Detection(mask=mask, score=float(score), label=results.names[int(cls_id)]))
        return detections
