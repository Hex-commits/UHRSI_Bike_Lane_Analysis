"""DEPRECATED -- fine-tunes a YOLO-seg model nothing in the pipeline loads.

Kept for the record, not for use. The inference half of this path was retired:
`detect.py` (the root YOLO pipeline) and
`scripts/detection/yolo_seg_detector.py` are both gone, so the weights this
writes to YOLO_SEG_TRAINED_WEIGHTS_PATH are read by no code. Running it will
train a model and change nothing downstream.

**This is not the CNN the pipeline uses, and never was.** Detection runs on
the frozen Swin V2-B backbone in `scripts/texture_embedding.py` -- pretrained
on NAIP aerial imagery and used as a feature extractor with no training at
all, classifying by arithmetic on frozen embeddings against the reference
crops in `data/input/textures/`. Nothing there needs this script, and
deleting this file would not affect it.

Why the YOLO path was retired: recall on this project's sparse "sample"
annotations is too low to find the real tracks -- it misses the validated
cycle track outright -- so bike-lane geometry now comes from the cached edge
mask instead (see `scripts/detect.py`).

Retained deliberately, because it is the only record of how the CVAT
annotations in ANNOTATIONS_DIR become a trainable dataset, and that negative
result is worth being able to reproduce. The annotations themselves, and
`data/training/`, are untouched by the retirement.

    uv run python train.py   # still works; produces weights nothing reads
"""

from ultralytics import YOLO

from scripts.config import (
    ANNOTATIONS_DIR,
    DETECTION_INPUT_DIR,
    TRAINING_CHIP_OVERLAP_PX,
    TRAINING_CHIP_SIZE_PX,
    TRAINING_DIR,
    TRAINING_VAL_FRACTION,
    YOLO_SEG_BASE_CHECKPOINT,
    YOLO_SEG_TRAINED_WEIGHTS_PATH,
)
from scripts.detection.dataset import export_dataset


def main():
    print(
        "DEPRECATED: this trains a YOLO-seg model that no longer has a consumer.\n"
        "  The inference side (detect.py, detection/yolo_seg_detector.py) was retired;\n"
        "  the pipeline's detector is the frozen backbone in scripts/texture_embedding.py,\n"
        "  which requires no training. See this file's docstring.\n"
    )
    class_names = export_dataset(
        ANNOTATIONS_DIR,
        TRAINING_DIR,
        DETECTION_INPUT_DIR,
        TRAINING_CHIP_SIZE_PX,
        TRAINING_CHIP_OVERLAP_PX,
        TRAINING_VAL_FRACTION,
    )
    print(f"Classes: {class_names}")

    run_dir = YOLO_SEG_TRAINED_WEIGHTS_PATH.parent.parent.parent
    run_name = YOLO_SEG_TRAINED_WEIGHTS_PATH.parent.parent.name

    model = YOLO(YOLO_SEG_BASE_CHECKPOINT)
    model.train(
        data=str(TRAINING_DIR / "dataset.yaml"),
        epochs=100,
        imgsz=TRAINING_CHIP_SIZE_PX,
        project=str(run_dir),
        name=run_name,
        exist_ok=True,
    )
    print(f"Trained weights: {YOLO_SEG_TRAINED_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
