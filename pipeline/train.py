from ultralytics import YOLO

from pipeline.config import (
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
