from ultralytics import YOLO

# Path to dataset YAML
DATA_YAML = "processed_dataset/data.yaml"

# Path to trained weights
MODEL_PATH = "runs/detect/yolov8s_underwater/weights/best.pt"

def main():
    # Load trained model
    model = YOLO(MODEL_PATH)

    # Evaluate on test set
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        conf=0.25,
        iou=0.45,
        device=0
    )

    print("\n========== EVALUATION RESULTS ==========")
    print(f"mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")
    print(f"Precision     : {metrics.box.mp:.4f}")
    print(f"Recall        : {metrics.box.mr:.4f}")

if __name__ == "__main__":
    main()