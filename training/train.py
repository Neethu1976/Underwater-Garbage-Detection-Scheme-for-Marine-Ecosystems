from ultralytics import YOLO

# Path to processed dataset YAML
DATA_YAML = "processed_dataset/data.yaml"

def main():
    # Load YOLOv8-Small pretrained weights
    model = YOLO("yolov8s.pt")

    # Train with paper-aligned configuration
    results = model.train(
        data=DATA_YAML,
        epochs=300,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        cos_lr=True,
        mosaic=1.0,
        fliplr=0.5,
        scale=0.5,
        label_smoothing=0.1,
        conf=0.25,
        iou=0.45,
        workers=2,
        device=0,
        project="runs",
        name="yolov8s_underwater",
        exist_ok=True,
        verbose=True
    )

if __name__ == "__main__":
    main()