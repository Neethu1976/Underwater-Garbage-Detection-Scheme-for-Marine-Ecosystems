from ultralytics import YOLO

# Load YOLOv8-Small pretrained on COCO (transfer learning base)
model = YOLO("yolov8s.pt")

# Train with paper-aligned hyperparameters
results = model.train(
    data    = "/kaggle/working/processed_dataset/data.yaml",
    epochs  = 300,
    imgsz   = 640,
    batch   = 64,
    optimizer = "Adam",
    lr0     = 0.0001,
    weight_decay = 0.0005,
    cos_lr  = True,           # Cosine decay scheduler
    mosaic  = 1.0,            # Mosaic augmentation ON (imbalance handling)
    fliplr  = 0.5,            # Horizontal flip
    scale   = 0.5,            # Random scaling
    label_smoothing = 0.1,    # Default YOLOv8 label smoothing
    conf    = 0.25,           # Confidence threshold
    iou     = 0.45,           # NMS IoU threshold
    workers = 2,              # Kaggle-safe worker count
    device  = 0,              # GPU
    project = "/kaggle/working/runs",
    name    = "yolov8s_underwater",
    exist_ok = True,
    verbose  = True
)
