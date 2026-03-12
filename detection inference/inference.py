import os
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────
# LOAD TRAINED MODEL
# ─────────────────────────────────────────
model = YOLO("/kaggle/working/runs/yolov8s_underwater/weights/best.pt")

# Class names (paper-aligned order)
CLASS_NAMES = ['Mask','can','cellphone','electronics','gbottle','glove',
               'metal','misc','net','pbag','pbottle','plastic','rod',
               'sunglasses','tire']

# Distinct color per class for clean visualization
COLORS = plt.cm.get_cmap('tab20', 15)

# ─────────────────────────────────────────
# PICK RANDOM SAMPLE IMAGES FROM TEST SET
# ─────────────────────────────────────────
test_img_dir = Path("/kaggle/working/processed_dataset/test/images")
all_images   = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))

# Pick 12 random images for visualization
sample_images = random.sample(all_images, min(12, len(all_images)))

# ─────────────────────────────────────────
# RUN INFERENCE & PLOT
# ─────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(24, 18))
fig.suptitle("YOLOv8s — Underwater Garbage Detection Inference\n(Test Set Samples)",
             fontsize=18, fontweight='bold', y=1.01)
axes = axes.flatten()

for idx, img_path in enumerate(sample_images):
    # Run inference
    results = model.predict(
        source     = str(img_path),
        conf       = 0.25,
        iou        = 0.45,
        verbose    = False
    )

    result  = results[0]
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    ax = axes[idx]
    ax.imshow(img_rgb)

    # Draw each detection
    boxes      = result.boxes
    detected_classes = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_score      = float(box.conf[0].cpu().numpy())
            cls_id          = int(box.cls[0].cpu().numpy())
            cls_name        = CLASS_NAMES[cls_id]
            color           = COLORS(cls_id)[:3]

            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth = 2,
                edgecolor = color,
                facecolor = (*color, 0.08)  # slight fill
            )
            ax.add_patch(rect)

            # Label with class + confidence
            ax.text(
                x1, y1 - 5,
                f"{cls_name} {conf_score:.2f}",
                color     = 'white',
                fontsize  = 8,
                fontweight= 'bold',
                bbox      = dict(facecolor=color, alpha=0.85, pad=1.5, edgecolor='none')
            )
            detected_classes.append(cls_name)

    # Title per image
    unique_classes = list(set(detected_classes))
    ax.set_title(
        f"Detected: {', '.join(unique_classes) if unique_classes else 'Nothing detected'}",
        fontsize=9, color='darkblue', pad=4
    )
    ax.axis('off')

plt.tight_layout()
plt.savefig("/kaggle/working/inference_visualization.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Inference visualization saved → /kaggle/working/inference_visualization.png")