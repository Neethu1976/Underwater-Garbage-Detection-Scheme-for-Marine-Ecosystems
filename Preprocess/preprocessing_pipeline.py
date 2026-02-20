import cv2
import numpy as np
import os
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DATASET_ROOT = "/kaggle/input/datasets/siddharth2305ego/underwater-garbagedebris/Underwater_garbage"   # ← Change to your dataset folder name
OUTPUT_ROOT  = "/kaggle/working/processed_dataset"
TARGET_SIZE  = (640, 640)

# CLAHE parameters (paper-aligned)
CLAHE_CLIP_LIMIT    = 2.0   # Limits noise amplification & backscatter
CLAHE_TILE_GRID     = (8, 8)  # 8×8 non-overlapping localized tiles

SPLITS = ["train", "valid", "test"]

# ──────────────────────────────────────────────
# CLAHE + RESIZE FUNCTION
# ──────────────────────────────────────────────
def apply_clahe_and_resize(img_bgr, clip_limit=2.0, tile_grid=(8,8), size=(640,640)):
    """
    1. Convert BGR → LAB color space
    2. Apply CLAHE only on L-channel (luminance) — preserves color integrity
    3. Merge back → BGR
    4. Resize to 640×640
    """
    # Step 1: BGR → LAB
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Step 2: CLAHE on L-channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l)

    # Step 3: Merge and convert back to BGR
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Step 4: Resize to 640×640
    resized = cv2.resize(enhanced_bgr, size, interpolation=cv2.INTER_LINEAR)

    return resized

# ──────────────────────────────────────────────
# COPY LABELS AS-IS (bounding boxes unchanged)
# ──────────────────────────────────────────────
def copy_labels(src_label_dir, dst_label_dir):
    """Labels are YOLO-normalized — no modification needed after resize."""
    import shutil
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    if src_label_dir.exists():
        for lbl_file in src_label_dir.glob("*.txt"):
            shutil.copy2(lbl_file, dst_label_dir / lbl_file.name)

# ──────────────────────────────────────────────
# MAIN PROCESSING PIPELINE
# ──────────────────────────────────────────────
def process_dataset():
    total_processed = 0
    total_failed    = 0

    for split in SPLITS:
        src_img_dir   = Path(DATASET_ROOT) / split / "images"
        src_label_dir = Path(DATASET_ROOT) / split / "labels"
        dst_img_dir   = Path(OUTPUT_ROOT)  / split / "images"
        dst_label_dir = Path(OUTPUT_ROOT)  / split / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.exists():
            print(f"[SKIP] {split}/images not found — skipping.")
            continue

        img_files = list(src_img_dir.glob("*.jpg")) + \
                    list(src_img_dir.glob("*.jpeg")) + \
                    list(src_img_dir.glob("*.png"))

        print(f"\n[{split.upper()}] Found {len(img_files)} images → Processing...")

        split_processed = 0
        split_failed    = 0

        for img_path in img_files:
            try:
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError(f"Cannot read: {img_path.name}")

                # Apply CLAHE + Resize
                result = apply_clahe_and_resize(
                    img,
                    clip_limit=CLAHE_CLIP_LIMIT,
                    tile_grid=CLAHE_TILE_GRID,
                    size=TARGET_SIZE
                )

                # Save processed image (keep original filename)
                out_path = dst_img_dir / img_path.name
                cv2.imwrite(str(out_path), result)
                split_processed += 1

            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")
                split_failed += 1

        # Copy labels unchanged
        copy_labels(src_label_dir, dst_label_dir)

        print(f"  ✅ Processed : {split_processed}")
        print(f"  ❌ Failed    : {split_failed}")
        print(f"  📁 Labels copied from {src_label_dir}")

        total_processed += split_processed
        total_failed    += split_failed

    print("\n" + "="*50)
    print(f"  TOTAL PROCESSED : {total_processed}")
    print(f"  TOTAL FAILED    : {total_failed}")
    print(f"  OUTPUT PATH     : {OUTPUT_ROOT}")
    print("="*50)

# ──────────────────────────────────────────────
# COPY data.yaml TO OUTPUT ROOT
# ──────────────────────────────────────────────
def copy_and_update_yaml():
    import shutil
    src_yaml = Path(DATASET_ROOT) / "data.yaml"
    dst_yaml = Path(OUTPUT_ROOT)  / "data.yaml"

    if src_yaml.exists():
        shutil.copy2(src_yaml, dst_yaml)

        # Update paths inside yaml to point to processed dataset
        with open(dst_yaml, "r") as f:
            content = f.read()

        content = content.replace(
            "../train/images", f"{OUTPUT_ROOT}/train/images"
        ).replace(
            "../valid/images", f"{OUTPUT_ROOT}/valid/images"
        ).replace(
            "../test/images",  f"{OUTPUT_ROOT}/test/images"
        )

        with open(dst_yaml, "w") as f:
            f.write(content)

        print(f"\n✅ data.yaml copied and paths updated → {dst_yaml}")
    else:
        print(f"\n[WARNING] data.yaml not found at {src_yaml}")

# ──────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────
process_dataset()
copy_and_update_yaml()