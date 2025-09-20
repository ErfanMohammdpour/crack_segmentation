import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
DATASET_ROOT = Path("/home/cv_project/dataset_all")   # original full dataset (images + masks)
OUTPUT_ROOT = Path("/home/cv_project/dataset_split")  # new split dataset root
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Collect image-mask pairs
image_exts = [".jpg", ".png", ".jpeg"]
images = [p for p in DATASET_ROOT.iterdir() if p.suffix.lower() in image_exts and "_mask" not in p.stem]

pairs = []
for img in images:
    mask = DATASET_ROOT / f"{img.stem}_mask{img.suffix}"
    if mask.exists():
        pairs.append((img, mask))

print(f"Found {len(pairs)} image-mask pairs")

# Split 70/15/15
train_pairs, temp_pairs = train_test_split(pairs, test_size=0.30, random_state=42, shuffle=True)
valid_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.50, random_state=42, shuffle=True)

splits = {"train": train_pairs, "valid": valid_pairs, "test": test_pairs}

# Copy into folders
for split, split_pairs in splits.items():
    img_dir = OUTPUT_ROOT / split / "images"
    mask_dir = OUTPUT_ROOT / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for img, mask in split_pairs:
        shutil.copy(img, img_dir / img.name)
        shutil.copy(mask, mask_dir / mask.name)

    print(f"{split}: {len(split_pairs)} samples saved")
