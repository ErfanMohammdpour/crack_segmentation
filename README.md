Crack Segmentation (CPU‑Only, Windows‑Friendly)

Overview
- Binary crack segmentation with clean, reproducible training on CPU.
- Two variants: Baseline U‑Net Mini and U‑Net Mini with Dropout. Optional SegFormer‑Lite (if timm installed).
- COCO dataset per split; positive class is any category name containing "crack" (case‑insensitive).

What You’ll Learn
- How to load COCO segmentation, build a binary mask via category name rules.
- How to implement OpenCV‑only augmentations correctly for images and masks.
- How to train a modern segmentation model with proper losses, metrics, and early stopping.
- How to evaluate and visualize results (curves, overlays, and triptychs).

Repository Layout
- crackseg/
  - data/: dataset loader, augmentations, I/O helpers
  - models/: UNet Mini, UNet Mini Dropout, optional SegFormer‑Lite
  - losses.py, metrics.py
  - train.py, evaluate.py, infer.py, vis.py
  - config.yaml
- Root wrappers: train.py, evaluate.py, infer.py (call into crackseg/*)

Requirements (CPU‑only)
- Python >= 3.9
- PyTorch (CPU wheels)
- OpenCV‑Python
- pycocotools‑windows (Windows) or pycocotools (Linux/macOS)
- numpy, pyyaml, tqdm, matplotlib

Installation (Windows PowerShell)
1) Create a virtual environment
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
2) Install dependencies (CPU wheels)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install opencv-python numpy pyyaml tqdm matplotlib
   pip install pycocotools-windows  # for Windows (Linux/mac: pip install pycocotools)

Dataset Setup (COCO per split)
<DATA_ROOT>/
  train/
    _annotations.coco.json
    *.jpg
  valid/
    _annotations.coco.json
    *.jpg
  test/
    _annotations.coco.json
    *.jpg

Binary rule: positive = any category whose name contains "crack" (case‑insensitive). All else is background. Masks are unions of positive polygons (annToMask) with fallback to bbox.

Config
- Edit crackseg/config.yaml (CPU defaults included):
  - DATA_ROOT: absolute path to your dataset (or use the provided sample_data for a smoke test)
  - IMG_SIZE: 512 (reduce to 256 on CPU for speed)
  - BATCH_SIZE: 2, NUM_WORKERS: 0 (Windows‑safe)
  - EPOCHS: 30 (increase later)
  - MODEL_NAME: unet_mini | unet_mini_dropout | segformer_lite
  - LOSS: bce_dice (default), threshold: 0.5

Quick Start (Baseline)
1) Point DATA_ROOT in crackseg/config.yaml to your dataset.
2) Train Baseline (UNetMini):
   python train.py --config crackseg/config.yaml --model unet_mini
3) Evaluate on test split:
   python evaluate.py --config crackseg/config.yaml --weights runs/unet_mini/best.pth
4) Inference on images/folder:
   python infer.py --config crackseg/config.yaml --weights runs/unet_mini/best.pth --input <DATA_ROOT>/test --save ./outputs/infer

Dropout Variant
- Train with dropout (helps generalization):
  python train.py --config crackseg/config.yaml --model unet_mini_dropout --dropout 0.3
- Evaluate:
  python evaluate.py --config crackseg/config.yaml --weights runs/unet_mini_dropout/best.pth

Optional: SegFormer‑Lite
- Requires: pip install timm
- Run:
  python train.py --config crackseg/config.yaml --model segformer_lite --encoder mobilenetv3_large_100 --pretrained 1

Outputs
- runs/<model_name>/: best.pth, config_snapshot.yaml, logs.csv, plots/, visuals/
- outputs/metrics_test.json (from evaluate.py)

Visualizations
- train.py automatically saves curves and 8 qualitative triptychs under runs/<model_name>/.
- You can regenerate curves from logs.csv:
  python -c "from crackseg.vis import plot_curves; import pathlib; plot_curves(pathlib.Path('runs/unet_mini/logs.csv'), pathlib.Path('runs/unet_mini/plots'))"

Reproducibility & Determinism
- Seeds for random/numpy/torch set to 42; cuDNN flags are guarded and disabled on CPU.
- AMP is disabled on CPU; gradient clipping enabled.

Troubleshooting
- pycocotools error on Windows: pip install pycocotools-windows
- Slow CPU: set IMG_SIZE=256, EPOCHS=10 for quick tests.
- Memory errors: lower BATCH_SIZE or IMG_SIZE.
- Empty masks: check category names in your COCO file (must contain "crack", case‑insensitive).

Learn As You Go (Recommended Path)
1) Segmentation Basics
   - Read about IoU and Dice and why thin structures (cracks) are challenging.
   - Exercise: print per‑image IoU/Dice and inspect failure cases (thin cracks, low contrast).
2) UNet Architecture
   - Study encoder‑decoder with skip connections; understand why ConvTranspose2d is used.
   - Exercise: change base channels (e.g., 16/32/64) and compare metrics vs. speed.
3) Loss Functions
   - Compare BCE+Dice vs. Focal vs. Tversky on class imbalance.
   - Exercise: switch LOSS in config and log metric changes.
4) Data Augmentation
   - Understand geometric vs. photometric transforms; masks use nearest‑neighbor resize only.
   - Exercise: change augmentation strength and see if overfitting reduces.
5) Evaluation & Visualization
   - Inspect overlays; annotate false positives/negatives.
   - Exercise: adjust THRESHOLD (0.3–0.7) to trade off precision/recall.
6) Transfer Learning (Optional)
   - Try SegFormer‑Lite (timm) to improve thin‑crack detection; freeze vs. fine‑tune encoder.

Sample Data (for Smoke Tests)
- A small synthetic COCO set generator is provided:
  python crackseg/tools/generate_sample_data.py
- Set DATA_ROOT: ./crackseg/sample_data and run Baseline/Dropout training to verify the pipeline end‑to‑end on CPU.

Next Steps
- Run both Baseline and Dropout, compare metrics and curves, and add a short analysis to crackseg/README.md.
- If you enable SegFormer‑Lite, include it in a 3‑way comparison.

