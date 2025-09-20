Crack Segmentation (CPU Only, Windows Friendly)

Why this repo
- Simple, reproducible crack segmentation on CPU-only machines.
- Three practical model choices: ED-Plus, U-Net Mini, and SegFormer-Lite.
- Human-friendly recipes, visuals, and ready-to-run commands.

Models at a Glance
- ED-Plus (a.k.a. scratch_ed_plus): lightweight encoder–decoder tuned for thin structures like cracks; fast and compact.
- U-Net Mini: classic U-Net downsized for CPU; solid baseline with skip connections. Dropout variant improves generalization.
- SegFormer-Lite: transformer-based encoder (from timm) with a light head; highest capacity if you can afford a bit more compute.

Training Curves (live from runs)
![ED-Plus Metrics](runs/scratch_ed_plus/plots/metrics_curves.png)
![U-Net Mini Metrics](runs/unet_mini/plots/metrics_curves.png)
![SegFormer-Lite Metrics](runs/segformer_lite/plots/metrics_curves.png)

Qualitative Samples (from runs/.../visuals)
![ED-Plus Sample](runs/scratch_ed_plus/visuals/sample_0.png)
![U-Net Mini Sample](runs/unet_mini/visuals/sample_0.png)
![SegFormer-Lite Sample](runs/segformer_lite/visuals/sample_0.png)

Repository Layout
- crackseg/
  - data/: dataset loader, augmentations, I/O helpers
  - models/: ED/ScratchED family, U-Net Mini, optional SegFormer-Lite head
  - losses.py, metrics.py
  - train.py, evaluate.py, infer.py, vis.py
  - config.yaml
- Root wrappers: train.py, evaluate.py, infer.py (call into crackseg/*)

Requirements (CPU only)
- Python >= 3.9
- PyTorch (CPU wheels)
- OpenCV-Python
- pycocotools-windows (Windows) or pycocotools (Linux/macOS)
- numpy, pyyaml, tqdm, matplotlib

Installation (Windows PowerShell)
1) Create a virtual environment
   `python -m venv .venv`
   `.\\.venv\\Scripts\\Activate.ps1`
2) Install dependencies (CPU wheels)
   `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
   `pip install opencv-python numpy pyyaml tqdm matplotlib`
   `pip install pycocotools-windows`  # Linux/mac: `pip install pycocotools`

Dataset Setup (COCO per split)
`<DATA_ROOT>/`
  `train/`  `_annotations.coco.json`, images
  `valid/`  `_annotations.coco.json`, images
  `test/`   `_annotations.coco.json`, images

Binary rule: positive = any category whose name contains "crack" (case-insensitive). All else is background.

Config
- Edit `crackseg/config.yaml`:
  - `DATA_ROOT`: absolute dataset path (or use sample_data for a smoke test)
  - `IMG_SIZE`: 512 (or 256 for faster CPU runs)
  - `BATCH_SIZE`: 2, `NUM_WORKERS`: 0 (Windows-safe)
  - `EPOCHS`: 30 (adjust as needed)
  - `MODEL_NAME`: `scratch_ed_plus | unet_mini | unet_mini_dropout | segformer_lite`
  - `LOSS`: `bce_dice` (default), `threshold`: 0.5

Quick Start
1) Set `DATA_ROOT` in `crackseg/config.yaml`
2) Train your chosen model (see commands below)
3) Evaluate on the test split
4) Run inference to export overlays

Model Guides (commands adapted from COMMANDS.txt)

ED-Plus (alias: `scratch_ed_plus`)
- Train (no dropout):
  `python -m crackseg.train --config crackseg/config.yaml --model scratch_ed_plus --dropout 0.0`
- Train (with dropout, e.g., 0.3):
  `python -m crackseg.train --config crackseg/config.yaml --model scratch_ed_plus --dropout 0.3`
- Evaluate:
  `python -m crackseg.evaluate --config crackseg/config.yaml --weights runs/scratch_ed_plus/best.pth --model scratch_ed_plus`
- Infer (folder):
  `python -m crackseg.infer --config crackseg/config.yaml --weights runs/scratch_ed_plus/best.pth --model scratch_ed_plus --input <DATA_ROOT>/test --save ./outputs/infer_scratch_plus`

U-Net Mini (baseline)
- Train:
  `python -m crackseg.train --config crackseg/config.yaml --model unet_mini`
- Train + Dropout (e.g., 0.3):
  `python -m crackseg.train --config crackseg/config.yaml --model unet_mini_dropout --dropout 0.3`
- Evaluate:
  `python -m crackseg.evaluate --config crackseg/config.yaml --weights runs/unet_mini/best.pth --model unet_mini`
- Infer (folder):
  `python -m crackseg.infer --config crackseg/config.yaml --weights runs/unet_mini/best.pth --model unet_mini --input <DATA_ROOT>/test --save ./outputs/infer_unet`

SegFormer-Lite (requires `timm`)
- Install backbones: `pip install timm`
- Train (with encoder + pretrained):
  `python -m crackseg.train --config crackseg/config.yaml --model segformer_lite --encoder segformer_b0 --pretrained 1`
- Freeze warm-up + dual LR groups:
  `python -m crackseg.train --config crackseg/config.yaml --model segformer_lite --encoder segformer_b0 --pretrained 1 --freeze-epochs 5 --lr-head 1e-3 --lr-encoder 1e-4`
- Evaluate:
  `python -m crackseg.evaluate --config crackseg/config.yaml --weights runs/segformer_lite/best.pth --model segformer_lite --encoder segformer_b0 --pretrained 1`
- Infer (folder):
  `python -m crackseg.infer --config crackseg/config.yaml --weights runs/segformer_lite/best.pth --model segformer_lite --encoder segformer_b0 --pretrained 1 --input <DATA_ROOT>/test --save ./outputs/infer_segformer`

Outputs
- `runs/<model_name>/`: `best.pth`, `config_snapshot.yaml`, `logs.csv`, `plots/`, `visuals/`
- `outputs/metrics_test.json` (from `evaluate.py`)

Visualizations
- Training saves curves and 8 qualitative triptychs under `runs/<model_name>/`.
- Regenerate curves from `logs.csv`:
  `python -c "from crackseg.vis import plot_curves; import pathlib; plot_curves(pathlib.Path('runs/unet_mini/logs.csv'), pathlib.Path('runs/unet_mini/plots'))"`

Reproducibility & Determinism
- Seeds for random/numpy/torch set to 42; cuDNN flags are disabled on CPU.
- AMP disabled on CPU; gradient clipping enabled.

Troubleshooting
- Windows COCO API: `pip install pycocotools-windows`
- Slow CPU: set `IMG_SIZE=256`, `EPOCHS=10` for quick tests.
- OOM: lower `BATCH_SIZE` or `IMG_SIZE`.
- Empty masks: ensure category names contain "crack".

Augmentation Notes
- Online, OpenCV-only (rotate ±30°, scale 0.8–1.2, brightness/contrast ±20%, Gaussian noise σ≈10/255) on train split only.
- Intensity/duplication via `AUG_MULTIPLIER` in YAML (e.g., 5).

Offline Aug Audit (do not use for metrics)
`python crackseg/tools/offline_augment.py --config crackseg/config.yaml --split train --save ./outputs/offline_aug5_train --seed 42`
`python crackseg/tools/offline_augment.py --config crackseg/config.yaml --split valid --save ./outputs/offline_aug5_valid --seed 42`
`python crackseg/tools/offline_augment.py --config crackseg/config.yaml --split test  --save ./outputs/offline_aug5_test  --seed 42`

