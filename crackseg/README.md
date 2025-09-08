Crack Segmentation – High-Accuracy Binary Segmentation

Overview
- End-to-end crack segmentation with clean, reproducible training and evaluation.
- Two model variants: Baseline U-Net Mini and Dropout U-Net Mini. Optional SegFormer-Lite.
- COCO-format dataset per split; positive class is any category name containing "crack" (case-insensitive).

Environment
- Python >= 3.9
- PyTorch >= 2.0, TorchVision
- OpenCV-Python
- PyCOCOTools
- NumPy, PyYAML, TQDM, Matplotlib

Dataset Layout (COCO per split)
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

Install (CPU-only, Windows-friendly)
- Create and activate a venv, then install deps:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install opencv-python numpy pyyaml tqdm matplotlib
  pip install pycocotools-windows  # Windows; on Linux/macOS use: pip install pycocotools

Config
- Edit paths in crackseg/config.yaml. Key options:
  - data_root: dataset root path
  - img_size: [512, 512]
  - batch_size, epochs, lr, weight_decay
  - model: unet_mini | unet_mini_dropout | segformer_lite

Commands
- Baseline:  python crackseg/train.py --config crackseg/config.yaml --model unet_mini
- Dropout:   python crackseg/train.py --config crackseg/config.yaml --model unet_mini_dropout --dropout 0.3
- SegFormer: python crackseg/train.py --config crackseg/config.yaml --model segformer_lite --encoder mobilenet_v3_small --pretrained 0
- Evaluate:  python crackseg/evaluate.py --config crackseg/config.yaml --weights runs/unet_mini/best.pth
- Infer:     python crackseg/infer.py --config crackseg/config.yaml --weights runs/unet_mini/best.pth --input ./samples --save ./outputs/infer

Windows/CPU Tips
- Set NUM_WORKERS=0 (default). Reduce IMG_SIZE to 256 for speed.
- AMP is disabled on CPU. Training uses AdamW and ReduceLROnPlateau with early stopping.

Outputs
- runs/<model_name>/
  - config.yaml snapshot
  - best.pth (best-by-IoU)
  - logs.csv
  - plots/: loss and IoU curves
  - metrics_test.json (from evaluate.py)
  - visuals/: qualitative overlays

Notes
- Augmentation uses only OpenCV (rotation, scaling, brightness/contrast, Gaussian noise).
- Masks are resized with INTER_NEAREST and kept as {0,1}.
- Mixed precision (AMP) enabled; deterministic seeds set.
