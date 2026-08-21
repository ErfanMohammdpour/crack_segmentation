# Crack Segmentation

CPU-friendly pavement / concrete **crack segmentation** with three models: **ED-Plus**, **U-Net Mini**, **SegFormer-Lite**.

Course / applied CV project. Real training logs under `runs/`.

## Dataset (ours)

We **collected / curated** this pavement-crack set ourselves for the course project, then labeled it **manually as COCO polygon segmentation** on Roboflow (not auto-generated masks).

- **Corrected export:** `crack segmentation.v3-version2` (COCO segmentation) — image–mask mismatches fixed.
- Layout after unzip: `train/` · `valid/` · `test/` each with `_annotations.coco.json` + images.
- Loader rule: positive = category name containing `crack` (case-insensitive).

### Size (this export)

| Split | Images | COCO annotations |
|-------|--------|------------------|
| **train** | **2078** | 2471 |
| **valid** | **8** | 8 |
| **test** | **27** | 28 |
| **Total** | **2113** | 2507 |

### Download

```bash
curl -L "https://app.roboflow.com/ds/035L0MXWQ7?key=Ar7SlkKyBk" > roboflow.zip
unzip roboflow.zip && rm roboflow.zip
```

Link: [Roboflow export `035L0MXWQ7`](https://app.roboflow.com/ds/035L0MXWQ7?key=Ar7SlkKyBk)

Set `DATA_ROOT` in `crackseg/config.yaml` to the unzipped folder.

### Samples from each split

| train | valid | test |
|:---:|:---:|:---:|
| ![train](docs/images/dataset_samples/split_train.jpg) | ![valid](docs/images/dataset_samples/split_valid.jpg) | ![test](docs/images/dataset_samples/split_test.jpg) |

### Manual COCO labels (image | polygon mask)

Green = hand-drawn COCO segmentation polygon · red = bbox from the same annotation.

![COCO labels grid](docs/images/dataset_samples/coco_labels_grid.jpg)

More per split:

| train | valid | test |
|:---:|:---:|:---:|
| ![ct1](docs/images/dataset_samples/coco_train_01.jpg) | ![cv1](docs/images/dataset_samples/coco_valid_01.jpg) | ![cte1](docs/images/dataset_samples/coco_test_01.jpg) |
| ![ct2](docs/images/dataset_samples/coco_train_02.jpg) | ![cv2](docs/images/dataset_samples/coco_valid_02.jpg) | ![cte2](docs/images/dataset_samples/coco_test_02.jpg) |

## Results (validation, from committed `runs/*/logs.csv`)

| Model | Best val IoU | Best val Dice | Epoch |
|-------|--------------|---------------|-------|
| **SegFormer-Lite** | **0.548** | **0.670** | 21 |
| ED-Plus (`scratch_ed_plus`) | 0.529 | 0.655 | 21 |
| U-Net Mini | 0.515 | 0.643 | 28 |

Validation curves during training — not a public test leaderboard. Weights (`.pth`) not in git; logs + figs are.

### Qualitative predictions

![SegFormer sample](docs/images/segformer_sample.jpg)
![ED-Plus sample](docs/images/ed_plus_sample.jpg)
![U-Net sample](docs/images/unet_sample.jpg)

## Models

- **ED-Plus** — light encoder–decoder for thin cracks  
- **U-Net Mini** — small U-Net baseline (+ optional dropout)  
- **SegFormer-Lite** — `timm` SegFormer-B0 + light head  

Loss: BCE + Dice. Seed 42.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy pyyaml tqdm matplotlib pycocotools
# optional SegFormer: pip install timm

# download dataset (see above), set DATA_ROOT in crackseg/config.yaml
python -m crackseg.train --config crackseg/config.yaml --model segformer_lite --encoder segformer_b0 --pretrained 1
python -m crackseg.evaluate --config crackseg/config.yaml --weights runs/segformer_lite/best.pth --model segformer_lite --encoder segformer_b0
python -m crackseg.infer --config crackseg/config.yaml --weights runs/segformer_lite/best.pth --model segformer_lite --input "$DATA_ROOT/test" --save ./outputs/infer
```

More recipes: `COMMANDS.txt`.

## Layout

```
crackseg/                      package (data, models, train/eval/infer)
runs/                          training logs per model
docs/images/                   prediction samples + metric plots
docs/images/dataset_samples/   train/valid/test frames + COCO label overlays
```
