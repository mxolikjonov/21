# 21 — Histopathology Classification

12-class H&E biopsy image classification using CNN (EfficientNet-B0) with PyTorch Lightning.

---

## Project Structure

```
21/
├── 21/                                  # test images for submission
│   ├── 1476.png
│   └── 1477.png
├── 21_test_ground_truth.xlsx            # generated predictions
├── classification/
│   └── train/
│       ├── 0/  ← class 0 images
│       ├── 1/
│       ...
│       └── 11/
└── models/
    └── classification/
        ├── data.py                      # data pipeline
        ├── model.py                     # CNN model
        ├── train.py                     # training script
        ├── classify.py                  # inference script
        ├── best_classification_model.ckpt
        └── requirements.txt
```

---

## 1. Install Dependencies

```bash
pip install torch torchvision pytorch-lightning timm monai torchmetrics numpy pillow pandas openpyxl
```

Or via requirements.txt:

```bash
pip install -r models/classification/requirements.txt
```

---

## 2. Train the Model

Make sure your training images are in `classification/train/{0..11}/` folders.

```bash
cd /Users/gulom/Desktop/21

python models/classification/train.py \
    --data_root . \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --image_size 256
```

After training, the best model is saved to:
```
models/classification/best_classification_model.ckpt
```

### Optional training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `.` | Root directory with `classification/train/` |
| `--epochs` | `50` | Max training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--lr` | `0.001` | Learning rate |
| `--image_size` | `256` | Resize target (px) |
| `--num_workers` | `4` | DataLoader workers |
| `--val_split` | `0.15` | Validation fraction |
| `--focal_gamma` | `2.0` | Focal Loss gamma |
| `--precision` | `16-mixed` | Training precision |

---

## 3. Run Inference

Place test images in the `21/` folder, then run:

```bash
python models/classification/classify.py \
    --input_dir 21/ \
    --model_path models/classification/best_classification_model.ckpt \
    --output 21_test_ground_truth.xlsx \
    --tta
```

This generates `21_test_ground_truth.xlsx` with columns:

| Image_ID | Label |
|----------|-------|
| 1476 | 3 |
| 1477 | 7 |

### Inference arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir` | required | Folder with test PNG images |
| `--model_path` | required | Path to `.ckpt` checkpoint |
| `--output` | `21_test_ground_truth.xlsx` | Output Excel file |
| `--image_size` | `256` | Must match training size |
| `--tta` | on | Test-Time Augmentation |
| `--no_tta` | — | Disable TTA |
| `--device` | `auto` | `cpu`, `cuda`, `mps`, or `auto` |

---

## 4. Troubleshooting

| Problem | Solution |
|---------|----------|
| `No images found` | Check that `classification/train/0/` contains `.png` files |
| `CUDA out of memory` | Lower `--batch_size` to `16` or `8` |
| MPS error (Mac) | Add `--device cpu` |
| `Module not found` | Run `pip install -r models/classification/requirements.txt` |
