"""
Inference script for histopathology classification.

Reads PNG images from an input directory, runs the saved CNN model
(with optional Test-Time Augmentation), and writes predictions to
21_test_ground_truth.xlsx with columns: Image_ID, Label.

Usage:
    python models/classification/classify.py \
        --input_dir  21/ \
        --model_path models/classification/best_classification_model.ckpt \
        --output     21_test_ground_truth.xlsx \
        --tta
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd

from monai.transforms import Compose, Resize, NormalizeIntensity, ToTensor, RandRotate90, RandFlip
from model import HistoCNNClassifier


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def base_transform(image_size: int = 256) -> Compose:
    return Compose([
        Resize(spatial_size=(image_size, image_size)),
        NormalizeIntensity(nonzero=False, channel_wise=True),
        ToTensor(),
    ])


def tta_transforms(image_size: int = 256) -> list:
    """Return a list of augmented transforms for TTA (8 versions)."""
    base = [
        Resize(spatial_size=(image_size, image_size)),
        NormalizeIntensity(nonzero=False, channel_wise=True),
    ]
    variants = []
    for flip_h in [False, True]:
        for flip_v in [False, True]:
            ops = list(base)
            if flip_h:
                ops.append(RandFlip(spatial_axis=0, prob=1.0))
            if flip_v:
                ops.append(RandFlip(spatial_axis=1, prob=1.0))
            ops.append(ToTensor())
            variants.append(Compose(ops))
    return variants


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load PNG/JPG as float32 C x H x W array in [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0   # H x W x C
    return arr.transpose(2, 0, 1)                    # C x H x W


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_single(
    model: HistoCNNClassifier,
    img_arr: np.ndarray,
    device: torch.device,
    image_size: int = 256,
    use_tta: bool = True,
) -> int:
    """Return predicted class index (0-11) for one image."""
    model.eval()

    if use_tta:
        transforms = tta_transforms(image_size)
    else:
        transforms = [base_transform(image_size)]

    probs_list = []
    for t in transforms:
        tensor = t(img_arr)                              # C x H x W
        tensor = tensor.unsqueeze(0).to(device)          # 1 x C x H x W
        logits = model(tensor)                           # 1 x 12
        probs = F.softmax(logits, dim=-1).cpu().numpy()  # 1 x 12
        probs_list.append(probs)

    avg_probs = np.mean(probs_list, axis=0)              # 1 x 12
    return int(np.argmax(avg_probs, axis=-1)[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Classify H&E biopsy images")
    p.add_argument("--input_dir", type=str, required=True,
                   help="Directory containing test PNG images")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to saved .ckpt checkpoint")
    p.add_argument("--output", type=str, default="21_test_ground_truth.xlsx",
                   help="Output Excel file path")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--tta", action="store_true", default=True,
                   help="Use Test-Time Augmentation (default: on)")
    p.add_argument("--no_tta", dest="tta", action="store_false",
                   help="Disable TTA")
    p.add_argument("--device", type=str, default="auto",
                   help="'cpu', 'cuda', 'mps', or 'auto'")
    return p.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # ---- load model ----
    model = HistoCNNClassifier.load_from_checkpoint(str(model_path), map_location=device)
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")

    # ---- collect images ----
    image_paths = sorted(
        list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    )
    if not image_paths:
        raise RuntimeError(f"No PNG/JPG images found in {input_dir}")
    print(f"Found {len(image_paths)} images")

    # ---- inference ----
    records = []
    for img_path in image_paths:
        try:
            arr = load_image(img_path)
            label = predict_single(
                model, arr, device,
                image_size=args.image_size,
                use_tta=args.tta,
            )
            image_id = img_path.stem          # filename without extension
            records.append({"Image_ID": image_id, "Label": label})
            print(f"  {image_id:>12s}  →  class {label}")
        except Exception as e:
            print(f"  WARNING: skipping {img_path.name}: {e}")

    # ---- save Excel ----
    df = pd.DataFrame(records, columns=["Image_ID", "Label"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(str(output_path), index=False)
    print(f"\nPredictions saved to: {output_path}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
