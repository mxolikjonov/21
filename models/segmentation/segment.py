"""
Inference script for binary histopathology segmentation.

Reads images from segmentation/testing/images/, runs the trained UNet++ model
with Test-Time Augmentation (TTA: original + 3 flips, averaged), and saves
binary PNG masks to the output folder (default: 21/).

Output masks:
  - Same filename as input image
  - Same spatial size as original image
  - PNG format, grayscale, values 0 (background) or 255 (foreground)

Usage:
    python models/segmentation/segment.py \
        --checkpoint models/segmentation/best_segmentation_model.ckpt \
        --input_dir  segmentation/testing/images \
        --output_dir 21 \
        --image_size 512 \
        --tta
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import HistoSegmenter


# ---------------------------------------------------------------------------
# Transforms for inference (no augmentation, only resize + normalise)
# ---------------------------------------------------------------------------

def _make_transform(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# TTA helpers — 4 deterministic flips, averaged in probability space
# ---------------------------------------------------------------------------

def _tta_predict(model: HistoSegmenter, tensor: torch.Tensor) -> torch.Tensor:
    """
    Run 4-fold TTA (original + H-flip + V-flip + HV-flip).
    Returns averaged sigmoid probability map, shape (1, H, W).
    """
    with torch.no_grad():
        p_orig = torch.sigmoid(model(tensor))
        p_hflip = torch.sigmoid(model(torch.flip(tensor, dims=[-1]))).flip(dims=[-1])
        p_vflip = torch.sigmoid(model(torch.flip(tensor, dims=[-2]))).flip(dims=[-2])
        p_hvflip = torch.sigmoid(
            model(torch.flip(tensor, dims=[-1, -2]))
        ).flip(dims=[-1, -2])

    return (p_orig + p_hflip + p_vflip + p_hvflip) / 4.0


# ---------------------------------------------------------------------------
# Single-image prediction
# ---------------------------------------------------------------------------

def predict_mask(
    model: HistoSegmenter,
    image_np: np.ndarray,
    transform: A.Compose,
    original_size: tuple[int, int],
    device: torch.device,
    threshold: float = 0.5,
    use_tta: bool = True,
) -> np.ndarray:
    """
    Args:
        image_np:      HxWx3 uint8 numpy array
        original_size: (W, H) in PIL convention

    Returns:
        Binary mask as HxW uint8 numpy array (0 or 255)
    """
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)  # (1,3,H,W)

    if use_tta:
        prob = _tta_predict(model, tensor)    # (1, 1, H, W)
    else:
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor))

    prob_np = prob.squeeze().cpu().numpy()    # (H, W)

    # Resize probability map back to original resolution before thresholding
    prob_pil = Image.fromarray((prob_np * 255).astype(np.uint8), mode="L")
    prob_pil = prob_pil.resize(original_size, resample=Image.BILINEAR)

    binary = (np.array(prob_pil) >= int(threshold * 255)).astype(np.uint8) * 255
    return binary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Segment histopathology images → binary masks")
    p.add_argument("--checkpoint", type=str,
                   default="models/segmentation/best_segmentation_model.ckpt",
                   help="Path to trained .ckpt file")
    p.add_argument("--input_dir", type=str,
                   default="segmentation/testing/images",
                   help="Folder with test PNG images (default: segmentation/testing/images)")
    p.add_argument("--output_dir", type=str, default="21",
                   help="Output folder for binary masks (default: 21)")
    p.add_argument("--image_size", type=int, default=512,
                   help="Model input resolution used during training")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binarization threshold on sigmoid output (default: 0.5)")
    p.add_argument("--tta", action="store_true", default=True,
                   help="Use Test-Time Augmentation (default: on)")
    p.add_argument("--no_tta", dest="tta", action="store_false",
                   help="Disable TTA")
    p.add_argument("--device", type=str, default="auto",
                   help="'cuda', 'mps', 'cpu', or 'auto'")
    return p.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()

    # ---------------------------------------------------------------- device
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # --------------------------------------------------------------- model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = HistoSegmenter.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval()
    model.to(device)
    print(f"Loaded checkpoint : {ckpt_path}")
    print(f"TTA enabled       : {args.tta}")

    # ------------------------------------------------------------ transform
    transform = _make_transform(args.image_size)

    # -------------------------------------------------------------- output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ inference
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = sorted(
        list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    print(f"Found {len(image_paths)} images → saving masks to '{output_dir}/'")

    for i, img_path in enumerate(image_paths, 1):
        image_pil = Image.open(img_path).convert("RGB")
        original_size = image_pil.size          # (W, H)
        image_np = np.array(image_pil, dtype=np.uint8)

        binary_mask = predict_mask(
            model, image_np, transform, original_size, device,
            threshold=args.threshold,
            use_tta=args.tta,
        )

        # Output filename matches input filename exactly, PNG format
        out_path = output_dir / (img_path.stem + ".png")
        Image.fromarray(binary_mask, mode="L").save(out_path)

        if i % 20 == 0 or i == len(image_paths):
            print(f"  [{i:>4}/{len(image_paths)}] {img_path.name} → {out_path.name}")

    print(f"\nDone. {len(image_paths)} masks saved to '{output_dir}/'")


if __name__ == "__main__":
    main()
