#!/usr/bin/env python3
"""
Classification inference script.

Usage:
    python classify.py <image_dir> [--cuda]

Arguments:
    image_dir    Directory containing test images (PNG/JPG)
    --cuda       Use GPU for inference if available (optional)

Output:
    21_test_ground_truth.xlsx in the root project directory
    Columns: Image_ID, Label
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

# Root project directory: two levels up from this script
# models/classification/classify.py -> root = /Users/.../21
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MODEL_DIR = os.path.join(SCRIPT_DIR, "21")  # saved model package

sys.path.insert(0, MODEL_DIR)
import torchxrayvision as xrv

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OUTPUT_FILENAME = "21_test_ground_truth.xlsx"


def load_model(weights: str = "densenet121-res224-all") -> torch.nn.Module:
    model = xrv.models.get_model(weights)
    model.eval()
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    img = xrv.utils.load_image(image_path)
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    img = transform(img)
    return torch.from_numpy(img).unsqueeze(0)


def get_label(probs: np.ndarray, pathologies: list) -> str:
    valid = [(p, float(s)) for p, s in zip(pathologies, probs) if p]
    if not valid:
        return "Normal"
    return max(valid, key=lambda x: x[1])[0]


def run_inference(image_dir: str, model: torch.nn.Module, device: torch.device) -> list:
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        print(f"No supported images (.png/.jpg/.jpeg) found in: {image_dir}")
        sys.exit(1)

    model = model.to(device)
    pathologies = getattr(model, "pathologies", xrv.datasets.default_pathologies)

    results = []
    for filename in tqdm(image_files, desc="Classifying"):
        image_path = os.path.join(image_dir, filename)
        try:
            img_tensor = preprocess_image(image_path).to(device)
            with torch.no_grad():
                preds = model(img_tensor).cpu().numpy()[0]
            label = get_label(preds, pathologies)
            image_id = os.path.splitext(filename)[0]
            results.append({"Image_ID": image_id, "Label": label})
        except Exception as e:
            print(f"  Warning: could not process '{filename}': {e}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Chest X-ray classification inference.")
    parser.add_argument(
        "image_dir",
        type=str,
        help="Directory containing test images (PNG/JPG).",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Use GPU if available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"Error: image directory not found: {args.image_dir}")
        sys.exit(1)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Images : {args.image_dir}")

    print("Loading model (densenet121-res224-all) ...")
    model = load_model("densenet121-res224-all")

    results = run_inference(args.image_dir, model, device)

    output_path = os.path.join(ROOT_DIR, OUTPUT_FILENAME)
    df = pd.DataFrame(results, columns=["Image_ID", "Label"])
    df.to_excel(output_path, index=False)

    print(f"\nSaved {len(df)} predictions -> {output_path}")


if __name__ == "__main__":
    main()
