"""
Evaluation script for histopathology classification.

Loads the trained model checkpoint, runs inference on the validation split
(same 15% held-out split as training), and prints a full classification report
with per-class precision, recall, F1-score, and a confusion matrix.

Usage:
    python models/classification/evaluate.py \
        --data_root . \
        --model_path models/classification/best_classification_model.ckpt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ClassificationDataModule, val_transforms
from model import HistoCNNClassifier

# sklearn for metrics
try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
except ImportError:
    raise ImportError("Run: pip install scikit-learn")


CLASS_NAMES = [str(i) for i in range(12)]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate classification model on validation set")
    p.add_argument("--data_root", type=str, default=".",
                   help="Root dir that contains classification/train/{0..11}/")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to saved .ckpt checkpoint")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.15,
                   help="Must match val_split used during training (default: 0.15)")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


@torch.no_grad()
def run_inference(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch_idx, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")

    return np.array(all_labels), np.array(all_preds)


def print_confusion_matrix(cm, class_names):
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    header = "       " + "  ".join(f"{c:>4}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>4}" for v in row)
        print(f"  [{class_names[i]:>2}]  {row_str}")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # ---- load model ----
    model = HistoCNNClassifier.load_from_checkpoint(args.model_path, map_location=device)
    model.to(device)
    model.eval()
    print(f"Loaded model: {args.model_path}")

    # ---- build val split (same seed=42 as training) ----
    dm = ClassificationDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_split=args.val_split,
    )
    dm.setup()

    val_loader = dm.val_dataloader()
    print(f"Validation set size: {len(dm.val_ds)} images")

    # ---- inference ----
    print("\nRunning inference on validation set...")
    y_true, y_pred = run_inference(model, val_loader, device)

    # ---- metrics ----
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print(f"  Overall Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Macro          : {f1_macro:.4f}")
    print(f"  F1 Weighted       : {f1_weighted:.4f}")
    print("=" * 60)

    # ---- per-class report ----
    present_classes = sorted(set(y_true.tolist()))
    present_names = [CLASS_NAMES[i] for i in present_classes]

    print("\nPer-Class Report:")
    print(classification_report(
        y_true, y_pred,
        labels=present_classes,
        target_names=[f"class_{n}" for n in present_names],
        zero_division=0,
    ))

    # ---- confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    print_confusion_matrix(cm, present_names)

    # ---- worst classes ----
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    print("\nPer-class accuracy (sorted worst → best):")
    sorted_idx = np.argsort(per_class_acc)
    for idx in sorted_idx:
        cls = present_classes[idx]
        n = cm.sum(axis=1)[idx]
        print(f"  class {cls:>2}: {per_class_acc[idx]*100:5.1f}%  ({int(cm.diagonal()[idx])}/{int(n)} correct)")


if __name__ == "__main__":
    main()
