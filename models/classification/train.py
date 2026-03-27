"""
Training script for histopathology classification.

Usage:
    python models/classification/train.py --data_root . --epochs 50 --batch_size 32 --lr 3e-4
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data import ClassificationDataModule
from model import HistoCNNClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Train 12-class histopathology CNN")
    p.add_argument("--data_root", type=str, default=".",
                   help="Root dir that contains classification/train/{0..11}/")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="LR per GPU. With 2 GPUs effective LR = lr * num_gpus (auto-scaled in trainer)")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--checkpoint_dir", type=str, default="models/classification",
                   help="Where to save the best model checkpoint")
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--drop_path_rate", type=float, default=0.2,
                   help="Stochastic depth rate for ConvNeXt")
    p.add_argument("--precision", type=str, default="16-mixed",
                   help="Trainer precision: '32', '16-mixed', 'bf16-mixed'")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ data
    dm = ClassificationDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_split=args.val_split,
    )
    dm.setup()

    # Scale LR linearly with number of GPUs (linear scaling rule)
    import torch
    num_gpus = max(1, torch.cuda.device_count())
    effective_lr = args.lr * num_gpus

    # ----------------------------------------------------------------- model
    model = HistoCNNClassifier(
        num_classes=12,
        lr=effective_lr,
        weight_decay=args.weight_decay,
        class_weights=dm.class_weights,
        focal_gamma=args.focal_gamma,
        t_max=args.epochs,
        drop_path_rate=args.drop_path_rate,
    )

    # --------------------------------------------------------------- callbacks
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best_classification_model",
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        verbose=True,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        mode="max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # --------------------------------------------------------------- trainer
    logger = TensorBoardLogger(
        save_dir="logs/classification",
        name="run",
        version="auto",          # auto → version_0, version_1, version_2 …
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        logger=logger,
        precision=args.precision,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)

    print(f"\nBest model saved to: {checkpoint_cb.best_model_path}")
    print(f"Best val_accuracy  : {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
