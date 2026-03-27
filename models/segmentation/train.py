"""
Training script for binary histopathology segmentation.

Data structure expected:
    <data_root>/segmentation/train/images/   *.png
    <data_root>/segmentation/train/masks/    *.png  (grayscale 0/255)

Each run saves logs to logs/segmentation/run/version_N/ automatically.

Usage:
    python models/segmentation/train.py --data_root . --epochs 50 --batch_size 16
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data import SegmentationDataModule
from model import HistoSegmenter


def parse_args():
    p = argparse.ArgumentParser(description="Train binary segmentation model")
    p.add_argument("--data_root", type=str, default=".",
                   help="Root dir with segmentation/train/images/ and masks/")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--encoder", type=str, default="efficientnet-b4")
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--bce_weight", type=float, default=0.5)
    p.add_argument("--checkpoint_dir", type=str, default="models/segmentation")
    p.add_argument("--precision", type=str, default="16-mixed")
    return p.parse_args()


def main():
    args = parse_args()

    dm = SegmentationDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_split=args.val_split,
    )
    dm.setup()

    model = HistoSegmenter(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        lr=args.lr,
        weight_decay=args.weight_decay,
        bce_weight=args.bce_weight,
        t_max=args.epochs,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best_segmentation_model",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    early_stop_cb = EarlyStopping(monitor="val_dice", patience=12, mode="max", verbose=True)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # New versioned log dir per run: logs/segmentation/run/version_0, version_1, …
    logger = TensorBoardLogger(
        save_dir="logs/segmentation",
        name="run",
        version="auto",
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

    print(f"\nBest model saved to : {checkpoint_cb.best_model_path}")
    print(f"Best val_dice       : {checkpoint_cb.best_model_score:.4f}")
    print(f"Logs saved to       : {logger.log_dir}")


if __name__ == "__main__":
    main()
