"""
Data pipeline for binary histopathology segmentation.

Expected folder structure:
    <data_root>/segmentation/train/images/    *.png  (RGB)
    <data_root>/segmentation/train/masks/     *.png  (grayscale, 0 or 255)
    <data_root>/segmentation/testing/images/  *.png  (RGB, inference only)

Mask filename must match image filename exactly.
"""

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def train_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Spatial distortions — important for tissue morphology variance
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.ElasticTransform(alpha=80, sigma=8, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.25),
        # Colour / intensity — histopathology staining variation
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.35),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15),
            A.RandomGamma(gamma_limit=(80, 120)),
        ], p=0.5),
        # Noise / blur
        A.GaussNoise(var_limit=(5, 40), p=0.25),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=3),
        ], p=0.2),
        # Occlusion — prevents over-reliance on local texture
        A.CoarseDropout(max_holes=6, max_height=32, max_width=32, fill_value=0, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def val_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """
    Paired image + binary mask dataset.
    Masks should be grayscale PNG (0 = background, 255 = foreground).
    """

    def __init__(self, image_dir: Path, mask_dir: Path, transform: A.Compose = None):
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_paths = sorted(
            list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image = out["image"]                   # (3, H, W) float32 tensor
            mask = out["mask"].unsqueeze(0)        # (1, H, W) float32 tensor

        return image, mask


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class SegmentationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for binary segmentation.

    Args:
        data_root:   path containing segmentation/train/images/ and masks/
        batch_size:  mini-batch size
        num_workers: DataLoader workers
        image_size:  spatial resize target
        val_split:   fraction used for validation
    """

    def __init__(
        self,
        data_root: str | Path = ".",
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: int = 512,
        val_split: float = 0.15,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split

        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        img_dir = self.data_root / "segmentation" / "train" / "images"
        msk_dir = self.data_root / "segmentation" / "train" / "masks"

        for d in (img_dir, msk_dir):
            if not d.exists():
                raise FileNotFoundError(f"Required directory not found: {d}")

        full_ds = SegmentationDataset(img_dir, msk_dir, transform=None)

        n_val = max(1, int(len(full_ds) * self.val_split))
        n_train = len(full_ds) - n_val

        generator = torch.Generator().manual_seed(42)
        train_sub, val_sub = torch.utils.data.random_split(
            full_ds, [n_train, n_val], generator=generator
        )

        self.train_ds = _TransformSubset(train_sub, train_transforms(self.image_size))
        self.val_ds = _TransformSubset(val_sub, val_transforms(self.image_size))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ---------------------------------------------------------------------------
# Helper: apply transforms lazily to a random_split Subset
# ---------------------------------------------------------------------------

class _TransformSubset(Dataset):
    def __init__(self, subset, transform: A.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, mask = self.subset[idx]
        # img and mask are numpy arrays from the underlying SegmentationDataset
        image_np = img if isinstance(img, np.ndarray) else np.array(img)
        mask_np = mask if isinstance(mask, np.ndarray) else np.array(mask)

        out = self.transform(image=image_np, mask=mask_np)
        return out["image"], out["mask"].unsqueeze(0)
