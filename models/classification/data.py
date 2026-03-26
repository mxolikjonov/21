"""
Data pipeline for histopathology classification.
Training folder: classification/train/{0..11}/
"""

from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np

from monai.transforms import (
    Compose,
    RandRotate90,
    RandFlip,
    RandZoom,
    RandGaussianNoise,
    ScaleIntensity,
    ToTensor,
    Resize,
    NormalizeIntensity,
    RandAffine,
    RandAdjustContrast,
)
from torchvision import transforms as T


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HistoDataset(Dataset):
    """Reads PNG images from classification/train/{class_id}/ folders."""

    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples = []  # (path, label)

        for label_dir in sorted(root.iterdir()):
            if not label_dir.is_dir():
                continue
            try:
                label = int(label_dir.name)
            except ValueError:
                continue
            for img_path in label_dir.glob("*.png"):
                self.samples.append((img_path, label))
            for img_path in label_dir.glob("*.jpg"):
                self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Cannot open {path}: {e}")

        img = np.array(img, dtype=np.float32) / 255.0   # H x W x C  in [0,1]
        img = img.transpose(2, 0, 1)                    # C x H x W

        if self.transform:
            img = self.transform(img)

        return img, label

    # ------------------------------------------------------------------
    # Class weights for sampler / loss
    # ------------------------------------------------------------------
    def class_weights(self) -> torch.Tensor:
        labels = [s[1] for s in self.samples]
        counts = np.bincount(labels, minlength=12).astype(np.float32)
        counts = np.where(counts == 0, 1, counts)
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def sample_weights(self) -> list:
        cw = self.class_weights().numpy()
        return [float(cw[s[1]]) for s in self.samples]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def train_transforms(image_size: int = 256) -> Compose:
    return Compose([
        Resize(spatial_size=(image_size, image_size)),          # C x H x W
        RandRotate90(prob=0.5),
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandZoom(min_zoom=0.85, max_zoom=1.15, prob=0.4),
        RandAffine(
            prob=0.3,
            rotate_range=(0.2,),
            shear_range=(0.1,),
            padding_mode="reflection",
        ),
        RandAdjustContrast(prob=0.3, gamma=(0.7, 1.5)),
        RandGaussianNoise(prob=0.2, mean=0.0, std=0.05),
        NormalizeIntensity(nonzero=False, channel_wise=True),
        ToTensor(),
    ])


def val_transforms(image_size: int = 256) -> Compose:
    return Compose([
        Resize(spatial_size=(image_size, image_size)),
        NormalizeIntensity(nonzero=False, channel_wise=True),
        ToTensor(),
    ])


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class ClassificationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for 12-class histopathology classification.

    Args:
        data_root: path to the directory that contains 'classification/train/'
        batch_size: mini-batch size
        num_workers: DataLoader workers
        image_size: spatial resize target (default 256)
        val_split: fraction of training data used for validation
    """

    def __init__(
        self,
        data_root: str | Path = ".",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 256,
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

    # ------------------------------------------------------------------

    def setup(self, stage=None):
        train_root = self.data_root / "classification" / "train"
        if not train_root.exists():
            raise FileNotFoundError(f"Training folder not found: {train_root}")

        full_dataset = HistoDataset(train_root, transform=None)

        n_total = len(full_dataset)
        n_val = max(1, int(n_total * self.val_split))
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val], generator=generator
        )

        # Wrap subsets with proper transforms
        self.train_ds = _TransformSubset(train_subset, train_transforms(self.image_size))
        self.val_ds = _TransformSubset(val_subset, val_transforms(self.image_size))

        # Store class weights for loss function access
        self.class_weights = full_dataset.class_weights()

        # Build weighted sampler for training (handles imbalance)
        all_sw = full_dataset.sample_weights()
        train_sw = [all_sw[i] for i in train_subset.indices]
        self._sampler = WeightedRandomSampler(
            weights=train_sw,
            num_samples=len(train_sw),
            replacement=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=self._sampler,
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
# Helper: apply transforms lazily to a Subset
# ---------------------------------------------------------------------------

class _TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
