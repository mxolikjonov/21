"""
Binary segmentation model.
Architecture: UNet++ with pretrained EfficientNet-B4 encoder.
Loss: 0.5 * BCE + 0.5 * Dice (combined).
Scheduler: CosineAnnealingWarmRestarts for stable LR across full training.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics import Dice, JaccardIndex


# ---------------------------------------------------------------------------
# Combined BCE + Dice Loss
# ---------------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice loss.
    Works on raw logits (no sigmoid needed before calling).
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        flat_p = probs.view(probs.size(0), -1)
        flat_t = targets.view(targets.size(0), -1)
        intersection = (flat_p * flat_t).sum(dim=1)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            flat_p.sum(dim=1) + flat_t.sum(dim=1) + self.smooth
        )

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss.mean()


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class HistoSegmenter(pl.LightningModule):
    """
    UNet++ with EfficientNet-B4 encoder for binary histopathology segmentation.

    Args:
        encoder_name:    SMP encoder (default: 'efficientnet-b4')
        encoder_weights: pretrained weights ('imagenet')
        lr:              initial learning rate
        weight_decay:    L2 regularisation
        bce_weight:      BCE weight in combined loss (rest → Dice)
        t_max:           CosineAnnealingWarmRestarts full period (epochs)
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        bce_weight: float = 0.5,
        t_max: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type="scse",
        )

        self.criterion = BCEDiceLoss(bce_weight=bce_weight)

        self.train_dice = Dice(threshold=0.5)
        self.val_dice = Dice(threshold=0.5)
        self.val_iou = JaccardIndex(task="binary", threshold=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = (torch.sigmoid(logits) > 0.5).long()
        self.train_dice(preds, masks.long())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):  # noqa: ARG002
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = (torch.sigmoid(logits) > 0.5).long()
        self.val_dice(preds, masks.long())
        self.val_iou(preds, masks.long())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_dice", self.val_dice, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_iou", self.val_iou, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, self.hparams.t_max // 2),
            T_mult=1,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
