"""
CNN classification model for 12-class histopathology images.
Backbone: EfficientNet-B0 (timm) with Focal Loss + CosineAnnealingLR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from torchmetrics import Accuracy


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        num_classes: int = 12,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha          # shape (num_classes,) or None
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), targets: (B,)
        log_p = F.log_softmax(logits, dim=-1)               # (B, C)
        p = torch.exp(log_p)                                 # (B, C)

        # Gather probabilities of the true class
        log_pt = log_p.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
        pt = p.gather(1, targets.view(-1, 1)).squeeze(1)          # (B,)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt                              # (B,)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]       # (B,)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# CNN Model (EfficientNet-B0 backbone)
# ---------------------------------------------------------------------------

class HistoCNNClassifier(pl.LightningModule):
    """
    Transfer-learning CNN for 12-class H&E biopsy classification.

    Args:
        num_classes:   number of output classes (12)
        lr:            initial learning rate
        weight_decay:  L2 regularisation
        class_weights: 1-D tensor of per-class inverse-frequency weights
                       (passed in from DataModule after setup)
        focal_gamma:   gamma parameter for Focal Loss
        t_max:         CosineAnnealingLR period (in epochs)
    """

    def __init__(
        self,
        num_classes: int = 12,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor | None = None,
        focal_gamma: float = 2.0,
        t_max: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        # ---- backbone ----
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,           # remove classifier head
            global_pool="avg",
        )
        in_features = self.backbone.num_features  # 1280 for eff-b0

        # ---- custom head ----
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

        # ---- loss ----
        alpha = class_weights  # may be None
        self.criterion = FocalLoss(
            num_classes=num_classes,
            gamma=focal_gamma,
            alpha=alpha,
        )

        # ---- metrics ----
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)

        self.train_acc(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)

        self.val_acc(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_acc, on_epoch=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.t_max,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
