"""
Custom loss functions for deepfake classification.

Three loss options are provided to address the Real ↔ AI Edited class confusion:

1. WeightedCELoss   — Standard CrossEntropy with per-class weights.
                      Up-weighting Real and AI Edited makes the gradient signal
                      for those classes proportionally larger, so the optimizer
                      invests more capacity in the hard boundary.

2. FocalLoss        — Down-weights easy, confidently-correct examples so the
                      optimizer focuses almost entirely on ambiguous ones.
                      A 95%-confident correct prediction contributes only
                      (1-0.95)^2 = 0.0025× its normal loss — effectively ignored.
                      Hard misclassifications contribute full loss or more.

3. WeightedFocalLoss — Combines both: per-class weights AND focal scaling.
                       This is the recommended default: weights fix the class
                       imbalance in gradient magnitude, focal fixes the easy/hard
                       imbalance within each class.

Class index convention (must match DeepfakeDataset):
    0 = Real
    1 = AI Generated
    2 = AI Edited

Default weights [1.5, 1.0, 1.5]:
    - Real and AI Edited get 1.5× gradient — they share the hardest decision boundary
    - AI Generated gets 1.0× — already achieves 0.92 F1, doesn't need extra emphasis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# Default class weights: Real=1.5, AI Gen=1.0, AI Edited=1.5
DEFAULT_WEIGHTS = torch.tensor([1.5, 1.0, 1.5], dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017 — "Focal Loss for Dense Object Detection").

    L_focal = -(1 - p_t)^gamma * log(p_t)

    where p_t is the model's probability for the correct class and gamma is the
    focusing parameter.  As gamma increases, the loss increasingly down-weights
    well-classified examples.

    gamma=0  → equivalent to standard CrossEntropyLoss
    gamma=1  → linear down-weighting of easy examples
    gamma=3  → (default) stronger (cubic) down-weighting — more aggressive focusing

    Args:
        gamma:   Focusing parameter (0.0 = standard CE, 2.0 recommended).
        weight:  Per-class weights tensor of shape [num_classes].
                 Applied before focal scaling — same semantics as
                 nn.CrossEntropyLoss(weight=...).
        reduction: 'mean' (default) | 'sum' | 'none'
        label_smoothing: Smoothing factor in [0, 1).  Prevents overconfident
                         predictions by spreading probability mass to wrong classes.
                         0.1 is a common default.
    """

    def __init__(
        self,
        gamma: float = 3.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C] raw (unnormalised) class scores
            targets: [B]    integer class indices

        Returns:
            Scalar loss value.
        """
        # Compute per-sample CE loss (no reduction yet)
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        # p_t = exp(-CE) is the model's probability assigned to the correct class
        # (for the unweighted case; approximation holds with weights/smoothing too)
        pt = torch.exp(-ce)

        # Focal scaling: (1 - pt)^gamma suppresses easy examples
        focal_loss = (1.0 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_criterion(loss_type: str, device: str, label_smoothing: float = 0.0,
                    class_weights: list = None, gamma: float = 3.0) -> nn.Module:
    """
    Factory function — returns the right loss module for the given loss_type string.

    Args:
        loss_type: One of:
                'ce'             — Standard CrossEntropyLoss (original baseline)
                'weighted'       — CrossEntropyLoss with class weights [1.5, 1.0, 1.5]
                'focal'          — FocalLoss(gamma=2) no class weights
                'weighted_focal' — FocalLoss(gamma=2) + class weights [1.5, 1.0, 1.5]
        device:    'cuda' or 'cpu' — weights tensor is moved to device.
        label_smoothing: Applied to all loss types that support it.
        class_weights: Optional list of 3 floats [w_real, w_ai_gen, w_ai_edit].
                    Overrides DEFAULT_WEIGHTS when provided.

    Returns:
        nn.Module loss callable: loss = criterion(logits, labels)
    """
    if class_weights is not None:
        weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        weights = DEFAULT_WEIGHTS.to(device)

    if loss_type == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    elif loss_type == "weighted":
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    elif loss_type == "focal":
        return FocalLoss(gamma=gamma, label_smoothing=label_smoothing)

    elif loss_type == "weighted_focal":
        return FocalLoss(gamma=gamma, weight=weights, label_smoothing=label_smoothing)

    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. "
            "Choose from: ce, weighted, focal, weighted_focal"
        )
