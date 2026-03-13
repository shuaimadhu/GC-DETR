"""
IoU Prediction Quality Head (IOUPQH) for GC-DETR.

IOUPQH tackles a well-known problem in object detection: the misalignment
between classification confidence and localisation quality.  A detector
can be highly confident about the *class* of a box while placing that box
poorly, or vice-versa.  This misalignment leads to sub-optimal Non-Maximum
Suppression (NMS) results, since NMS relies on classification scores to
rank proposals.

IOUPQH addresses this by:

1. Predicting a per-box IoU quality score in the range [0, 1] from the
   decoder query representation.
2. Re-weighting the raw classification logits by the predicted IoU so
   that the final score reflects *both* categorical certainty and
   localisation accuracy.

During training the IoU branch is supervised with the actual IoU between the
predicted box and its assigned ground-truth box, using a binary cross-entropy
loss over the sigmoid-activated IoU logit.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class IoUPredictionQualityHead(nn.Module):
    """IoU Prediction Quality Head (IOUPQH).

    Produces joint classification-quality scores by reweighting the
    classification logits with a predicted IoU quality score.

    Args:
        embed_dim: Dimensionality of the input decoder query features.
        num_classes: Number of object categories.
        hidden_dim: Hidden dimension of the MLP branches.
        num_layers: Number of hidden layers in each MLP.
        prior_prob: Prior probability for the classification focal initialisation
            (mimics RetinaNet-style bias init to stabilise early training).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        prior_prob: float = 0.01,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # ------------------------------------------------------------------
        # Classification branch
        # ------------------------------------------------------------------
        cls_layers: list = []
        in_dim = embed_dim
        for _ in range(num_layers - 1):
            cls_layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        cls_layers.append(nn.Linear(in_dim, num_classes))
        self.cls_head = nn.Sequential(*cls_layers)

        # ------------------------------------------------------------------
        # IoU quality branch
        # ------------------------------------------------------------------
        iou_layers: list = []
        in_dim = embed_dim
        for _ in range(num_layers - 1):
            iou_layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        iou_layers.append(nn.Linear(in_dim, 1))
        self.iou_head = nn.Sequential(*iou_layers)

        self._reset_parameters(prior_prob)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _reset_parameters(self, prior_prob: float) -> None:
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # Classification head
        for layer in self.cls_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        # Apply focal-style bias to the final classification layer.
        nn.init.constant_(self.cls_head[-1].bias, bias_value)

        # IoU head
        for layer in self.iou_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, query_features: Tensor) -> dict:
        """Compute joint classification-quality scores.

        Args:
            query_features: Decoder output features with shape (B, N, C).

        Returns:
            A dictionary containing:

            - ``"cls_logits"`` (Tensor): Raw classification logits of shape
              (B, N, num_classes).  These are the un-scaled per-class scores
              before IoU reweighting; useful for training the classification
              loss directly.
            - ``"iou_logits"`` (Tensor): IoU prediction logits of shape
              (B, N, 1).  Apply ``sigmoid`` to obtain the predicted IoU score
              in [0, 1].
            - ``"scores"`` (Tensor): Joint detection scores of shape
              (B, N, num_classes), computed as
              ``sigmoid(cls_logits) * sigmoid(iou_logits)``.
              This is the quantity used for NMS ranking at inference time.
        """
        cls_logits = self.cls_head(query_features)  # (B, N, num_classes)
        iou_logits = self.iou_head(query_features)  # (B, N, 1)

        # Reweight classification probability by predicted IoU quality.
        cls_prob = torch.sigmoid(cls_logits)
        iou_quality = torch.sigmoid(iou_logits)
        scores = cls_prob * iou_quality  # (B, N, num_classes)

        return {
            "cls_logits": cls_logits,
            "iou_logits": iou_logits,
            "scores": scores,
        }

    def loss(
        self,
        cls_logits: Tensor,
        iou_logits: Tensor,
        target_classes: Tensor,
        target_iou: Tensor,
        cls_weight: float = 1.0,
        iou_weight: float = 1.0,
    ) -> dict:
        """Compute combined classification and IoU prediction losses.

        Args:
            cls_logits: Classification logits of shape (N_pos + N_neg, num_classes).
            iou_logits: IoU prediction logits of shape (N_pos, 1).
            target_classes: One-hot target class labels of shape
                (N_pos + N_neg, num_classes).  For negative (background) samples
                all entries should be 0.
            target_iou: Ground-truth IoU values of shape (N_pos, 1) in [0, 1].
            cls_weight: Loss weight for the classification term.
            iou_weight: Loss weight for the IoU prediction term.

        Returns:
            Dictionary with keys:
            - ``"loss_cls"``: Binary cross-entropy classification loss.
            - ``"loss_iou_pred"``: Binary cross-entropy IoU prediction loss.
            - ``"loss_total"``: Weighted sum of the above.
        """
        loss_cls = F.binary_cross_entropy_with_logits(
            cls_logits,
            target_classes.to(cls_logits.dtype),
            reduction="mean",
        )

        loss_iou_pred = F.binary_cross_entropy_with_logits(
            iou_logits,
            target_iou.to(iou_logits.dtype),
            reduction="mean",
        )

        loss_total = cls_weight * loss_cls + iou_weight * loss_iou_pred

        return {
            "loss_cls": loss_cls,
            "loss_iou_pred": loss_iou_pred,
            "loss_total": loss_total,
        }
