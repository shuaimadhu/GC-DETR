"""
CNN backbone for GC-DETR.

Provides a ResNet-based feature extractor that returns multi-scale feature maps
at three stages (C3, C4, C5), which feed into the detection neck.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as tv_models


class ResNetBackbone(nn.Module):
    """ResNet feature extractor.

    Returns feature maps from the last three stages (layer2, layer3, layer4)
    of the chosen ResNet variant, corresponding to strides 8, 16 and 32
    respectively.

    Args:
        name: ResNet variant.  One of ``"resnet18"``, ``"resnet34"``,
            ``"resnet50"``, ``"resnet101"``.
        pretrained: Whether to initialise from ImageNet pre-trained weights.
        freeze_bn: If *True* the batch-norm statistics are frozen after the
            first batch-norm layer (i.e. from ``layer1`` onwards).
        return_layers: Which stages to return.  Defaults to
            ``["layer2", "layer3", "layer4"]``.
    """

    # Output channels for each stage, keyed by backbone name.
    _out_channels: Dict[str, List[int]] = {
        "resnet18": [128, 256, 512],
        "resnet34": [128, 256, 512],
        "resnet50": [512, 1024, 2048],
        "resnet101": [512, 1024, 2048],
    }

    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = False,
        freeze_bn: bool = False,
        return_layers: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        assert name in self._out_channels, (
            f"Unsupported backbone '{name}'. "
            f"Choose from {list(self._out_channels.keys())}."
        )

        self.name = name
        self.return_layers = return_layers or ["layer2", "layer3", "layer4"]

        # Build the underlying torchvision ResNet.
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = getattr(tv_models, name)(weights=weights)

        # Keep only the layers we need.
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if freeze_bn:
            self._freeze_bn()

        # Expose output channel sizes.
        self.out_channels: List[int] = [
            self._out_channels[name][i]
            for i, lyr in enumerate(["layer2", "layer3", "layer4"])
            if lyr in self.return_layers
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _freeze_bn(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input image tensor with shape (B, 3, H, W).

        Returns:
            Ordered dictionary mapping layer name to feature tensor.
            Feature shapes: layer2 → (B, C2, H/8, W/8),
                            layer3 → (B, C3, H/16, W/16),
                            layer4 → (B, C4, H/32, W/32).
        """
        x = self.stem(x)
        x = self.layer1(x)

        features: Dict[str, Tensor] = {}
        x = self.layer2(x)
        if "layer2" in self.return_layers:
            features["layer2"] = x
        x = self.layer3(x)
        if "layer3" in self.return_layers:
            features["layer3"] = x
        x = self.layer4(x)
        if "layer4" in self.return_layers:
            features["layer4"] = x

        return features
