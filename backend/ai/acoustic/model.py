"""Acoustic deepfake detection CNN model architecture.

Per 05 §2.3:
  - Backbone: efficientnet_b0 (timm), ImageNet-pretrained, in_chans=3
  - Custom binary classification head (Class 0 = bona fide, Class 1 = spoof)
  - Exposed gradcam_target_layer (conv_head for EfficientNet-B0; layer4[-1] for ResNet-18)
  - Swappable fallback to resnet18 via config

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class AcousticDeepfakeCNN(nn.Module):
    """Binary classifier over log-mel spectrograms.

    Class 0 = bona fide, Class 1 = spoof.
    Input shape: (B, 3, 128, 400)
    Output shape: (B, 2) logits
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
        )
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Spectrogram tensor of shape (B, 3, 128, 400).

        Returns:
            Logits of shape (B, 2).
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Final convolutional block before pooling for Grad-CAM explainability."""
        if "efficientnet" in self.backbone_name:
            # EfficientNet conv_head
            if hasattr(self.backbone, "conv_head"):
                return self.backbone.conv_head
            raise AttributeError("EfficientNet backbone has no conv_head attribute")
        elif "resnet" in self.backbone_name:
            # ResNet layer4[-1]
            if hasattr(self.backbone, "layer4"):
                return self.backbone.layer4[-1]
            raise AttributeError("ResNet backbone has no layer4 attribute")
        else:
            # Generic fallback: find the last Conv2d layer in backbone
            for module in reversed(list(self.backbone.modules())):
                if isinstance(module, nn.Conv2d):
                    return module
            raise AttributeError(f"Could not locate target conv layer for backbone {self.backbone_name}")
