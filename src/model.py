import os
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


def build_25d_efficientnet_b0(
    in_channels: int = 5, weights_path: Optional[str] = None
) -> nn.Sequential:
    """Adapts an ImageNet-pretrained EfficientNet-B0 stem to accept 5-slice 2.5D MRI stacks."""
    backbone = models.efficientnet_b0(weights=None)

    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state_dict)

    orig_conv = backbone.features[0][0]
    adapted_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=orig_conv.out_channels,
        kernel_size=orig_conv.kernel_size,
        stride=orig_conv.stride,
        padding=orig_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        adapted_conv.weight[:, :3, :, :] = orig_conv.weight
        if in_channels > 3:
            extra = in_channels - 3
            adapted_conv.weight[:, 3:, :, :] = orig_conv.weight[:, :extra, :, :]

    backbone.features[0][0] = adapted_conv
    return nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten())


class DualBranchKneeNet(nn.Module):
    """Late-fusion dual-plane (Sagittal + Coronal) 2.5D CNN for 12 knee pathologies."""

    def __init__(
        self,
        num_classes: int = 12,
        in_channels: int = 5,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.sag_branch = build_25d_efficientnet_b0(in_channels, weights_path)
        self.cor_branch = build_25d_efficientnet_b0(in_channels, weights_path)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280 * 2, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, sag: torch.Tensor, cor: torch.Tensor) -> torch.Tensor:
        feat_sag = self.sag_branch(sag)
        feat_cor = self.cor_branch(cor)
        fused = torch.cat([feat_sag, feat_cor], dim=1)
        return self.classifier(fused)
