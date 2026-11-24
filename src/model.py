import os
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class SliceAttentionGate(nn.Module):
    """Channel-wise Squeeze-and-Excitation gate across the 2.5D slice stack."""

    def __init__(self, num_slices: int = 5) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_slices, num_slices * 2),
            nn.SiLU(),
            nn.Linear(num_slices * 2, num_slices),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.gate(x).unsqueeze(-1).unsqueeze(-1)
        return x * weights


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


class CrossPlaneGatedFusion(nn.Module):
    """Bidirectional cross-plane attention gate between Sagittal and Coronal embeddings."""

    def __init__(self, feat_dim: int = 1280) -> None:
        super().__init__()
        self.sag_to_cor = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.Sigmoid())
        self.cor_to_sag = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.Sigmoid())

    def forward(self, feat_sag: torch.Tensor, feat_cor: torch.Tensor) -> torch.Tensor:
        gated_sag = feat_sag * self.cor_to_sag(feat_cor)
        gated_cor = feat_cor * self.sag_to_cor(feat_sag)
        return torch.cat([feat_sag + gated_sag, feat_cor + gated_cor], dim=1)


class DualBranchKneeNet(nn.Module):
    """Slice-attended dual-plane (Sagittal + Coronal) 2.5D CNN with cross-plane gated fusion."""

    def __init__(
        self,
        num_classes: int = 12,
        in_channels: int = 5,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.sag_slice_attn = SliceAttentionGate(in_channels)
        self.cor_slice_attn = SliceAttentionGate(in_channels)

        self.sag_branch = build_25d_efficientnet_b0(in_channels, weights_path)
        self.cor_branch = build_25d_efficientnet_b0(in_channels, weights_path)
        self.fusion = CrossPlaneGatedFusion(feat_dim=1280)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280 * 2, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, sag: torch.Tensor, cor: torch.Tensor) -> torch.Tensor:
        feat_sag = self.sag_branch(self.sag_slice_attn(sag))
        feat_cor = self.cor_branch(self.cor_slice_attn(cor))
        fused = self.fusion(feat_sag, feat_cor)
        return self.classifier(fused)
