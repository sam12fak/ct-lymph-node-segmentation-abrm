"""
Auxiliary Boundary Refinement Module (ABRM) and ABRM-CST Architecture.

Research Motivation:
Small lymph nodes have ambiguous boundaries and low contrast against surrounding mediastinal
soft tissue and vasculature. Although boundary-aware loss functions (e.g. BoundaryLoss) help
penalize distant errors, they lack an explicit boundary representation.

ABRM introduces:
1. An explicit boundary prediction branch directly coupled with the high-resolution decoder features.
2. A boundary-guided refinement module that conditions final segmentation logits on both
   spatial semantic features and predicted boundary confidence.
3. Residual formulation ensuring baseline feature retention while sharpening topological boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cst import CST, DoubleConv, Down, Up


def extract_boundary_gt(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Extracts a 1-voxel/2-voxel boundary contour band from binary segmentation mask
    using morphological gradient (Dilation - Erosion) via GPU/CPU max pooling.

    Args:
        mask: Tensor of shape [B, 1, H, W] with values in {0, 1}
        kernel_size: Size of structuring element (default 3)

    Returns:
        Tensor of shape [B, 1, H, W] representing the boundary contour band.
    """
    pad = kernel_size // 2
    dilation = F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad)
    erosion = 1.0 - F.max_pool2d(1.0 - mask.float(), kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (dilation - erosion).clamp(0.0, 1.0)
    return boundary


class BoundaryHead(nn.Module):
    """
    Lightweight convolutional boundary prediction head.
    Predicts a 1-channel boundary probability map from decoder features.
    """
    def __init__(self, in_channels: int = 64, mid_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, x):
        feat = self.conv1(x)
        boundary_logits = self.out_conv(feat)
        return boundary_logits, feat


class BoundaryGuidedRefinement(nn.Module):
    """
    Refines initial segmentation logits using predicted boundary features and probabilities.
    Employs a residual formulation: final_logits = initial_logits + delta.
    """
    def __init__(self, feat_channels: int = 64, boundary_feat_channels: int = 32, mid_channels: int = 32):
        super().__init__()
        in_ch = feat_channels + 2 + boundary_feat_channels  # feat + initial_prob + boundary_prob + boundary_feat
        self.refine_block = nn.Sequential(
            nn.Conv2d(in_ch, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, mid_channels), mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, dec_feat, initial_logits, boundary_logits, boundary_feat):
        initial_prob = torch.sigmoid(initial_logits)
        boundary_prob = torch.sigmoid(boundary_logits)
        
        combined = torch.cat([dec_feat, initial_prob, boundary_prob, boundary_feat], dim=1)
        residual_delta = self.refine_block(combined)
        final_logits = initial_logits + residual_delta
        return final_logits


class ABRMCST(nn.Module):
    """
    CST with Auxiliary Boundary Refinement Module (ABRM).
    Preserves original CST backbone and adds:
    1. Auxiliary boundary prediction head.
    2. Boundary-guided refinement module.

    Outputs:
        final_logits: Refined segmentation logits [B, 1, H, W]
        initial_logits: Baseline segmentation logits [B, 1, H, W]
        boundary_logits: Predicted boundary logits [B, 1, H, W]
        ds1: Deep supervision head 1 [B, 1, H/2, W/2]
        ds2: Deep supervision head 2 [B, 1, H/4, W/4]
        weights: CST slice weights [B, num_slices]
        diff_energy: Slice difference energy [B, num_slices]
    """
    def __init__(
        self,
        num_slices: int = 6,
        base_channels: int = 64,
        hidden_dim: int = 128,
        transformer_layers: int = 3,
        num_heads: int = 8,
        variant: str = "full",
        enable_refinement: bool = True
    ):
        super().__init__()
        self.cst = CST(
            num_slices=num_slices,
            base_channels=base_channels,
            hidden_dim=hidden_dim,
            transformer_layers=transformer_layers,
            num_heads=num_heads,
            variant=variant,
        )
        self.enable_refinement = enable_refinement
        C = base_channels
        self.boundary_head = BoundaryHead(in_channels=C, mid_channels=C // 2)

        if self.enable_refinement:
            self.refinement_module = BoundaryGuidedRefinement(
                feat_channels=C,
                boundary_feat_channels=C // 2,
                mid_channels=C // 2
            )

    def forward(self, x):
        # Forward pass through CST encoder and decoder
        fused, weights, diff_energy = self.cst.slice_encoder(x)
        x1 = self.cst.inc(fused)
        x2 = self.cst.down1(x1)
        x3 = self.cst.down2(x2)
        x4 = self.cst.down3(x3)
        d1 = self.cst.up1(x4, x3)
        d2 = self.cst.up2(d1, x2)
        d3 = self.cst.up3(d2, x1)

        initial_logits = self.cst.out_conv(d3)
        ds1 = self.cst.ds_head1(d2)
        ds2 = self.cst.ds_head2(d1)

        # Boundary head prediction
        boundary_logits, boundary_feat = self.boundary_head(d3)

        if self.enable_refinement:
            final_logits = self.refinement_module(d3, initial_logits, boundary_logits, boundary_feat)
        else:
            final_logits = initial_logits

        return (final_logits, initial_logits, boundary_logits, ds1, ds2, weights, diff_energy)

