"""
Cross-Slice Transformer UNet (CST)
Reproduced from: "Cross-Slice Transformer with Adaptive Slice Fusion for Lymph Node Segmentation in CT Imaging"
IEEE Access 2026.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

VALID_VARIANTS = ("full", "no_interaction", "mlp_fusion", "fixed_weights")


class ContextualSliceTransformer(nn.Module):
    """
    Computes per-slice importance weights from slice statistical descriptors
    via a lightweight Transformer encoder, dynamically fusing 2.5D slice context.
    """
    def __init__(
        self,
        num_slices: int = 6,
        hidden_dim: int = 128,
        transformer_layers: int = 3,
        num_heads: int = 8,
        out_channels: int = 64,
        variant: str = "full"
    ):
        super().__init__()
        assert variant in VALID_VARIANTS
        self.num_slices = num_slices
        self.variant = variant

        n_desc = 3 if variant == "no_interaction" else 4

        if variant != "fixed_weights":
            self.token_embed = nn.Sequential(
                nn.Linear(n_desc, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )

        if variant in ("full", "no_interaction"):
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
            self.slice_weights = nn.Linear(hidden_dim, 1)
        elif variant == "mlp_fusion":
            self.mlp_weights = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

        self.context_conv = nn.Sequential(
            nn.Conv2d(num_slices + 2, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.num_slices, f"Expected {self.num_slices} channels, got {c}"

        mean = x.mean(dim=(2, 3))
        std = x.view(b, c, -1).std(dim=-1)
        diff = torch.zeros(b, c, device=x.device)
        if c > 1:
            adj = (x[:, 1:] - x[:, :-1]).abs().mean(dim=(2, 3))
            diff[:, 1:] = adj
            diff[:, 0] = adj[:, 0]

        if self.variant == "fixed_weights":
            weights = torch.full((b, c), 1.0 / c, device=x.device)
        elif self.variant == "mlp_fusion":
            descs = torch.stack([mean, std + 1e-6, diff, mean * std], dim=-1)
            tokens = self.token_embed(descs)
            weights = torch.softmax(self.mlp_weights(tokens).squeeze(-1), dim=1)
        elif self.variant == "no_interaction":
            descs = torch.stack([mean, std + 1e-6, diff], dim=-1)
            tokens = self.transformer(self.token_embed(descs))
            weights = torch.softmax(self.slice_weights(tokens).squeeze(-1), dim=1)
        else:
            descs = torch.stack([mean, std + 1e-6, diff, mean * std], dim=-1)
            tokens = self.transformer(self.token_embed(descs))
            weights = torch.softmax(self.slice_weights(tokens).squeeze(-1), dim=1)

        fused = (x * weights.view(b, c, 1, 1)).sum(dim=1, keepdim=True)
        diff_map = (
            (x[:, 1:] - x[:, :-1]).abs().mean(dim=1, keepdim=True)
            if c > 1
            else torch.zeros(b, 1, h, w, device=x.device)
        )
        features = self.context_conv(torch.cat([x, fused, diff_map], dim=1))

        diff_energy = diff / (diff.sum(dim=1, keepdim=True) + 1e-6)
        return features, weights, diff_energy


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with GroupNorm and ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downsample: MaxPool2d(2) followed by DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample, concatenation of skip connection, followed by DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class CST(nn.Module):
    """
    Full Cross-Slice Transformer U-Net architecture.
    Input: [B, num_slices, H, W] (e.g. 5 axial CT slices + 1 anatomy prior = 6 channels).
    Output: (logits, ds1, ds2, weights, diff_energy)
    """
    def __init__(
        self,
        num_slices: int = 6,
        base_channels: int = 64,
        hidden_dim: int = 128,
        transformer_layers: int = 3,
        num_heads: int = 8,
        variant: str = "full"
    ):
        super().__init__()
        C = base_channels
        self.slice_encoder = ContextualSliceTransformer(
            num_slices=num_slices,
            hidden_dim=hidden_dim,
            transformer_layers=transformer_layers,
            num_heads=num_heads,
            out_channels=C,
            variant=variant,
        )
        self.inc = DoubleConv(C, C)
        self.down1 = Down(C, C * 2)
        self.down2 = Down(C * 2, C * 4)
        self.down3 = Down(C * 4, C * 8)
        self.up1 = Up(C * 8, C * 4)
        self.up2 = Up(C * 4, C * 2)
        self.up3 = Up(C * 2, C)
        self.out_conv = nn.Conv2d(C, 1, kernel_size=1)
        self.ds_head1 = nn.Conv2d(C * 2, 1, kernel_size=1)
        self.ds_head2 = nn.Conv2d(C * 4, 1, kernel_size=1)

    def forward(self, x):
        fused, weights, diff_energy = self.slice_encoder(x)
        x1 = self.inc(fused)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        logits = self.out_conv(d3)
        ds1 = self.ds_head1(d2)
        ds2 = self.ds_head2(d1)
        return (logits, ds1, ds2, weights, diff_energy)

