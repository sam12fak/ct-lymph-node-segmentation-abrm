"""
Original Loss Formulation for Cross-Slice Transformer (CST)
Reproduced from: "Cross-Slice Transformer with Adaptive Slice Fusion for Lymph Node Segmentation in CT Imaging"
IEEE Access 2026.

L_original = 0.40 * Tversky + 0.30 * Soft Dice + 0.20 * Boundary Loss + 0.10 * Focal Loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


class SoftDiceLoss(nn.Module):
    """Differentiable Soft Dice on sigmoid probability outputs."""
    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        vm = valid_mask.float() if valid_mask is not None else torch.ones_like(prob)
        prob_m = prob * vm
        gt_m = target.float() * vm
        inter = (prob_m * gt_m).sum(dim=(1, 2, 3))
        denom = prob_m.sum(dim=(1, 2, 3)) + gt_m.sum(dim=(1, 2, 3)) + self.eps
        dice = (2.0 * inter + self.eps) / denom
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Alpha-balanced Focal Loss."""
    def __init__(self, gamma: float = 1.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        tgt = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        p_t = prob * tgt + (1.0 - prob) * (1.0 - tgt)
        alpha_t = self.alpha * tgt + (1.0 - self.alpha) * (1.0 - tgt)
        focal_e = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        if valid_mask is not None:
            vm = valid_mask.float()
            vm_sum = vm.sum() + 1e-6
            return (focal_e * vm).sum() / vm_sum
        return focal_e.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss: alpha controls penalty for False Positives,
    beta controls penalty for False Negatives.
    For small lymph nodes, beta=0.7 penalizes missed nodes heavily.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        vm = valid_mask.float() if valid_mask is not None else torch.ones_like(prob)
        prob_m = prob * vm
        gt_m = target.float() * vm
        tp = (prob_m * gt_m).sum(dim=(1, 2, 3))
        fp = (prob_m * (1.0 - gt_m)).sum(dim=(1, 2, 3))
        fn = ((1.0 - prob_m) * gt_m).sum(dim=(1, 2, 3))
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1.0 - tversky).mean()


class SDTBoundaryLoss(nn.Module):
    """
    Signed Distance Transform (SDT) boundary loss.
    sdt < 0 inside GT, sdt > 0 outside GT.
    """
    def forward(self, logits: torch.Tensor, sdt: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) * sdt).mean()


def compute_sdt_batch(target_np: np.ndarray, clip: float = 50.0) -> np.ndarray:
    """
    Computes per-sample signed distance transform of target masks, normalized to [-1, 1].
    target_np: numpy array of shape [B, 1, H, W]
    """
    B = target_np.shape[0]
    sdt = np.zeros_like(target_np, dtype=np.float32)
    for i in range(B):
        mask = target_np[i, 0]
        n_pos = mask.sum()
        if 0 < n_pos < mask.size:
            dt_out = distance_transform_edt(1.0 - mask)
            dt_in = distance_transform_edt(mask)
            raw = dt_out - dt_in
            sdt[i, 0] = np.clip(raw, -clip, clip) / clip
        elif n_pos == 0:
            sdt[i, 0] = 1.0
        else:
            sdt[i, 0] = -1.0
    return sdt


class OriginalCombinedLoss(nn.Module):
    """
    Exact loss from paper:
    L_original = 0.40 * L_Tversky + 0.30 * L_Dice + 0.20 * L_Boundary + 0.10 * L_Focal
    """
    def __init__(self, tversky_alpha: float = 0.3, tversky_beta: float = 0.7, focal_gamma: float = 1.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.dice = SoftDiceLoss()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=0.75)
        self.boundary = SDTBoundaryLoss()

    def forward(
        self,
        model_output,
        target: torch.Tensor,
        valid_mask: torch.Tensor = None,
        tier_flags: torch.Tensor = None,
        sdt: torch.Tensor = None,
    ) -> torch.Tensor:
        logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        vm = valid_mask.float() if valid_mask is not None else torch.ones_like(logits)

        # 1. Tversky Loss (0.40)
        l_tv = self.tversky(logits, target, vm)

        # 2. Soft Dice Loss (0.30)
        l_dice = self.dice(logits, target, vm)

        # 3. Focal Loss (0.10)
        l_focal = self.focal(logits, target, vm)

        # 4. SDT Boundary Loss (0.20)
        l_boundary = torch.tensor(0.0, device=logits.device)
        if sdt is not None:
            if tier_flags is not None and tier_flags.any():
                t1_w = tier_flags.view(-1, 1, 1, 1).float()
                l_boundary = self.boundary(logits * t1_w, sdt * t1_w)
            elif tier_flags is None:
                l_boundary = self.boundary(logits, sdt)

        total_loss = 0.40 * l_tv + 0.30 * l_dice + 0.20 * l_boundary + 0.10 * l_focal
        return total_loss

