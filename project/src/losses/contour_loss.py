"""
Boundary Contour Loss and Total ABRM Loss.

Provides:
1. ContourLoss: Combined Binary Cross Entropy + Dice Loss for the auxiliary boundary prediction head.
2. ABRMLoss: Total multi-task training objective:
   L_total = L_original(final_logits) + lambda_boundary * L_contour(boundary_logits, boundary_gt)
   with optional intermediate supervision on initial_logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.original_loss import OriginalCombinedLoss, SoftDiceLoss
from src.models.abr_module import extract_boundary_gt


class BoundaryDiceLoss(nn.Module):
    """Soft Dice loss specifically tuned for thin boundary contours."""
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


class ContourLoss(nn.Module):
    """
    Supervision for the explicit boundary prediction branch.
    Combines BCE loss with Dice loss to handle spatial boundary sparsity.
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, pos_weight: float = 4.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.dice_loss = BoundaryDiceLoss()

    def forward(self, boundary_logits: torch.Tensor, boundary_gt: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        tgt = boundary_gt.float()
        vm = valid_mask.float() if valid_mask is not None else torch.ones_like(boundary_logits)

        # Weighted BCE for sparse boundaries
        bce = F.binary_cross_entropy_with_logits(
            boundary_logits, tgt, pos_weight=self.pos_weight, reduction="none"
        )
        vm_sum = vm.sum() + 1e-6
        loss_bce = (bce * vm).sum() / vm_sum

        # Boundary Dice
        loss_dice = self.dice_loss(boundary_logits, tgt, vm)

        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class ABRMLoss(nn.Module):
    """
    Complete ABRM multi-task loss:
    L_total = L_original(final_logits) + lambda_aux * L_original(initial_logits) + lambda_b * L_contour(boundary_logits)
    """
    def __init__(
        self,
        lambda_boundary: float = 0.2,
        supervise_initial: bool = True,
        lambda_initial: float = 0.5,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        focal_gamma: float = 1.0,
    ):
        super().__init__()
        self.lambda_boundary = lambda_boundary
        self.supervise_initial = supervise_initial
        self.lambda_initial = lambda_initial
        self.original_loss = OriginalCombinedLoss(
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            focal_gamma=focal_gamma,
        )
        self.contour_loss = ContourLoss()

    def forward(
        self,
        model_output,
        target: torch.Tensor,
        valid_mask: torch.Tensor = None,
        tier_flags: torch.Tensor = None,
        sdt: torch.Tensor = None,
    ):
        """
        model_output: tuple of (final_logits, initial_logits, boundary_logits, ds1, ds2, weights, diff_energy)
        """
        final_logits, initial_logits, boundary_logits = model_output[0], model_output[1], model_output[2]

        # 1. Primary segmentation loss on refined mask
        l_orig_final = self.original_loss(final_logits, target, valid_mask, tier_flags, sdt)

        # 2. Auxiliary segmentation loss on initial mask
        l_orig_initial = torch.tensor(0.0, device=final_logits.device)
        if self.supervise_initial:
            l_orig_initial = self.original_loss(initial_logits, target, valid_mask, tier_flags, sdt)

        # 3. Ground truth boundary extraction
        boundary_gt = extract_boundary_gt(target)

        # 4. Auxiliary boundary contour loss
        l_contour = self.contour_loss(boundary_logits, boundary_gt, valid_mask)

        # Total multi-task loss
        l_total = l_orig_final + self.lambda_initial * l_orig_initial + self.lambda_boundary * l_contour

        loss_dict = {
            "loss": l_total,
            "l_final": l_orig_final.item(),
            "l_initial": l_orig_initial.item() if self.supervise_initial else 0.0,
            "l_contour": l_contour.item(),
        }
        return l_total, loss_dict

