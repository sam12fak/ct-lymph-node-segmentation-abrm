"""
Visualization Tools for Lymph Node Segmentation and Boundary Analysis.

Generates comprehensive high-resolution comparison figures showing:
1. Original CT image (soft tissue mediastinal window)
2. Ground-truth lymph node mask
3. Original CST prediction
4. Predicted boundary map (ABRM auxiliary branch)
5. ABRM refined segmentation
6. Ground-truth vs Prediction overlay
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_qualitative_figure(
    ct_slice: np.ndarray,
    gt_mask: np.ndarray,
    cst_pred: np.ndarray,
    boundary_pred: np.ndarray,
    abrm_pred: np.ndarray,
    save_path: str | Path,
    title: str = "Lymph Node Segmentation & Boundary-Guided Refinement",
    hu_window: Tuple[float, float] = (-50, 300)
):
    """
    Renders and saves a 6-panel qualitative comparison figure.

    Args:
        ct_slice: 2D array [H, W] normalized or in HU
        gt_mask: 2D binary array [H, W]
        cst_pred: 2D binary array [H, W] (baseline CST output)
        boundary_pred: 2D float array [H, W] (predicted boundary probability)
        abrm_pred: 2D binary array [H, W] (ABRM refined output)
        save_path: Output file path (.png)
        title: Figure title
        hu_window: Windowing range for CT display
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 6, figsize=(24, 4.5), dpi=300)

    # 1. Original CT
    axes[0].imshow(ct_slice, cmap="gray", vmin=0, vmax=1 if ct_slice.max() <= 1.5 else hu_window[1])
    axes[0].set_title("1. Original CT Slice", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2. Ground Truth Mask
    axes[1].imshow(ct_slice, cmap="gray", alpha=0.7)
    axes[1].imshow(np.ma.masked_where(gt_mask == 0, gt_mask), cmap="Greens", vmin=0, vmax=1, alpha=0.7)
    axes[1].set_title("2. Ground Truth (Green)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # 3. Original CST Prediction
    axes[2].imshow(ct_slice, cmap="gray", alpha=0.7)
    axes[2].imshow(np.ma.masked_where(cst_pred == 0, cst_pred), cmap="Blues", vmin=0, vmax=1, alpha=0.7)
    axes[2].set_title("3. Baseline CST (Blue)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # 4. Predicted Boundary Map
    axes[3].imshow(ct_slice, cmap="gray", alpha=0.5)
    im_b = axes[3].imshow(boundary_pred, cmap="hot", vmin=0.0, vmax=1.0, alpha=0.8)
    axes[3].set_title("4. Predicted Boundary Map", fontsize=12, fontweight="bold")
    axes[3].axis("off")
    plt.colorbar(im_b, ax=axes[3], fraction=0.046, pad=0.04)

    # 5. ABRM Refined Prediction
    axes[4].imshow(ct_slice, cmap="gray", alpha=0.7)
    axes[4].imshow(np.ma.masked_where(abrm_pred == 0, abrm_pred), cmap="Reds", vmin=0, vmax=1, alpha=0.7)
    axes[4].set_title("5. Proposed ABRM (Red)", fontsize=12, fontweight="bold")
    axes[4].axis("off")

    # 6. Overlay Comparison: GT (Green) vs ABRM (Red), Overlap (Yellow/Orange)
    overlay = np.zeros((*ct_slice.shape, 3), dtype=np.float32)
    ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-6)
    overlay[..., 0] = ct_norm * 0.6
    overlay[..., 1] = ct_norm * 0.6
    overlay[..., 2] = ct_norm * 0.6

    # Red channel for pred, Green channel for GT
    overlay[gt_mask > 0, 1] = 1.0
    overlay[abrm_pred > 0, 0] = 1.0

    axes[5].imshow(overlay)
    axes[5].set_title("6. Overlay (GT=G, Pred=R, Match=Y)", fontsize=12, fontweight="bold")
    axes[5].axis("off")

    plt.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved qualitative figure to: {save_path}")

