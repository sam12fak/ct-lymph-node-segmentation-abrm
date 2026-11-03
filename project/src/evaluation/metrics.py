"""
Evaluation Metrics for Lymph Node CT Segmentation.

Implements:
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Precision & Recall
- Boundary F1 Score (BF1)
- 95th Percentile Hausdorff Distance (HD95)
- Parameter count and inference latency benchmarking
- Size-stratified evaluation:
  - Small (< 10 mm / equivalent volume)
  - Medium (10 - 15 mm)
  - Large (>= 15 mm)
"""

from __future__ import annotations
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label


def compute_dice_iou(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Computes (Dice, IoU, Precision, Recall) on binary numpy masks.
    """
    p = pred.astype(bool)
    t = target.astype(bool)

    tp = float((p & t).sum())
    fp = float((p & ~t).sum())
    fn = float((~p & t).sum())

    denom_dice = 2.0 * tp + fp + fn
    dice = (2.0 * tp / denom_dice) if denom_dice > 0 else (1.0 if not t.any() and not p.any() else 0.0)

    denom_iou = tp + fp + fn
    iou = (tp / denom_iou) if denom_iou > 0 else (1.0 if not t.any() and not p.any() else 0.0)

    prec = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if not p.any() else 0.0)
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if not t.any() else 0.0)

    return dice, iou, prec, rec


def extract_surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Extract surface boundary voxels from a 2D or 3D binary mask."""
    from scipy.ndimage import binary_erosion
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask)
    return mask ^ eroded


def compute_boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance_px: float = 2.0) -> float:
    """
    Boundary F1 score measures contour boundary alignment within a tolerance distance.
    """
    p = pred.astype(bool)
    t = target.astype(bool)

    if not t.any() and not p.any():
        return 1.0
    if not t.any() or not p.any():
        return 0.0

    surf_pred = extract_surface_voxels(p)
    surf_gt = extract_surface_voxels(t)

    if not surf_pred.any() or not surf_gt.any():
        return 0.0

    # Distance from pred surface to GT surface
    dt_gt = distance_transform_edt(~surf_gt)
    # Distance from GT surface to pred surface
    dt_pred = distance_transform_edt(~surf_pred)

    pred_precision = float((dt_gt[surf_pred] <= tolerance_px).mean())
    gt_recall = float((dt_pred[surf_gt] <= tolerance_px).mean())

    denom = pred_precision + gt_recall
    bf1 = (2.0 * pred_precision * gt_recall / denom) if denom > 0 else 0.0
    return bf1


def compute_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float, float] = (1.5, 0.5, 0.5)
) -> float:
    """
    Computes the 95th percentile Hausdorff Distance (HD95) in mm.
    """
    p = pred.astype(bool)
    t = target.astype(bool)

    if not t.any() and not p.any():
        return 0.0
    if not t.any() or not p.any():
        return 100.0  # Penalty value for complete miss

    surf_pred = extract_surface_voxels(p)
    surf_gt = extract_surface_voxels(t)

    # Physical distance maps in mm
    dt_gt = distance_transform_edt(~surf_gt, sampling=spacing)
    dt_pred = distance_transform_edt(~surf_pred, sampling=spacing)

    distances_pred_to_gt = dt_gt[surf_pred]
    distances_gt_to_pred = dt_pred[surf_gt]

    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    if len(all_distances) == 0:
        return 0.0

    hd95 = float(np.percentile(all_distances, 95))
    return hd95


def size_stratified_evaluation(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float, float] = (1.5, 0.5, 0.5)
) -> Dict[str, Dict[str, float]]:
    """
    Stratifies lymph node evaluation by size:
    - Small: < 10 mm max diameter (< 523 mm3 sphere eq)
    - Medium: 10 - 15 mm (523 - 1767 mm3 sphere eq)
    - Large: >= 15 mm (>= 1767 mm3 sphere eq)
    """
    vox_vol_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    labeled_target, num_gt = label(target.astype(bool))

    bins = {
        "small (<10mm)": (0.0, 523.0),
        "medium (10-15mm)": (523.0, 1767.0),
        "large (>=15mm)": (1767.0, float("inf")),
    }

    results = {}
    for bin_name, (lo, hi) in bins.items():
        # Mask GT containing only nodes in this size category
        bin_mask_gt = np.zeros_like(target, dtype=bool)
        count = 0
        for l in range(1, num_gt + 1):
            comp = (labeled_target == l)
            vol = float(comp.sum()) * vox_vol_mm3
            if lo <= vol < hi:
                bin_mask_gt |= comp
                count += 1

        if not bin_mask_gt.any():
            results[bin_name] = {"dice": float("nan"), "iou": float("nan"), "count": 0}
            continue

        # Overlapping prediction components
        dice, iou, prec, rec = compute_dice_iou(pred, bin_mask_gt)
        results[bin_name] = {
            "dice": dice,
            "iou": iou,
            "precision": prec,
            "recall": rec,
            "count": count,
        }

    return results


def benchmark_model(model: torch.nn.Module, input_shape: Tuple[int, ...] = (1, 6, 256, 256), device: str = "cuda:0", n_runs: int = 50):
    """
    Calculates parameter count and average inference latency.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy_input = torch.randn(*input_shape, device=dev)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / n_runs * 1000.0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "latency_ms_per_slice": latency_ms,
        "fps": 1000.0 / latency_ms if latency_ms > 0 else 0.0,
    }

