"""
CT Lymph Node Preprocessing Utilities.

Provides:
- Soft-tissue mediastinal HU windowing and clipping [-50, 300] HU.
- Per-case foreground z-score intensity normalization.
- 2.5D multi-slice stacking (5 axial CT slices + 1 anatomical prior channel = 6 channels).
- Synthetic/sample patient volume generator for verification and testing.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def normalize_ct_slice(
    slice_ct: np.ndarray,
    hu_min: float = -50.0,
    hu_max: float = 300.0,
    per_case_norm: bool = True,
    norm_mu: float = 50.0,
    norm_sigma: float = 80.0
) -> np.ndarray:
    """
    Normalizes a CT slice according to the paper's mediastinal preprocessing protocol.
    """
    arr = slice_ct.astype(np.float32)
    if per_case_norm:
        arr = np.clip((arr - norm_mu) / (norm_sigma + 1e-6), -3.0, 3.0)
        arr = (arr + 3.0) / 6.0
    else:
        arr = (np.clip(arr, hu_min, hu_max) - hu_min) / (hu_max - hu_min)
    return np.clip(arr, 0.0, 1.0)


def extract_slice_stack(
    ct_vol: np.ndarray,
    prior_vol: np.ndarray,
    z: int,
    context_offsets: List[int] = (-2, -1, 0, 1, 2),
    hu_min: float = -50.0,
    hu_max: float = 300.0,
    per_case_norm: bool = True,
    norm_mu: float = 50.0,
    norm_sigma: float = 80.0
) -> np.ndarray:
    """
    Extracts a [6, H, W] tensor for a given axial slice z:
    - 5 CT context slices [z-2, z-1, z, z+1, z+2]
    - 1 Anatomy prior channel at z
    """
    Z, H, W = ct_vol.shape
    channels = []

    for dz in context_offsets:
        zz = max(0, min(Z - 1, z + dz))
        ct_norm = normalize_ct_slice(
            ct_vol[zz],
            hu_min=hu_min,
            hu_max=hu_max,
            per_case_norm=per_case_norm,
            norm_mu=norm_mu,
            norm_sigma=norm_sigma
        )
        channels.append(ct_norm)

    # 6th channel: Anatomy prior
    pr = prior_vol[z].astype(np.float32)
    channels.append(pr)

    return np.stack(channels, axis=0)  # [6, H, W]


def create_mock_patient_record(
    patient_id: str,
    output_dir: Path,
    n_slices: int = 40,
    h: int = 256,
    w: int = 256,
    tier: int = 1,
    seed: int = 42
) -> Dict:
    """
    Generates a realistic synthetic mediastinal CT patient volume for unit tests and smoke tests.
    Creates mediastinal corridor anatomy, realistic HU intensity values, and synthetic lymph nodes.
    """
    # Check if arrays already exist on disk
    ct_path = output_dir / f"{patient_id}_ct.npy"
    seg_path = output_dir / f"{patient_id}_seg.npy"
    prior_path = output_dir / f"{patient_id}_prior.npy"
    valid_path = output_dir / f"{patient_id}_valid.npy"

    if ct_path.exists() and seg_path.exists() and prior_path.exists() and valid_path.exists():
        seg_arr = np.load(str(seg_path))
        pos_slices = [int(z) for z in range(seg_arr.shape[0]) if seg_arr[z].sum() > 0]
        neg_slices = [z for z in range(seg_arr.shape[0]) if z not in pos_slices]
        prior_arr = np.load(str(prior_path))
        return {
            "patient_id": patient_id,
            "ct_path": str(ct_path),
            "seg_path": str(seg_path),
            "prior_path": str(prior_path),
            "valid_mask_path": str(valid_path),
            "tier": tier,
            "n_total_slices": int(seg_arr.shape[0]),
            "n_positive_slices": len(pos_slices),
            "positive_slices_voi": pos_slices,
            "negative_slices_voi": neg_slices,
            "prior_coverage": float(prior_arr.mean()),
        }

    # Generate synthetic mediastinal corridor and lymph nodes
    rng = np.random.RandomState(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    ct = rng.normal(loc=35.0, scale=40.0, size=(n_slices, h, w)).astype(np.float32)
    prior = np.zeros((n_slices, h, w), dtype=np.uint8)
    seg = np.zeros((n_slices, h, w), dtype=np.uint8)
    valid = np.ones((n_slices, h, w), dtype=np.uint8)

    # Simulate mediastinal prior corridor (oval in center)
    y, x = np.ogrid[:h, :w]
    mediastinum_mask = ((y - 128)**2 / (60**2) + (x - 128)**2 / (45**2)) <= 1.0
    prior[:] = mediastinum_mask.astype(np.uint8)

    # Simulate realistic lymph nodes (small 8-15px ellipsoids inside mediastinum)
    pos_slices = []
    node_z_start = n_slices // 3
    node_z_end = node_z_start + 8
    for z in range(node_z_start, node_z_end):
        r_y = 6 + int(2 * np.sin((z - node_z_start) / 7 * np.pi))
        r_x = 7 + int(3 * np.sin((z - node_z_start) / 7 * np.pi))
        cy, cx = 120 + (z % 3), 130 - (z % 2)
        node_mask = ((y - cy)**2 / (r_y**2) + (x - cx)**2 / (r_x**2)) <= 1.0
        seg[z, node_mask] = 1
        ct[z, node_mask] = rng.normal(loc=70.0, scale=15.0, size=node_mask.sum())
        pos_slices.append(z)

    # Save arrays
    np.save(str(ct_path), ct)
    np.save(str(seg_path), seg)
    np.save(str(prior_path), prior)
    np.save(str(valid_path), valid)

    neg_slices = [z for z in range(n_slices) if z not in pos_slices]

    record = {
        "patient_id": patient_id,
        "ct_path": str(ct_path),
        "seg_path": str(seg_path),
        "prior_path": str(prior_path),
        "valid_mask_path": str(valid_path),
        "tier": tier,
        "n_total_slices": n_slices,
        "n_positive_slices": len(pos_slices),
        "positive_slices_voi": pos_slices,
        "negative_slices_voi": neg_slices,
        "prior_coverage": float(prior.mean()),
    }
    return record

