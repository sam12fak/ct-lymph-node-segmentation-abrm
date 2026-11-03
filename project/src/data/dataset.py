"""
Lymph Node Dataset Loader with 2.5D Slice Context and Augmentation.

Each sample returns:
- images: [6, 256, 256] float32 in [0, 1] (5 CT context slices + 1 anatomy prior)
- mask: [1, 256, 256] float32 {0, 1} (ground truth lymph node mask at center slice)
- valid_mask: [1, 256, 256] float32 {0, 1}
- tier: int
"""

from __future__ import annotations
import functools
import random
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.data.preprocessing import normalize_ct_slice


@functools.lru_cache(maxsize=16)
def _load_cached_volumes(ct_path: str, seg_path: str, prior_path: str, valid_path: str):
    ct = np.load(ct_path, mmap_mode="r")
    seg = np.load(seg_path, mmap_mode="r")
    prior = np.load(prior_path, mmap_mode="r")
    valid = np.load(valid_path, mmap_mode="r")
    return ct, seg, prior, valid


class LymphNodeDataset(Dataset):
    """
    2.5D Dataset yielding 6-channel stacks [B, 6, 256, 256]
    for centre-slice lymph node segmentation.
    """
    def __init__(
        self,
        records: List[Dict],
        config: Dict,
        training: bool = True,
    ):
        self.records = records
        self.config = config
        self.training = training

        pre = config.get("preprocessing", {})
        self.hu_min = float(pre.get("hu_min", -50.0))
        self.hu_max = float(pre.get("hu_max", 300.0))
        self.per_case_norm = bool(pre.get("per_case_norm", True))
        self.context_offsets = list(pre.get("context_offsets", [-2, -1, 0, 1, 2]))

        aug = config.get("augmentation", {})
        self.hflip_prob = float(aug.get("hflip_prob", 0.5)) if training else 0.0
        self.rotation_prob = float(aug.get("rotation_prob", 0.3)) if training else 0.0
        self.rotation_limit = float(aug.get("rotation_limit", 10.0))
        self.intensity_shift_prob = float(aug.get("intensity_shift_prob", 0.5)) if training else 0.0
        self.gaussian_noise_prob = float(aug.get("gaussian_noise_prob", 0.3)) if training else 0.0

        # Construct flat sample indexing
        self.samples = []
        self.weights = []
        fg_weight = float(config.get("training", {}).get("foreground_sample_weight", 5.0))

        for pi, rec in enumerate(records):
            pos_set = set(rec.get("positive_slices_voi", []))
            tier = int(rec.get("tier", 1))
            n_z = int(rec.get("n_total_slices", 0))

            for z in range(n_z):
                is_pos = z in pos_set
                self.samples.append((pi, z, tier, is_pos))
                if is_pos:
                    self.weights.append(fg_weight)
                else:
                    self.weights.append(1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        pi, z, tier, is_pos = self.samples[idx]
        rec = self.records[pi]

        ct_vol, seg_vol, prior_vol, valid_vol = _load_cached_volumes(
            rec["ct_path"], rec["seg_path"], rec["prior_path"], rec["valid_mask_path"]
        )

        Z, H, W = ct_vol.shape

        # Per-case stats for normalization
        if self.per_case_norm:
            flat = np.asarray(ct_vol).ravel()
            fg = flat[(flat > self.hu_min) & (flat < self.hu_max)]
            norm_mu = float(fg.mean()) if len(fg) > 500 else 50.0
            norm_sigma = float(fg.std()) if len(fg) > 500 else 80.0
        else:
            norm_mu, norm_sigma = 50.0, 80.0

        # Build 6-channel stack
        channels = []
        for dz in self.context_offsets:
            zz = max(0, min(Z - 1, z + dz))
            raw_slice = ct_vol[zz].astype(np.float32)

            # Augmentation on CT channels (intensity shift / noise)
            if self.training:
                if random.random() < self.intensity_shift_prob:
                    shift = random.uniform(-15.0, 15.0)
                    raw_slice = raw_slice + shift
                if random.random() < self.gaussian_noise_prob:
                    noise = np.random.normal(0.0, 5.0, size=raw_slice.shape).astype(np.float32)
                    raw_slice = raw_slice + noise

            norm_slice = normalize_ct_slice(
                raw_slice,
                hu_min=self.hu_min,
                hu_max=self.hu_max,
                per_case_norm=self.per_case_norm,
                norm_mu=norm_mu,
                norm_sigma=norm_sigma,
            )
            channels.append(norm_slice)

        # 6th channel: Anatomy prior
        pr = prior_vol[z].astype(np.float32)
        channels.append(pr)

        img_stack = np.stack(channels, axis=0)  # [6, H, W]
        mask = (seg_vol[z] > 0).astype(np.float32)[np.newaxis, ...]  # [1, H, W]
        valid_mask = (valid_vol[z] > 0).astype(np.float32)[np.newaxis, ...]  # [1, H, W]

        # Spatial augmentations (joint across img, mask, valid)
        if self.training:
            # Horizontal flip
            if random.random() < self.hflip_prob:
                img_stack = np.flip(img_stack, axis=2).copy()
                mask = np.flip(mask, axis=2).copy()
                valid_mask = np.flip(valid_mask, axis=2).copy()

            # Small rotation
            if random.random() < self.rotation_prob:
                angle = random.uniform(-self.rotation_limit, self.rotation_limit)
                M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
                # Rotate channels
                for c in range(img_stack.shape[0]):
                    img_stack[c] = cv2.warpAffine(img_stack[c], M, (W, H), borderMode=cv2.BORDER_REPLICATE)
                mask[0] = (cv2.warpAffine(mask[0], M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0.5).astype(np.float32)
                valid_mask[0] = (cv2.warpAffine(valid_mask[0], M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0.5).astype(np.float32)

        return (
            torch.from_numpy(img_stack).float(),
            torch.from_numpy(mask).float(),
            torch.from_numpy(valid_mask).float(),
            tier,
        )


def make_data_loader(records: List[Dict], config: Dict, training: bool = True, batch_size: int = 4, num_workers: int = 0):
    dataset = LymphNodeDataset(records, config, training=training)
    if training:
        sampler = WeightedRandomSampler(
            weights=dataset.weights,
            num_samples=len(dataset),
            replacement=True
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True if len(dataset) >= batch_size else False,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
        )
    return loader

