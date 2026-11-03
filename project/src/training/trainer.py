"""
Training and Validation Engine for Baseline CST and ABRM-CST.

Features:
- Automatic Mixed Precision (FP16) optimized for 4-6 GB VRAM GPUs
- Gradient accumulation for configurable effective batch size
- Exponential Moving Average (EMA) weight tracking (decay=0.999)
- Patient-level evaluation with Dice, IoU, Precision, Recall
- Checkpointing, logging, and GPU memory tracking
"""

from __future__ import annotations
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.losses.original_loss import compute_sdt_batch
from src.evaluation.metrics import compute_dice_iou, compute_boundary_f1, compute_hd95

if torch.cuda.is_available():
    dev_name = torch.cuda.get_device_name(0).upper()
    if "1650" in dev_name or "1660" in dev_name:
        torch.backends.cudnn.enabled = False


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup.clear()

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


def warmup_cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    warmup_epochs: int,
    max_epochs: int,
    lr_base: float,
    lr_min: float = 1e-6
) -> float:
    """Warmup + Cosine Annealing learning rate schedule."""
    if epoch < warmup_epochs:
        lr = lr_min + (lr_base - lr_min) * (epoch + 1) / max(1, warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        lr = lr_min + 0.5 * (lr_base - lr_min) * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    ema: EMA,
    scaler: GradScaler,
    device: torch.device,
    accum_steps: int = 1,
    grad_clip: float = 1.0,
    epoch: int = 0,
    is_abrm: bool = False,
    sdt_interval: int = 4,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    accum_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True, leave=False)
    for step, (images, masks, valid_masks, tiers) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid_masks = valid_masks.to(device, non_blocking=True)
        tier_flags = (tiers == 1).to(device)

        # SDT computation for boundary loss (every sdt_interval steps, matching paper)
        sdt = None
        if (step % sdt_interval == 0) and tier_flags.any():
            if masks.any():
                sdt_np = compute_sdt_batch(masks.cpu().numpy())
                sdt = torch.from_numpy(sdt_np).float().to(device)
            else:
                sdt = torch.ones_like(masks)

        use_cuda = device.type == "cuda"
        # AMP FP16
        with autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.float16 if use_cuda else torch.bfloat16):
            output = model(images)
            if is_abrm:
                loss, loss_dict = loss_fn(output, masks, valid_masks, tier_flags, sdt)
            else:
                loss = loss_fn(output, masks, valid_masks, tier_flags, sdt)

        scaler.scale(loss / accum_steps).backward()
        accum_loss += loss.item()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {"loss": total_loss / max(1, len(loader))}


@torch.no_grad()
def evaluate_patient_volume(
    model: nn.Module,
    rec: Dict,
    config: Dict,
    device: torch.device,
    batch_size: int = 8,
    is_abrm: bool = False,
) -> Dict[str, float]:
    """Runs 2.5D slice inference over an entire patient volume."""
    model.eval()
    ct = np.load(rec["ct_path"])
    seg = np.load(rec["seg_path"]).astype(np.uint8)
    prior = np.load(rec["prior_path"])

    Z, H, W = ct.shape
    pre = config.get("preprocessing", {})
    hu_min = float(pre.get("hu_min", -50.0))
    hu_max = float(pre.get("hu_max", 300.0))
    per_case_norm = bool(pre.get("per_case_norm", True))
    context_offsets = list(pre.get("context_offsets", [-2, -1, 0, 1, 2]))

    flat = ct.ravel()
    fg = flat[(flat > hu_min) & (flat < hu_max)]
    norm_mu = float(fg.mean()) if len(fg) > 500 else 50.0
    norm_sigma = float(fg.std()) if len(fg) > 500 else 80.0

    prob_vol = np.zeros((Z, H, W), dtype=np.float32)
    boundary_vol = np.zeros((Z, H, W), dtype=np.float32) if is_abrm else None

    # Process in batches of slices
    use_cuda = device.type == "cuda"
    for z_start in range(0, Z, batch_size):
        z_end = min(Z, z_start + batch_size)
        batch_slices = []
        for z in range(z_start, z_end):
            channels = []
            for dz in context_offsets:
                zz = max(0, min(Z - 1, z + dz))
                sl = ct[zz].astype(np.float32)
                if per_case_norm:
                    sl = np.clip((sl - norm_mu) / (norm_sigma + 1e-6), -3.0, 3.0)
                    sl = (sl + 3.0) / 6.0
                else:
                    sl = (np.clip(sl, hu_min, hu_max) - hu_min) / (hu_max - hu_min)
                channels.append(np.clip(sl, 0.0, 1.0))
            channels.append(prior[z].astype(np.float32))
            batch_slices.append(np.stack(channels, axis=0))

        batch_tensor = torch.from_numpy(np.stack(batch_slices, axis=0)).to(device)
        with autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.float16 if use_cuda else torch.bfloat16):
            output = model(batch_tensor)
            logits = output[0]  # Final prediction
            prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            prob_vol[z_start:z_end] = prob

            if is_abrm and len(output) > 2:
                b_logits = output[2]
                b_prob = torch.sigmoid(b_logits).squeeze(1).cpu().numpy()
                boundary_vol[z_start:z_end] = b_prob

    pred_binary = (prob_vol >= 0.5).astype(np.uint8)
    dice, iou, prec, rec = compute_dice_iou(pred_binary, seg)
    bf1 = compute_boundary_f1(pred_binary, seg)
    hd95 = compute_hd95(pred_binary, seg)

    metrics = {
        "dice": dice,
        "iou": iou,
        "precision": prec,
        "recall": rec,
        "boundary_f1": bf1,
        "hd95": hd95,
    }
    return metrics


@torch.no_grad()
def validate_dataset(
    model: nn.Module,
    val_records: List[Dict],
    config: Dict,
    device: torch.device,
    is_abrm: bool = False,
    batch_size: int = 8,
) -> Dict[str, float]:
    """Validates model across all validation patient volumes with patient-level metrics."""
    model.eval()
    all_metrics = []
    for rec in val_records:
        m = evaluate_patient_volume(model, rec, config, device, batch_size=batch_size, is_abrm=is_abrm)
        all_metrics.append(m)

    mean_metrics = {
        "dice": float(np.mean([m["dice"] for m in all_metrics])),
        "iou": float(np.mean([m["iou"] for m in all_metrics])),
        "precision": float(np.mean([m["precision"] for m in all_metrics])),
        "recall": float(np.mean([m["recall"] for m in all_metrics])),
        "boundary_f1": float(np.mean([m["boundary_f1"] for m in all_metrics])),
        "hd95": float(np.mean([m["hd95"] for m in all_metrics])),
        "n_patients": len(all_metrics),
    }
    return mean_metrics


def save_checkpoint(path: str | Path, epoch: int, model: nn.Module, ema: EMA, optimizer: torch.optim.Optimizer, val_metrics: Dict, config: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    state = {
        "epoch": epoch,
        "val_metrics": val_metrics,
        "model_state": raw.state_dict(),
        "ema_state": ema.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(state, str(path))
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str | Path, model: nn.Module, ema: Optional[EMA] = None, optimizer: Optional[torch.optim.Optimizer] = None, device: str = "cpu"):
    state = torch.load(str(path), map_location=device, weights_only=False)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(state["model_state"])
    if ema is not None and "ema_state" in state:
        ema.load_state_dict(state["ema_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    return state.get("epoch", 0), state.get("val_metrics", {})
