"""
Experiment 1: Train Original Cross-Slice Transformer (CST) Baseline.

Loss: L_original = 0.40 * Tversky + 0.30 * Soft Dice + 0.20 * Boundary Loss + 0.10 * Focal Loss
Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
Precision: FP16 Automatic Mixed Precision
Saves: Checkpoints to checkpoints/cst_baseline_best.pt and logs to results/e1_baseline_metrics.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.amp import GradScaler

from src.models.cst import CST
from src.losses.original_loss import OriginalCombinedLoss
from src.data.dataset import make_data_loader
from src.data.preprocessing import create_mock_patient_record
from src.training.trainer import EMA, train_one_epoch, validate_dataset, save_checkpoint, warmup_cosine_lr


def main():
    parser = argparse.ArgumentParser(description="Train CST Baseline (E1)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "results" / "e1_baseline"))
    parser.add_argument("--num-patients", type=int, default=8, help="Number of cohort patients")
    parser.add_argument("--n-slices", type=int, default=20, help="Slices per patient")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0).upper()
        if "1650" in dev_name or "1660" in dev_name:
            torch.backends.cudnn.enabled = False
    print(f"=== Starting Experiment 1: CST Baseline ===")
    print(f"Device: {device} | Epochs: {args.epochs} | Physical Batch: {args.batch_size} | Accum: {args.accum_steps} (Effective: {args.batch_size * args.accum_steps})")

    # Prepare cohort data
    data_dir = ROOT / "data" / "cohort"
    data_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(args.num_patients):
        rec = create_mock_patient_record(f"PATIENT_{i:03d}", data_dir, n_slices=args.n_slices, seed=42 + i)
        records.append(rec)

    # 80/20 train/val split strictly by patient
    n_train = max(1, int(len(records) * 0.8))
    train_recs = records[:n_train]
    val_recs = records[n_train:]
    print(f"Patient-level split: {len(train_recs)} Train patients | {len(val_recs)} Val patients (No leakage)")

    train_loader = make_data_loader(train_recs, config, training=True, batch_size=args.batch_size)

    # Model, Loss, Optimizer
    model = CST(num_slices=6, base_channels=64).to(device)
    loss_fn = OriginalCombinedLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ema = EMA(model, decay=0.999)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    best_val_dice = -1.0
    history = []
    t_start = time.time()

    for epoch in range(args.epochs):
        t_ep_start = time.time()
        lr = warmup_cosine_lr(optimizer, epoch, 1, args.epochs, args.lr, 1e-6)

        train_res = train_one_epoch(
            model, train_loader, optimizer, loss_fn, ema, scaler, device,
            accum_steps=args.accum_steps, grad_clip=1.0, epoch=epoch, is_abrm=False
        )

        ema.apply_shadow(model)
        val_res = validate_dataset(model, val_recs, config, device, is_abrm=False, batch_size=args.batch_size)
        ema.restore(model)

        ep_duration = time.time() - t_ep_start
        vram_mb = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0

        epoch_stat = {
            "epoch": epoch + 1,
            "train_loss": train_res["loss"],
            "val_dice": val_res["dice"],
            "val_iou": val_res["iou"],
            "val_precision": val_res["precision"],
            "val_recall": val_res["recall"],
            "val_hd95": val_res["hd95"],
            "val_boundary_f1": val_res["boundary_f1"],
            "lr": lr,
            "vram_mb": vram_mb,
            "epoch_duration_s": ep_duration,
        }
        history.append(epoch_stat)

        print(f"[E1 Epoch {epoch+1:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_res['loss']:.4f} | "
              f"Val Dice: {val_res['dice']:.4f} | "
              f"IoU: {val_res['iou']:.4f} | "
              f"HD95: {val_res['hd95']:.2f}mm | "
              f"VRAM: {vram_mb:.1f}MB | "
              f"Time: {ep_duration:.1f}s")

        if val_res["dice"] > best_val_dice:
            best_val_dice = val_res["dice"]
            save_checkpoint(ckpt_dir / "cst_baseline_best.pt", epoch, model, ema, optimizer, val_res, config)

    total_training_time = time.time() - t_start
    print(f"\nExperiment 1 Completed! Total Time: {total_training_time:.1f}s | Best Val Dice: {best_val_dice:.4f}")

    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "experiment": "E1: Original CST Baseline",
            "total_time_s": total_training_time,
            "best_val_dice": best_val_dice,
            "history": history,
            "config": config,
        }, f, indent=2)
    print(f"Saved E1 training log to: {metrics_file}")


if __name__ == "__main__":
    main()
