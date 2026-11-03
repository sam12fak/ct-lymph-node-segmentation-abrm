"""
Train Proposed Architectures:
E2: Original CST + Auxiliary Boundary Prediction Head (no refinement)
E3: Original CST + Complete ABRM Refinement (Boundary head + Refinement module)

Loss: L_total = L_original + lambda_boundary * L_contour
Saves:
- Checkpoints to checkpoints/cst_boundary_aux_best.pt and checkpoints/cst_abrm_best.pt
- Experiment metrics to results/
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

from src.models.abr_module import ABRMCST
from src.losses.contour_loss import ABRMLoss
from src.data.dataset import make_data_loader
from src.data.preprocessing import create_mock_patient_record
from src.training.trainer import EMA, train_one_epoch, validate_dataset, save_checkpoint, warmup_cosine_lr


def run_abrm_experiment(experiment_type: str, epochs: int = 10, batch_size: int = 4, accum_steps: int = 4, lr: float = 1e-4, num_patients: int = 8, n_slices: int = 20, lambda_b: float = 0.2):
    assert experiment_type in ("E2", "E3"), "experiment_type must be E2 or E3"
    enable_refinement = (experiment_type == "E3")
    exp_name = "E3_cst_abrm_full" if enable_refinement else "E2_cst_boundary_aux"
    ckpt_name = "cst_abrm_best.pt" if enable_refinement else "cst_boundary_aux_best.pt"

    out_dir = ROOT / "results" / exp_name
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
    print(f"\n{'='*60}")
    print(f"=== Starting Experiment {experiment_type}: {'ABRM Full Refinement' if enable_refinement else 'Boundary Head Only'} ===")
    print(f"{'='*60}")
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size} | Accum: {accum_steps} | Lambda_b: {lambda_b}")

    # Use identical cohort data and identical patient splits for fair comparison
    data_dir = ROOT / "data" / "cohort"
    data_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(num_patients):
        rec = create_mock_patient_record(f"PATIENT_{i:03d}", data_dir, n_slices=n_slices, seed=42 + i)
        records.append(rec)

    n_train = max(1, int(len(records) * 0.8))
    train_recs = records[:n_train]
    val_recs = records[n_train:]
    print(f"Patient-level split: {len(train_recs)} Train patients | {len(val_recs)} Val patients")

    train_loader = make_data_loader(train_recs, config, training=True, batch_size=batch_size)

    # Initialize model with/without refinement
    model = ABRMCST(num_slices=6, base_channels=64, enable_refinement=enable_refinement).to(device)
    loss_fn = ABRMLoss(lambda_boundary=lambda_b, supervise_initial=enable_refinement).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ema = EMA(model, decay=0.999)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    best_val_dice = -1.0
    history = []
    t_start = time.time()

    for epoch in range(epochs):
        t_ep_start = time.time()
        lr_curr = warmup_cosine_lr(optimizer, epoch, 1, epochs, lr, 1e-6)

        train_res = train_one_epoch(
            model, train_loader, optimizer, loss_fn, ema, scaler, device,
            accum_steps=accum_steps, grad_clip=1.0, epoch=epoch, is_abrm=True
        )

        ema.apply_shadow(model)
        val_res = validate_dataset(model, val_recs, config, device, is_abrm=True, batch_size=batch_size)
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
            "lr": lr_curr,
            "vram_mb": vram_mb,
            "epoch_duration_s": ep_duration,
        }
        history.append(epoch_stat)

        print(f"[{experiment_type} Epoch {epoch+1:02d}/{epochs:02d}] "
              f"Train Loss: {train_res['loss']:.4f} | "
              f"Val Dice: {val_res['dice']:.4f} | "
              f"IoU: {val_res['iou']:.4f} | "
              f"BF1: {val_res['boundary_f1']:.4f} | "
              f"HD95: {val_res['hd95']:.2f}mm | "
              f"VRAM: {vram_mb:.1f}MB | "
              f"Time: {ep_duration:.1f}s")

        if val_res["dice"] > best_val_dice:
            best_val_dice = val_res["dice"]
            save_checkpoint(ckpt_dir / ckpt_name, epoch, model, ema, optimizer, val_res, config)

    total_training_time = time.time() - t_start
    print(f"\nExperiment {experiment_type} Completed! Total Time: {total_training_time:.1f}s | Best Val Dice: {best_val_dice:.4f}")

    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "experiment": f"{experiment_type}: {'ABRM Full Refinement' if enable_refinement else 'Boundary Head Only'}",
            "enable_refinement": enable_refinement,
            "lambda_boundary": lambda_b,
            "total_time_s": total_training_time,
            "best_val_dice": best_val_dice,
            "history": history,
            "config": config,
        }, f, indent=2)
    print(f"Saved {experiment_type} training log to: {metrics_file}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train ABRM Models (E2 and/or E3)")
    parser.add_argument("--mode", choices=["e2", "e3", "both"], default="both", help="Which experiment to run")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--accum-steps", type=int, default=4, help="Accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda-b", type=float, default=0.2, help="Auxiliary boundary loss weight")
    args = parser.parse_args()

    if args.mode in ("e2", "both"):
        run_abrm_experiment("E2", epochs=args.epochs, batch_size=args.batch_size, accum_steps=args.accum_steps, lr=args.lr, lambda_b=args.lambda_b)

    if args.mode in ("e3", "both"):
        run_abrm_experiment("E3", epochs=args.epochs, batch_size=args.batch_size, accum_steps=args.accum_steps, lr=args.lr, lambda_b=args.lambda_b)


if __name__ == "__main__":
    main()
