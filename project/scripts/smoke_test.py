"""
Two-Epoch Smoke Test for Cross-Slice Transformer and ABRM.

Verifies:
1. CUDA availability, GPU device identification, and FP16 support.
2. Tensor shape verification for CST baseline and ABRM-CST.
3. Gradient backward pass and optimizer step.
4. Loss function verification (OriginalCombinedLoss and ABRMLoss).
5. 2-epoch end-to-end training and evaluation cycle.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root and src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.amp import GradScaler

from src.models.cst import CST
from src.models.abr_module import ABRMCST, extract_boundary_gt
from src.losses.original_loss import OriginalCombinedLoss, compute_sdt_batch
from src.losses.contour_loss import ABRMLoss
from src.data.preprocessing import create_mock_patient_record
from src.data.dataset import make_data_loader
from src.training.trainer import EMA, train_one_epoch, validate_dataset, save_checkpoint


def run_smoke_test():
    print("=" * 60)
    print("PHASE 4: TWO-EPOCH SMOKE TEST (CST BASELINE & ABRM)")
    print("=" * 60)

    # 1. Device and CUDA Verification
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0).upper()
        if "1650" in dev_name or "1660" in dev_name:
            torch.backends.cudnn.enabled = False
            print("Note: GTX 1650/1660 detected -> disabled cuDNN FP16 conv to prevent native kernel NaN bug.")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
        print(f"VRAM Allocated: {torch.cuda.memory_allocated(0) / (1024**2):.1f} MB")
    else:
        print("WARNING: CUDA is not available. Running on CPU.")

    # 2. Config Loading
    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    # Override for quick smoke test
    config["training"]["epochs"] = 2
    config["training"]["warmup_epochs"] = 1
    config["training"]["batch_size"] = 2
    config["training"]["grad_accum_steps"] = 1

    # 3. Model Architecture Verification
    print("\n--- Model Architecture & Forward/Backward Verification ---")
    x = torch.randn(2, 6, 256, 256, device=device)
    target_mask = (torch.rand(2, 1, 256, 256, device=device) > 0.9).float()
    valid_mask = torch.ones(2, 1, 256, 256, device=device)
    tier_flags = torch.ones(2, dtype=torch.bool, device=device)

    # Baseline CST
    cst_model = CST(num_slices=6, base_channels=64).to(device)
    out_cst = cst_model(x)
    print(f"CST forward pass OK. Output logits shape: {out_cst[0].shape}, DS1: {out_cst[1].shape}, DS2: {out_cst[2].shape}")
    assert out_cst[0].shape == (2, 1, 256, 256), f"Wrong CST output shape: {out_cst[0].shape}"

    # ABRM CST
    abrm_model = ABRMCST(num_slices=6, base_channels=64, enable_refinement=True).to(device)
    out_abrm = abrm_model(x)
    print(f"ABRM forward pass OK. Final logits shape: {out_abrm[0].shape}, Initial: {out_abrm[1].shape}, Boundary: {out_abrm[2].shape}")
    assert out_abrm[0].shape == (2, 1, 256, 256)
    assert out_abrm[2].shape == (2, 1, 256, 256)

    # Loss verification
    orig_loss_fn = OriginalCombinedLoss().to(device)
    l_orig = orig_loss_fn(out_cst, target_mask, valid_mask, tier_flags)
    l_orig.backward()
    print(f"Original loss backward pass OK: {l_orig.item():.4f}")

    abrm_loss_fn = ABRMLoss(lambda_boundary=0.2).to(device)
    l_abrm, loss_dict = abrm_loss_fn(out_abrm, target_mask, valid_mask, tier_flags)
    l_abrm.backward()
    print(f"ABRM loss backward pass OK: {l_abrm.item():.4f} (l_final: {loss_dict['l_final']:.4f}, l_contour: {loss_dict['l_contour']:.4f})")

    # 4. Generate Smoke Test Data
    print("\n--- Generating Small Synthetic Cohort for Smoke Testing ---")
    data_dir = ROOT / "data" / "smoke_cohort"
    data_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(4):  # 3 train, 1 val
        rec = create_mock_patient_record(f"SMOKE_PATIENT_{i:03d}", data_dir, n_slices=20, seed=100 + i)
        records.append(rec)

    train_recs = records[:3]
    val_recs = records[3:]
    print(f"Created {len(train_recs)} training patients and {len(val_recs)} validation patient.")

    train_loader = make_data_loader(train_recs, config, training=True, batch_size=2)
    val_loader = make_data_loader(val_recs, config, training=False, batch_size=2)

    # 5. Run 2-Epoch Training on CST Baseline
    print("\n--- Running 2-Epoch Smoke Training: CST Baseline ---")
    optimizer = torch.optim.AdamW(cst_model.parameters(), lr=1e-4, weight_decay=1e-4)
    ema = EMA(cst_model, decay=0.999)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    t0 = time.time()
    for ep in range(2):
        train_res = train_one_epoch(
            cst_model, train_loader, optimizer, orig_loss_fn, ema, scaler, device,
            accum_steps=1, grad_clip=1.0, epoch=ep, is_abrm=False
        )
        ema.apply_shadow(cst_model)
        val_res = validate_dataset(cst_model, val_recs, config, device, is_abrm=False, batch_size=4)
        ema.restore(cst_model)
        print(f"[CST Epoch {ep+1}/2] Train Loss: {train_res['loss']:.4f} | Val Dice: {val_res['dice']:.4f} | Val IoU: {val_res['iou']:.4f}")

    print(f"CST Smoke Test Completed in {time.time() - t0:.1f}s.")

    # 6. Run 2-Epoch Training on ABRM
    print("\n--- Running 2-Epoch Smoke Training: Proposed ABRM-CST ---")
    optimizer_abrm = torch.optim.AdamW(abrm_model.parameters(), lr=1e-4, weight_decay=1e-4)
    ema_abrm = EMA(abrm_model, decay=0.999)
    scaler_abrm = GradScaler("cuda" if device.type == "cuda" else "cpu")

    t0 = time.time()
    for ep in range(2):
        train_res = train_one_epoch(
            abrm_model, train_loader, optimizer_abrm, abrm_loss_fn, ema_abrm, scaler_abrm, device,
            accum_steps=1, grad_clip=1.0, epoch=ep, is_abrm=True
        )
        ema_abrm.apply_shadow(abrm_model)
        val_res = validate_dataset(abrm_model, val_recs, config, device, is_abrm=True, batch_size=4)
        ema_abrm.restore(abrm_model)
        print(f"[ABRM Epoch {ep+1}/2] Train Loss: {train_res['loss']:.4f} | Val Dice: {val_res['dice']:.4f} | Boundary F1: {val_res['boundary_f1']:.4f}")

    print(f"ABRM Smoke Test Completed in {time.time() - t0:.1f}s.")
    print("\n>>> ALL SMOKE TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_smoke_test()
