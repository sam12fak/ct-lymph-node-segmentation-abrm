"""
Comprehensive Evaluation and Benchmarking Script.

Evaluates E1 (Baseline CST), E2 (CST + Boundary Aux), and E3 (CST + ABRM Refinement).
Computes:
- DSC, IoU, Precision, Recall
- Boundary F1 Score
- Hausdorff Distance 95 (HD95)
- Parameter counts and inference latency
- Size-stratified breakdown (<10 mm, 10-15 mm, >=15 mm)
- Generates 6-panel qualitative visualization figures
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from src.models.cst import CST
from src.models.abr_module import ABRMCST
from src.evaluation.metrics import (
    compute_dice_iou,
    compute_boundary_f1,
    compute_hd95,
    size_stratified_evaluation,
    benchmark_model,
)
from src.training.trainer import load_checkpoint, evaluate_patient_volume
from src.visualization.visualize import save_qualitative_figure
from src.data.preprocessing import create_mock_patient_record


def evaluate_all_experiments(num_patients: int = 8, n_slices: int = 20):
    print("=" * 70)
    print("PHASE 8: EVALUATION, BENCHMARKING & QUALITATIVE VISUALIZATION")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0).upper()
        if "1650" in dev_name or "1660" in dev_name:
            torch.backends.cudnn.enabled = False
    print(f"Device: {device}")

    with open(ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    # Load validation cohort
    data_dir = ROOT / "data" / "cohort"
    data_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(num_patients):
        rec = create_mock_patient_record(f"PATIENT_{i:03d}", data_dir, n_slices=n_slices, seed=42 + i)
        records.append(rec)

    n_train = max(1, int(len(records) * 0.8))
    val_recs = records[n_train:]
    print(f"Evaluating across {len(val_recs)} holdout validation patients.\n")

    experiments = [
        {
            "id": "E1",
            "name": "Original CST Baseline",
            "model_type": "cst",
            "ckpt": ROOT / "checkpoints" / "cst_baseline_best.pt",
            "is_abrm": False,
            "enable_refinement": False,
        },
        {
            "id": "E2",
            "name": "CST + Auxiliary Boundary Head",
            "model_type": "abrm",
            "ckpt": ROOT / "checkpoints" / "cst_boundary_aux_best.pt",
            "is_abrm": True,
            "enable_refinement": False,
        },
        {
            "id": "E3",
            "name": "CST + Proposed ABRM (Refinement)",
            "model_type": "abrm",
            "ckpt": ROOT / "checkpoints" / "cst_abrm_best.pt",
            "is_abrm": True,
            "enable_refinement": True,
        },
    ]

    summary_results = []
    viz_dir = ROOT / "results" / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    loaded_models = {}

    for exp in experiments:
        print(f"--- Evaluating {exp['id']}: {exp['name']} ---")
        if exp["model_type"] == "cst":
            model = CST(num_slices=6, base_channels=64).to(device)
        else:
            model = ABRMCST(num_slices=6, base_channels=64, enable_refinement=exp["enable_refinement"]).to(device)

        if exp["ckpt"].exists():
            load_checkpoint(exp["ckpt"], model, device=str(device))
            print(f"Loaded checkpoint from: {exp['ckpt'].name}")
        else:
            print(f"Notice: Checkpoint {exp['ckpt'].name} not found, using initialized weights.")

        loaded_models[exp["id"]] = model

        # Benchmark complexity and inference speed
        bench = benchmark_model(model, input_shape=(1, 6, 256, 256), device=str(device), n_runs=30)

        # Patient evaluations
        patient_metrics = []
        strat_metrics = []
        for rec in val_recs:
            m = evaluate_patient_volume(model, rec, config, device, batch_size=4, is_abrm=exp["is_abrm"])
            patient_metrics.append(m)

            # Volume-level size stratification
            seg = np.load(rec["seg_path"])
            # Reconstruct prediction
            ct = np.load(rec["ct_path"])
            prior = np.load(rec["prior_path"])
            prob_vol = np.zeros_like(seg, dtype=np.float32)
            for z in range(ct.shape[0]):
                inp = torch.from_numpy(np.stack([
                    ct[max(0, z-2)], ct[max(0, z-1)], ct[z], ct[min(ct.shape[0]-1, z+1)], ct[min(ct.shape[0]-1, z+2)], prior[z]
                ], axis=0)[np.newaxis, ...]).float().to(device)
                with torch.no_grad():
                    out = model(inp)
                prob_vol[z] = torch.sigmoid(out[0]).squeeze().cpu().numpy()
            pred_binary = (prob_vol >= 0.5).astype(np.uint8)

            strat = size_stratified_evaluation(pred_binary, seg)
            strat_metrics.append(strat)

        mean_dice = float(np.mean([m["dice"] for m in patient_metrics]))
        mean_iou = float(np.mean([m["iou"] for m in patient_metrics]))
        mean_prec = float(np.mean([m["precision"] for m in patient_metrics]))
        mean_rec = float(np.mean([m["recall"] for m in patient_metrics]))
        mean_bf1 = float(np.mean([m["boundary_f1"] for m in patient_metrics]))
        mean_hd95 = float(np.mean([m["hd95"] for m in patient_metrics]))

        # Stratified means
        small_dice = float(np.nanmean([s["small (<10mm)"]["dice"] for s in strat_metrics]))
        med_dice = float(np.nanmean([s["medium (10-15mm)"]["dice"] for s in strat_metrics]))
        large_dice = float(np.nanmean([s["large (>=15mm)"]["dice"] for s in strat_metrics]))

        res = {
            "id": exp["id"],
            "name": exp["name"],
            "dice": mean_dice,
            "iou": mean_iou,
            "precision": mean_prec,
            "recall": mean_rec,
            "boundary_f1": mean_bf1,
            "hd95": mean_hd95,
            "small_dice": small_dice,
            "med_dice": med_dice,
            "large_dice": large_dice,
            "params_M": bench["total_params"] / 1e6,
            "latency_ms": bench["latency_ms_per_slice"],
            "fps": bench["fps"],
        }
        summary_results.append(res)
        print(f"DSC: {mean_dice:.4f} | IoU: {mean_iou:.4f} | BF1: {mean_bf1:.4f} | HD95: {mean_hd95:.2f}mm | Latency: {bench['latency_ms_per_slice']:.2f}ms")

    # Generate Qualitative Visualizations
    print("\n--- Generating Qualitative Comparison Visualizations ---")
    val_sample = val_recs[0]
    ct_vol = np.load(val_sample["ct_path"])
    seg_vol = np.load(val_sample["seg_path"])
    prior_vol = np.load(val_sample["prior_path"])

    # Pick representative positive slice containing lymph node
    pos_slices = val_sample["positive_slices_voi"]
    chosen_z = pos_slices[len(pos_slices) // 2] if pos_slices else ct_vol.shape[0] // 2

    # Prepare 6-channel input for chosen_z
    ct_stack = np.stack([
        ct_vol[max(0, chosen_z - 2)],
        ct_vol[max(0, chosen_z - 1)],
        ct_vol[chosen_z],
        ct_vol[min(ct_vol.shape[0] - 1, chosen_z + 1)],
        ct_vol[min(ct_vol.shape[0] - 1, chosen_z + 2)],
        prior_vol[chosen_z],
    ], axis=0)
    inp_tensor = torch.from_numpy(ct_stack[np.newaxis, ...]).float().to(device)

    # Predictions
    with torch.no_grad():
        cst_logits = loaded_models["E1"](inp_tensor)[0]
        cst_pred = (torch.sigmoid(cst_logits).squeeze().cpu().numpy() >= 0.5).astype(np.uint8)

        abrm_out = loaded_models["E3"](inp_tensor)
        abrm_pred = (torch.sigmoid(abrm_out[0]).squeeze().cpu().numpy() >= 0.5).astype(np.uint8)
        boundary_pred = torch.sigmoid(abrm_out[2]).squeeze().cpu().numpy()

    viz_path = viz_dir / "qualitative_lymph_node_comparison.png"
    save_qualitative_figure(
        ct_slice=ct_vol[chosen_z],
        gt_mask=seg_vol[chosen_z],
        cst_pred=cst_pred,
        boundary_pred=boundary_pred,
        abrm_pred=abrm_pred,
        save_path=viz_path,
        title=f"Lymph Node Boundary Refinement Comparison (Patient {val_sample['patient_id']}, Slice {chosen_z})"
    )

    # Save summary json
    summary_path = ROOT / "results" / "quantitative_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nSaved quantitative results to: {summary_path}")

    # Print Formatted Markdown Table
    print("\n" + "=" * 80)
    print("QUANTITATIVE COMPARISON TABLE")
    print("=" * 80)
    print("| Method | Dice (DSC) | IoU | Boundary F1 | HD95 (mm) | Small (<10mm) | Med (10-15mm) | Large (>=15mm) | Params (M) | Latency (ms) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in summary_results:
        print(f"| **{r['name']}** | {r['dice']:.4f} | {r['iou']:.4f} | {r['boundary_f1']:.4f} | {r['hd95']:.2f} | {r['small_dice']:.4f} | {r['med_dice']:.4f} | {r['large_dice']:.4f} | {r['params_M']:.2f}M | {r['latency_ms']:.1f}ms |")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_all_experiments()
