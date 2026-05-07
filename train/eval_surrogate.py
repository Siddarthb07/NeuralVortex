#!/usr/bin/env python3
"""Evaluate surrogate RMSE on held-out HDF5 samples (physical units, pooled scalars)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train.field_dataset import NeuralVortexFieldDataset, params_to_grid_tensor  # noqa: E402
from train.hdf5_loader import pooled_features  # noqa: E402
from train.infer_surrogate import load_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=Path("data/smoke.h5"))
    ap.add_argument("--ckpt", type=Path, default=Path("runs/tfno_train/surrogate_best.pt"))
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--no-tfno", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/surrogate_eval_pooled.json"))
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu_only or not torch.cuda.is_available() else "cuda")
    if not args.ckpt.is_file():
        print(f"Missing checkpoint {args.ckpt} — run train/train_tfno.py first.")
        return 1

    ds = NeuralVortexFieldDataset(args.h5)
    n = len(ds)
    if n < 2:
        print("Need >=2 samples.")
        return 1

    val_n = max(1, int(round(n * args.val_frac)))
    train_n = n - val_n
    if train_n < 1:
        train_n, val_n = n - 1, 1
    _, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))

    model, ckpt = load_checkpoint(args.ckpt, prefer_tfno=not args.no_tfno, device=device)

    errs = []
    for i in range(len(val_ds)):
        xb, yb = val_ds[i]
        idx = val_ds.indices[i]
        inp, tgt_np = ds.samples[idx]
        rpm_lo, rpm_hi = ds.rpm_lo, ds.rpm_hi
        grid_shape = ds.spatial_shape
        x = params_to_grid_tensor(inp, rpm_lo=rpm_lo, rpm_hi=rpm_hi, grid_shape=grid_shape).unsqueeze(0).to(device)
        mean = ckpt["norm"]["mean"].to(device)
        std = ckpt["norm"]["std"].to(device)
        with torch.no_grad():
            pred_n = model(x)[0]
            pred = pred_n * std + mean
        pred_np = pred.cpu().numpy()
        gt = torch.from_numpy(tgt_np).to(device)
        gt_np = gt.cpu().numpy()
        pe = pooled_features({"velocity": pred_np[:3], "pressure": pred_np[3]})
        ge = pooled_features({"velocity": gt_np[:3], "pressure": gt_np[3]})
        errs.append(np.abs(pe - ge))

    errs = np.stack(errs, axis=0)
    mae = errs.mean(axis=0).tolist()
    metrics = {
        "n_val": len(val_ds),
        "mae_mean_speed": mae[0],
        "mae_mean_pressure": mae[1],
        "mae_max_speed": mae[2],
        "checkpoint": str(args.ckpt.resolve()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
