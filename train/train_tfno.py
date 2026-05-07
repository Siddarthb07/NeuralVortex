#!/usr/bin/env python3
"""
Phase 2 — Train a Fourier / spectral surrogate on NeuralVortex HDF5 fields.

Prefers `neuralop.models.TFNO`; falls back to a shallow Conv3D stack if TFNO is missing.

Optional Weights & Biases logging via `--wandb` (requires `WANDB_API_KEY`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train.field_dataset import NeuralVortexFieldDataset  # noqa: E402
from train.surrogate_models import TinyConv3dSurrogate, build_field_surrogate  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--h5", type=Path, default=Path("data/smoke.h5"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    p.add_argument("--wandb-project", type=str, default="NeuralVortex")
    p.add_argument("--out-dir", type=Path, default=Path("runs/tfno_train"))
    p.add_argument("--no-tfno", action="store_true", help="Force Conv3D surrogate")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    if not args.cpu_only and torch.cuda.is_available():
        device = torch.device("cuda")

    ds = NeuralVortexFieldDataset(args.h5)
    n = len(ds)
    if n < 2:
        print("Need >=2 HDF5 samples for train/val split.")
        return 1

    val_n = max(1, int(round(n * args.val_frac)))
    train_n = n - val_n
    if train_n < 1:
        train_n, val_n = n - 1, 1
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=min(args.batch_size, train_n), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=min(args.batch_size, val_n), shuffle=False)

    model = build_field_surrogate(
        prefer_tfno=not args.no_tfno,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    wb = None
    if args.wandb:
        if not os.environ.get("WANDB_API_KEY"):
            print("WANDB_API_KEY not set; disabling --wandb.")
        else:
            import wandb

            wb = wandb.init(project=args.wandb_project, config=vars(args))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    metrics_path = args.out_dir / "metrics.json"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.detach()) * len(xb)
        train_loss /= train_n

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_loss += float(loss_fn(pred, yb)) * len(xb)
        val_loss /= val_n

        if wb:
            wb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"epoch {epoch:03d} train_mse_norm={train_loss:.6f} val_mse_norm={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            arch = "tiny_conv3d" if isinstance(model, TinyConv3dSurrogate) else "tfno"
            ckpt = {
                "model_state": model.state_dict(),
                "norm": {"mean": ds.target_mean.cpu(), "std": ds.target_std.cpu()},
                "rpm_bounds": [ds.rpm_lo, ds.rpm_hi],
                "h5": str(args.h5.resolve()),
                "spatial_shape": list(ds.spatial_shape),
                "arch": arch,
            }
            torch.save(ckpt, args.out_dir / "surrogate_best.pt")
            metrics_path.write_text(
                json.dumps(
                    {
                        "best_val_mse_normalized": best_val,
                        "epochs_ran": epoch,
                        "checkpoint": str((args.out_dir / "surrogate_best.pt").resolve()),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    if wb:
        wb.finish()

    print(f"Best val MSE (normalized targets): {best_val:.6f}")
    print(f"Wrote {(args.out_dir / 'surrogate_best.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
