#!/usr/bin/env python3
"""
Generate PNG assets under docs/assets/demo/ for README / DEMO.md.

Runs a short surrogate training pass (stdout captured) and evaluation,
then plots velocity slice + loss curves + pooled MAE bars.

Requires: matplotlib, h5py, numpy, torch (same as training stack).
Run from repo root: python scripts/capture_demo_assets.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "demo"
H5 = ROOT / "data" / "smoke.h5"
RUN_DIR = ROOT / "runs" / "demo_asset_gen"


def plot_velocity_slice() -> None:
    if not H5.is_file():
        raise FileNotFoundError(f"Missing {H5} — run scripts/smoke_test.sh or data/generate.py first.")

    with h5py.File(H5, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        grp = f[keys[0]]
        vel = np.asarray(grp["velocity"], dtype=np.float64)
    # vel: [3, nx, ny, nz]
    speed = np.linalg.norm(vel, axis=0)
    nx, ny, nz = speed.shape
    sl = speed[nx // 2, :, :]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(sl, cmap="viridis", origin="lower")
    ax.set_title("Velocity magnitude — mid X slice (smoke sample)")
    ax.set_xlabel("Z index")
    ax.set_ylabel("Y index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "velocity_slice.png", dpi=150)
    plt.close(fig)


def run_train_capture_losses() -> tuple[list[float], list[float]]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "train" / "train_tfno.py"),
        "--no-tfno",
        "--epochs",
        "12",
        "--cpu-only",
        "--h5",
        str(H5),
        "--out-dir",
        str(RUN_DIR),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + "\n" + proc.stderr)

    train_losses: list[float] = []
    val_losses: list[float] = []
    pat = re.compile(r"epoch (\d+) train_mse_norm=([\d.]+) val_mse_norm=([\d.]+)")
    for line in proc.stdout.splitlines():
        m = pat.search(line)
        if m:
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))
    return train_losses, val_losses


def plot_losses(train_losses: list[float], val_losses: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train MSE (norm)", marker="o", ms=3)
    ax.plot(epochs, val_losses, label="val MSE (norm)", marker="o", ms=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Surrogate training (Conv3D fallback, smoke HDF5)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "training_loss.png", dpi=150)
    plt.close(fig)


def run_eval_bar() -> None:
    ckpt = RUN_DIR / "surrogate_best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    eval_out = RUN_DIR / "eval_pooled.json"
    cmd = [
        sys.executable,
        str(ROOT / "train" / "eval_surrogate.py"),
        "--no-tfno",
        "--cpu-only",
        "--h5",
        str(H5),
        "--ckpt",
        str(ckpt),
        "--out",
        str(eval_out),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    metrics = json.loads(eval_out.read_text(encoding="utf-8"))
    labels = ["mean_speed", "mean_pressure", "max_speed"]
    vals = [metrics["mae_mean_speed"], metrics["mae_mean_pressure"], metrics["mae_max_speed"]]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(labels, vals, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_ylabel("MAE vs pooled solver scalars (val shard)")
    ax.set_title("Surrogate vs HDF5 ground truth (smoke scale)")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(OUT / "pooled_mae.png", dpi=150)
    plt.close(fig)


def copy_checkpoint_for_gradio() -> None:
    dst_dir = ROOT / "runs" / "tfno_train"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUN_DIR / "surrogate_best.pt", dst_dir / "surrogate_best.pt")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_velocity_slice()
    train_losses, val_losses = run_train_capture_losses()
    if not train_losses:
        print("Could not parse training losses from stdout.", file=sys.stderr)
        return 1
    plot_losses(train_losses, val_losses)
    run_eval_bar()
    copy_checkpoint_for_gradio()
    print(f"Wrote assets under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
