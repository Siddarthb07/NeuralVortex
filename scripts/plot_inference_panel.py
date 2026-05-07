#!/usr/bin/env python3
"""Render pooled surrogate prediction as a static panel (Gradio-equivalent sliders)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.infer_surrogate import forward_pooled, load_checkpoint  # noqa: E402


def main() -> int:
    ckpt_path = ROOT / "runs" / "demo_asset_gen" / "surrogate_best.pt"
    if not ckpt_path.is_file():
        ckpt_path = ROOT / "runs" / "tfno_train" / "surrogate_best.pt"
    if not ckpt_path.is_file():
        print("No checkpoint found — run scripts/capture_demo_assets.py first.", file=sys.stderr)
        return 1

    out = ROOT / "docs" / "assets" / "demo" / "surrogate_inference_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model, ckpt = load_checkpoint(ckpt_path, prefer_tfno=False, device=device)
    inp = {"rpm": 4500.0, "blades": 4.0, "pitch_deg": 12.0, "v_inflow": 5.0}
    pools = forward_pooled(model, ckpt, inp, device=device)

    fig = plt.figure(figsize=(6.5, 4.0))
    fig.patch.set_facecolor("#f7f7f7")
    ax = fig.add_subplot(111)
    ax.axis("off")
    lines = [
        "NeuralVortex surrogate — pooled outputs",
        "(matches demo/app_gradio.py sliders)",
        "",
        f"rpm = {inp['rpm']:.0f}   blades = {inp['blades']:.0f}",
        f"pitch_deg = {inp['pitch_deg']:.1f}   v_inflow = {inp['v_inflow']:.1f} m/s",
        "",
        f"mean_speed     = {pools['mean_speed']:.6g}",
        f"mean_pressure  = {pools['mean_pressure']:.6g}",
        f"max_speed      = {pools['max_speed']:.6g}",
        "",
        f"checkpoint: {ckpt_path.name}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=11, verticalalignment="top", family="monospace")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
