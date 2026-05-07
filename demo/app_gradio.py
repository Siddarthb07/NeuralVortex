#!/usr/bin/env python3
"""
Phase 4 — Gradio UI for NeuralVortex surrogate (pooled flow scalars).

Requires a checkpoint from `train/train_tfno.py` (`runs/tfno_train/surrogate_best.pt` by default).

Install: `pip install gradio torch` (+ train deps). Run from repo root:

    python demo/app_gradio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    try:
        import gradio as gr
        import torch
    except ImportError as e:
        raise SystemExit(f"Missing dependency: {e}. pip install gradio torch") from e

    from train.infer_surrogate import forward_pooled, load_checkpoint

    ckpt_path = ROOT_DIR / "runs" / "tfno_train" / "surrogate_best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefer_tfno = True

    def load_or_none():
        if not ckpt_path.is_file():
            return None, None
        model, ckpt = load_checkpoint(ckpt_path, prefer_tfno=prefer_tfno, device=device)
        return model, ckpt

    model, ckpt = load_or_none()

    def predict(rpm: float, blades: float, pitch_deg: float, v_inflow: float):
        if model is None or ckpt is None:
            return (
                "### Checkpoint missing\nTrain first:\n\n"
                "```bash\npip install -r requirements-train.txt\n"
                "python train/train_tfno.py --no-tfno --epochs 30 --cpu-only\n```",
                {},
            )
        inp = {"rpm": rpm, "blades": blades, "pitch_deg": pitch_deg, "v_inflow": v_inflow}
        pools = forward_pooled(model, ckpt, inp, device=device)
        md = (
            "| Metric | Value |\n| --- | --- |\n"
            f"| Mean speed | {pools['mean_speed']:.6g} |\n"
            f"| Mean pressure | {pools['mean_pressure']:.6g} |\n"
            f"| Max speed | {pools['max_speed']:.6g} |\n"
        )
        return md, pools

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Slider(1000, 10000, value=4500, step=100, label="rpm"),
            gr.Slider(2, 6, value=4, step=1, label="blades"),
            gr.Slider(5, 25, value=12, step=0.5, label="pitch_deg"),
            gr.Slider(0, 15, value=5, step=0.5, label="v_inflow (m/s)"),
        ],
        outputs=[gr.Markdown(), gr.JSON(label="raw_json")],
        title="NeuralVortex surrogate (pooled outputs)",
        description=f"Loads `{ckpt_path}` when present. Conv3D fallback training matches `--no-tfno` checkpoints.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
