#!/usr/bin/env python3
"""
Phase 4 — Gradio dashboard covering NeuralVortex Phases 1–4.

Phases in the UI mirror the project roadmap:
  1. HDF5 dataset metadata + nearest solver sample vs sliders
  2. Surrogate full-field stats + pooled scalars + |v| slice
  3. Divergence diagnostics (surrogate vs nearest HDF5 velocity)
  4. Reproduction commands + doc pointers

Environment:
  NEURALVORTEX_CKPT — checkpoint path (default runs/tfno_train/surrogate_best.pt)
  NEURALVORTEX_H5   — HDF5 for Phase 1 panels (default data/smoke.h5)

Run from repo root:
    pip install -r requirements-demo.txt
    pip install -r requirements-train.txt
    python demo/app_gradio.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_deep_dive_text() -> str:
    p = ROOT_DIR / "docs" / "DEEP_DIVE.md"
    if not p.is_file():
        return "*Missing `docs/DEEP_DIVE.md` — add it from the repo.*"
    return p.read_text(encoding="utf-8")


def _channels_table(rows: list[dict]) -> str:
    md = "| Channel | mean | std | min | max |\n| --- | --- | --- | --- | --- |\n"
    for r in rows:
        md += (
            f"| `{r['channel']}` | {r['mean']:.6g} | {r['std']:.6g} | "
            f"{r['min']:.6g} | {r['max']:.6g} |\n"
        )
    return md


def _div_block(title: str, vnorm: float, dnorm: float, ratio: float) -> str:
    return (
        f"#### {title}\n\n"
        f"| Quantity | Value |\n| --- | --- |\n"
        f"| $\\|v\\|_2$ | {vnorm:.6e} |\n"
        f"| $\\|\\nabla\\cdot v\\|_2$ (central FD) | {dnorm:.6e} |\n"
        f"| ratio | {ratio:.6e} |\n"
    )


def main() -> None:
    try:
        import gradio as gr
        import torch
    except ImportError as e:
        raise SystemExit(f"Missing dependency: {e}. pip install gradio torch") from e

    from PIL import Image

    from demo.dashboard_data import attrs_markdown, nearest_markdown, nearest_solver_sample, read_global_attrs
    from train.field_metrics import divergence_norms, velocity_magnitude_slice_png
    from train.hdf5_loader import pooled_features
    from train.infer_surrogate import forward_full_field, load_checkpoint

    _env_ckpt = os.environ.get("NEURALVORTEX_CKPT", "").strip()
    ckpt_path = Path(_env_ckpt) if _env_ckpt else (ROOT_DIR / "runs" / "tfno_train" / "surrogate_best.pt")
    _env_h5 = os.environ.get("NEURALVORTEX_H5", "").strip()
    h5_path = Path(_env_h5) if _env_h5 else (ROOT_DIR / "data" / "smoke.h5")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefer_tfno = True

    def load_or_none():
        if not ckpt_path.is_file():
            return None, None
        model, ckpt = load_checkpoint(ckpt_path, prefer_tfno=prefer_tfno, device=device)
        return model, ckpt

    model, ckpt = load_or_none()
    deep_md = _load_deep_dive_text()

    default_rpm_lo, default_rpm_hi = 1000.0, 10000.0
    if ckpt is not None:
        default_rpm_lo = float(ckpt["rpm_bounds"][0])
        default_rpm_hi = float(ckpt["rpm_bounds"][1])

    def run_dashboard(rpm: float, blades: float, pitch_deg: float, v_inflow: float, slice_axis: int):
        inp = {"rpm": rpm, "blades": blades, "pitch_deg": pitch_deg, "v_inflow": v_inflow}
        slice_axis = int(max(0, min(2, slice_axis)))

        attrs_md = attrs_markdown(read_global_attrs(h5_path))
        nearest_md = "*Load a checkpoint and HDF5 to compare nearest solver sample.*"
        phase2_md = "### Surrogate output\n\n*No checkpoint — train with `train/train_tfno.py`.*"
        phase3_md = "### Divergence\n\n*Need surrogate velocity and HDF5 for comparison.*"
        plot_out = None
        serial: dict = {"controls": inp, "checkpoint": str(ckpt_path), "h5": str(h5_path), "errors": []}

        rpm_lo, rpm_hi = default_rpm_lo, default_rpm_hi
        if ckpt is not None:
            rpm_lo = float(ckpt["rpm_bounds"][0])
            rpm_hi = float(ckpt["rpm_bounds"][1])

        if h5_path.is_file():
            try:
                dist, key, sinp, arrs, prop = nearest_solver_sample(
                    h5_path, inp, rpm_lo=rpm_lo, rpm_hi=rpm_hi
                )
                nearest_md = nearest_markdown(dist, key, sinp, prop)
                pools_h5 = pooled_features(arrs)
                serial["nearest_sample"] = {
                    "distance": dist,
                    "group": key,
                    "inputs": sinp,
                    "propeller": prop,
                    "pooled_solver": {
                        "mean_speed": float(pools_h5[0]),
                        "mean_pressure": float(pools_h5[1]),
                        "max_speed": float(pools_h5[2]),
                    },
                }
                vn_h5, dn_h5, r_h5 = divergence_norms(arrs["velocity"])
                phase3_md = _div_block("Nearest HDF5 velocity (solver truth)", vn_h5, dn_h5, r_h5)
            except Exception as e:
                serial["errors"].append(f"nearest_sample: {e}")
                nearest_md = f"*Could not read nearest sample:* `{e}`"
                phase3_md = f"*HDF5 error:* `{e}`"
        else:
            nearest_md = f"*HDF5 not found:* `{h5_path}` — set `NEURALVORTEX_H5`."
            serial["errors"].append("h5_missing")

        if model is not None and ckpt is not None:
            try:
                ff = forward_full_field(model, ckpt, inp, device=device)
                phase2_md = (
                    "### Pooled scalars (Phase 2 summary)\n\n"
                    "| Metric | Value |\n| --- | --- |\n"
                    f"| Mean speed | {ff['pooled']['mean_speed']:.6g} |\n"
                    f"| Mean pressure | {ff['pooled']['mean_pressure']:.6g} |\n"
                    f"| Max speed | {ff['pooled']['max_speed']:.6g} |\n\n"
                    "### Per-channel statistics (full volumetric prediction)\n\n"
                    + _channels_table(ff["channels"])
                )
                png = velocity_magnitude_slice_png(ff["velocity"], axis=slice_axis)
                plot_out = Image.open(io.BytesIO(png))

                vn_m, dn_m, r_m = divergence_norms(ff["velocity"])
                if phase3_md.startswith("### Divergence") or "*HDF5 error*" in phase3_md:
                    phase3_md = ""
                phase3_md += "\n\n" + _div_block("Surrogate predicted velocity", vn_m, dn_m, r_m)

                serial["surrogate"] = {
                    "pooled": ff["pooled"],
                    "channels": ff["channels"],
                    "divergence": {"vnorm": vn_m, "div_norm": dn_m, "ratio": r_m},
                    "velocity_shape": list(ff["velocity"].shape),
                    "pressure_shape": list(ff["pressure"].shape),
                }
            except Exception as e:
                serial["errors"].append(f"surrogate: {e}")
                phase2_md = f"*Surrogate forward failed:* `{e}`"
        elif model is None:
            serial["errors"].append("checkpoint_missing")

        phase4_md = (
            "## Phase 4 — Reproduce & ship\n\n"
            "### Commands\n\n"
            "```bash\n"
            "pip install -r requirements.txt\n"
            "pip install -r requirements-train.txt\n"
            "pip install -r requirements-demo.txt\n"
            "python data/generate.py --n 8 --grid-res 16 --out data/smoke.h5\n"
            "python train/train_tfno.py --no-tfno --epochs 40 --cpu-only --h5 data/smoke.h5\n"
            "python train/eval_surrogate.py --no-tfno --cpu-only\n"
            "python train/pinn_residual_smoke.py --h5 data/smoke.h5\n"
            "python demo/app_gradio.py\n"
            "```\n\n"
            "### Docs in this repo\n\n"
            "- [`docs/DEEP_DIVE.md`](docs/DEEP_DIVE.md) — narrative map (also embedded in **Overview**).\n"
            "- [`docs/report.md`](docs/report.md) — interim technical report.\n"
            "- [`docs/DEMO.md`](docs/DEMO.md) — screenshots & asset regeneration.\n\n"
            "### Hugging Face Space\n\n"
            "Mirror `demo/README.md`: pin `torch`, `gradio`, `numpy`, `h5py`, `matplotlib`, ship a **small** checkpoint "
            "and HDF5, set `PYTHONPATH` to repo root or flatten imports.\n"
        )

        footer = (
            f"**Checkpoint:** `{ckpt_path}` {'✓' if ckpt_path.is_file() else '(missing)'} · "
            f"**HDF5:** `{h5_path}` {'✓' if h5_path.is_file() else '(missing)'} · "
            f"**Device:** `{device}`"
        )

        return (
            attrs_md,
            nearest_md,
            phase2_md,
            phase3_md,
            phase4_md,
            plot_out,
            json.dumps(serial, indent=2),
            footer,
        )

    header = (
        "# NeuralVortex — phased dashboard\n\n"
        f"Loads **`{ckpt_path}`** when present (`NEURALVORTEX_CKPT`). "
        f"HDF5 panels use **`{h5_path}`** (`NEURALVORTEX_H5`). "
        "Use the tabs below to walk Phases 1–4 in order."
    )

    with gr.Blocks(theme=gr.themes.Soft(), title="NeuralVortex") as demo:
        gr.Markdown(header)
        with gr.Row():
            rpm = gr.Slider(1000, 10000, value=4500, step=100, label="rpm")
            blades = gr.Slider(2, 6, value=4, step=1, label="blades")
            pitch_deg = gr.Slider(5, 25, value=12, step=0.5, label="pitch_deg")
            v_inflow = gr.Slider(0, 15, value=5, step=0.5, label="v_inflow (m/s)")
        slice_axis = gr.Radio(
            choices=[("Slice normal ≈ X (axis 0)", 0), ("Slice normal ≈ Y (axis 1)", 1), ("Slice normal ≈ Z (axis 2)", 2)],
            value=0,
            label="Velocity magnitude slice orientation",
        )
        run_btn = gr.Button("Run / refresh all phases", variant="primary")

        with gr.Tabs():
            with gr.Tab("Overview — deep dive"):
                gr.Markdown(deep_md)
            with gr.Tab("Phase 1 — HDF5 data"):
                out_attrs = gr.Markdown()
                out_nearest = gr.Markdown()
            with gr.Tab("Phase 2 — Surrogate field"):
                out_p2 = gr.Markdown()
                out_plot = gr.Image(type="pil", label="|v| mid-slice (surrogate)")
            with gr.Tab("Phase 3 — Divergence"):
                out_p3 = gr.Markdown()
            with gr.Tab("Phase 4 — Ship & docs"):
                out_p4 = gr.Markdown()

        out_json = gr.JSON(label="Structured summary (no raw tensors)")
        status = gr.Markdown()

        run_btn.click(
            fn=run_dashboard,
            inputs=[rpm, blades, pitch_deg, v_inflow, slice_axis],
            outputs=[out_attrs, out_nearest, out_p2, out_p3, out_p4, out_plot, out_json, status],
        )

        demo.load(
            fn=run_dashboard,
            inputs=[rpm, blades, pitch_deg, v_inflow, slice_axis],
            outputs=[out_attrs, out_nearest, out_p2, out_p3, out_p4, out_plot, out_json, status],
        )

    demo.launch()


if __name__ == "__main__":
    main()
