# Phase 2–4 — Training & shipping (next)

Phase **1** on `main` produces labelled HDF5 (`velocity`, `pressure`, propeller scalars) via `data/generate.py`.

## Phase 2 — FNO baseline (planned)

1. Install extras: `pip install -r requirements-train.txt` (PyTorch + [`neuraloperator`](https://github.com/neuraloperator/neuraloperator)).
2. Generate ≥ a few hundred LHS samples (GPU farm / overnight laptop).
3. Implement `train/fno_darcy_style.py` (to be added): reads HDF5, normalizes fields, trains a 3D FNO with L2 + divergence penalty.
4. Log to Weights & Biases (`wandb login`).

The repo does **not** yet ship a trained checkpoint — that requires a committed GPU budget.

## Phase 3 — PINN stretch (optional)

Use DeepXDE or manual autograd on NS residuals; treat as research stretch per portfolio plan.

## Phase 4 — Paper + demo

- `docs/report/` LaTeX or Markdown → export PDF.
- Gradio app loading ONNX/TorchScript weights for interactive sliders.
