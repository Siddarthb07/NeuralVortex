# Training & evaluation (Phases 2–4)

Phase **1** HDF5 comes from `data/generate.py`. These scripts consume it.

## Phase 2 — Surrogate training (`train_tfno.py`)

Install ML stack:

```bash
pip install -r requirements.txt
pip install -r requirements-train.txt
```

Train (Conv3D fallback — reliable on CPU laptops):

```bash
python train/train_tfno.py --no-tfno --epochs 40 --cpu-only --h5 data/smoke.h5
```

Train with **TFNO** when `neuralop` imports cleanly (often GPU-friendly):

```bash
python train/train_tfno.py --h5 data/smoke.h5 --epochs 40
```

Optional **Weights & Biases**: `--wandb` with `WANDB_API_KEY` exported.

Artifacts:

- `runs/tfno_train/surrogate_best.pt` — weights + normalization tensors + `arch` tag (`tiny_conv3d` | `tfno`).

## Phase 3 — PINN sanity (`pinn_residual_smoke.py`)

Not full PINN training — reports discrete divergence energy on a sampled velocity field:

```bash
python train/pinn_residual_smoke.py --h5 data/smoke.h5
```

See `docs/pinn_notes.md`.

## Evaluation vs solver (pooled scalars)

```bash
python train/eval_surrogate.py --no-tfno --cpu-only --ckpt runs/tfno_train/surrogate_best.pt
```

Writes `runs/surrogate_eval_pooled.json` (MAE on mean speed / mean pressure / max speed).

## Phase 4 — Gradio

```bash
pip install -r requirements-demo.txt
python demo/app_gradio.py
```

See `demo/README.md` for HF Space notes.

## Written report

Human-readable Markdown report: `docs/report.md` (export to PDF externally if needed).
