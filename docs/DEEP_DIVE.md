# NeuralVortex — project deep dive

This note is for anyone opening the repo cold and asking: **what problem does this solve, what shipped in each phase, and what is intentionally *not* claimed**. Treat it as the narrative companion to [`report.md`](report.md) and [`DEMO.md`](DEMO.md).

## The elevator pitch

**NeuralVortex** is a small research-style pipeline that:

1. **Generates** synthetic volumetric flow snapshots (3-component velocity + pressure on a 3D grid) from a handful of propeller controls, and writes them to **HDF5**.
2. **Trains** a neural **field surrogate** — map four scalars (RPM, blade count, pitch, inflow speed) to the full 4-channel volume — using PyTorch (optional Fourier Neural Operator path when `neuralop` works; otherwise a compact Conv3D stack).
3. **Sanity-checks** physics-ish behavior with a **finite-difference divergence** diagnostic on velocity (Phase 3 smoke — not full PINN training on Navier–Stokes).
4. **Democratizes** inspection via **Gradio**: sliders, HDF5 metadata, nearest solver sample, surrogate channel statistics, a 2D slice of predicted speed, divergence numbers, and links back to docs.

The **product** is reproducibility and a clear story for a portfolio — **not** certifiable CFD accuracy.

---

## End-to-end mental model

```
Controls (rpm, blades, pitch, v_inflow)
        │
        ▼
┌───────────────────────┐     Phase 1      ┌─────────────┐
│ data/generate.py      │ ───────────────► │  *.h5       │
│ + unified solvers     │                  │  samples    │
└───────────────────────┘                  └──────┬──────┘
                                                  │
                     Phase 2                       │
        ┌──────────────────────────────────────────┘
        ▼
┌───────────────────────┐     checkpoint    ┌──────────────────┐
│ train/train_tfno.py   │ ───────────────► │ surrogate_best.pt │
│ L2 on normalized vol  │                   └─────────┬────────┘
└───────────────────────┘                             │
                                                      │ Phase 4 UI
                                                      ▼
                                           ┌──────────────────────┐
                                           │ demo/app_gradio.py   │
                                           │ + eval / divergence  │
                                           └──────────────────────┘
```

**Important distinction:** the early UI only surfaced **three pooled scalars** (mean speed, mean pressure, max speed). Those summarize the **full 3D fields** — they are not the whole model output. The updated dashboard exposes **per-channel statistics** and a **velocity magnitude slice** so you can see structure beyond pooling.

---

## Phase 1 — Data generation (HDF5)

**Goal:** Turn parameterized propeller settings into labeled volumetric tensors suitable for supervised learning.

**Where to look**

- `data/generate.py` — orchestrates batches / LHS-style sweeps (see repo configs).
- `data/solvers.py` — reduced-order physics glue (vortex-ring-style velocity + simple pressure proxy; documented limitations in `report.md`).
- Output format: HDF5 groups `sample_*` with datasets `velocity` `[3, nx, ny, nz]`, `pressure` `[nx, ny, nz]`, attrs for controls, optional `propeller/` subgroup (`thrust`, `torque`, `efficiency`).

**What “success” means here**

- Deterministic, versioned artifacts (`schema_version`, `git_sha`, `config_hash` in file attrs).
- Enough samples to train without memorizing a single cube (smoke sets are for wiring only).

**Honest limits**

- This is **not** RANS/LES CFD; do not compare absolute coefficients to wind tunnel data without a dedicated calibration study.

---

## Phase 2 — Surrogate training (field operator)

**Goal:** Learn \( f(\text{controls}) \approx (v_x, v_y, v_z, p) \) on the grid.

**Mechanics**

- Inputs are **broadcast** to four spatial channels (normalized RPM, blades, pitch, inflow) over `spatial_shape` stored in the checkpoint.
- Targets are **normalized** channel-wise (mean/std from training HDF5) before loss; inference **denormalizes** back to physical-scale tensors in `train/infer_surrogate.py`.
- Entry point: `train/train_tfno.py` — `--no-tfno` forces the Conv3D path when TFNO wheels are painful on your machine.

**Evaluation**

- `train/eval_surrogate.py` compares **pooled** targets vs predictions on a held-out shard — cheap regression signal, not spectral field error.

---

## Phase 3 — PINN scope (what shipped vs fantasy)

**Shipped:** `train/pinn_residual_smoke.py` (and shared `train/field_metrics.py`) compute a **central finite-difference divergence** of a velocity snapshot and report \(\|v\|_2\), \(\|\nabla\cdot v\|_2\), and their ratio.

**Not shipped:** automated DeepXDE / multi-term Navier–Stokes residual training with adaptive collocation. That would be a new project phase with its own validation story.

**Why divergence still matters**

- For incompressible intuition, velocity fields with moderate discrete divergence are easier to justify coupling to pressure via Poisson-style corrections later.
- The metric is a **sanity lamp**, not a pass/fail certificate.

---

## Phase 4 — Demo, docs, and Hugging Face outline

**Gradio app:** `demo/app_gradio.py`

- **Overview** tab embeds this deep dive (`docs/DEEP_DIVE.md`).
- **Phase 1** tab — global HDF5 attrs + nearest solver sample (uses same normalization as training when a checkpoint is loaded).
- **Phase 2** tab — surrogate **full-field** statistics table + pooled scalars + mid-slice plot of \(|\mathbf{v}|\).
- **Phase 3** tab — divergence metrics for **surrogate** vs **nearest HDF5** velocity (apples-to-apples grid).
- **Phase 4** tab — reproduction commands and pointers to `report.md` / `DEMO.md`.

**Environment**

| Variable | Purpose |
| --- | --- |
| `NEURALVORTEX_CKPT` | Path to `surrogate_best.pt` (default `runs/tfno_train/surrogate_best.pt`). |
| `NEURALVORTEX_H5` | Path to HDF5 for Phase 1 / nearest-sample panels (default `data/smoke.h5`). |

Static screenshots live under `assets/demo/`; regenerate per `docs/DEMO.md`.

---

## Where you might feel “lost” (and the fix)

If it seemed like the project was **only three numbers**, that was the **old UI contract** — pooled summaries for quick sanity. The **actual trained tensor** is a full grid; pooling was documentation-friendly, not scientifically complete.

Use this priority when exploring:

1. Read **this file** + **`docs/report.md`** (5–10 minutes).
2. Open **`demo/app_gradio.py`** and walk the tabs left-to-right (Phases 1→4).
3. Run **`train/eval_surrogate.py`** for quantitative pooled error on held-out data.
4. Run **`train/pinn_residual_smoke.py`** on a real sample index if you question divergence scale.

---

## Suggested reading order in-repo

| Order | Path | Why |
| --- | --- | --- |
| 1 | `docs/DEEP_DIVE.md` | Orientation (you are here). |
| 2 | `docs/report.md` | Formal-ish phase summary + limitations. |
| 3 | `configs/default.yaml` | Knobs for generation grid size / sampling. |
| 4 | `train/infer_surrogate.py` | Exact inference / denormalization contract. |
| 5 | `demo/app_gradio.py` | How Phases 1–4 map to UI widgets. |

Welcome back — the repo should feel smaller once you see **HDF5 → checkpoint → volumetric prediction → pooled/slice/divergence lenses** as one pipeline rather than three mysterious floats.
