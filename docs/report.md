# NeuralVortex — interim technical report (Markdown)

This document substitutes for a formal PDF until LaTeX export is wired. It summarizes Phases 1–4 at a **portfolio / reproducibility** level rather than claiming peer-reviewed CFD fidelity.

## Abstract

NeuralVortex couples two reduced-order propeller/vortex simulators into a single HDF5 dataset and trains an operator-style surrogate to approximate volumetric velocity and pressure fields from four scalar controls (RPM, blade count, pitch, inflow). Phase 1 validates data generation; Phase 2 adds a torch surrogate with optional Fourier Neural Operator weights via `neuralop`; Phase 3 documents PINN-style sanity checks without claiming full Navier–Stokes calibration; Phase 4 ships a phased Gradio dashboard (HDF5 metadata, full-field channel stats, divergence, docs) plus [`docs/DEEP_DIVE.md`](docs/DEEP_DIVE.md).

## 1. Data generation

The unified solver (`data/solvers.py`) samples `(velocity [3,N³], pressure [N³])` on a cubic grid with documented attribution to the parent vortex-ring and BEMT repositories. Latin hypercube sweeps are driven by `data/generate.py` and `configs/default.yaml`.

## 2. Surrogate model (Phase 2)

Inputs are broadcast onto the grid as four channels (normalized RPM, blades, pitch, inflow). Targets concatenate velocity and pressure (four channels). Training minimizes L2 on **normalized** targets with channel-wise mean/variance computed from the training HDF5. Default export: `runs/tfno_train/surrogate_best.pt`.

Optional **Weights & Biases** logging is enabled with `--wandb` when `WANDB_API_KEY` is present.

When `neuralop` is unavailable or incompatible, a shallow **Conv3D** residual stack trains identically — useful for laptops without resolving TFNO ABI drift.

## 3. Physics sanity (Phase 3)

Full PINN training on Navier–Stokes residuals is **not** shipped as stable automation. Instead `train/pinn_residual_smoke.py` reports a discrete divergence norm of a sampled velocity field as a qualitative sanity metric on the generated data.

## 4. Evaluation vs solver

`train/eval_surrogate.py` compares **pooled** ground-truth scalars (mean speed, mean pressure, max speed) against surrogate predictions on a held-out shard of the HDF5. This is **not** a spectral-field benchmark; it is a cheap regression contract that survives tiny smoke sets.

## 5. Demo (Phase 4)

`demo/app_gradio.py` is a **Blocks** dashboard aligned with Phases 1–4: HDF5 file metadata + nearest solver sample, surrogate **pooled** scalars plus **per-channel** volumetric statistics and a velocity-magnitude slice, finite-difference **divergence** vs nearest HDF5 velocity, and a reproduction / shipping tab. Environment variables: **`NEURALVORTEX_CKPT`** (checkpoint), **`NEURALVORTEX_H5`** (dataset). Narrative orientation: [`docs/DEEP_DIVE.md`](DEEP_DIVE.md).

Full interactive volumetric visualization (volume rendering / ONNX export) remains future work.

## Limitations (explicit)

- Reduced-order vortex rings ≠ high-fidelity CFD.
- Smoke HDF5 (few samples) cannot stress-test generalization; production runs need hundreds–thousands of LHS draws.
- Execution accuracy numbers belong to the dataset generator wall-clock, not published timing tables yet.

## Reproduction checklist

```bash
pip install -r requirements.txt
python data/generate.py --n 8 --grid-res 16 --out data/smoke.h5
pip install -r requirements-train.txt
pip install -r requirements-demo.txt
python train/train_tfno.py --no-tfno --epochs 40 --cpu-only --h5 data/smoke.h5
python train/eval_surrogate.py --no-tfno --cpu-only
python train/pinn_residual_smoke.py --h5 data/smoke.h5
python demo/app_gradio.py
```
