# Phase 3 — PINN scope (honest)

Physics-informed neural networks (PINNs) minimize PDE residuals alongside data loss. NeuralVortex does **not** currently ship an automated DeepXDE / JAX training loop that enforces incompressible Navier–Stokes on the voxel grid.

What *is* included:

- `train/pinn_residual_smoke.py` — discrete divergence magnitude on a Biot–Savart sampled velocity field as a **data sanity** probe.
- This README note — so nobody mistakes the repo for a calibrated NS solver.

Reasonable next steps if revisiting PINNs:

1. Downsample fields or supervise on boundary strips only (memory).
2. Pair residual loss with sparse CFD anchors if higher-fidelity references appear.
3. Treat residual weights as hyperparameters; unstable PINNs are normal without curriculum.
