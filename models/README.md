# models/

Placeholder. FNO + PINN baselines land in Phase 2-3.

Planned contents (Phase 2+):

- `fno3d.py` — 4-layer 3D Fourier Neural Operator (32 modes/axis, ~5M params), built on the [`neuraloperator`](https://github.com/neuraloperator/neuraloperator) library.
- `pinn.py` — DeepXDE-based PINN with incompressible Navier-Stokes residual loss (stretch goal).
- `train.py` — single-entrypoint training loop with W&B logging.
- `eval.py` — benchmark harness: solver wall-clock vs. surrogate inference time, RMSE on velocity/pressure, incompressibility violation.

Phase 1 (this commit) ships only the data pipeline that feeds these models. No weights, no training code yet.
