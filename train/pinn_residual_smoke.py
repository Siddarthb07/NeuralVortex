#!/usr/bin/env python3
"""
Phase 3 smoke — finite-difference divergence of a velocity snapshot.

This is **not** full PINN training (DeepXDE + Navier-Stokes residuals). It sanity-checks how
"divergence-light" the sampled Biot-Savart velocity field is on the HDF5 grid, which is useful
when judging whether a PINN residual penalty is worth the integration cost later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train.hdf5_loader import iter_samples  # noqa: E402


def central_div(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Cartesian central differences on a regular grid; spacing cancels when reporting ||div|| / ||v||."""
    # Arrays indexed [z, y, x] matching loaders.
    dvx_dx = np.zeros_like(vx)
    dvy_dy = np.zeros_like(vy)
    dvz_dz = np.zeros_like(vz)

    dvx_dx[..., 1:-1] = (vx[..., 2:] - vx[..., :-2]) / (2.0 * spacing)
    dvy_dy[:, 1:-1, :] = (vy[:, 2:, :] - vy[:, :-2, :]) / (2.0 * spacing)
    dvz_dz[1:-1, :, :] = (vz[2:, :, :] - vz[:-2, :, :]) / (2.0 * spacing)

    return dvx_dx + dvy_dy + dvz_dz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=Path("data/smoke.h5"))
    ap.add_argument("--sample-index", type=int, default=0)
    args = ap.parse_args()

    rows = list(iter_samples(args.h5))
    if not rows:
        print("No samples.")
        return 1
    _, arrs = rows[args.sample_index % len(rows)]
    v = np.asarray(arrs["velocity"], dtype=np.float64)
    vx, vy, vz = v[0], v[1], v[2]
    div = central_div(vx, vy, vz)
    vnorm = float(np.linalg.norm(v))
    dnorm = float(np.linalg.norm(div))
    ratio = dnorm / max(vnorm, 1e-12)
    print(f"sample_index={args.sample_index % len(rows)}")
    print(f"||v||_2={vnorm:.6e}  ||div(v)||_2={dnorm:.6e}  ratio={ratio:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
