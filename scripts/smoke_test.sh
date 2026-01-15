#!/usr/bin/env bash
# NeuralVortex Phase-1 smoke test.
#
# Generates a tiny dataset (4 samples, 16-cube grids) and verifies that
# the resulting HDF5 file exists and has the expected number of samples
# with sane shapes. Should finish in well under 2 minutes on a laptop.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT="data/smoke.h5"
N=4
GRID=16

echo "[smoke] generating $N samples at grid_res=$GRID -> $OUT"
python data/generate.py --n "$N" --grid-res "$GRID" --out "$OUT"

echo "[smoke] verifying $OUT"
python - <<PY
import sys
import h5py

path = "$OUT"
with h5py.File(path, "r") as f:
    n = f.attrs["n_samples"]
    grid_res = f.attrs["grid_res"]
    samples = sorted(k for k in f.keys() if k.startswith("sample_"))
    assert n == $N, f"expected n_samples=$N, got {n}"
    assert len(samples) == $N, f"expected $N sample groups, got {len(samples)}"
    s0 = f[samples[0]]
    v_shape = s0["velocity"].shape
    p_shape = s0["pressure"].shape
    assert v_shape == (3, $GRID, $GRID, $GRID), f"velocity shape {v_shape}"
    assert p_shape == ($GRID, $GRID, $GRID),     f"pressure shape {p_shape}"
    for k in ("rpm", "blades", "pitch_deg", "v_inflow", "n_rings"):
        assert k in s0.attrs, f"missing attr {k!r}"
    print(f"[smoke] OK: {len(samples)} samples, velocity {v_shape}, pressure {p_shape}")
PY

echo "[smoke] all checks passed"
