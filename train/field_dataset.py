"""Torch Dataset over NeuralVortex HDF5 fields: broadcast params → grid channels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from train.hdf5_loader import iter_samples


def _rpm_norm_bounds(h5_path: Path) -> tuple[float, float]:
    rpm_lo, rpm_hi = 1e9, -1e9
    for inp, _ in iter_samples(h5_path):
        rpm_lo = min(rpm_lo, inp["rpm"])
        rpm_hi = max(rpm_hi, inp["rpm"])
    return rpm_lo, rpm_hi


def params_to_grid_tensor(
    inp: dict[str, float],
    *,
    rpm_lo: float,
    rpm_hi: float,
    grid_shape: tuple[int, int, int],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Shape (4, X, Y, Z) matching HDF5 velocity layout (channels first)."""
    rpm_n = (inp["rpm"] - rpm_lo) / max(rpm_hi - rpm_lo, 1e-6)
    blades_n = (inp["blades"] - 2.0) / 4.0
    pitch_n = (inp["pitch_deg"] - 5.0) / 25.0
    v_n = inp["v_inflow"] / 15.0
    dz, dy, dx = grid_shape
    x = torch.zeros((4, dz, dy, dx), dtype=torch.float32, device=device)
    x[0].fill_(float(rpm_n))
    x[1].fill_(float(blades_n))
    x[2].fill_(float(pitch_n))
    x[3].fill_(float(v_n))
    return x


def stack_velocity_pressure(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """(4, Z, Y, X) float32: vx, vy, vz, p."""
    v = np.asarray(arrays["velocity"], dtype=np.float32)
    p = np.asarray(arrays["pressure"], dtype=np.float32)[np.newaxis, ...]
    return np.concatenate([v, p], axis=0)


class NeuralVortexFieldDataset(Dataset):
    """Pairs broadcast-parameter grids with concatenated velocity+pressure."""

    def __init__(self, h5_path: str | Path, *, targets_std_eps: float = 1e-6):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.samples: list[tuple[dict[str, float], np.ndarray]] = []
        rpm_lo, rpm_hi = _rpm_norm_bounds(self.h5_path)
        self.rpm_lo, self.rpm_hi = rpm_lo, rpm_hi

        for inp, arrs in iter_samples(self.h5_path):
            tgt = stack_velocity_pressure(arrs)
            self.samples.append((inp, tgt))

        if not self.samples:
            raise ValueError(f"No samples found in {self.h5_path}")

        self._shape = self.samples[0][1].shape
        if self._shape[0] != 4:
            raise ValueError(f"Expected 4 channels (vx,vy,vz,p); got {self._shape}")

        flat = np.stack([s[1].reshape(-1) for s in self.samples], axis=0)
        mean = flat.mean(axis=0, dtype=np.float64)
        std = flat.std(axis=0, dtype=np.float64) + targets_std_eps
        self.target_mean = torch.from_numpy(mean.reshape(self._shape).astype(np.float32))
        self.target_std = torch.from_numpy(std.reshape(self._shape).astype(np.float32))

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        _, sx, sy, sz = self._shape
        return sx, sy, sz

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp, tgt = self.samples[idx]
        grid_shape = self.spatial_shape
        x = params_to_grid_tensor(inp, rpm_lo=self.rpm_lo, rpm_hi=self.rpm_hi, grid_shape=grid_shape)
        y = torch.from_numpy(tgt)
        yn = (y - self.target_mean) / self.target_std
        return x, yn
