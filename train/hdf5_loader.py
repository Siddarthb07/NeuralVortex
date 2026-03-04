"""Load NeuralVortex HDF5 samples for training baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


def iter_samples(h5_path: str | Path) -> Iterator[tuple[dict[str, float], dict[str, np.ndarray]]]:
    """Yield (attrs_dict, arrays dict with velocity, pressure) per sample group."""
    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        for k in keys:
            grp = f[k]
            inp = {
                "rpm": float(grp.attrs["rpm"]),
                "blades": float(grp.attrs["blades"]),
                "pitch_deg": float(grp.attrs["pitch_deg"]),
                "v_inflow": float(grp.attrs["v_inflow"]),
            }
            vel = np.asarray(grp["velocity"], dtype=np.float64)
            pr = np.asarray(grp["pressure"], dtype=np.float64)
            yield inp, {"velocity": vel, "pressure": pr}


def pooled_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Scalar summary per sample: mean speed, mean pressure, max speed."""
    v = arrays["velocity"]
    speed = np.linalg.norm(v, axis=0)
    p = arrays["pressure"]
    return np.array(
        [float(np.mean(speed)), float(np.mean(p)), float(np.max(speed))],
        dtype=np.float64,
    )


def stack_xy(h5_path: str | Path):
    """Build X (n,4) normalized inputs and Y (n,3) pooled targets."""
    xs, ys = [], []
    rpm_l, rpm_h = 1e9, -1e9
    # pass 1: ranges for normalization
    raw = list(iter_samples(h5_path))
    if not raw:
        raise ValueError("HDF5 has no sample_* groups")
    for inp, _ in raw:
        rpm_l = min(rpm_l, inp["rpm"])
        rpm_h = max(rpm_h, inp["rpm"])
    for inp, arrs in raw:
        rpm_n = (inp["rpm"] - rpm_l) / max(rpm_h - rpm_l, 1e-6)
        x = np.array(
            [
                rpm_n,
                (inp["blades"] - 2.0) / 4.0,
                (inp["pitch_deg"] - 5.0) / 25.0,
                inp["v_inflow"] / 15.0,
            ],
            dtype=np.float64,
        )
        xs.append(x)
        ys.append(pooled_features(arrs))
    return np.stack(xs), np.stack(ys)
