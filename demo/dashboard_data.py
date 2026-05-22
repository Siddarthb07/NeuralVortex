"""HDF5 lookups for the phased Gradio dashboard (Phase 1: solver truth vs sliders)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def normalize_params(inp: dict[str, float], rpm_lo: float, rpm_hi: float) -> np.ndarray:
    rpm_n = (float(inp["rpm"]) - rpm_lo) / max(rpm_hi - rpm_lo, 1e-6)
    return np.array(
        [
            rpm_n,
            (float(inp["blades"]) - 2.0) / 4.0,
            (float(inp["pitch_deg"]) - 5.0) / 25.0,
            float(inp["v_inflow"]) / 15.0,
        ],
        dtype=np.float64,
    )


def read_global_attrs(h5_path: Path) -> dict[str, Any]:
    if not h5_path.is_file():
        return {}
    out: dict[str, Any] = {}
    with h5py.File(h5_path, "r") as f:
        for k in f.attrs:
            v = f.attrs[k]
            try:
                if hasattr(v, "shape") and getattr(v, "shape", ()) != ():
                    out[str(k)] = np.asarray(v).tolist()
                else:
                    out[str(k)] = v.item() if hasattr(v, "item") else str(v)
            except Exception:
                out[str(k)] = str(v)
    return out


def attrs_markdown(attrs: dict[str, Any]) -> str:
    if not attrs:
        return "*HDF5 missing or has no file-level attributes — set `NEURALVORTEX_H5` to a valid dataset.*"
    rows = "| Attribute | Value |\n| --- | --- |\n"
    for k in sorted(attrs.keys()):
        rows += f"| `{k}` | `{attrs[k]}` |\n"
    return rows


def nearest_solver_sample(
    h5_path: Path,
    inp: dict[str, float],
    *,
    rpm_lo: float,
    rpm_hi: float,
) -> tuple[float, str, dict[str, float], dict[str, np.ndarray], dict[str, float]]:
    """Distance in normalized parameter space, sample key, inputs, arrays, propeller scalars (optional)."""
    if not h5_path.is_file():
        raise FileNotFoundError(str(h5_path))
    target = normalize_params(inp, rpm_lo, rpm_hi)
    best_d = float(np.inf)
    best_key = ""
    best_inp: dict[str, float] = {}
    best_arrs: dict[str, np.ndarray] = {}
    best_prop: dict[str, float] = {}

    with h5py.File(h5_path, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        for k in keys:
            grp = f[k]
            sinp = {
                "rpm": float(grp.attrs["rpm"]),
                "blades": float(grp.attrs["blades"]),
                "pitch_deg": float(grp.attrs["pitch_deg"]),
                "v_inflow": float(grp.attrs["v_inflow"]),
            }
            d = float(np.linalg.norm(normalize_params(sinp, rpm_lo, rpm_hi) - target))
            if d < best_d:
                best_d = d
                best_key = k
                best_inp = sinp
                best_arrs = {
                    "velocity": np.asarray(grp["velocity"], dtype=np.float64),
                    "pressure": np.asarray(grp["pressure"], dtype=np.float64),
                }
                best_prop = {}
                if "propeller" in grp:
                    pg = grp["propeller"]
                    for name in ("thrust", "torque", "efficiency"):
                        if name in pg:
                            best_prop[name] = float(pg[name][()])
    return best_d, best_key, best_inp, best_arrs, best_prop


def nearest_markdown(
    dist: float,
    sample_key: str,
    sinp: dict[str, float],
    prop: dict[str, float],
) -> str:
    lines = [
        "### Nearest solver sample (same normalization as training)",
        "",
        f"- **Match distance** (L2 in normalized RPM/blades/pitch/inflow space): `{dist:.6g}`",
        f"- **HDF5 group**: `{sample_key}`",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
    ]
    for k in ("rpm", "blades", "pitch_deg", "v_inflow"):
        lines.append(f"| {k} | {sinp.get(k, float('nan'))} |")
    if prop:
        lines.extend(["", "**Propeller scalars (if present)**", "", "| Metric | Value |", "| --- | --- |"])
        for k, v in sorted(prop.items()):
            lines.append(f"| {k} | {v:.6g} |")
    return "\n".join(lines)
