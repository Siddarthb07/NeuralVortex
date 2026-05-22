"""Scalar diagnostics on volumetric velocity / pressure fields."""

from __future__ import annotations

import numpy as np


def central_div(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Finite-difference divergence (interior nodes); spacing cancels in ratio ||div||/||v||."""
    dvx_dx = np.zeros_like(vx)
    dvy_dy = np.zeros_like(vy)
    dvz_dz = np.zeros_like(vz)

    dvx_dx[..., 1:-1] = (vx[..., 2:] - vx[..., :-2]) / (2.0 * spacing)
    dvy_dy[:, 1:-1, :] = (vy[:, 2:, :] - vy[:, :-2, :]) / (2.0 * spacing)
    dvz_dz[1:-1, :, :] = (vz[2:, :, :] - vz[:-2, :, :]) / (2.0 * spacing)

    return dvx_dx + dvy_dy + dvz_dz


def divergence_norms(velocity: np.ndarray) -> tuple[float, float, float]:
    """Returns ||v||_2, ||div(v)||_2, ratio on stacked velocity [3, *grid]."""
    v = np.asarray(velocity, dtype=np.float64)
    vx, vy, vz = v[0], v[1], v[2]
    div = central_div(vx, vy, vz)
    vnorm = float(np.linalg.norm(v))
    dnorm = float(np.linalg.norm(div))
    ratio = dnorm / max(vnorm, 1e-12)
    return vnorm, dnorm, ratio


def channel_statistics(velocity: np.ndarray, pressure: np.ndarray) -> list[dict[str, float]]:
    """Per-channel mean / std / min / max for vx, vy, vz, p."""
    vel = np.asarray(velocity, dtype=np.float64)
    pr = np.asarray(pressure, dtype=np.float64)
    out = []
    names = ["vx", "vy", "vz"]
    for i, nm in enumerate(names):
        a = vel[i]
        out.append(
            {
                "channel": nm,
                "mean": float(a.mean()),
                "std": float(a.std()),
                "min": float(a.min()),
                "max": float(a.max()),
            }
        )
    out.append(
        {
            "channel": "pressure",
            "mean": float(pr.mean()),
            "std": float(pr.std()),
            "min": float(pr.min()),
            "max": float(pr.max()),
        }
    )
    speed = np.linalg.norm(vel, axis=0)
    out.append(
        {
            "channel": "|v|",
            "mean": float(speed.mean()),
            "std": float(speed.std()),
            "min": float(speed.min()),
            "max": float(speed.max()),
        }
    )
    return out


def velocity_magnitude_slice_png(velocity: np.ndarray, *, axis: int = 0, index: int | None = None) -> bytes:
    """PNG bytes: velocity magnitude, mid-slice along axis 0=native X for array [3,nx,ny,nz]."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = np.asarray(velocity, dtype=np.float64)
    speed = np.linalg.norm(v, axis=0)
    if index is None:
        index = speed.shape[axis] // 2
    if axis == 0:
        sl = speed[index, :, :]
        xlab, ylab = "Y idx", "Z idx"
    elif axis == 1:
        sl = speed[:, index, :]
        xlab, ylab = "X idx", "Z idx"
    else:
        sl = speed[:, :, index]
        xlab, ylab = "X idx", "Y idx"

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(sl, cmap="viridis", origin="lower")
    ax.set_title(f"|v| slice (axis={axis}, idx={index})")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()
