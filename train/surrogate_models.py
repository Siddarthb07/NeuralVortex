"""Surrogate models for NeuralVortex fields (TFNO preferred; Conv3D fallback)."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn


class TinyConv3dSurrogate(nn.Module):
    """Lightweight baseline when `neuralop` / TFNO is unavailable."""

    def __init__(self, in_ch: int = 4, out_ch: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_field_surrogate(
    *,
    in_channels: int = 4,
    out_channels: int = 4,
    n_modes: tuple[int, int, int] = (8, 8, 8),
    hidden_channels: int = 32,
    prefer_tfno: bool = True,
) -> nn.Module:
    """
    Returns TFNO from `neuralop` when installed; otherwise warns once and returns TinyConv3dSurrogate.
    """
    if prefer_tfno:
        try:
            from neuralop.models import TFNO  # type: ignore

            return TFNO(
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
            )
        except Exception as err:  # pragma: no cover - optional dependency path
            warnings.warn(
                f"TFNO unavailable ({type(err).__name__}: {err}); "
                "falling back to TinyConv3dSurrogate. Install `requirements-train.txt` for neuralop.",
                stacklevel=2,
            )
    return TinyConv3dSurrogate(in_ch=in_channels, out_ch=out_channels, hidden=hidden_channels)
