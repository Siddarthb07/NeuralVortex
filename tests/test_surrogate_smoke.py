#!/usr/bin/env python3
"""Smoke: Conv3D surrogate forward pass matches HDF5 grid shape."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from train.surrogate_models import TinyConv3dSurrogate  # noqa: E402


def test_conv3d_forward_smoke():
    m = TinyConv3dSurrogate()
    x = torch.randn(2, 4, 8, 8, 8)
    y = m(x)
    assert y.shape == x.shape
