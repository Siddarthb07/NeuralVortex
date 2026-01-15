"""Smoke tests for the unified NeuralVortex solver API."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.solvers import simulate_propeller, simulate_vortex_ring  # noqa: E402


def test_simulate_propeller_smoke():
    out = simulate_propeller(rpm=5000.0, blades=3, pitch=15.0, v_inflow=2.0)

    assert set(out.keys()) >= {"thrust", "torque", "efficiency", "metadata"}
    for k in ("thrust", "torque", "efficiency"):
        v = out[k]
        assert isinstance(v, float), f"{k} should be float, got {type(v)}"
        assert np.isfinite(v), f"{k} should be finite, got {v}"

    # Static thrust at non-trivial RPM must be strictly positive.
    assert out["thrust"] > 0.0
    # Power (= torque * omega) should be positive; torque must therefore be > 0.
    assert out["torque"] > 0.0

    meta = out["metadata"]
    for required_key in ("power_W", "tip_mach", "rotor_radius_m", "advance_ratio"):
        assert required_key in meta


def test_simulate_vortex_ring_smoke_shapes():
    grid_res = 8
    out = simulate_vortex_ring(
        rpm=5000.0, blades=3, pitch=15.0, v_inflow=2.0,
        grid_res=grid_res, t_final=0.3,
    )

    assert set(out.keys()) >= {"velocity", "pressure", "metadata"}

    v = out["velocity"]
    p = out["pressure"]
    assert v.shape == (3, grid_res, grid_res, grid_res), v.shape
    assert p.shape == (grid_res, grid_res, grid_res), p.shape
    assert v.dtype == np.float32
    assert p.dtype == np.float32

    # No NaNs / infs anywhere — the elliptic-integral kernel must stay finite.
    assert np.all(np.isfinite(v))
    assert np.all(np.isfinite(p))

    meta = out["metadata"]
    for required_key in (
        "rpm", "blades", "pitch_deg", "v_inflow",
        "grid_res", "t_final", "n_rings",
        "domain_x_min", "domain_x_max",
        "domain_y_min", "domain_y_max",
        "domain_z_min", "domain_z_max",
        "rotor_radius_m", "rho_air",
        "thrust_steady_N", "torque_N_m", "efficiency",
    ):
        assert required_key in meta, f"missing metadata key {required_key!r}"

    # At a non-trivial RPM with a sane geometry we should have emitted at
    # least one ring during t_final = 0.3 s (emission cadence is 0.12 s).
    assert meta["n_rings"] >= 1


def test_zero_rpm_returns_inflow_only():
    grid_res = 6
    v_inflow = 3.0
    out = simulate_vortex_ring(
        rpm=0.0, blades=3, pitch=15.0, v_inflow=v_inflow,
        grid_res=grid_res, t_final=0.2,
    )
    # No thrust -> no rings -> velocity field should be pure inflow on -z.
    assert out["metadata"]["n_rings"] == 0
    v = out["velocity"]
    assert np.allclose(v[0], 0.0)
    assert np.allclose(v[1], 0.0)
    assert np.allclose(v[2], -v_inflow)
