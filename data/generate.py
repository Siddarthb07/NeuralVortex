"""
NeuralVortex dataset generator.

Sweeps the (rpm, blades, pitch_deg, v_inflow) input space, runs both the
vortex-ring and BEMT solvers per sample, and writes the results to a
single HDF5 file. Two sampling modes are supported:

  - `--n N`        : random Latin-hypercube sample of size N (for smoke
                     tests and Phase-1 sanity checks).
  - `--config ...` with `mode: grid` in the YAML : full Cartesian grid
                     sweep over the configured ranges (the Phase-2
                     prerequisite ~16k-sample run).

Examples
--------
    # Smoke test (default; ~30 s on a laptop)
    python data/generate.py --n 4 --grid-res 16 --out data/smoke.h5

    # Larger smoke run
    python data/generate.py --n 8 --grid-res 32 --out data/smoke32.h5

    # Full Phase-2 prerequisite (NOT executed in Phase 1)
    python data/generate.py --config configs/default.yaml \\
        --out data/full.h5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import h5py
import numpy as np
import yaml
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
ROOT_DIR = THIS_FILE.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.solvers import (  # noqa: E402  (sys.path tweak above)
    ROTOR_RADIUS,
    simulate_propeller,
    simulate_vortex_ring,
)


# =============================================================================
# Config loading
# =============================================================================

@dataclass
class SweepRanges:
    rpm: Tuple[float, float]
    blades: List[int]
    pitch_deg: Tuple[float, float]
    v_inflow: Tuple[float, float]
    rpm_step: float
    pitch_step: float
    v_inflow_step: float


@dataclass
class SweepConfig:
    mode: str            # "grid" or "lhs"
    seed: int
    grid_res: int
    t_final: float
    ranges: SweepRanges
    raw_text: str        # original YAML text, for hashing


def load_config(path: Path) -> SweepConfig:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    r = data["ranges"]
    return SweepConfig(
        mode=str(data.get("mode", "lhs")),
        seed=int(data.get("seed", 42)),
        grid_res=int(data.get("grid_res", 32)),
        t_final=float(data.get("t_final", 0.6)),
        ranges=SweepRanges(
            rpm=(float(r["rpm"]["min"]), float(r["rpm"]["max"])),
            rpm_step=float(r["rpm"].get("step", 500.0)),
            blades=[int(b) for b in r["blades"]["values"]],
            pitch_deg=(float(r["pitch_deg"]["min"]), float(r["pitch_deg"]["max"])),
            pitch_step=float(r["pitch_deg"].get("step", 2.0)),
            v_inflow=(float(r["v_inflow"]["min"]), float(r["v_inflow"]["max"])),
            v_inflow_step=float(r["v_inflow"].get("step", 1.0)),
        ),
        raw_text=raw,
    )


def default_config() -> SweepConfig:
    """Built-in fallback when --config is not supplied."""
    raw = (
        "mode: lhs\nseed: 42\nranges:\n"
        "  rpm: {min: 1000.0, max: 10000.0, step: 500.0}\n"
        "  blades: {values: [2, 3, 4, 5, 6]}\n"
        "  pitch_deg: {min: 5.0, max: 25.0, step: 2.0}\n"
        "  v_inflow: {min: 0.0, max: 15.0, step: 1.0}\n"
        "grid_res: 32\nt_final: 0.6\n"
    )
    return SweepConfig(
        mode="lhs",
        seed=42,
        grid_res=32,
        t_final=0.6,
        ranges=SweepRanges(
            rpm=(1000.0, 10000.0), rpm_step=500.0,
            blades=[2, 3, 4, 5, 6],
            pitch_deg=(5.0, 25.0), pitch_step=2.0,
            v_inflow=(0.0, 15.0), v_inflow_step=1.0,
        ),
        raw_text=raw,
    )


# =============================================================================
# Sampling strategies
# =============================================================================

def _latin_hypercube_unit(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Plain LHS in the unit cube [0, 1]^dim; shape (n, dim)."""
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.uniform(size=(n, dim))
    sample = cut[:n, None] + u * (1.0 / n)
    # Permute each column independently to randomise the LHS.
    for j in range(dim):
        rng.shuffle(sample[:, j])
    return sample


def sample_lhs(cfg: SweepConfig, n: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed)
    u = _latin_hypercube_unit(n, 4, rng)
    rpm_lo, rpm_hi = cfg.ranges.rpm
    pitch_lo, pitch_hi = cfg.ranges.pitch_deg
    v_lo, v_hi = cfg.ranges.v_inflow
    blades_arr = np.array(cfg.ranges.blades)

    samples = []
    for i in range(n):
        rpm = float(rpm_lo + u[i, 0] * (rpm_hi - rpm_lo))
        pitch = float(pitch_lo + u[i, 2] * (pitch_hi - pitch_lo))
        v_inflow = float(v_lo + u[i, 3] * (v_hi - v_lo))
        b_idx = int(np.floor(u[i, 1] * len(blades_arr)))
        b_idx = max(0, min(len(blades_arr) - 1, b_idx))
        samples.append({
            "rpm": rpm,
            "blades": int(blades_arr[b_idx]),
            "pitch_deg": pitch,
            "v_inflow": v_inflow,
        })
    return samples


def sample_grid(cfg: SweepConfig) -> List[Dict[str, Any]]:
    r = cfg.ranges
    rpm_vals = np.arange(r.rpm[0], r.rpm[1] + 0.5 * r.rpm_step, r.rpm_step)
    pitch_vals = np.arange(
        r.pitch_deg[0], r.pitch_deg[1] + 0.5 * r.pitch_step, r.pitch_step
    )
    v_vals = np.arange(
        r.v_inflow[0], r.v_inflow[1] + 0.5 * r.v_inflow_step, r.v_inflow_step
    )
    samples = []
    for rpm in rpm_vals:
        for b in r.blades:
            for pitch in pitch_vals:
                for v in v_vals:
                    samples.append({
                        "rpm": float(rpm),
                        "blades": int(b),
                        "pitch_deg": float(pitch),
                        "v_inflow": float(v),
                    })
    return samples


# =============================================================================
# HDF5 writer
# =============================================================================

def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def write_sample(
    f: h5py.File,
    idx: int,
    inputs: Dict[str, Any],
    vortex: Dict[str, Any],
    prop: Dict[str, Any],
) -> None:
    grp = f.create_group(f"sample_{idx:06d}")
    grp.create_dataset(
        "velocity", data=vortex["velocity"], compression="gzip", compression_opts=4
    )
    grp.create_dataset(
        "pressure", data=vortex["pressure"], compression="gzip", compression_opts=4
    )
    prop_grp = grp.create_group("propeller")
    prop_grp.create_dataset("thrust", data=float(prop["thrust"]))
    prop_grp.create_dataset("torque", data=float(prop["torque"]))
    prop_grp.create_dataset("efficiency", data=float(prop["efficiency"]))
    for k, v in prop["metadata"].items():
        prop_grp.attrs[k] = v

    grp.attrs["rpm"] = float(inputs["rpm"])
    grp.attrs["blades"] = int(inputs["blades"])
    grp.attrs["pitch_deg"] = float(inputs["pitch_deg"])
    grp.attrs["v_inflow"] = float(inputs["v_inflow"])
    for k, v in vortex["metadata"].items():
        if k in ("rpm", "blades", "pitch_deg", "v_inflow"):
            continue
        grp.attrs[k] = v


def write_file_attrs(
    f: h5py.File, cfg: SweepConfig, n_samples: int, repo_root: Path
) -> None:
    f.attrs["schema_version"] = "1.0"
    f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
    f.attrs["grid_res"] = int(cfg.grid_res)
    f.attrs["t_final"] = float(cfg.t_final)
    f.attrs["rotor_radius_m"] = float(ROTOR_RADIUS)
    f.attrs["n_samples"] = int(n_samples)
    f.attrs["config_hash"] = hashlib.sha256(cfg.raw_text.encode("utf-8")).hexdigest()
    f.attrs["git_sha"] = _git_sha(repo_root)


# =============================================================================
# Main
# =============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeuralVortex dataset generator (Phase 1)")
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of LHS samples (forces mode=lhs). If omitted, uses the "
             "sampling mode from --config.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a sweep YAML. If omitted, uses a built-in default range.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/dataset.h5"),
        help="Output HDF5 path.",
    )
    p.add_argument(
        "--grid-res",
        type=int,
        default=None,
        help="Override grid resolution (samples per axis). Default 32 for "
             "smoke, 64 for full sweep — taken from config if not given.",
    )
    p.add_argument(
        "--t-final",
        type=float,
        default=None,
        help="Override vortex-ring evolution time in seconds.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.config is not None:
        cfg = load_config(args.config)
    else:
        cfg = default_config()

    if args.grid_res is not None:
        cfg.grid_res = int(args.grid_res)
    if args.t_final is not None:
        cfg.t_final = float(args.t_final)

    if args.n is not None:
        cfg.mode = "lhs"
        samples = sample_lhs(cfg, args.n)
    elif cfg.mode == "grid":
        samples = sample_grid(cfg)
    elif cfg.mode == "lhs":
        raise SystemExit(
            "config mode=lhs requires --n to be set on the command line"
        )
    else:
        raise SystemExit(f"unknown sampling mode: {cfg.mode!r}")

    n = len(samples)
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = ROOT_DIR
    print(
        f"NeuralVortex generate: mode={cfg.mode}, n={n}, grid_res={cfg.grid_res}, "
        f"t_final={cfg.t_final}s, out={out_path}"
    )

    t0 = time.time()
    with h5py.File(out_path, "w") as f:
        write_file_attrs(f, cfg, n_samples=n, repo_root=repo_root)
        for i, inputs in enumerate(tqdm(samples, desc="solving", unit="sample")):
            prop = simulate_propeller(
                rpm=inputs["rpm"],
                blades=inputs["blades"],
                pitch=inputs["pitch_deg"],
                v_inflow=inputs["v_inflow"],
            )
            vortex = simulate_vortex_ring(
                rpm=inputs["rpm"],
                blades=inputs["blades"],
                pitch=inputs["pitch_deg"],
                v_inflow=inputs["v_inflow"],
                grid_res=cfg.grid_res,
                t_final=cfg.t_final,
            )
            write_sample(f, i, inputs, vortex, prop)

    dt = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"DONE: {n} samples in {dt:.1f}s "
        f"({dt / max(n, 1):.2f} s/sample), {size_mb:.2f} MB at {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
