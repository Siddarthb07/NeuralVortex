# data/

Generates the NeuralVortex training dataset by sweeping the four-parameter input space and running both physics solvers per sample.

## Input parameter space

| Parameter | Symbol | Units | Default range |
|---|---|---|---|
| Rotor RPM | `rpm` | rev/min | 1000 - 10000 (step 500) |
| Blade count | `blades` | - | {2, 3, 4, 5, 6} |
| Pitch angle | `pitch_deg` | degrees | 5 - 25 (step 2) |
| Inflow velocity | `v_inflow` | m/s | 0 - 15 (step 1) |

Full grid sweep ~ 16k samples. Smoke test draws a small Latin-hypercube sample via `--n`.

## Output HDF5 schema

Each sample is stored under integer-keyed group `sample_{i:06d}` with this layout:

```
sample_000000/
    velocity      float32  [3, X, Y, Z]   # (u, v, w) in m/s
    pressure      float32  [X, Y, Z]      # Pa (gauge, p_inf = 0)
    propeller/
        thrust          float64  N
        torque          float64  N*m
        efficiency      float64  -
    attrs:
        rpm, blades, pitch_deg, v_inflow,
        grid_res, t_final, n_rings,
        domain_x_min, domain_x_max,
        domain_y_min, domain_y_max,
        domain_z_min, domain_z_max,
        rotor_radius, rho_air
```

File-level attributes:

```
attrs:
    schema_version  = "1.0"
    created_utc     ISO-8601 timestamp
    grid_res        int
    t_final         float
    rotor_radius    float
    n_samples       int
    config_hash     SHA-256 of the config YAML
    git_sha         (best-effort) git HEAD of the generating repo
```

## Solvers

- `solvers.simulate_vortex_ring(rpm, blades, pitch, v_inflow, grid_res, t_final)` - runs the Helmholtz thin-ring vortex evolution from `Drone-Vortex-Ring-Simulation/vortex_rings_simulation.py` headlessly for `t_final` seconds, then samples the induced 3D velocity field onto the requested grid via a textbook Biot-Savart elliptic-integral kernel. Pressure is recovered from the incompressible Bernoulli relation.
- `solvers.simulate_propeller(rpm, blades, pitch, v_inflow)` - runs the BEMT loop from `Propeller-simulator/main.py`'s `PropellerPhysics` and returns `{thrust, torque, efficiency, metadata}`.

The CFD math (Kelvin circulation scaling, Helmholtz self-induction, viscous decay, BEMT) is copied **verbatim** from the parent repos; only the field-sampling read-out is new. See `solvers.py` header for the line-by-line citations.

## Usage

```bash
# Smoke test (4 samples, coarse 16-cube grid, < 2 min on a laptop)
python data/generate.py --n 4 --grid-res 16 --out data/smoke.h5

# Full Phase-2 prerequisite run (NOT executed in Phase 1)
python data/generate.py --config configs/default.yaml --out data/full.h5
```
