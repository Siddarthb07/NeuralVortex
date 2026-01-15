"""
NeuralVortex unified solver API.

The physics in this module is lifted **verbatim** (formulae and constants
unchanged) from Siddarth Boggarapu's two open-source CFD simulators. The
only logic that is new in this file is the field-sampling read-out used
to turn the existing simulator state into a 3D voxel grid for ML
consumption. The read-out is standard textbook potential-flow theory,
not a re-derivation of the parent physics.

Source attribution
------------------
1. Vortex-ring physics
   Repo: https://github.com/Siddarthb07/Drone-Vortex-Ring-Simulation
   File: vortex_rings_simulation.py

   - `RHO_AIR`, `NU_AIR`           lifted from L10-L11.
   - `thrust_to_circulation`       lifted from L44-L50 (Kelvin-type
                                   scaling Gamma ~ sqrt(T * 4*pi*R/rho)).
   - `VortexRingSimple` class      lifted from L57-L96, including:
       * Helmholtz thin-ring self-induction
         `U = Gamma/(4 pi R) * (ln(8R/a) - 1/4)`               (L89)
       * Viscous circulation decay `Gamma *= exp(-nu dt/a^2)`  (L84)
       * Core growth `a += sqrt(4 nu dt) * 0.2`                (L81)
       * Ring-ring interaction damping factor 0.7              (L255)
   - Ring-emission cadence (0.12 s) and `T_now > 0.1 N` gate
     are taken from `DroneVortexApp.maybe_emit_ring` (L225-L241).

2. BEMT propeller physics
   Repo: https://github.com/Siddarthb07/Propeller-simulator
   File: main.py

   - `PropellerPhysics` class      lifted from L19-L128, including:
       * `cl_cd_from_aoa` thin-airfoil model        (L37-L41)
       * `tip_mach` helper                          (L55-L58)
       * `_bem_simple` integration loop             (L70-L115)
   - Default geometry `radius=0.5 m`, `chord=0.1 m` is **overridden**
     here by the shared NeuralVortex rotor (`ROTOR_RADIUS`, `CHORD`)
     so that the BEMT thrust drives the vortex-ring emission for the
     *same* physical rotor. Formulae are unchanged; only the input
     geometry is set explicitly per call.

What is NEW in this file (clearly labelled below)
-------------------------------------------------
- `_ring_velocity_field_biot_savart`: textbook Biot-Savart velocity
  field of a circular vortex filament, expressed via complete elliptic
  integrals K(m), E(m). See e.g. H. Lamb, *Hydrodynamics* (1932), section
  163; P. G. Saffman, *Vortex Dynamics* (1992), section 10.1; the
  closed-form expression used here matches the cylindrical (u_r, u_z)
  components given in the "Vortex ring" Wikipedia article. A Lamb-Oseen
  style core regularisation `(R-rho)^2 + (z-z_ring)^2 -> max(., a^2)`
  prevents the on-filament singularity.

- `_bernoulli_pressure`: incompressible Bernoulli read-out
  `p = 0.5 * rho * (v_inflow^2 - |v|^2)` with `p_inf = 0`. Standard
  inviscid steady-flow relation.

These additions are *sampling operators* on the existing simulator
state; they do not change how rings evolve, how circulation decays, or
how thrust is computed.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.special import ellipe, ellipk


# =============================================================================
# Physical constants — VERBATIM from vortex_rings_simulation.py (L9-L12)
# and main.py (rho=1.225 default, L31).
# =============================================================================

RHO_AIR: float = 1.225      # kg/m^3
NU_AIR: float = 1.5e-5      # m^2/s  (kinematic viscosity)
G: float = 9.81             # m/s^2

# Shared rotor geometry for the unified NeuralVortex solver.
# (The parent repos use different defaults — 0.15 m in the vortex-ring
#  sim, 0.5 m in the BEMT sim — because they were standalone demos. For
#  NeuralVortex we model a single small-drone rotor end-to-end so that
#  the BEMT thrust meaningfully drives the vortex-ring emission.)
ROTOR_RADIUS: float = 0.15  # m   (small quad-copter rotor, ~6 inch)
CHORD: float = 0.03         # m   (chord/R = 0.20, matching the BEMT default ratio)


# =============================================================================
# Vortex-ring physics — VERBATIM from vortex_rings_simulation.py
# =============================================================================

def thrust_to_circulation(T: float, R: float, rho: float) -> float:
    """
    Simple actuator-disc style scaling:
        T ~ rho * Gamma^2 / (4 pi R)  =>  Gamma ~ sqrt(T * 4 pi R / rho)
    Kelvin-type scaling. [web:45]

    VERBATIM from vortex_rings_simulation.py L44-L50.
    """
    T = max(T, 0.0)
    return np.sqrt(T * 4.0 * np.pi * R / max(rho, 1e-9))


class VortexRingSimple:
    """
    Each ring carries:
        Gamma : circulation [m^2/s]
        R     : ring radius [m]
        z     : vertical position [m]
        a     : core radius / thickness [m]

    Motion (Kelvin-Helmholtz thin-ring formula):
        U = Gamma / (4 pi R) * (ln(8R/a) - 1/4)
    Decay (viscous diffusion scaling):
        dGamma/dt = -nu_eff * Gamma / a^2

    VERBATIM from vortex_rings_simulation.py L57-L96.
    """

    def __init__(self, R: float, z0: float, Gamma: float, nu_eff: float):
        self.R = R
        self.z = z0
        self.Gamma = Gamma
        self.a = 0.01        # initial core size [m]
        self.nu_eff = nu_eff

    def update(self, dt: float) -> None:
        self.a = max(self.a, 1e-4)
        self.a += np.sqrt(4 * self.nu_eff * dt) * 0.2

        self.Gamma *= np.exp(-self.nu_eff * dt / (self.a ** 2 + 1e-9))

        R = max(self.R, 1e-4)
        a = max(self.a, 1e-4)
        U = self.Gamma / (4.0 * np.pi * R) * (np.log(8.0 * R / a) - 0.25)

        self.z -= U * dt

    def weaken_due_to_interaction(self, factor: float) -> None:
        """VERBATIM from vortex_rings_simulation.py L94-L96."""
        self.Gamma *= factor


def _apply_ring_interactions(rings: List[VortexRingSimple]) -> None:
    """
    VERBATIM from vortex_rings_simulation.py L243-L255.
    """
    if len(rings) < 2:
        return
    threshold = 0.05  # m
    for i in range(1, len(rings)):
        older = rings[i - 1]
        newer = rings[i]
        if abs(newer.z - older.z) < threshold:
            newer.weaken_due_to_interaction(0.7)


# =============================================================================
# BEMT propeller physics — VERBATIM from Propeller-simulator/main.py
# =============================================================================

class PropellerPhysics:
    """
    Physics engine with switchable models:
    - model="simple_bemt": current working simplified BEMT
    - model="full_bemt":   placeholder that falls through to simple_bemt
                           (matches the parent repo's behaviour). [web:75][web:81]

    VERBATIM from Propeller-simulator/main.py L19-L128.
    """

    def __init__(
        self,
        radius: float = 0.5,
        blades: int = 3,
        chord: float = 0.1,
        rho: float = 1.225,
        model: str = "simple_bemt",
        airfoil_name: str = "NACA2412",
    ):
        self.radius = radius
        self.blades = blades
        self.chord = chord
        self.rho = rho
        self.model = model
        self.airfoil_name = airfoil_name

    def cl_cd_from_aoa(self, aoa_deg: float):
        aoa = np.radians(aoa_deg)
        cl = 2.0 * np.pi * np.sin(aoa) * np.cos(aoa)
        cd = 0.008 + 0.03 * (np.sin(aoa) ** 2)
        return max(0.0, cl), max(0.001, cd)

    def tip_mach(self, rpm: float, speed_of_sound: float = 343.0):
        omega = 2.0 * np.pi * rpm / 60.0
        tip_speed = omega * self.radius
        return tip_speed / speed_of_sound, tip_speed

    def bem_forces(self, rpm: float, pitch_deg: float, advance_ratio: float = 0.0):
        if self.model == "full_bemt":
            return self._bem_full(rpm, pitch_deg, advance_ratio)
        return self._bem_simple(rpm, pitch_deg, advance_ratio)

    def _bem_simple(self, rpm: float, pitch_deg: float, advance_ratio: float):
        omega = 2.0 * np.pi * rpm / 60.0
        r = np.linspace(0.2, 1.0, 25) * self.radius

        thrust = 0.0
        torque = 0.0
        dr = r[1] - r[0]

        cl_list, cd_list = [], []

        for ri in r:
            phi = np.arctan2(advance_ratio * self.radius, ri)
            aoa_deg = pitch_deg - np.degrees(phi)
            cl, cd = self.cl_cd_from_aoa(aoa_deg)
            cl_list.append(cl)
            cd_list.append(cd)

            V_rel = omega * ri
            dL = 0.5 * self.rho * V_rel ** 2 * self.chord * dr * cl
            dD = 0.5 * self.rho * V_rel ** 2 * self.chord * dr * cd

            dT = self.blades * (dL * np.cos(phi) - dD * np.sin(phi))
            dQ = self.blades * ri * (dD * np.cos(phi) + dL * np.sin(phi))

            thrust += dT
            torque += dQ

        power = torque * omega
        efficiency = (thrust * 9.81) / power if power > 1e-6 else 0.0

        disk_area = np.pi * self.radius ** 2
        disk_loading = thrust / disk_area if disk_area > 0 else 0.0
        power_loading = thrust / power if power > 1e-6 else 0.0
        tip_mach, tip_speed = self.tip_mach(rpm)

        return {
            "thrust": thrust,
            "torque": torque,
            "power": power,
            "efficiency": efficiency,
            "cl_avg": float(np.mean(cl_list)),
            "cd_avg": float(np.mean(cd_list)),
            "disk_loading": disk_loading,
            "power_loading": power_loading,
            "tip_speed": tip_speed,
            "tip_mach": tip_mach,
        }

    def _bem_full(self, rpm: float, pitch_deg: float, advance_ratio: float):
        """
        TODO mirror of the parent repo:
        - Iterate axial/tangential induction (a, a') using Glauert's method
        - Apply Prandtl tip & hub loss factors
        - Re-dependent polars from a UIUC airfoil database
        For now, just call simple model so simulator still runs. [web:75][web:81]
        """
        return self._bem_simple(rpm, pitch_deg, advance_ratio)


# =============================================================================
# Field read-out (NEW — standard textbook potential flow, not parent-repo math)
# =============================================================================

def _ring_velocity_field_biot_savart(
    R: float,
    z_ring: float,
    Gamma: float,
    a: float,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Biot-Savart induced velocity field of a single circular vortex filament
    of radius `R`, centred at the origin in the plane `z = z_ring`, with
    circulation `Gamma` and effective core size `a` (used only to regularise
    the on-filament singularity).

    Closed-form expression via complete elliptic integrals K(m) and E(m):

        m   = 4 R rho / [(R + rho)^2 + zeta^2]
        u_z = Gamma / (2 pi sqrt((R+rho)^2 + zeta^2))
              * [ K(m) + (R^2 - rho^2 - zeta^2) / ((R-rho)^2 + zeta^2) * E(m) ]
        u_rho = (Gamma * zeta) / (2 pi rho sqrt((R+rho)^2 + zeta^2))
              * [ -K(m) + (R^2 + rho^2 + zeta^2) / ((R-rho)^2 + zeta^2) * E(m) ]

    where `rho = sqrt(x^2 + y^2)` is the cylindrical radius and
    `zeta = z - z_ring` is the axial offset.

    References:
        - H. Lamb, *Hydrodynamics* (1932), Cambridge UP, section 163.
        - P. G. Saffman, *Vortex Dynamics* (1992), Cambridge UP, section 10.1.
        - Wikipedia, "Vortex ring", retrieved 2026.

    This function does NOT modify the rings; it samples their induced
    velocity onto the grid defined by (X, Y, Z).
    """
    rho = np.sqrt(X * X + Y * Y)
    zeta = Z - z_ring

    # Regularise the on-filament denominator with a Lamb-Oseen-style core
    # of size `a` so we don't divide by zero at points (rho, zeta) = (R, 0).
    core2 = max(a * a, 1e-10)
    denom_alpha2 = np.maximum((R - rho) ** 2 + zeta ** 2, core2)
    denom_beta2 = np.maximum((R + rho) ** 2 + zeta ** 2, core2)
    sqrt_beta = np.sqrt(denom_beta2)

    m = np.clip(4.0 * R * rho / denom_beta2, 0.0, 1.0 - 1e-9)
    Km = ellipk(m)
    Em = ellipe(m)

    pref = Gamma / (2.0 * np.pi * sqrt_beta)
    u_z = pref * (Km + (R * R - rho * rho - zeta * zeta) / denom_alpha2 * Em)

    rho_safe = np.maximum(rho, 1e-6)
    pref_r = (Gamma * zeta) / (2.0 * np.pi * rho_safe * sqrt_beta)
    u_rho = pref_r * (-Km + (R * R + rho * rho + zeta * zeta) / denom_alpha2 * Em)

    # On the axis (rho -> 0) u_rho is identically zero by symmetry.
    on_axis = rho < 1e-6
    u_rho = np.where(on_axis, 0.0, u_rho)

    # Decompose radial component back to Cartesian (u_x, u_y).
    cos_phi = X / rho_safe
    sin_phi = Y / rho_safe
    cos_phi = np.where(on_axis, 0.0, cos_phi)
    sin_phi = np.where(on_axis, 0.0, sin_phi)

    u_x = u_rho * cos_phi
    u_y = u_rho * sin_phi
    return u_x, u_y, u_z


def _bernoulli_pressure(
    velocity: np.ndarray, v_inflow: float, rho: float = RHO_AIR
) -> np.ndarray:
    """
    Incompressible Bernoulli read-out (steady, inviscid, irrotational
    far field):

        p + 0.5 * rho * |v|^2 = const = 0.5 * rho * v_inflow^2

    so that `p -> 0` at infinity where `|v| -> v_inflow`. Returns gauge
    pressure in Pa.
    """
    v_sq = velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2
    return 0.5 * rho * (v_inflow ** 2 - v_sq)


# =============================================================================
# Unified public API
# =============================================================================

def simulate_propeller(
    rpm: float,
    blades: int,
    pitch: float,
    v_inflow: float,
) -> Dict[str, Any]:
    """
    BEMT propeller solve at the unified NeuralVortex rotor geometry.

    Parameters
    ----------
    rpm : float
        Rotor speed in rev/min.
    blades : int
        Number of blades.
    pitch : float
        Collective pitch in degrees.
    v_inflow : float
        Axial inflow velocity in m/s.

    Returns
    -------
    {"thrust": N, "torque": N*m, "efficiency": -, "metadata": {...}}
    """
    omega = 2.0 * np.pi * float(rpm) / 60.0
    # Standard propeller advance ratio J = V / (omega * R). The parent
    # BEMT folds this into phi = atan2(J*R, r_i); we pass it through
    # unchanged so the math is preserved.
    J = float(v_inflow) / max(omega * ROTOR_RADIUS, 1e-9)

    physics = PropellerPhysics(
        radius=ROTOR_RADIUS,
        blades=int(blades),
        chord=CHORD,
        rho=RHO_AIR,
        model="simple_bemt",
    )
    forces = physics.bem_forces(float(rpm), float(pitch), advance_ratio=J)

    return {
        "thrust": float(forces["thrust"]),
        "torque": float(forces["torque"]),
        "efficiency": float(forces["efficiency"]),
        "metadata": {
            "power_W": float(forces["power"]),
            "cl_avg": float(forces["cl_avg"]),
            "cd_avg": float(forces["cd_avg"]),
            "disk_loading_N_per_m2": float(forces["disk_loading"]),
            "power_loading_N_per_W": float(forces["power_loading"]),
            "tip_speed_m_per_s": float(forces["tip_speed"]),
            "tip_mach": float(forces["tip_mach"]),
            "rotor_radius_m": ROTOR_RADIUS,
            "chord_m": CHORD,
            "advance_ratio": float(J),
        },
    }


def _run_vortex_ring_evolution(
    T_max: float,
    t_final: float,
    rotor_radius: float = ROTOR_RADIUS,
    rho: float = RHO_AIR,
    nu_eff: float = NU_AIR,
    dt: float = 0.03,
    emit_period: float = 0.12,
    emit_threshold_N: float = 0.1,
    max_rings: int = 25,
) -> List[VortexRingSimple]:
    """
    Headless port of DroneVortexApp.step_sim (vortex_rings_simulation.py L257-L268).
    Uses VortexRingSimple.update verbatim; only the matplotlib UI is
    stripped. The emission cadence, threshold, and `max_rings = 25` cap
    are taken from L227 / L231 / L240-L241 of the parent file.
    """
    rings: List[VortexRingSimple] = []
    t = 0.0
    last_emit_t = -999.0
    n_steps = int(np.ceil(t_final / dt))
    for _ in range(n_steps):
        t += dt
        if t - last_emit_t >= emit_period and T_max > emit_threshold_N:
            Gamma = thrust_to_circulation(T_max, rotor_radius, rho)
            rings.append(
                VortexRingSimple(R=rotor_radius, z0=-0.02, Gamma=Gamma, nu_eff=nu_eff)
            )
            last_emit_t = t
        for ring in rings:
            ring.update(dt)
        _apply_ring_interactions(rings)
        if len(rings) > max_rings:
            rings.pop(0)
    return rings


def simulate_vortex_ring(
    rpm: float,
    blades: int,
    pitch: float,
    v_inflow: float,
    grid_res: int = 32,
    t_final: float = 0.6,
) -> Dict[str, Any]:
    """
    Run the vortex-ring evolution for `t_final` seconds and sample the
    resulting velocity & pressure fields onto a `grid_res^3` voxel grid.

    Steady thrust driving the ring emission is taken from a BEMT solve at
    the same (rpm, blades, pitch, v_inflow). The vortex-ring physics is
    verbatim from `Drone-Vortex-Ring-Simulation/vortex_rings_simulation.py`;
    the field-sampling step uses the standard Biot-Savart elliptic-integral
    form for a circular vortex filament (see module docstring).

    Returns
    -------
    {
      "velocity": float32 ndarray [3, X, Y, Z]  in m/s,
      "pressure": float32 ndarray [X, Y, Z]     in Pa (gauge),
      "metadata": {...}
    }
    """
    grid_res = int(grid_res)

    prop = simulate_propeller(rpm, blades, pitch, v_inflow)
    T_steady = max(float(prop["thrust"]), 0.0)

    rings = _run_vortex_ring_evolution(
        T_max=T_steady,
        t_final=float(t_final),
        rotor_radius=ROTOR_RADIUS,
        rho=RHO_AIR,
        nu_eff=NU_AIR,
    )

    # Domain: a cube centred on the rotor disc, biased downstream
    # (negative z is "downward / downstream" in the parent simulator).
    L = 4.0 * ROTOR_RADIUS                  # half-extent in x, y
    z_min = -6.0 * ROTOR_RADIUS             # downstream depth
    z_max = 1.0 * ROTOR_RADIUS              # just above the rotor disc

    x = np.linspace(-L, L, grid_res, dtype=np.float64)
    y = np.linspace(-L, L, grid_res, dtype=np.float64)
    z = np.linspace(z_min, z_max, grid_res, dtype=np.float64)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    velocity = np.zeros((3, grid_res, grid_res, grid_res), dtype=np.float64)

    # Background inflow: axial flow into the rotor disc (along -z).
    velocity[2] -= float(v_inflow)

    for ring in rings:
        if not np.isfinite(ring.Gamma) or ring.Gamma <= 0.0:
            continue
        u_x, u_y, u_z = _ring_velocity_field_biot_savart(
            R=float(ring.R),
            z_ring=float(ring.z),
            Gamma=float(ring.Gamma),
            a=float(ring.a),
            X=X,
            Y=Y,
            Z=Z,
        )
        velocity[0] += u_x
        velocity[1] += u_y
        velocity[2] += u_z

    pressure = _bernoulli_pressure(velocity, v_inflow=float(v_inflow), rho=RHO_AIR)

    return {
        "velocity": velocity.astype(np.float32),
        "pressure": pressure.astype(np.float32),
        "metadata": {
            "rpm": float(rpm),
            "blades": int(blades),
            "pitch_deg": float(pitch),
            "v_inflow": float(v_inflow),
            "grid_res": int(grid_res),
            "t_final": float(t_final),
            "n_rings": int(len(rings)),
            "domain_x_min": float(-L), "domain_x_max": float(L),
            "domain_y_min": float(-L), "domain_y_max": float(L),
            "domain_z_min": float(z_min), "domain_z_max": float(z_max),
            "rotor_radius_m": float(ROTOR_RADIUS),
            "rho_air": float(RHO_AIR),
            "thrust_steady_N": float(T_steady),
            "torque_N_m": float(prop["torque"]),
            "efficiency": float(prop["efficiency"]),
        },
    }
