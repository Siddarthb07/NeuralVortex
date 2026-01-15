"""NeuralVortex data-generation package."""

from .solvers import (
    ROTOR_RADIUS,
    CHORD,
    RHO_AIR,
    NU_AIR,
    simulate_propeller,
    simulate_vortex_ring,
)

__all__ = [
    "ROTOR_RADIUS",
    "CHORD",
    "RHO_AIR",
    "NU_AIR",
    "simulate_propeller",
    "simulate_vortex_ring",
]
