"""Thermodynamically constrained dual-potential viscoelastic framework."""

from .constitutive import DualPotentialMaterial, MaterialParameters
from .dissipation import (
    NeuralDissipationPICNN,
    Stage2DissipationPICNN,
    Stage6GatedDissipationPotential,
    T2GroupedPowerPotential,
)
from .energies import (
    FixedOgdenEnergy,
    LowParameterOgdenEnergy,
    NeuralIsochoricEnergy,
    ZhangLiEquilibriumEnergy,
)
from .integrator import ImplicitExponentialIntegrator, NewtonOptions
from .solvers import MaterialPointSolver

__all__ = [
    "DualPotentialMaterial",
    "MaterialParameters",
    "FixedOgdenEnergy",
    "LowParameterOgdenEnergy",
    "NeuralIsochoricEnergy",
    "ZhangLiEquilibriumEnergy",
    "NeuralDissipationPICNN",
    "Stage2DissipationPICNN",
    "Stage6GatedDissipationPotential",
    "T2GroupedPowerPotential",
    "ImplicitExponentialIntegrator",
    "NewtonOptions",
    "MaterialPointSolver",
]
