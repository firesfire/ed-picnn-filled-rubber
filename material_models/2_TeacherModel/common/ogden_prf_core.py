"""Shared one-term incompressible Ogden nonequilibrium PRF implementation.

The equilibrium network remains the frozen Zhang--Li model.  This module is
shared by T1-Ogden-1 and T2-Ogden-1 so that their only structural difference is
the flow-parameter grouping.  It deliberately does not import or reuse the
Zhang--Li nonequilibrium Mandel force.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from teacher_model.flow import dev, flow_rate
from teacher_model.integrator import NewtonOptions, integrate_with_subdivision
from teacher_model.model import ModelState
from teacher_model.parameters import FlowParameters, ZhangLiParameters
from teacher_model.zhang_li import energy as zhang_li_energy
from teacher_model.zhang_li import small_strain_moduli
from teacher_model.zhang_li import unpressurized_cauchy as zhang_li_cauchy


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(A, float) + np.asarray(A, float).T)


def _spectral_power(A: np.ndarray, exponent: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(_sym(A))
    if np.min(values) <= 0:
        raise ValueError("Ogden elastic metric must be positive definite")
    return (vectors * np.power(values, exponent)) @ vectors.T


def isochoric_metric(A: np.ndarray) -> tuple[np.ndarray, float]:
    A = _sym(A)
    determinant = float(np.linalg.det(A))
    if determinant <= 0:
        raise ValueError("elastic metric must have positive determinant")
    J = np.sqrt(determinant)
    return A * determinant ** (-1.0 / 3.0), J


def ogden_unit_energy(Ce: np.ndarray, a_ogden: float) -> float:
    """Dimensionless normalized Ogden energy with unit small-strain shear modulus."""
    if abs(a_ogden) < 1e-8:
        raise ValueError("a_ogden is too close to the logarithmic singular limit")
    Cbar, _ = isochoric_metric(Ce)
    values = np.linalg.eigvalsh(Cbar)
    principal_sum = float(np.sum(np.power(values, 0.5 * a_ogden)))
    return float(2.0 / a_ogden**2 * (principal_sum - 3.0))


def ogden_kirchhoff_deviator(Fe: np.ndarray, shear_modulus_MPa: float,
                             a_ogden: float) -> np.ndarray:
    """Deviatoric Kirchhoff stress of the normalized one-term Ogden solid."""
    be = np.asarray(Fe, float) @ np.asarray(Fe, float).T
    bbar, _ = isochoric_metric(be)
    trial = (2.0 * shear_modulus_MPa / a_ogden) * _spectral_power(
        bbar, 0.5 * a_ogden)
    return dev(_sym(trial))


def ogden_cauchy_deviator(Fe: np.ndarray, shear_modulus_MPa: float,
                          a_ogden: float) -> np.ndarray:
    determinant = float(np.linalg.det(Fe))
    if determinant <= 0:
        raise ValueError("elastic deformation must have positive determinant")
    return ogden_kirchhoff_deviator(Fe, shear_modulus_MPa, a_ogden) / determinant


def ogden_mandel_deviator(Fe: np.ndarray, shear_modulus_MPa: float,
                          a_ogden: float) -> np.ndarray:
    """Thermodynamic driving force rederived from the Ogden branch energy."""
    Ce = np.asarray(Fe, float).T @ np.asarray(Fe, float)
    Cbar, _ = isochoric_metric(Ce)
    trial = (2.0 * shear_modulus_MPa / a_ogden) * _spectral_power(
        Cbar, 0.5 * a_ogden)
    return dev(_sym(trial))


def ogden_uniaxial_nominal(engineering_strain, shear_modulus_MPa: float,
                           a_ogden: float):
    strain = np.asarray(engineering_strain, float)
    stretch = 1.0 + strain
    if np.any(stretch <= 0):
        raise ValueError("nonpositive uniaxial stretch")
    return (2.0 * shear_modulus_MPa / a_ogden) * (
        stretch ** (a_ogden - 1.0)
        - stretch ** (-0.5 * a_ogden - 1.0))


def ogden_simple_shear(gamma, shear_modulus_MPa: float, a_ogden: float):
    values = []
    for scalar in np.atleast_1d(np.asarray(gamma, float)):
        F = np.array([[1.0, scalar, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        values.append(ogden_cauchy_deviator(F, shear_modulus_MPa, a_ogden)[0, 1])
    result = np.asarray(values)
    return float(result[0]) if np.ndim(gamma) == 0 else result


@dataclass(frozen=True)
class OgdenBranch:
    tau_s: float
    p: float

    def validate(self) -> None:
        if self.tau_s <= 0 or self.p <= 0:
            raise ValueError("Ogden branch requires positive tau and share")


@dataclass(frozen=True)
class OgdenPRFParameters:
    zhang_li: ZhangLiParameters
    Gneq_MPa: float
    a_ogden: float
    branches: tuple[OgdenBranch, ...]
    shared_flow: FlowParameters | None = None
    fast_flow: FlowParameters | None = None
    slow_flow: FlowParameters | None = None
    transition_tau_s: float = 1.0
    model_name: str = "Ogden-PRF"

    def validate(self) -> None:
        self.zhang_li.validate()
        if self.Gneq_MPa <= 0 or not (0.1 <= self.a_ogden <= 30.0):
            raise ValueError("invalid Ogden stiffness or exponent")
        if len(self.branches) not in (1, 7):
            raise ValueError("production Ogden models require seven branches")
        for branch in self.branches:
            branch.validate()
        tau = np.asarray([branch.tau_s for branch in self.branches])
        shares = np.asarray([branch.p for branch in self.branches])
        if np.any(np.diff(tau) <= 0):
            raise ValueError("branch times must increase")
        if len(self.branches) == 7 and not np.isclose(np.sum(shares), 1.0, atol=1e-10):
            raise ValueError("seven branch shares must sum to one")
        if self.shared_flow is not None:
            self.shared_flow.validate()
            if self.fast_flow is not None or self.slow_flow is not None:
                raise ValueError("shared and grouped flow cannot coexist")
        else:
            if self.fast_flow is None or self.slow_flow is None:
                raise ValueError("grouped model needs fast and slow flows")
            self.fast_flow.validate(); self.slow_flow.validate()
        if self.transition_tau_s <= 0:
            raise ValueError("transition time must be positive")

    def flow_for(self, tau_s: float) -> FlowParameters:
        if self.shared_flow is not None:
            return self.shared_flow
        return self.fast_flow if tau_s <= self.transition_tau_s else self.slow_flow

    def to_dict(self) -> dict:
        return asdict(self)


class OgdenPRFModel:
    def __init__(self, parameters: OgdenPRFParameters, newton: NewtonOptions | None = None):
        parameters.validate()
        self.parameters = parameters
        self.newton = newton or NewtonOptions()
        self.G0, self.E0 = small_strain_moduli(parameters.zhang_li)

    def _branch(self, F: np.ndarray, Fv: np.ndarray, branch: OgdenBranch):
        Fe = np.asarray(F, float) @ np.linalg.inv(np.asarray(Fv, float))
        modulus = self.parameters.Gneq_MPa * branch.p
        energy = modulus * ogden_unit_energy(Fe.T @ Fe, self.parameters.a_ogden)
        X = ogden_mandel_deviator(Fe, modulus, self.parameters.a_ogden)
        eta = max(3.0 * modulus * branch.tau_s, 1e-14)
        # Equivalent dimensionless Zhang--Li branch scale.  This keeps the old
        # threshold units and the linear time constant while Gneq changes.
        drive_scale = max(modulus / self.G0, 1e-14)
        flow = self.parameters.flow_for(branch.tau_s)
        Dv, diagnostics = flow_rate(
            X, Fv, eta, flow, 1.0, drive_scale, self.G0, branch.tau_s)
        stress = ogden_cauchy_deviator(Fe, modulus, self.parameters.a_ogden)
        return Dv, diagnostics, stress, Fe, X, energy

    def advance(self, state: ModelState, F0: np.ndarray, F1: np.ndarray, dt: float):
        new_state, diagnostics = [], []
        for Fv, branch in zip(state.Fv, self.parameters.branches):
            def rate(F, V):
                return self._branch(F, V, branch)[:2]
            result = integrate_with_subdivision(Fv, F0, F1, dt, rate, self.newton)
            new_state.append(result.Fv)
            diagnostics.append({
                **result.diagnostics,
                "newton_iterations": result.iterations,
                "residual": result.residual_norm,
                "subdivisions": result.subdivisions,
                "detFv": float(np.linalg.det(result.Fv)),
                "group": "shared" if self.parameters.shared_flow is not None else
                         ("fast" if branch.tau_s <= self.parameters.transition_tau_s else "slow"),
            })
        return ModelState(new_state), diagnostics

    def response(self, F: np.ndarray, state: ModelState) -> dict:
        stress = zhang_li_cauchy(F, self.parameters.zhang_li)
        total_energy = zhang_li_energy(F.T @ F, self.parameters.zhang_li)
        branch_stress, branch_diagnostics = [], []
        for Fv, branch in zip(state.Fv, self.parameters.branches):
            Dv, diagnostics, sigma, _Fe, X, branch_energy = self._branch(F, Fv, branch)
            stress = stress + sigma
            total_energy += branch_energy
            branch_stress.append(sigma)
            branch_diagnostics.append({**diagnostics, "X": X, "Dv": Dv})
        return {
            "cauchy_unpressurized": stress,
            "branch_cauchy": branch_stress,
            "branch": branch_diagnostics,
            "energy_MPa": float(total_energy),
        }

    @staticmethod
    def deformation(value: float, mode: str) -> np.ndarray:
        if mode == "uniaxial":
            stretch = 1.0 + value
            if stretch <= 0:
                raise ValueError("nonpositive stretch")
            return np.diag([stretch, stretch ** -0.5, stretch ** -0.5])
        if mode == "shear":
            return np.array([[1.0, value, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        raise ValueError(mode)

    def measured_stress(self, F: np.ndarray, state: ModelState, mode: str) -> float:
        stress = self.response(F, state)["cauchy_unpressurized"]
        if mode == "uniaxial":
            return float((stress[0, 0] - 0.5 * (stress[1, 1] + stress[2, 2])) / F[0, 0])
        if mode == "shear":
            return float(stress[0, 1])
        raise ValueError(mode)


def normalized_shares(weights) -> np.ndarray:
    weights = np.asarray(weights, float)
    if np.any(weights <= 0) or np.sum(weights) <= 0:
        raise ValueError("old branch weights must be positive")
    return weights / np.sum(weights)


def build_t1_ogden_parameters(zhang_li: dict, flow: dict, tau_s, old_weights,
                              Gneq_MPa: float, a_ogden: float) -> OgdenPRFParameters:
    shares = normalized_shares(old_weights)
    result = OgdenPRFParameters(
        ZhangLiParameters(**zhang_li), float(Gneq_MPa), float(a_ogden),
        tuple(OgdenBranch(float(tau), float(p)) for tau, p in zip(tau_s, shares)),
        shared_flow=FlowParameters(**flow), model_name="T1-Ogden-1")
    result.validate()
    return result


def build_t2_ogden_parameters(zhang_li: dict, fast_flow: dict, slow_flow: dict,
                              tau_s, old_weights, Gneq_MPa: float,
                              a_ogden: float, transition_tau_s: float = 1.0) -> OgdenPRFParameters:
    shares = normalized_shares(old_weights)
    result = OgdenPRFParameters(
        ZhangLiParameters(**zhang_li), float(Gneq_MPa), float(a_ogden),
        tuple(OgdenBranch(float(tau), float(p)) for tau, p in zip(tau_s, shares)),
        fast_flow=FlowParameters(**fast_flow), slow_flow=FlowParameters(**slow_flow),
        transition_tau_s=float(transition_tau_s), model_name="T2-Ogden-1")
    result.validate()
    return result

