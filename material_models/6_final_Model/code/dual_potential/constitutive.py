"""Common constitutive assembly for traditional and neural dual potentials."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn

from .dissipation import DualDissipationPotential
from .energies import FreeEnergy, ZhangLiEquilibriumEnergy
from .tensor import dev, first_piola, pressure_project, sym


@dataclass(frozen=True)
class MaterialParameters:
    zhang_li: dict
    tau_s: tuple[float, ...]
    shares: tuple[float, ...]
    Gneq_MPa: float
    a_ogden: float
    fast_flow: dict
    slow_flow: dict
    transition_tau_s: float

    @classmethod
    def from_json(cls, path: str | Path) -> "MaterialParameters":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(raw["zhang_li"], tuple(raw["tau_s"]), tuple(raw["p_alpha"]),
                   float(raw["Gneq_MPa"]), float(raw["a_ogden"]),
                   raw["flows"]["fast"], raw["flows"]["slow"],
                   float(raw["transition_tau_s"]))

    @property
    def branch_moduli_MPa(self) -> tuple[float, ...]:
        return tuple(self.Gneq_MPa * value for value in self.shares)

    @property
    def equilibrium_shear_MPa(self) -> float:
        p = self.zhang_li
        return 2.0 * (p["D"] + p["B"] * p["beta"] * p["N"] ** (p["beta"] - 1.0))


class DualPotentialMaterial(nn.Module):
    """Seven-branch multiplicative model driven solely by two potentials."""

    def __init__(self, equilibrium_energy: ZhangLiEquilibriumEnergy,
                 nonequilibrium_energy: FreeEnergy,
                 dissipation_potential: DualDissipationPotential,
                 tau_s, *, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.equilibrium_energy = equilibrium_energy
        self.nonequilibrium_energy = nonequilibrium_energy
        self.dissipation_potential = dissipation_potential
        self.register_buffer("tau_s", torch.as_tensor(tau_s, dtype=dtype, device=device))
        if len(self.tau_s) != 7:
            raise ValueError("the frozen chapter-3 core spectrum has seven branches")
        self.stop_rollout_physics_gradient = False

    @property
    def dtype(self):
        return self.tau_s.dtype

    @property
    def device(self):
        return self.tau_s.device

    def identity_state(self) -> torch.Tensor:
        return torch.eye(3, dtype=self.dtype, device=self.device).repeat(7, 1, 1)

    @staticmethod
    def _energy_derivatives(energy: FreeEnergy, deformation: torch.Tensor,
                            branch: int | None = None, *, create_graph: bool = True):
        with torch.enable_grad():
            A = deformation if deformation.requires_grad else deformation.detach().clone().requires_grad_(True)
            psi = energy(A, branch)
            derivative = torch.autograd.grad(psi.sum(), A, create_graph=create_graph)[0]
            J = torch.linalg.det(A)
            tau = derivative @ A.transpose(-1, -2)
            cauchy = sym(tau) / J[..., None, None]
            mandel = sym(A.transpose(-1, -2) @ derivative)
        return psi, derivative, cauchy, mandel

    def branch_kinematics(self, F: torch.Tensor, Fv: torch.Tensor,
                          *, create_graph: bool = True) -> dict[str, torch.Tensor]:
        Fe = F.unsqueeze(0) @ torch.linalg.inv(Fv)
        energies, stresses, mandels = [], [], []
        for branch in range(7):
            psi, _, sigma, M = self._energy_derivatives(
                self.nonequilibrium_energy, Fe[branch], branch, create_graph=create_graph)
            energies.append(psi); stresses.append(sigma); mandels.append(M)
        M = torch.stack(mandels)
        X = dev(M)
        return {"Fe": Fe, "branch_energy_MPa": torch.stack(energies),
                "branch_cauchy_unpressurized_MPa": torch.stack(stresses),
                "mandel_MPa": M, "driving_force_MPa": X}

    def branch_flow(self, F: torch.Tensor, Fv: torch.Tensor, branch: int,
                    *, create_graph: bool = True) -> dict[str, torch.Tensor]:
        Fe = F @ torch.linalg.inv(Fv)
        psi, _, sigma, M = self._energy_derivatives(
            self.nonequilibrium_energy, Fe, branch, create_graph=create_graph)
        X = dev(M)
        C, Ce, Cv = F.transpose(-1, -2) @ F, Fe.transpose(-1, -2) @ Fe, Fv.transpose(-1, -2) @ Fv
        context = {
            "total_I1_minus_3": torch.diagonal(C).sum() - 3.0,
            "elastic_I1_minus_3": torch.diagonal(Ce).sum() - 3.0,
            "viscous_I1_minus_3": torch.diagonal(Cv).sum() - 3.0,
            "hardening_variable": torch.sqrt(torch.clamp(
                2.0 * torch.sum((0.5 * torch.log(torch.linalg.eigvalsh(Cv).clamp_min(1e-30)) -
                                 0.5 * torch.log(torch.linalg.eigvalsh(Cv).clamp_min(1e-30)).mean()).square()) / 3.0,
                min=0.0)),
        }
        flow_force = X.detach() if create_graph and self.stop_rollout_physics_gradient else X
        flow_context = ({name:value.detach() for name,value in context.items()}
                        if create_graph and self.stop_rollout_physics_gradient else context)
        rate, phi = self.dissipation_potential.flow(
            flow_force, Fv, branch, context=flow_context, create_graph=create_graph)
        dissipation = torch.einsum("...ij,...ij->...", X, rate)
        return {"Fe": Fe, "branch_energy_MPa": psi,
                "branch_cauchy_unpressurized_MPa": sigma, "mandel_MPa": M,
                "driving_force_MPa": X, "flow_rate_per_s": rate,
                "dual_potential_MPa_per_s": phi,
                "dissipation_MPa_per_s": dissipation}

    def response(self, F: torch.Tensor, Fv: torch.Tensor, mode: str,
                 *, create_graph: bool = True) -> dict[str, torch.Tensor]:
        eq_psi, _, eq_sigma, _ = self._energy_derivatives(
            self.equilibrium_energy, F, create_graph=create_graph)
        branches = self.branch_kinematics(F, Fv, create_graph=create_graph)
        unpressurized = eq_sigma + branches["branch_cauchy_unpressurized_MPa"].sum(0)
        cauchy = pressure_project(unpressurized, mode)
        P = first_piola(cauchy, F)
        S = torch.linalg.inv(F) @ P
        measured = cauchy[0, 1] if mode == "shear" else P[0, 0]
        flow, potential, dissipation = [], [], []
        for branch in range(7):
            C = F.transpose(-1, -2) @ F
            Fe = branches["Fe"][branch]
            Ce = Fe.transpose(-1, -2) @ Fe
            Cv = Fv[branch].transpose(-1, -2) @ Fv[branch]
            principal = 0.5 * torch.log(torch.linalg.eigvalsh(Cv).clamp_min(1e-30))
            context = {
                "total_I1_minus_3": torch.diagonal(C).sum() - 3.0,
                "elastic_I1_minus_3": torch.diagonal(Ce).sum() - 3.0,
                "viscous_I1_minus_3": torch.diagonal(Cv).sum() - 3.0,
                "hardening_variable": torch.sqrt(torch.clamp(
                    2.0 * torch.sum((principal-principal.mean()).square()) / 3.0, min=0.0)),
            }
            rate, phi = self.dissipation_potential.flow(
                branches["driving_force_MPa"][branch], Fv[branch], branch,
                context=context, create_graph=create_graph)
            flow.append(rate); potential.append(phi)
            dissipation.append(torch.einsum("ij,ij->", branches["driving_force_MPa"][branch], rate))
        return {**branches, "equilibrium_energy_MPa": eq_psi,
                "total_energy_MPa": eq_psi + branches["branch_energy_MPa"].sum(),
                "equilibrium_cauchy_unpressurized_MPa": eq_sigma,
                "cauchy_unpressurized_MPa": unpressurized,
                "cauchy_MPa": cauchy, "first_piola_MPa": P,
                "second_piola_MPa": sym(S), "measured_stress_MPa": measured,
                "flow_rate_per_s": torch.stack(flow),
                "dual_potential_MPa_per_s": torch.stack(potential),
                "branch_dissipation_MPa_per_s": torch.stack(dissipation),
                "total_dissipation_MPa_per_s": torch.stack(dissipation).sum()}
