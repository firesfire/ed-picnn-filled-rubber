"""Local implicit exponential-map integration with Newton and subdivision."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from .constitutive import DualPotentialMaterial
from .tensor import dev, symmetric_deviatoric_basis, sym


@dataclass(frozen=True)
class NewtonOptions:
    absolute_tolerance: float = 1e-10
    relative_tolerance: float = 1e-8
    maximum_iterations: int = 40
    maximum_subdivision_depth: int = 10
    line_search_steps: int = 12


@dataclass
class NewtonDiagnostics:
    converged: bool
    iterations: int
    residual_norm: float
    subdivisions: int
    determinant_error: float
    line_search_reductions: int


class ImplicitExponentialIntegrator:
    def __init__(self, material: DualPotentialMaterial, options: NewtonOptions | None = None,
                 jacobian_mode: str = "finite_difference"):
        if jacobian_mode not in ("finite_difference", "autograd"):
            raise ValueError("jacobian_mode must be finite_difference or autograd")
        self.material = material
        self.options = options or NewtonOptions()
        self.jacobian_mode = jacobian_mode
        self.basis = symmetric_deviatoric_basis(material.dtype, material.device)

    def vector_to_tensor(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...k,kij->...ij", z, self.basis)

    def tensor_to_vector(self, A: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...ij,kij->...k", dev(sym(A)), self.basis)

    def _residual(self, z: torch.Tensor, Fv0: torch.Tensor, F1: torch.Tensor,
                  dt: torch.Tensor, branch: int, create_graph: bool):
        Fv = torch.matrix_exp(self.vector_to_tensor(z)) @ Fv0
        # The local Jacobian differentiates a flow gradient composed with an
        # energy gradient, so both gradients must retain their graph even when
        # the converged state itself is later detached.
        response = self.material.branch_flow(F1, Fv, branch, create_graph=True)
        residual = z - dt * self.tensor_to_vector(response["flow_rate_per_s"])
        return residual, Fv

    def _single(self, Fv0: torch.Tensor, F1: torch.Tensor, dt: float, branch: int,
                differentiable: bool) -> tuple[torch.Tensor, NewtonDiagnostics]:
        if dt <= 0:
            return Fv0, NewtonDiagnostics(True, 0, 0.0, 0, float(abs(torch.linalg.det(Fv0)-1)), 0)
        step = torch.tensor(dt, dtype=Fv0.dtype, device=Fv0.device)
        initial = self.material.branch_flow(F1, Fv0, branch, create_graph=False)["flow_rate_per_s"]
        z = (step * self.tensor_to_vector(initial)).clamp(-0.35, 0.35)
        reductions = 0
        for iteration in range(1, self.options.maximum_iterations + 1):
            z = z.requires_grad_(True)
            residual, Fv = self._residual(z, Fv0, F1, step, branch, differentiable)
            norm = torch.linalg.vector_norm(residual)
            scalar = float(norm.detach().cpu())
            # Scale by the actual converging logarithmic increment, not by a
            # potentially enormous explicit fast-branch predictor. The latter
            # can declare convergence with an inaccurate implicit sensitivity.
            scale = max(1.0, float(torch.linalg.vector_norm(z).detach().cpu()))
            if scalar <= self.options.absolute_tolerance + self.options.relative_tolerance * scale:
                error = float(torch.abs(torch.linalg.det(Fv) - 1.0).detach().cpu())
                return Fv, NewtonDiagnostics(True, iteration, scalar, 0, error, reductions)
            def function(value):
                return self._residual(value, Fv0, F1, step, branch, False)[0]
            if self.jacobian_mode == "autograd":
                J = torch.autograd.functional.jacobian(function, z, create_graph=False).detach()
            else:
                # Central differences avoid unnecessary second derivatives of
                # repeated principal stretches during ordinary rollout.
                h = 1e-7 * (1.0 + z.detach().abs())
                columns = []
                for column in range(5):
                    delta = torch.zeros_like(z); delta[column] = h[column]
                    plus = function(z + delta); minus = function(z - delta)
                    columns.append((plus-minus)/(2.0*h[column]))
                J = torch.stack(columns, dim=-1).detach()
            try:
                dz = torch.linalg.solve(J, -residual)
            except torch.linalg.LinAlgError:
                dz = torch.linalg.lstsq(J, -residual).solution
            accepted = False
            alpha = 1.0
            for reduction in range(self.options.line_search_steps):
                candidate = z + alpha * dz
                try:
                    candidate_residual, candidate_Fv = self._residual(
                        candidate, Fv0, F1, step, branch, differentiable)
                    candidate_norm = float(torch.linalg.vector_norm(candidate_residual).detach().cpu())
                except (RuntimeError, ValueError, FloatingPointError, torch.linalg.LinAlgError):
                    candidate_norm = float("inf")
                if candidate_norm < (1.0 - 1e-4 * alpha) * scalar:
                    z, Fv = candidate, candidate_Fv
                    reductions += reduction
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break
        return Fv0, NewtonDiagnostics(False, self.options.maximum_iterations,
                                       float("inf"), 0, float("inf"), reductions)

    @staticmethod
    def _midpoint(F0: torch.Tensor, F1: torch.Tensor) -> torch.Tensor:
        midpoint = 0.5 * (F0 + F1)
        determinant = torch.linalg.det(midpoint)
        return midpoint * determinant.pow(-1.0 / 3.0)

    def _advance_branch(self, Fv0, F0, F1, dt, branch, differentiable, level=0):
        try:
            state, diagnostics = self._single(Fv0, F1, dt, branch, differentiable)
        except (RuntimeError, ValueError, FloatingPointError, torch.linalg.LinAlgError):
            diagnostics = NewtonDiagnostics(False, self.options.maximum_iterations,
                                            float("inf"), level, float("inf"), 0)
            state = Fv0
        if diagnostics.converged:
            diagnostics.subdivisions = level
            return state, diagnostics
        if level >= self.options.maximum_subdivision_depth:
            raise RuntimeError(f"branch {branch} Newton failed after {level} subdivisions")
        midpoint = self._midpoint(F0, F1)
        half, first = self._advance_branch(Fv0, F0, midpoint, 0.5*dt, branch, differentiable, level+1)
        state, second = self._advance_branch(half, midpoint, F1, 0.5*dt, branch, differentiable, level+1)
        second.iterations += first.iterations
        second.line_search_reductions += first.line_search_reductions
        second.subdivisions = max(first.subdivisions, second.subdivisions)
        return state, second

    def advance(self, Fv0: torch.Tensor, F0: torch.Tensor, F1: torch.Tensor,
                dt: float, *, differentiable: bool = False):
        states, diagnostics = [], []
        for branch in range(7):
            state, item = self._advance_branch(Fv0[branch], F0, F1, dt, branch, differentiable)
            states.append(state); diagnostics.append(item)
        result = torch.stack(states)
        return (result if differentiable else result.detach()), diagnostics
