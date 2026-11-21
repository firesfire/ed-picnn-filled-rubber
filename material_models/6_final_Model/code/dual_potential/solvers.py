"""Material-point paths, stress control, periodic steady state and DMA output."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .constitutive import DualPotentialMaterial
from .integrator import ImplicitExponentialIntegrator, NewtonDiagnostics
from .tensor import deformation_gradient


@dataclass
class HistoryResult:
    time_s: torch.Tensor
    primary: torch.Tensor
    F: torch.Tensor
    Fv: torch.Tensor
    measured_stress_MPa: torch.Tensor
    total_energy_MPa: torch.Tensor
    branch_dissipation_MPa_per_s: torch.Tensor
    total_dissipation_MPa_per_s: torch.Tensor
    driving_force_MPa: torch.Tensor
    flow_rate_per_s: torch.Tensor
    diagnostics: list[list[NewtonDiagnostics]]


@dataclass(frozen=True)
class DynamicModuli:
    frequency_hz: float
    amplitude: float
    storage_MPa: float
    loss_MPa: float
    loss_factor: float
    cycle_change: float
    steady: bool
    harmonic_ratios: dict[int, float]


class MaterialPointSolver:
    def __init__(self, material: DualPotentialMaterial,
                 integrator: ImplicitExponentialIntegrator | None = None,
                 maximum_scalar_increment: float = 0.025):
        self.material = material
        self.integrator = integrator or ImplicitExponentialIntegrator(material)
        self.maximum_scalar_increment = maximum_scalar_increment

    def solve_history(self, primary: torch.Tensor, time_s: torch.Tensor, mode: str,
                      *, secondary: torch.Tensor | None = None,
                      initial_Fv: torch.Tensor | None = None,
                      differentiable: bool = False) -> HistoryResult:
        if len(primary) != len(time_s) or bool(torch.any(torch.diff(time_s) <= 0)):
            raise ValueError("history sizes must match and time must increase")
        secondary = torch.zeros_like(primary) if secondary is None else secondary
        F = deformation_gradient(primary, mode, secondary,
                                 dtype=self.material.dtype, device=self.material.device)
        state = self.material.identity_state() if initial_Fv is None else initial_Fv
        states, responses, diagnostics = [state], [], [[]]
        responses.append(self.material.response(F[0], state, mode, create_graph=differentiable))
        for index in range(1, len(time_s)):
            dt = float(time_s[index] - time_s[index-1])
            distance = max(float(abs(primary[index]-primary[index-1])),
                           float(abs(secondary[index]-secondary[index-1])))
            pieces = max(1, math.ceil(distance / self.maximum_scalar_increment))
            step_diagnostics = []
            for piece in range(pieces):
                a, b = piece / pieces, (piece + 1) / pieces
                p0 = primary[index-1] + a * (primary[index]-primary[index-1])
                p1 = primary[index-1] + b * (primary[index]-primary[index-1])
                s0 = secondary[index-1] + a * (secondary[index]-secondary[index-1])
                s1 = secondary[index-1] + b * (secondary[index]-secondary[index-1])
                F0 = deformation_gradient(p0, mode, s0, dtype=self.material.dtype, device=self.material.device)
                F1 = deformation_gradient(p1, mode, s1, dtype=self.material.dtype, device=self.material.device)
                state, diag = self.integrator.advance(state, F0, F1, dt/pieces,
                                                      differentiable=differentiable)
                step_diagnostics.extend(diag)
            states.append(state); diagnostics.append(step_diagnostics)
            responses.append(self.material.response(F[index], state, mode, create_graph=differentiable))
        return HistoryResult(time_s, primary, F, torch.stack(states),
            torch.stack([r["measured_stress_MPa"] for r in responses]),
            torch.stack([r["total_energy_MPa"] for r in responses]),
            torch.stack([r["branch_dissipation_MPa_per_s"] for r in responses]),
            torch.stack([r["total_dissipation_MPa_per_s"] for r in responses]),
            torch.stack([r["driving_force_MPa"] for r in responses]),
            torch.stack([r["flow_rate_per_s"] for r in responses]), diagnostics)

    def relaxation(self, level: float, loading_time_s: float, hold_time_s: float,
                   mode: str, *, time_step_s: float) -> HistoryResult:
        loading_points = max(1, math.ceil(loading_time_s/time_step_s))
        holding_points = max(1, math.ceil(hold_time_s/time_step_s))
        load_t = torch.linspace(0, loading_time_s, loading_points+1,
                               dtype=self.material.dtype, device=self.material.device)
        hold_t = loading_time_s + torch.linspace(time_step_s, hold_time_s, holding_points,
                                                 dtype=self.material.dtype, device=self.material.device)
        time = torch.cat((load_t, hold_t))
        primary = torch.cat((level * load_t/loading_time_s,
                             torch.full_like(hold_t, level)))
        return self.solve_history(primary, time, mode)

    def multilevel(self, levels, durations_s, mode: str, *, time_step_s: float) -> HistoryResult:
        if len(levels) != len(durations_s):
            raise ValueError("one duration is required for every level")
        time = [0.0]; primary = [0.0]
        for level, duration in zip(levels, durations_s):
            count = max(1, math.ceil(duration/time_step_s))
            for _ in range(count):
                time.append(time[-1] + duration/count); primary.append(float(level))
        return self.solve_history(torch.tensor(primary, dtype=self.material.dtype, device=self.material.device),
                                  torch.tensor(time, dtype=self.material.dtype, device=self.material.device), mode)

    def creep(self, stress_history_MPa: torch.Tensor, time_s: torch.Tensor, mode: str,
              *, initial_strain: float = 0.0, tolerance_MPa: float = 1e-7,
              maximum_iterations: int = 20) -> HistoryResult:
        """Stress-controlled path using an outer scalar Newton at every step."""
        if len(stress_history_MPa) != len(time_s):
            raise ValueError("stress and time histories must match")
        state = self.material.identity_state(); strain = torch.tensor(initial_strain, dtype=self.material.dtype)
        strains = [strain]; states = [state]; diagnostics = [[]]
        F_previous = deformation_gradient(strain, mode, dtype=self.material.dtype)
        responses = [self.material.response(F_previous, state, mode, create_graph=False)]
        for index in range(1, len(time_s)):
            target = stress_history_MPa[index]; dt = float(time_s[index]-time_s[index-1])
            guess = strain.clone()
            accepted_state = None; accepted_diag = None
            for _ in range(maximum_iterations):
                def trial(value):
                    F_trial = deformation_gradient(value, mode, dtype=self.material.dtype)
                    candidate, diag = self.integrator.advance(state, F_previous, F_trial, dt)
                    stress = self.material.response(F_trial, candidate, mode, create_graph=False)["measured_stress_MPa"]
                    return stress, candidate, diag
                stress, candidate, diag = trial(guess)
                residual = stress-target
                if float(abs(residual).detach()) <= tolerance_MPa:
                    accepted_state, accepted_diag = candidate, diag; break
                step = 1e-5 * max(1.0, abs(float(guess.detach())))
                plus = trial(guess + step)[0]; minus = trial(guess - step)[0]
                tangent = (plus-minus)/(2*step)
                guess = torch.clamp(guess-residual/tangent.clamp(min=1e-8), min=-0.8, max=3.0)
            if accepted_state is None:
                raise RuntimeError(f"creep stress control failed at index {index}")
            strain, state = guess.detach(), accepted_state
            F_previous = deformation_gradient(strain, mode, dtype=self.material.dtype)
            strains.append(strain); states.append(state); diagnostics.append(accepted_diag)
            responses.append(self.material.response(F_previous, state, mode, create_graph=False))
        primary = torch.stack(strains)
        F = deformation_gradient(primary, mode, dtype=self.material.dtype)
        return HistoryResult(time_s, primary, F, torch.stack(states),
            torch.stack([r["measured_stress_MPa"] for r in responses]),
            torch.stack([r["total_energy_MPa"] for r in responses]),
            torch.stack([r["branch_dissipation_MPa_per_s"] for r in responses]),
            torch.stack([r["total_dissipation_MPa_per_s"] for r in responses]),
            torch.stack([r["driving_force_MPa"] for r in responses]),
            torch.stack([r["flow_rate_per_s"] for r in responses]), diagnostics)

    def periodic_shear(self, frequency_hz: float, amplitude: float, *, preload: float = 0.0,
                       points_per_cycle: int = 128, maximum_cycles: int = 30,
                       steady_tolerance: float = 1e-5):
        state = self.material.identity_state(); prior = None; histories = []; change = float("inf")
        period = 1.0/frequency_hz
        for cycle in range(maximum_cycles):
            local = torch.linspace(0, period, points_per_cycle+1, dtype=self.material.dtype)
            strain = preload + amplitude*torch.sin(2*torch.pi*frequency_hz*local)
            result = self.solve_history(strain, local, "shear", initial_Fv=state)
            state = result.Fv[-1]; stress = result.measured_stress_MPa[1:]
            histories.append(result)
            if prior is not None:
                scale = torch.clamp(torch.max(torch.abs(stress)), min=0.025)
                change = float((torch.max(torch.abs(stress-prior))/scale).detach())
                if change <= steady_tolerance:
                    return result, extract_dynamic_moduli(strain[1:], stress, frequency_hz,
                                                          amplitude, change, steady_tolerance)
            prior = stress
        return histories[-1], extract_dynamic_moduli(strain[1:], stress, frequency_hz,
                                                      amplitude, change, steady_tolerance)

    def local_linear_shear_moduli(self, frequencies_hz: torch.Tensor,
                                  *, perturbation: float = 1e-6):
        """Complex shear tangent from a numerical state-space linearization.

        The 35 state coordinates are the symmetric-deviatoric logarithmic
        generators of the seven viscous deformation gradients. Linearization
        is performed at the undeformed equilibrium state and is independent of
        periodic integration.
        """
        basis = self.integrator.basis
        z0 = torch.zeros(35, dtype=self.material.dtype, device=self.material.device)

        def evaluate(z, gamma):
            tensors = torch.einsum("bk,kij->bij", z.reshape(7, 5), basis)
            Fv = torch.matrix_exp(tensors)
            F = deformation_gradient(gamma, "shear", dtype=self.material.dtype,
                                     device=self.material.device)
            response = self.material.response(F, Fv, "shear", create_graph=False)
            zdot = torch.einsum("bij,kij->bk", response["flow_rate_per_s"], basis).reshape(-1)
            return zdot.detach(), response["measured_stress_MPa"].detach()

        A = torch.zeros((35, 35), dtype=self.material.dtype)
        C = torch.zeros(35, dtype=self.material.dtype)
        for column in range(35):
            delta = torch.zeros_like(z0); delta[column] = perturbation
            plus_f, plus_s = evaluate(z0 + delta, 0.0)
            minus_f, minus_s = evaluate(z0 - delta, 0.0)
            A[:, column] = (plus_f - minus_f) / (2 * perturbation)
            C[column] = (plus_s - minus_s) / (2 * perturbation)
        plus_f, plus_s = evaluate(z0, perturbation)
        minus_f, minus_s = evaluate(z0, -perturbation)
        B = (plus_f - minus_f) / (2 * perturbation)
        D = (plus_s - minus_s) / (2 * perturbation)
        complex_A = A.to(torch.complex128)
        identity = torch.eye(35, dtype=torch.complex128)
        moduli = []
        for frequency in frequencies_hz:
            omega = 2 * torch.pi * frequency
            state_gain = torch.linalg.solve(1j * omega * identity - complex_A,
                                            B.to(torch.complex128))
            moduli.append(D.to(torch.complex128) + C.to(torch.complex128) @ state_gain)
        values = torch.stack(moduli)
        return values.real, values.imag


def extract_dynamic_moduli(strain: torch.Tensor, stress: torch.Tensor, frequency_hz: float,
                           amplitude: float, cycle_change: float, steady_tolerance: float) -> DynamicModuli:
    count = len(stress)
    phase = 2*torch.pi*torch.arange(1, count+1, dtype=stress.dtype, device=stress.device)/count
    storage = 2*torch.mean(stress*torch.sin(phase))/amplitude
    loss = 2*torch.mean(stress*torch.cos(phase))/amplitude
    fundamental = torch.sqrt(storage.square()+loss.square()).clamp_min(1e-14)
    ratios = {}
    for order in (2, 3, 5):
        a = 2*torch.mean(stress*torch.sin(order*phase))/amplitude
        b = 2*torch.mean(stress*torch.cos(order*phase))/amplitude
        ratios[order] = float((torch.sqrt(a.square()+b.square())/fundamental).detach())
    return DynamicModuli(frequency_hz, amplitude, float(storage.detach()), float(loss.detach()),
                         float((loss/storage.clamp_min(1e-14)).detach()), cycle_change,
                         cycle_change <= steady_tolerance, ratios)


def relaxation_transform(time_s: torch.Tensor, modulus_MPa: torch.Tensor,
                         frequencies_hz: torch.Tensor, equilibrium_MPa: float):
    """Numerical transform of the decaying part of a relaxation modulus."""
    tail = modulus_MPa - equilibrium_MPa
    storage, loss = [], []
    for frequency in frequencies_hz:
        omega = 2*torch.pi*frequency
        storage.append(equilibrium_MPa + omega*torch.trapezoid(tail*torch.sin(omega*time_s), time_s))
        loss.append(omega*torch.trapezoid(tail*torch.cos(omega*time_s), time_s))
    return torch.stack(storage), torch.stack(loss)
