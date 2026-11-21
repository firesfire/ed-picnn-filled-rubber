"""Equilibrium, Ogden and constrained neural isochoric free energies."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .tensor import invariants, isochoric_metric, spectral_power


class FreeEnergy(nn.Module):
    """Scalar energy density in MPa as a function of a deformation gradient."""

    def forward(self, deformation: torch.Tensor, branch: int | None = None) -> torch.Tensor:
        raise NotImplementedError


class ZhangLiEquilibriumEnergy(FreeEnergy):
    def __init__(self, parameters: dict, *, dtype=torch.float64, device="cpu"):
        super().__init__()
        for name in ("D", "mu", "B", "N", "beta"):
            self.register_buffer(name, torch.tensor(float(parameters[name]), dtype=dtype, device=device))

    def forward(self, deformation: torch.Tensor, branch: int | None = None) -> torch.Tensor:
        C = deformation.transpose(-1, -2) @ deformation
        I1, I2, _ = invariants(C)
        base = I2 - 3.0 + self.N
        if bool(torch.any(base <= 0).detach().cpu()):
            raise FloatingPointError("invalid Zhang-Li invariant base")
        first = self.D * (I1 - 3.0) if abs(float(self.mu)) < 1e-12 else (
            self.D / self.mu * (1.0 - torch.exp(-self.mu * (I1 - 3.0))))
        return first + self.B * (base.pow(self.beta) - self.N.pow(self.beta))


class FixedOgdenEnergy(FreeEnergy):
    """One-term incompressible Ogden spectrum with fixed positive branch moduli."""

    def __init__(self, branch_moduli_MPa, exponent: float, *, dtype=torch.float64, device="cpu"):
        super().__init__()
        moduli = torch.as_tensor(branch_moduli_MPa, dtype=dtype, device=device)
        if bool(torch.any(moduli <= 0)) or abs(exponent) < 1e-6:
            raise ValueError("Ogden moduli and exponent must be nonzero and positive")
        self.register_buffer("branch_moduli_MPa", moduli)
        self.register_buffer("exponent", torch.tensor(float(exponent), dtype=dtype, device=device))

    def parameters_physical(self) -> dict[str, torch.Tensor]:
        return {"branch_moduli_MPa": self.branch_moduli_MPa, "exponent": self.exponent}

    def forward(self, deformation: torch.Tensor, branch: int | None = None) -> torch.Tensor:
        Cbar = isochoric_metric(deformation.transpose(-1, -2) @ deformation)
        # Use the symmetric spectral values directly. Reconstructing a matrix
        # power and then taking its trace differentiates arbitrary eigenvectors
        # at repeated reference eigenvalues and produces NaNs at C=I.
        values = torch.linalg.eigvalsh(Cbar).clamp_min(1.0e-14)
        principal_sum = values.pow(0.5 * self.exponent).sum(-1)
        unit = 2.0 * (principal_sum - 3.0) / self.exponent.square()
        modulus = self.branch_moduli_MPa if branch is None else self.branch_moduli_MPa[branch]
        if branch is None and deformation.ndim >= 3 and deformation.shape[-3] == len(self.branch_moduli_MPa):
            return modulus * unit
        return modulus * unit


class LowParameterOgdenEnergy(FixedOgdenEnergy):
    """Trainable positive total stiffness, simplex shares and bounded exponent."""

    def __init__(self, total_modulus_MPa: float, shares, exponent: float,
                 exponent_bounds=(0.25, 20.0), *, dtype=torch.float64, device="cpu"):
        nn.Module.__init__(self)
        shares = torch.as_tensor(shares, dtype=dtype, device=device)
        shares = shares / shares.sum()
        self.raw_total = nn.Parameter(torch.tensor(math.log(math.expm1(total_modulus_MPa)), dtype=dtype, device=device))
        self.raw_shares = nn.Parameter(torch.log(shares.clamp_min(1e-12)))
        low, high = exponent_bounds
        fraction = (exponent - low) / (high - low)
        self.raw_exponent = nn.Parameter(torch.tensor(math.log(fraction / (1.0 - fraction)), dtype=dtype, device=device))
        self.exponent_bounds = exponent_bounds

    @property
    def branch_moduli_MPa(self) -> torch.Tensor:
        return F.softplus(self.raw_total) * torch.softmax(self.raw_shares, dim=0)

    @property
    def exponent(self) -> torch.Tensor:
        low, high = self.exponent_bounds
        return low + (high - low) * torch.sigmoid(self.raw_exponent)


class GroupedLowParameterOgdenEnergy(LowParameterOgdenEnergy):
    """Positive simplex spectrum with separate bounded fast/slow exponents.

    The small-strain modulus remains the sum of the branch moduli.  Only the
    finite-strain shape is grouped, so this is a deliberately low-parameter
    diagnostic before introducing a neural free energy.
    """

    def __init__(self, total_modulus_MPa: float, shares, fast_exponent: float,
                 slow_exponent: float, tau_s, transition_tau_s: float = 1.0,
                 exponent_bounds=(0.25, 20.0), *, dtype=torch.float64, device="cpu"):
        nn.Module.__init__(self)
        shares = torch.as_tensor(shares, dtype=dtype, device=device)
        shares = shares / shares.sum()
        self.raw_total = nn.Parameter(torch.tensor(
            math.log(math.expm1(total_modulus_MPa)), dtype=dtype, device=device))
        self.raw_shares = nn.Parameter(torch.log(shares.clamp_min(1e-12)))
        low, high = exponent_bounds

        def raw(value):
            fraction = min(max((value-low)/(high-low), 1e-8), 1.0-1e-8)
            return math.log(fraction/(1.0-fraction))

        self.raw_fast_exponent = nn.Parameter(torch.tensor(raw(fast_exponent), dtype=dtype, device=device))
        self.raw_slow_exponent = nn.Parameter(torch.tensor(raw(slow_exponent), dtype=dtype, device=device))
        self.register_buffer("tau_s", torch.as_tensor(tau_s, dtype=dtype, device=device))
        self.transition_tau_s = float(transition_tau_s)
        self.exponent_bounds = exponent_bounds

    @property
    def fast_exponent(self) -> torch.Tensor:
        low, high = self.exponent_bounds
        return low + (high-low)*torch.sigmoid(self.raw_fast_exponent)

    @property
    def slow_exponent(self) -> torch.Tensor:
        low, high = self.exponent_bounds
        return low + (high-low)*torch.sigmoid(self.raw_slow_exponent)

    @property
    def exponent(self) -> torch.Tensor:
        return torch.where(self.tau_s <= self.transition_tau_s,
                           self.fast_exponent, self.slow_exponent)

    def forward(self, deformation: torch.Tensor, branch: int | None = None) -> torch.Tensor:
        Cbar = isochoric_metric(deformation.transpose(-1, -2) @ deformation)
        values = torch.linalg.eigvalsh(Cbar).clamp_min(1.0e-14)
        if branch is None:
            if deformation.ndim < 3 or deformation.shape[-3] != len(self.tau_s):
                raise ValueError("grouped Ogden energy expects an explicit branch axis")
            exponent = self.exponent
            modulus = self.branch_moduli_MPa
        else:
            exponent = self.exponent[branch]
            modulus = self.branch_moduli_MPa[branch]
        principal_sum = values.pow(0.5*exponent[..., None]).sum(-1)
        return modulus * 2.0*(principal_sum-3.0)/exponent.square()


class NeuralIsochoricEnergy(FreeEnergy):
    """Isotropic neural free energy convex and nondecreasing in isochoric invariants.

    Inputs I1_bar-3 and I2_bar-3 are objective, isotropic and nonnegative.
    The isochoric projection makes their first derivative vanish at identity,
    while positive first slopes retain a finite small-strain tangent.
    """

    def __init__(self, branch_count: int, width: int = 12, *, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.branch_count = branch_count
        self.raw_linear = nn.Parameter(torch.full((branch_count, 2), -2.0, dtype=dtype, device=device))
        self.raw_rates = nn.Parameter(torch.zeros((branch_count, width, 2), dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.linspace(-3.0, 1.0, width, dtype=dtype, device=device).repeat(branch_count, 1))
        self.raw_gains = nn.Parameter(torch.full((branch_count, width), -4.0, dtype=dtype, device=device))

    def _features(self, deformation: torch.Tensor) -> torch.Tensor:
        Cbar = isochoric_metric(deformation.transpose(-1, -2) @ deformation)
        I1, I2, _ = invariants(Cbar)
        return torch.stack(((I1 - 3.0).clamp_min(0.0), (I2 - 3.0).clamp_min(0.0)), -1)

    def forward(self, deformation: torch.Tensor, branch: int | None = None) -> torch.Tensor:
        if branch is None:
            if deformation.shape[-3] != self.branch_count:
                raise ValueError("batched neural energy expects a branch axis")
            indices = torch.arange(self.branch_count, device=deformation.device)
        else:
            indices = branch
        x = self._features(deformation)
        linear = F.softplus(self.raw_linear[indices])
        rates = F.softplus(self.raw_rates[indices]) + 1e-8
        gains = F.softplus(self.raw_gains[indices]) / self.raw_gains.shape[-1]
        bias = self.bias[indices]
        arguments = torch.einsum("...d,...kd->...k", x, rates) + bias
        shifted = F.softplus(arguments) - F.softplus(bias)
        return torch.einsum("...d,...d->...", linear, x) + torch.sum(gains * shifted, dim=-1)


class TangentSecantEnergy(FreeEnergy):
    """Stage-5B isochoric energy with fixed branch tangents and trainable shape.

    Positive saturating invariant ridges preserve the prescribed reference
    tangent.  A positive invariant-quadratic term has zero small-strain
    tangent and supplies finite-strain re-hardening and an energy-growth floor.
    Parameters are shared either by all branches or by fixed fast/slow groups.
    """

    def __init__(self, branch_moduli_MPa, tau_s, physical_groups: list[dict],
                 *, grouped: bool, b_bounds=(1e-3, 1e4), p_bounds=(0.5, 1.0),
                 transition_tau_s=1.0, share_delta_limit=0.25,
                 dtype=torch.float64, device="cpu"):
        super().__init__()
        moduli = torch.as_tensor(branch_moduli_MPa, dtype=dtype, device=device)
        times = torch.as_tensor(tau_s, dtype=dtype, device=device)
        if len(moduli) != 7 or bool(torch.any(moduli <= 0)):
            raise ValueError("TangentSecantEnergy requires seven positive branch moduli")
        self.register_buffer("branch_moduli_MPa", moduli)
        # Stage 5 keeps this parameter frozen.  Stage 6 may release it in a
        # bounded softmax block, preserving positivity, normalization and the
        # total reference nonequilibrium modulus.
        self.raw_share_delta = nn.Parameter(torch.zeros_like(moduli), requires_grad=False)
        self.share_delta_limit = float(share_delta_limit)
        self.register_buffer("tau_s", times)
        groups = (times > float(transition_tau_s)).long() if grouped else torch.zeros_like(times, dtype=torch.long)
        self.register_buffer("branch_group", groups)
        self.grouped = bool(grouped)
        self.transition_tau_s = float(transition_tau_s)
        self.log_b_bounds = tuple(math.log(float(value)) for value in b_bounds)
        self.p_bounds = tuple(float(value) for value in p_bounds)
        expected = 2 if grouped else 1
        if len(physical_groups) != expected:
            raise ValueError(f"expected {expected} physical parameter groups")

        def logit(value):
            value = torch.as_tensor(value, dtype=dtype, device=device).clamp(1e-8, 1.0-1e-8)
            return torch.log(value/(1.0-value))

        self.raw_weight = nn.ParameterList(); self.raw_rho = nn.ParameterList()
        self.raw_b = nn.ParameterList(); self.raw_p = nn.ParameterList()
        self.raw_hardening = nn.ParameterList(); self.raw_hardening_rho = nn.ParameterList()
        blo, bhi = self.log_b_bounds; plo, phi = self.p_bounds
        for item in physical_groups:
            weights = torch.as_tensor(item["weights"], dtype=dtype, device=device).clamp_min(1e-12)
            rho = torch.as_tensor(item["rho_I1"], dtype=dtype, device=device)
            b = torch.as_tensor(item["b"], dtype=dtype, device=device).clamp(math.exp(blo), math.exp(bhi))
            p = torch.as_tensor(item["p"], dtype=dtype, device=device).clamp(plo+1e-8, phi-1e-8)
            hardening = torch.as_tensor(float(item["hardening_MPa"]), dtype=dtype, device=device).clamp_min(1e-12)
            self.raw_weight.append(nn.Parameter(torch.log(weights/weights.sum())))
            self.raw_rho.append(nn.Parameter(logit(rho)))
            self.raw_b.append(nn.Parameter(logit((torch.log(b)-blo)/(bhi-blo))))
            self.raw_p.append(nn.Parameter(logit((p-plo)/(phi-plo))))
            self.raw_hardening.append(nn.Parameter(torch.log(torch.expm1(hardening))))
            self.raw_hardening_rho.append(nn.Parameter(logit(float(item["hardening_rho_I1"]))))

    @property
    def effective_branch_moduli_MPa(self) -> torch.Tensor:
        total = self.branch_moduli_MPa.sum()
        base = self.branch_moduli_MPa/total
        logits = torch.log(base.clamp_min(1e-30)) + self.share_delta_limit*torch.tanh(
            self.raw_share_delta)
        return total*torch.softmax(logits, 0)

    def group_physical(self, group: int) -> dict[str, torch.Tensor]:
        blo, bhi = self.log_b_bounds; plo, phi = self.p_bounds
        return {
            "weights": torch.softmax(self.raw_weight[group], 0),
            "rho_I1": torch.sigmoid(self.raw_rho[group]),
            "b": torch.exp(blo+(bhi-blo)*torch.sigmoid(self.raw_b[group])),
            "p": plo+(phi-plo)*torch.sigmoid(self.raw_p[group]),
            "hardening_MPa": F.softplus(self.raw_hardening[group]),
            "hardening_rho_I1": torch.sigmoid(self.raw_hardening_rho[group]),
        }

    @staticmethod
    def _saturation(z, b, p):
        return ((1.0+b*z).pow(p)-1.0)/(b*p)

    def invariant_quantities(self, x1: torch.Tensor, x2: torch.Tensor,
                             branch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        group_index = int(self.branch_group[branch])
        group = self.group_physical(group_index)
        moduli = self.effective_branch_moduli_MPa
        modulus = moduli[branch]
        group_modulus = moduli[self.branch_group == group_index].sum()
        hardening = group["hardening_MPa"]*modulus/group_modulus
        z = group["rho_I1"]*x1[..., None] + (1.0-group["rho_I1"])*x2[..., None]
        slope = (1.0+group["b"]*z).pow(group["p"]-1.0)
        psi = .5*modulus*torch.sum(group["weights"]*self._saturation(z, group["b"], group["p"]), -1)
        w1 = .5*modulus*torch.sum(group["weights"]*slope*group["rho_I1"], -1)
        w2 = .5*modulus*torch.sum(group["weights"]*slope*(1.0-group["rho_I1"]), -1)
        zh = group["hardening_rho_I1"]*x1 + (1.0-group["hardening_rho_I1"])*x2
        psi = psi + .25*hardening*zh.square()
        w1 = w1 + .5*hardening*zh*group["hardening_rho_I1"]
        w2 = w2 + .5*hardening*zh*(1.0-group["hardening_rho_I1"])
        return psi, w1, w2

    def invariant_quantities_all(self, x1: torch.Tensor, x2: torch.Tensor):
        """Vectorized energy and invariant slopes for a trailing branch axis."""
        if x1.shape != x2.shape or x1.shape[-1] != 7:
            raise ValueError("invariant_quantities_all expects a trailing seven-branch axis")
        outputs=[]
        for group_index in range(len(self.raw_weight)):
            mask=self.branch_group==group_index
            group=self.group_physical(group_index)
            modulus=self.effective_branch_moduli_MPa[mask]
            group_modulus=modulus.sum()
            hardening=group["hardening_MPa"]*modulus/group_modulus
            first=x1[...,mask];second=x2[...,mask]
            z=group["rho_I1"]*first[...,None]+(1.0-group["rho_I1"])*second[...,None]
            slope=(1.0+group["b"]*z).pow(group["p"]-1.0)
            psi=.5*modulus*torch.sum(group["weights"]*self._saturation(z,group["b"],group["p"]),-1)
            w1=.5*modulus*torch.sum(group["weights"]*slope*group["rho_I1"],-1)
            w2=.5*modulus*torch.sum(group["weights"]*slope*(1.0-group["rho_I1"]),-1)
            zh=group["hardening_rho_I1"]*first+(1.0-group["hardening_rho_I1"])*second
            psi=psi+.25*hardening*zh.square()
            w1=w1+.5*hardening*zh*group["hardening_rho_I1"]
            w2=w2+.5*hardening*zh*(1.0-group["hardening_rho_I1"])
            outputs.append((psi,w1,w2))
        return tuple(torch.cat([item[index] for item in outputs],-1) for index in range(3))

    def forward(self, deformation: torch.Tensor, branch: int | None = None) -> torch.Tensor:
        if branch is None:
            if deformation.ndim < 3 or deformation.shape[-3] != 7:
                raise ValueError("batched TangentSecantEnergy expects a seven-branch axis")
            values = [self.forward(deformation[..., index, :, :], index) for index in range(7)]
            return torch.stack(values, -1)
        Cbar = isochoric_metric(deformation.transpose(-1, -2) @ deformation)
        I1, I2, _ = invariants(Cbar)
        return self.invariant_quantities((I1-3.0).clamp_min(0.0),
                                         (I2-3.0).clamp_min(0.0), branch)[0]

    def physical_parameters(self) -> dict:
        groups = []
        for index in range(len(self.raw_weight)):
            item = self.group_physical(index)
            groups.append({name: (float(value.detach()) if value.ndim == 0 else
                                  [float(x) for x in value.detach().cpu()])
                           for name, value in item.items()})
        return {"family": "hierarchical_fast_slow_tangent_secant" if self.grouped else
                          "shared_tangent_secant",
                "branch_moduli_MPa": [float(x) for x in self.effective_branch_moduli_MPa.detach().cpu()],
                "base_branch_moduli_MPa": [float(x) for x in self.branch_moduli_MPa.detach().cpu()],
                "groups": groups}
