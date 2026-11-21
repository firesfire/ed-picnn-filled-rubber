"""Traditional and neural convex dual dissipation potentials."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .tensor import dev, sym


def equivalent_force(X: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(1.5 * torch.einsum("...ij,...ij->...", X, X), min=0.0))


def viscous_hencky_equivalent(Fv: torch.Tensor) -> torch.Tensor:
    values = torch.linalg.eigvalsh(sym(Fv.transpose(-1, -2) @ Fv)).clamp_min(1e-30)
    principal = 0.5 * torch.log(values)
    principal = principal - principal.mean(-1, keepdim=True)
    return torch.sqrt(torch.clamp(2.0 * torch.sum(principal.square(), -1) / 3.0, min=0.0))


class DualDissipationPotential(nn.Module):
    def potential(self, X: torch.Tensor, Fv: torch.Tensor, branch: int,
                  context: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        raise NotImplementedError

    def flow(self, X: torch.Tensor, Fv: torch.Tensor, branch: int,
             *, context: dict[str, torch.Tensor] | None = None,
             create_graph: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            force = X if X.requires_grad else X.detach().clone().requires_grad_(True)
            phi = self.potential(force, Fv, branch, context)
            rate = torch.autograd.grad(phi.sum(), force, create_graph=create_graph)[0]
        return dev(sym(rate)), phi


class T2GroupedPowerPotential(DualDissipationPotential):
    """Convex radial potential reproducing the frozen grouped power-law form."""

    def __init__(self, branch_moduli_MPa, tau_s, equilibrium_shear_MPa: float,
                 fast: dict, slow: dict, transition_tau_s: float = 1.0,
                 *, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.register_buffer("moduli", torch.as_tensor(branch_moduli_MPa, dtype=dtype, device=device))
        self.register_buffer("tau_s", torch.as_tensor(tau_s, dtype=dtype, device=device))
        self.register_buffer("equilibrium_shear", torch.tensor(equilibrium_shear_MPa, dtype=dtype, device=device))
        self.fast, self.slow = dict(fast), dict(slow)
        self.transition_tau_s = float(transition_tau_s)

    def _flow_parameters(self, branch: int) -> dict:
        return self.fast if float(self.tau_s[branch]) <= self.transition_tau_s else self.slow

    def potential(self, X: torch.Tensor, Fv: torch.Tensor, branch: int,
                  context: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        X = dev(sym(X))
        p = self._flow_parameters(branch)
        modulus, tau = self.moduli[branch], self.tau_s[branch]
        eta = torch.clamp(3.0 * modulus * tau, min=1e-14)
        drive_scale = torch.clamp(modulus / self.equilibrium_shear, min=1e-14)
        s = 0.5 * torch.einsum("...ij,...ij->...", X, X)
        h = viscous_hencky_equivalent(Fv)
        hardening = (1.0 + h / float(p["h0"])).pow(float(p["c"]))
        r = 0.5 * (float(p["m"]) - 1.0)
        eps2 = (float(p["q0_MPa"]) * drive_scale).square()
        denominator = (drive_scale * float(p["s0_MPa"])).pow(2.0 * r)
        integral = ((3.0 * s + eps2).pow(r + 1.0) - eps2.pow(r + 1.0)) / (3.0 * (r + 1.0))
        nonlinear = float(p["nonlinear_multiplier"]) * hardening * integral / denominator
        return 1.5 * (s + nonlinear) / eta


class NeuralDissipationPICNN(DualDissipationPotential):
    """Shared PICNN convex in j=X:X/(2 sigma_ref^2), conditioned on state."""

    def __init__(self, tau_s, sigma_ref_MPa: float, context_dim: int = 5,
                 width: int = 16, strict_convexity: float = 1e-8,
                 *, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.register_buffer("tau_s", torch.as_tensor(tau_s, dtype=dtype, device=device))
        self.register_buffer("sigma_ref", torch.tensor(sigma_ref_MPa, dtype=dtype, device=device))
        self.encoder = nn.Sequential(nn.Linear(context_dim, width, dtype=dtype, device=device), nn.Softplus(),
                                     nn.Linear(width, width, dtype=dtype, device=device), nn.Softplus())
        self.coefficients = nn.Linear(width, 3, dtype=dtype, device=device)
        self.bias_head = nn.Linear(width, width, dtype=dtype, device=device)
        self.gain_head = nn.Linear(width, width, dtype=dtype, device=device)
        self.raw_rates = nn.Parameter(torch.zeros(width, dtype=dtype, device=device))
        self.strict_convexity = strict_convexity

    def context(self, Fv: torch.Tensor, branch: int) -> torch.Tensor:
        Cv = Fv.transpose(-1, -2) @ Fv
        I1 = torch.diagonal(Cv, dim1=-2, dim2=-1).sum(-1) - 3.0
        I2 = 0.5 * ((I1 + 3.0).square() - torch.einsum("...ij,...ji->...", Cv, Cv)) - 3.0
        h = viscous_hencky_equivalent(Fv)
        logtau = torch.log10(self.tau_s[branch]).expand_as(h)
        inverse_I1 = torch.diagonal(torch.linalg.inv(Cv), dim1=-2, dim2=-1).sum(-1) - 3.0
        return torch.stack((I1, I2, h, logtau, inverse_I1), -1)

    def potential(self, X: torch.Tensor, Fv: torch.Tensor, branch: int,
                  context: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        X = dev(sym(X))
        j = 0.5 * torch.einsum("...ij,...ij->...", X, X) / self.sigma_ref.square()
        encoded = self.encoder(self.context(Fv, branch))
        c = F.softplus(self.coefficients(encoded))
        b = self.bias_head(encoded)
        g = F.softplus(self.gain_head(encoded)) / b.shape[-1]
        rates = F.softplus(self.raw_rates) + 1e-8
        shifted = F.softplus(j[..., None] * rates + b) - F.softplus(b)
        dimensionless = c[..., 0] * j + 0.5 * c[..., 1] * j.square() + c[..., 2] * j.pow(3) / 3.0
        dimensionless = dimensionless + torch.sum(g * shifted, -1) + 0.5 * self.strict_convexity * j.square()
        # tau_ref=1 s is explicit in the dimensional interpretation, hence
        # Phi* has units MPa/s and dPhi*/dX has units 1/s.
        return self.sigma_ref * dimensionless


STAGE2_FEATURES = {
    "PICNN-A": (),
    "PICNN-B": ("elastic_I1_minus_3", "viscous_I1_minus_3", "hardening_variable"),
    "PICNN-C": ("total_I1_minus_3", "elastic_I1_minus_3",
                "viscous_I1_minus_3", "hardening_variable"),
}


class Stage2DissipationPICNN(DualDissipationPotential):
    """Stage-2 shared convex potential with nested objective context ablations."""

    def __init__(self, architecture: str, tau_s, branch_moduli_MPa, sigma_ref_MPa: float,
                 normalization: dict[str, tuple[float, float]], encoder_width: int = 24,
                 basis_width: int = 12, strict_convexity: float = 1e-8,
                 *, dtype=torch.float64, device="cpu"):
        super().__init__()
        if architecture not in STAGE2_FEATURES:
            raise ValueError(f"unknown stage-2 architecture: {architecture}")
        self.architecture = architecture
        self.feature_names = STAGE2_FEATURES[architecture]
        all_names = ("log_tau_over_1s", *self.feature_names)
        centers = [normalization[name][0] for name in all_names]
        scales = [max(normalization[name][1], 1e-8) for name in all_names]
        self.register_buffer("centers", torch.tensor(centers, dtype=dtype, device=device))
        self.register_buffer("scales", torch.tensor(scales, dtype=dtype, device=device))
        self.register_buffer("tau_s", torch.as_tensor(tau_s, dtype=dtype, device=device))
        self.register_buffer("sigma_ref", torch.tensor(sigma_ref_MPa, dtype=dtype, device=device))
        self.register_buffer("branch_scale", torch.as_tensor(
            branch_moduli_MPa, dtype=dtype, device=device) / self.sigma_ref)
        condition_dim = len(all_names)
        self.encoder = nn.Sequential(
            nn.Linear(condition_dim, encoder_width, dtype=dtype, device=device), nn.Softplus(),
            nn.Linear(encoder_width, encoder_width, dtype=dtype, device=device), nn.Softplus())
        self.coefficient_head = nn.Linear(encoder_width, 3, dtype=dtype, device=device)
        self.bias_head = nn.Linear(encoder_width, basis_width, dtype=dtype, device=device)
        self.gain_head = nn.Linear(encoder_width, basis_width, dtype=dtype, device=device)
        self.raw_rates = nn.Parameter(torch.zeros(basis_width, dtype=dtype, device=device))
        self.strict_convexity = strict_convexity
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5); nn.init.zeros_(module.bias)
        nn.init.normal_(self.coefficient_head.weight, std=0.03)
        nn.init.zeros_(self.coefficient_head.bias)
        nn.init.normal_(self.bias_head.weight, std=0.03)
        nn.init.uniform_(self.bias_head.bias, -3.0, 1.0)
        nn.init.normal_(self.gain_head.weight, std=0.02)
        nn.init.constant_(self.gain_head.bias, -3.0)

    @property
    def parameter_count(self) -> int:
        return sum(item.numel() for item in self.parameters() if item.requires_grad)

    def normalized_condition(self, branch: torch.Tensor, features: dict[str, torch.Tensor]):
        log_tau = torch.log10(self.tau_s[branch])
        values = [log_tau, *(features[name] for name in self.feature_names)]
        return (torch.stack(values, -1) - self.centers) / self.scales

    def potential_and_slope(self, j: torch.Tensor, branch: torch.Tensor,
                            features: dict[str, torch.Tensor]):
        encoded = self.encoder(self.normalized_condition(branch, features))
        coefficients = torch.exp(self.coefficient_head(encoded).clamp(-20.0, 25.0))
        bias = self.bias_head(encoded)
        gain = torch.exp(self.gain_head(encoded).clamp(-20.0, 18.0)) / bias.shape[-1]
        rates = F.softplus(self.raw_rates) + 1e-8
        scale = self.branch_scale[branch].clamp_min(1e-14)
        u = j / scale.square()
        arguments = u[..., None] * rates + bias
        shifted = F.softplus(arguments) - F.softplus(bias)
        dimensionless = scale * (coefficients[..., 0] * u +
                         0.5 * coefficients[..., 1] * u.square() +
                         coefficients[..., 2] * u.pow(3) / 3.0 +
                         torch.sum(gain * shifted, -1) +
                         0.5 * self.strict_convexity * u.square())
        slope = (coefficients[..., 0] + coefficients[..., 1] * u +
                 coefficients[..., 2] * u.square() +
                 torch.sum(gain * rates * torch.sigmoid(arguments), -1) +
                 self.strict_convexity * u) / scale
        return self.sigma_ref * dimensionless, slope

    def potential(self, X: torch.Tensor, Fv: torch.Tensor, branch: int,
                  context: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if context is None:
            raise ValueError("stage-2 PICNN requires objective constitutive context")
        X = dev(sym(X))
        j = 0.5 * torch.einsum("...ij,...ij->...", X, X) / self.sigma_ref.square()
        index = torch.as_tensor(branch, dtype=torch.long, device=X.device).expand(j.shape)
        return self.potential_and_slope(j, index, context)[0]


class Stage6GatedDissipationPotential(DualDissipationPotential):
    """Convex Stage-6 correction around a frozen Stage-4 D-PICNN.

    The frozen neural potential retains the high-confidence teacher shape.  A
    deterministic branch calibration enforces the prescribed local-linear
    relaxation spectrum.  Trainable fast/slow state gates act only away from
    the reference state, while positive quadratic/cubic terms provide a convex
    large-force correction whose first derivative vanishes at zero force.

    For fixed constitutive state the multiplier is positive and independent of
    ``X``.  Consequently a positive multiple of the frozen convex potential plus
    the positive polynomial correction remains convex in the driving force.
    """

    FEATURE_NAMES = ("total_I1_minus_3", "elastic_I1_minus_3",
                     "viscous_I1_minus_3", "hardening_variable")

    def __init__(self, base: Stage2DissipationPICNN, branch_moduli_MPa,
                 tau_s, transition_tau_s: float = 1.0,
                 *, state_scales=(1.0, 1.0, 1.0, 0.5),
                 initial_gate: float = 1e-3,
                 initial_nonlinear: float = 1e-8):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        dtype, device = self.base.sigma_ref.dtype, self.base.sigma_ref.device
        moduli = torch.as_tensor(branch_moduli_MPa, dtype=dtype, device=device)
        times = torch.as_tensor(tau_s, dtype=dtype, device=device)
        self.register_buffer("branch_moduli_MPa", moduli)
        self.register_buffer("tau_s", times)
        self.register_buffer("branch_scale", moduli/self.base.sigma_ref)
        self.register_buffer("branch_group", (times > float(transition_tau_s)).long())
        self.register_buffer("state_scales", torch.as_tensor(
            state_scales, dtype=dtype, device=device))
        self.register_buffer("target_reference_slope",
            self.base.sigma_ref/(2.0*moduli*times))

        zero_features = {name: torch.zeros(7, dtype=dtype, device=device)
                         for name in self.FEATURE_NAMES}
        branches = torch.arange(7, dtype=torch.long, device=device)
        with torch.no_grad():
            zero_j = torch.zeros(7, dtype=dtype, device=device)
            _, base_slope = self.base.potential_and_slope(
                zero_j, branches, zero_features)
            calibration = self.target_reference_slope/base_slope.clamp_min(1e-30)
        self.register_buffer("reference_calibration", calibration)

        def inverse_softplus(value: float):
            tensor = torch.tensor(float(value), dtype=dtype, device=device)
            return torch.log(torch.expm1(tensor).clamp_min(1e-30))

        self.raw_state_gate = nn.Parameter(torch.full(
            (2, len(self.FEATURE_NAMES)), float(inverse_softplus(initial_gate)),
            dtype=dtype, device=device))
        self.raw_nonlinear = nn.Parameter(torch.full(
            (2, 2), float(inverse_softplus(initial_nonlinear)),
            dtype=dtype, device=device))
        self.state_network = nn.Sequential(
            nn.Linear(1+len(self.FEATURE_NAMES), 12, dtype=dtype, device=device),
            nn.Tanh(),
            nn.Linear(12, 1, dtype=dtype, device=device))
        nn.init.xavier_uniform_(self.state_network[0].weight, gain=0.2)
        nn.init.zeros_(self.state_network[0].bias)
        nn.init.zeros_(self.state_network[2].weight)
        nn.init.zeros_(self.state_network[2].bias)

    @property
    def sigma_ref(self):
        return self.base.sigma_ref

    def _state_multiplier(self, branch: torch.Tensor,
                          features: dict[str, torch.Tensor]) -> torch.Tensor:
        group = self.branch_group[branch]
        values = torch.stack([features[name].clamp_min(0.0)
                              for name in self.FEATURE_NAMES], -1)
        normalized = values/self.state_scales
        gate = F.softplus(self.raw_state_gate)[group]
        logtau = torch.log10(self.tau_s[branch]).unsqueeze(-1)/3.0
        network_state = torch.cat((logtau, torch.log1p(normalized)), -1)
        reference_state = torch.cat((logtau, torch.zeros_like(normalized)), -1)
        neural_delta = self.state_network(network_state).squeeze(-1)-self.state_network(
            reference_state).squeeze(-1)
        neural_delta = 3.0*torch.tanh(neural_delta/3.0)
        exponent = -torch.sum(gate*normalized, -1)+neural_delta
        return self.reference_calibration[branch]*torch.exp(exponent.clamp(-30.0, 0.0))

    def potential_and_slope(self, j: torch.Tensor, branch: torch.Tensor,
                            features: dict[str, torch.Tensor]):
        base_phi, base_slope = self.base.potential_and_slope(j, branch, features)
        multiplier = self._state_multiplier(branch, features)
        group = self.branch_group[branch]
        coefficients = F.softplus(self.raw_nonlinear)[group]
        scale = self.branch_scale[branch].clamp_min(1e-14)
        u = j/scale.square()
        correction_dimensionless = scale*(
            0.5*coefficients[..., 0]*u.square() +
            coefficients[..., 1]*u.pow(3)/3.0)
        correction_slope = (coefficients[..., 0]*u +
                            coefficients[..., 1]*u.square())/scale
        phi = multiplier*base_phi + self.sigma_ref*correction_dimensionless
        slope = multiplier*base_slope + correction_slope
        return phi, slope

    def potential(self, X: torch.Tensor, Fv: torch.Tensor, branch: int,
                  context: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        if context is None:
            raise ValueError("stage-6 gated potential requires objective context")
        X = dev(sym(X))
        j = 0.5*torch.einsum("...ij,...ij->...", X, X)/self.sigma_ref.square()
        index = torch.as_tensor(branch, dtype=torch.long, device=X.device).expand(j.shape)
        return self.potential_and_slope(j, index, context)[0]

    def physical_parameters(self) -> dict:
        return {
            "family": "stage6_gated_frozen_dpicnn",
            "state_gate": F.softplus(self.raw_state_gate).detach().cpu().tolist(),
            "nonlinear_coefficients": F.softplus(self.raw_nonlinear).detach().cpu().tolist(),
            "state_network_parameter_norm": float(torch.sqrt(sum(
                parameter.detach().square().sum() for parameter in self.state_network.parameters())).cpu()),
            "reference_calibration": self.reference_calibration.detach().cpu().tolist(),
            "target_reference_slope": self.target_reference_slope.detach().cpu().tolist(),
        }
