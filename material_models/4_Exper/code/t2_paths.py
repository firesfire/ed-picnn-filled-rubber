"""Grouped Ogden flow on the uniaxial and simple-shear paths."""
from contextlib import contextmanager
import numpy as np
import torch
import observations as existing
import generate_final_model_creep_preload_dma as solver
from stage3_real import _coordinate, _strain_from_coordinate, equilibrium_stress

def t2_response(material, strain, internal, mode):
    """Analytic Ogden stress, Mandel norm and original grouped potential slope."""
    g = material.nonequilibrium_energy.branch_moduli_MPa
    a = material.nonequilibrium_energy.exponent
    gap = _coordinate(strain, mode)[..., None] - internal
    if mode == "uniaxial":
        e1, e2 = torch.exp(a*gap), torch.exp(-0.5*a*gap)
        diff = 2.0*g/a*(e1-e2)
        branch_stress = diff/(1.0+strain)[..., None]
        force_component = 2.0*diff/3.0
        norm2 = 2.0*diff.square()/3.0
        h = internal.abs()
        vmult = 1.0
        linear_factor = 3.0
    elif mode == "shear":
        y = torch.asinh(gap/2.0)
        den = torch.sinh(2.0*y)
        resolved = den.abs() > 1e-12
        ratio = torch.where(resolved, torch.sinh(a*y)/torch.where(resolved, den,
                            torch.ones_like(den)), a/2.0)
        branch_stress = 2.0*g/a*ratio*gap
        force_component = branch_stress
        principal = 2.0*g[..., None]/a*torch.stack((torch.exp(a*y),
                       torch.exp(-a*y), torch.ones_like(y)), -1)
        norm2 = (principal-principal.mean(-1, keepdim=True)).square().sum(-1)
        h = 2.0/np.sqrt(3.0)*torch.asinh(internal/2.0).abs()
        vmult = 2.0
        linear_factor = 2.0
    else:
        raise ValueError(mode)
    pot = material.dissipation_potential
    fast = material.tau_s <= pot.transition_tau_s
    def grouped(key):
        return torch.where(fast, g.new_tensor(pot.fast[key]), g.new_tensor(pot.slow[key]))
    scale = g/pot.equilibrium_shear
    hardening = (1.0+h/grouped("h0")).pow(grouped("c"))
    rho = torch.sqrt(1.5*norm2+(grouped("q0_MPa")*scale).square())
    multiplier = 1.0+grouped("nonlinear_multiplier")*hardening*(
        rho/(scale*grouped("s0_MPa"))).pow(grouped("m")-1.0)
    coefficient = multiplier/(2.0*g*material.tau_s)
    sigma = pot.sigma_ref
    eq = equilibrium_stress(strain, mode, material.equilibrium_energy_parameters)
    return {"stress": eq+branch_stress.sum(-1), "equilibrium": eq,
            "branch_stress": branch_stress,
            "coordinate_velocity": vmult*coefficient*force_component,
            "dissipation": coefficient*norm2, "slope": coefficient*sigma,
            "gap": gap, "j": 0.5*norm2/sigma.square(),
            "linear_rate": linear_factor*g*coefficient}

def t2_rate(material, coordinate, internal, mode):
    response = t2_response(material, _strain_from_coordinate(coordinate, mode), internal, mode)
    gap = response["gap"]
    resolved = gap.abs() > 1e-8
    rate = torch.where(resolved, response["coordinate_velocity"]/
        torch.where(resolved, gap, torch.ones_like(gap)), response["linear_rate"])
    return rate.clamp(1e-8, 1e10)

def t2_advance(material, internal, strain0, strain1, dt, mode):
    x0, x1 = _coordinate(strain0, mode), _coordinate(strain1, mode)
    mid = 0.5*(x0+x1)
    initial = t2_rate(material, mid, internal, mode)
    predicted = x0[..., None]-(x0[..., None]-internal)*torch.exp(-0.5*dt[..., None]*initial)
    corrected = t2_rate(material, mid, predicted, mode)
    z = (dt[..., None]*torch.sqrt(initial*corrected)).clamp(1e-12, 1e10)
    gap = torch.exp(-z)*(x0[..., None]-internal)-torch.expm1(-z)/z*(x1-x0)[..., None]
    return x1[..., None]-gap

@contextmanager
def t2_interface():
    prior = [(module, module.reduced_response, module.advance_reduced)
             for module in (existing, solver)]
    try:
        for module, _, _ in prior:
            module.reduced_response = t2_response
            module.advance_reduced = t2_advance
        yield
    finally:
        for module, response, advance in prior:
            module.reduced_response = response
            module.advance_reduced = advance
