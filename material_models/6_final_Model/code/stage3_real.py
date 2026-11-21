"""Reduced differentiable paths and losses for fixed-energy stage-3 fine-tuning.

The scalar charts retain exact one-term Ogden stresses and driving-force
invariants for uniaxial and simple-shear paths. They accelerate optimization;
the selected checkpoint is subsequently checked with the full-tensor local
implicit exponential-map solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


DECADE_EDGES = (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, float("inf"))


@dataclass
class RelaxationCondition:
    key: str
    mode: str
    label: str
    split: str
    time_s: torch.Tensor
    strain: torch.Tensor
    stress_MPa: torch.Tensor
    equilibrium_MPa: float
    loading_end_index: int


@dataclass
class DMACondition:
    frequency_hz: float
    amplitude_fraction: float
    storage_MPa: float
    loss_MPa: float
    split: str


def equilibrium_stress(strain: torch.Tensor, mode: str, p: dict) -> torch.Tensor:
    if mode == "uniaxial":
        lam = 1.0 + strain
        i1, i2 = lam.square() + 2.0/lam, 2.0*lam + lam.pow(-2)
        w1 = p["D"] * torch.exp(-p["mu"]*(i1-3.0))
        w2 = p["B"]*p["beta"]*(i2-3.0+p["N"]).pow(p["beta"]-1.0)
        return 2.0*w1*(lam-lam.pow(-2)) + 2.0*w2*(1.0-lam.pow(-3))
    if mode == "shear":
        inv = 3.0 + strain.square()
        w1 = p["D"] * torch.exp(-p["mu"]*(inv-3.0))
        w2 = p["B"]*p["beta"]*(inv-3.0+p["N"]).pow(p["beta"]-1.0)
        return 2.0*(w1+w2)*strain
    raise ValueError(mode)


def equilibrium_shear_modulus(p: dict) -> float:
    return float(2.0*(p["D"] + p["B"]*p["beta"]*p["N"]**(p["beta"]-1.0)))


def _downsample_curve(row, p, points_per_decade, maximum_loading_points,
                      dtype, device):
    time = np.asarray(row["time_s"], float)
    strain = np.asarray(row["strain"], float)
    stress = np.asarray(row["nominal_stress_MPa"], float)
    if np.any(np.diff(time) <= 0) or not np.all(np.isfinite(np.c_[time,strain,stress])):
        raise ValueError(f"invalid real history: {row['key']}")
    peak = int(np.argmax(strain))
    loading = np.unique(np.r_[np.linspace(0, peak, min(peak+1, maximum_loading_points), dtype=int), peak])
    elapsed = time[peak:] - time[peak]
    positive = elapsed[elapsed > 0]
    first = max(float(positive.min()), 1e-4)
    duration = float(elapsed[-1])
    decades = max(1.0, np.log10(duration/first))
    count = max(3, int(np.ceil(decades*points_per_decade))+1)
    hold = np.unique(np.r_[0.0, np.logspace(np.log10(first), np.log10(duration), count), duration])
    selected_time = np.r_[time[loading], time[peak]+hold[1:]]
    selected_strain = np.interp(selected_time, time, strain)
    selected_stress = np.interp(selected_time, time, stress)
    peak_new = len(loading)-1
    peak_strain = float(selected_strain[peak_new])
    prototype = torch.tensor(peak_strain, dtype=dtype, device=device)
    equilibrium = float(equilibrium_stress(prototype, row["mode"], p).detach().cpu())
    if selected_stress[peak_new] <= equilibrium:
        raise ValueError(f"nonpositive nonequilibrium peak: {row['key']}")
    tensor = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
    return RelaxationCondition(row["key"], row["mode"], row["label"], row["split"],
        tensor(selected_time), tensor(selected_strain), tensor(selected_stress),
        equilibrium, peak_new)


def load_real_data(path, parameters, *, points_per_decade=6,
                   maximum_loading_points=18, dtype=torch.float64, device="cpu"):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw["frequency_dma_amplitude_fraction"] != 0.001 or raw["amplitude_labels_loaded"]:
        raise ValueError("stage-3 training cache violates the frozen DMA boundary")
    relaxation = tuple(_downsample_curve(row, parameters.zhang_li, points_per_decade,
        maximum_loading_points, dtype, device) for row in raw["relaxation"])
    frequency = tuple(DMACondition(float(row["frequency_hz"]), float(row["amplitude_fraction"]),
        float(row["storage_MPa"]), float(row["loss_MPa"]), row["split"])
        for row in raw["frequency_dma"])
    return raw, relaxation, frequency


def _coordinate(strain, mode):
    return torch.log1p(strain) if mode == "uniaxial" else strain


def _strain_from_coordinate(value, mode):
    return torch.expm1(value) if mode == "uniaxial" else value


def reduced_response(material, strain, internal, mode):
    """Exact Ogden scalar-chart stress and driving-force invariants."""
    potential = material.dissipation_potential
    moduli = material.nonequilibrium_energy.branch_moduli_MPa
    a = material.nonequilibrium_energy.exponent
    sigma_ref = potential.sigma_ref
    branch_shape = internal.shape
    branches = torch.arange(7, device=internal.device).expand(branch_shape)
    total_i1 = ((1.0+strain).square()+2.0/(1.0+strain)-3.0
                if mode == "uniaxial" else strain.square())
    if mode == "uniaxial":
        x = torch.log1p(strain)[...,None]
        gap = x-internal
        elastic = torch.exp(gap); viscous = torch.exp(internal)
        delta = 2.0*moduli/a*(elastic.pow(a)-elastic.pow(-0.5*a))
        force_component = 2.0*delta/3.0
        force_norm2 = 2.0*delta.square()/3.0
        branch_stress = delta/(1.0+strain)[...,None]
        elastic_i1 = elastic.square()+2.0/elastic-3.0
        viscous_i1 = viscous.square()+2.0/viscous-3.0
        hardening = internal.abs()
        velocity_multiplier = 1.0
    elif mode == "shear":
        gap = strain[...,None]-internal
        r = torch.asinh(gap/2.0)
        principal = torch.stack((torch.exp(a*r), torch.exp(-a*r), torch.ones_like(r)), -1)
        mandel_principal = 2.0*moduli[...,None]/a[...,None]*principal if a.ndim else 2.0*moduli[...,None]/a*principal
        force_norm2 = torch.sum(mandel_principal.square(),-1)-torch.sum(mandel_principal,-1).square()/3.0
        branch_stress = 2.0*moduli/a*torch.sinh(a*r)/torch.cosh(r)
        force_component = branch_stress
        elastic_i1 = gap.square()
        viscous_i1 = internal.square()
        hardening = 2.0/np.sqrt(3.0)*torch.asinh(internal/2.0).abs()
        velocity_multiplier = 2.0
    else:
        raise ValueError(mode)
    features = {"total_I1_minus_3": total_i1[...,None].expand_as(internal),
                "elastic_I1_minus_3": elastic_i1,
                "viscous_I1_minus_3": viscous_i1,
                "hardening_variable": hardening}
    j = 0.5*force_norm2/sigma_ref.square()
    _, slope = potential.potential_and_slope(j, branches, features)
    velocity = velocity_multiplier*slope*force_component/sigma_ref
    dissipation = slope*force_norm2/sigma_ref
    eq = equilibrium_stress(strain, mode, material.equilibrium_energy_parameters)
    return {"stress": eq+branch_stress.sum(-1), "equilibrium": eq,
            "branch_stress": branch_stress, "coordinate_velocity": velocity,
            "dissipation": dissipation, "slope": slope, "gap": gap,
            "features": features, "j": j}


def _effective_rate(material, coordinate, internal, mode):
    strain = _strain_from_coordinate(coordinate, mode)
    response = reduced_response(material, strain, internal, mode)
    gap = coordinate[...,None]-internal
    resolved = gap.abs() > 1e-8
    moduli = material.nonequilibrium_energy.branch_moduli_MPa
    fallback = 2.0*moduli*response["slope"]/material.dissipation_potential.sigma_ref
    rate = torch.where(resolved, response["coordinate_velocity"] /
                       torch.where(resolved, gap, torch.ones_like(gap)), fallback)
    return rate.clamp(1e-8, 1e10)


def advance_reduced(material, internal, strain0, strain1, dt, mode):
    x0, x1 = _coordinate(strain0,mode), _coordinate(strain1,mode)
    midpoint = 0.5*(x0+x1)
    initial = _effective_rate(material, midpoint, internal, mode)
    predicted = x0[...,None]-(x0[...,None]-internal)*torch.exp(-0.5*dt[...,None]*initial)
    corrected = _effective_rate(material, midpoint, predicted, mode)
    rate = torch.sqrt(initial*corrected)
    z = (dt[...,None]*rate).clamp(1e-12, 1e10)
    decay = torch.exp(-z); ramp = -torch.expm1(-z)/z
    elastic_gap = decay*(x0[...,None]-internal)+ramp*(x1-x0)[...,None]
    return x1[...,None]-elastic_gap


def relaxation_rollout(material, conditions, differentiable=True):
    if not conditions: return []
    indexed={id(condition):None for condition in conditions}
    for mode in ("uniaxial","shear"):
        group=[condition for condition in conditions if condition.mode==mode]
        if not group:continue
        lengths=[len(c.time_s) for c in group];maximum=max(lengths)
        pad=lambda value,count:torch.cat((value,value[-1:].repeat(count-len(value))))
        time=torch.stack([pad(c.time_s,maximum) for c in group])
        strain=torch.stack([pad(c.strain,maximum) for c in group])
        internal=torch.zeros((len(group),7),dtype=time.dtype,device=time.device)
        stresses=[];dissipations=[];states=[]
        for index in range(maximum):
            if index:
                dt=(time[:,index]-time[:,index-1]).clamp_min(0.0)
                advanced=advance_reduced(material,internal,strain[:,index-1],strain[:,index],dt,mode)
                active=torch.tensor([index<length for length in lengths],dtype=torch.bool,device=time.device)
                internal=torch.where(active[:,None],advanced,internal)
            response=reduced_response(material,strain[:,index],internal,mode)
            stresses.append(response["stress"]);dissipations.append(response["dissipation"]);states.append(internal)
            if not differentiable:internal=internal.detach()
        stresses=torch.stack(stresses,1);dissipations=torch.stack(dissipations,1);states=torch.stack(states,1)
        for row,c in enumerate(group):
            count=lengths[row]
            indexed[id(c)]={"condition":c,"stress":stresses[row,:count],
                "internal":states[row,:count],"dissipation":dissipations[row,:count]}
    return [indexed[id(condition)] for condition in conditions]


def relaxation_loss(predictions):
    q_losses=[]; shape_losses=[]; decade_losses=[]; total_losses=[]; loading_losses=[]
    rows=[]; q_by_mode={"uniaxial":[],"shear":[]}
    for item in predictions:
        c=item["condition"]; peak=c.loading_end_index
        eq=item["stress"].new_tensor(c.equilibrium_MPa); obs=c.stress_MPa
        obs_neq=(obs[peak]-eq).clamp_min(1e-10); pred_neq=(item["stress"][peak]-eq).clamp_min(1e-10)
        q_obs=obs_neq/eq.abs().clamp_min(1e-10); q_pred=pred_neq/eq.abs().clamp_min(1e-10)
        q_losses.append(torch.log(q_pred/q_obs).square())
        q_by_mode[c.mode].append((q_obs,q_pred,c.label))
        obs_shape=(obs[peak:]-eq)/obs_neq; pred_shape=(item["stress"][peak:]-eq)/pred_neq
        err=pred_shape-obs_shape; shape_losses.append(err.square().mean())
        elapsed=c.time_s[peak:]-c.time_s[peak]; bins=[]
        for left,right in zip(DECADE_EDGES[:-1],DECADE_EDGES[1:]):
            mask=(elapsed>=left)&(elapsed<right)
            if bool(mask.any()): bins.append(err[mask].square().mean())
        decade_losses.append(torch.stack(bins).mean())
        obs_total=obs[peak:]/obs[peak]; pred_total=item["stress"][peak:]/item["stress"][peak]
        total_losses.append((pred_total-obs_total).square().mean())
        loading_losses.append(((item["stress"][:peak+1]-obs[:peak+1])/
                               eq.abs().clamp_min(0.05)).square().mean())
        rows.append({"condition":c.key,"split":c.split,"mode":c.mode,"label":c.label,
            "experimental_q":float(q_obs.detach()),"predicted_q":float(q_pred.detach()),
            "Rneq_rmse":float(torch.sqrt(err.square().mean()).detach()),
            "R_rmse":float(torch.sqrt((pred_total-obs_total).square().mean()).detach()),
            "stress_rmse_MPa":float(torch.sqrt((item["stress"]-obs).square().mean()).detach()),
            "minimum_dissipation":float(item["dissipation"].min().detach())})
    ordering=[]
    for values in q_by_mode.values():
        values=sorted(values,key=lambda x:float(x[0].detach()))
        for left,right in zip(values[:-1],values[1:]):
            target_gap=(right[0]-left[0]).detach()
            ordering.append(torch.relu(0.2*target_gap-(right[1]-left[1])).square()/target_gap.square().clamp_min(1e-8))
    mean=lambda x:torch.stack(x).mean()
    terms={"q_log":mean(q_losses),"q_ordering":mean(ordering) if ordering else mean(q_losses)*0,
           "Rneq":mean(shape_losses),"Rneq_decade":mean(decade_losses),
           "R":mean(total_losses),"loading":mean(loading_losses)}
    loss=(.25*terms["q_log"]+.10*terms["q_ordering"]+.25*terms["Rneq"]+
          .25*terms["Rneq_decade"]+.10*terms["R"]+.05*terms["loading"])
    return {"loss":loss,"terms":terms,"rows":rows}


def dma_rollout(material, conditions, *, cycles=3, points_per_cycle=24,
                differentiable=True):
    if not conditions: return None
    device=material.tau_s.device; dtype=material.tau_s.dtype
    frequency=torch.tensor([x.frequency_hz for x in conditions],dtype=dtype,device=device)
    amplitude=torch.tensor([x.amplitude_fraction for x in conditions],dtype=dtype,device=device)
    dt=1.0/(frequency*points_per_cycle); internal=torch.zeros((len(conditions),7),dtype=dtype,device=device)
    phase_step=2.0*torch.pi/points_per_cycle; saved=[]; min_diss=torch.tensor(float("inf"),dtype=dtype,device=device)
    for step in range(1,cycles*points_per_cycle+1):
        s0=amplitude*torch.sin(torch.as_tensor((step-1)*phase_step,dtype=dtype,device=device))
        s1=amplitude*torch.sin(torch.as_tensor(step*phase_step,dtype=dtype,device=device))
        internal=advance_reduced(material,internal,s0,s1,dt,"shear")
        response=reduced_response(material,s1,internal,"shear")
        min_diss=torch.minimum(min_diss,response["dissipation"].min())
        if step>(cycles-2)*points_per_cycle:saved.append(response["stress"])
        if not differentiable:internal=internal.detach()
    values=torch.stack(saved,1); current=values[:,-points_per_cycle:]; previous=values[:,:points_per_cycle]
    phase=phase_step*torch.arange(1,points_per_cycle+1,dtype=dtype,device=device)
    storage=2.0*torch.mean(current*torch.sin(phase),1)/amplitude
    loss=2.0*torch.mean(current*torch.cos(phase),1)/amplitude
    change=torch.amax(torch.abs(current-previous),1)/torch.clamp(torch.amax(torch.abs(current),1),min=.025)
    third_s=2.0*torch.mean(current*torch.sin(3*phase),1)/amplitude
    third_c=2.0*torch.mean(current*torch.cos(3*phase),1)/amplitude
    third=torch.sqrt(third_s.square()+third_c.square())/torch.sqrt(storage.square()+loss.square()).clamp_min(1e-12)
    return {"storage":storage,"loss":loss,"cycle_change":change,"third_harmonic":third,
            "minimum_dissipation":min_diss}


def dma_loss(material, conditions, prediction):
    device=prediction["storage"].device; dtype=prediction["storage"].dtype
    exp_storage=torch.tensor([x.storage_MPa for x in conditions],dtype=dtype,device=device)
    exp_loss=torch.tensor([x.loss_MPa for x in conditions],dtype=dtype,device=device)
    geq=equilibrium_shear_modulus(material.equilibrium_energy_parameters)
    storage_scale=(exp_storage-geq).abs().clamp_min(.10)
    storage_error=(prediction["storage"]-exp_storage)/storage_scale
    loss_error=torch.log(prediction["loss"].clamp_min(1e-7))-torch.log(exp_loss.clamp_min(1e-7))
    total=.5*storage_error.square().mean()+.5*loss_error.square().mean()
    rows=[]
    for i,c in enumerate(conditions):
        rows.append({"frequency_hz":c.frequency_hz,"split":c.split,
          "experimental_storage_MPa":c.storage_MPa,"predicted_storage_MPa":float(prediction["storage"][i].detach()),
          "experimental_loss_MPa":c.loss_MPa,"predicted_loss_MPa":float(prediction["loss"][i].detach()),
          "storage_relative_error":float(((prediction["storage"][i]-exp_storage[i])/exp_storage[i]).detach()),
          "loss_relative_error":float(((prediction["loss"][i]-exp_loss[i])/exp_loss[i]).detach()),
          "cycle_change":float(prediction["cycle_change"][i].detach()),"third_harmonic":float(prediction["third_harmonic"][i].detach())})
    return {"loss":total,"storage_rmse_scaled":torch.sqrt(storage_error.square().mean()),
            "loss_log_rmse":torch.sqrt(loss_error.square().mean()),"rows":rows}


def real_objective(material, relaxation, frequency, split, *, cycles=3,
                   points_per_cycle=24, differentiable=True):
    r=[x for x in relaxation if x.split==split]; f=[x for x in frequency if x.split==split]
    rpred=relaxation_rollout(material,r,differentiable); rloss=relaxation_loss(rpred)
    fpred=dma_rollout(material,f,cycles=cycles,points_per_cycle=points_per_cycle,differentiable=differentiable)
    floss=dma_loss(material,f,fpred)
    return {"loss":.60*rloss["loss"]+.40*floss["loss"],"relaxation":rloss,
            "frequency":floss,"relaxation_predictions":rpred,"frequency_prediction":fpred}
