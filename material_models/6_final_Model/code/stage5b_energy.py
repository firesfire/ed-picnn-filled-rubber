"""Stage-5B tangent--secant energy with frozen Stage-4 dissipation potential."""

from __future__ import annotations

from project_paths import load_config

import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from dual_potential.energies import TangentSecantEnergy
from stage3_real import (DECADE_EDGES, _coordinate, _strain_from_coordinate,
                         equilibrium_shear_modulus, equilibrium_stress)
from stage4_energy import load_stage4_checkpoint


def load_stage5b_material(config: dict, candidate: str, device="cpu",
                          *, synchronize_branch_scale: bool = True):
    stage4_config = load_config(Path(config["stage4_config"]).read_text(encoding="utf-8"))
    material, stage4_saved, parameters = load_stage4_checkpoint(
        config["stage4_checkpoint"], stage4_config, device)
    spectrum = json.loads(Path(config["stage5a_spectrum_file"]).read_text(encoding="utf-8"))
    checkpoint = torch.load(config["stage5a_candidate_checkpoints"][candidate],
                            map_location=device, weights_only=False)
    grouped = candidate == "hierarchical_fast_slow_energy_network"
    energy = TangentSecantEnergy(
        spectrum["weights_MPa"], spectrum["tau_s"],
        checkpoint["physical_parameters"]["groups"], grouped=grouped,
        b_bounds=tuple(config["energy"]["saturation_b_bounds"]),
        p_bounds=tuple(config["energy"]["saturation_power_bounds"]),
        transition_tau_s=config["energy"]["group_transition_tau_s"],
        dtype=torch.float64, device=device)
    material.nonequilibrium_energy = energy
    for parameter in material.dissipation_potential.parameters():
        parameter.requires_grad_(False)
    old_scale = material.dissipation_potential.branch_scale.detach().clone()
    if synchronize_branch_scale:
        with torch.no_grad():
            material.dissipation_potential.branch_scale.copy_(
                energy.effective_branch_moduli_MPa/material.dissipation_potential.sigma_ref)
    return material, {"stage4": stage4_saved, "stage5a": checkpoint,
                      "spectrum": spectrum, "old_branch_scale": old_scale,
                      "branch_scale_synchronized": synchronize_branch_scale}, parameters


def _branch_invariants_and_mandel(energy, gap, mode):
    if mode == "uniaxial":
        lam=torch.exp(gap)
        eigen_C=torch.stack((lam.square(),lam.reciprocal(),lam.reciprocal()),-1)
        I1=eigen_C.sum(-1);I2=(eigen_C[...,0]*eigen_C[...,1]+
                              eigen_C[...,0]*eigen_C[...,2]+eigen_C[...,1]*eigen_C[...,2])
        _,w1,w2=energy.invariant_quantities_all(I1-3.0,I2-3.0)
        principal_M=2.0*w1[...,None]*eigen_C+2.0*w2[...,None]*(I1[...,None]*eigen_C-eigen_C.square())
        X=principal_M-principal_M.mean(-1,keepdim=True)
        branch_stress=principal_M[...,0]-.5*(principal_M[...,1]+principal_M[...,2])
        return branch_stress,X[...,0],X.square().sum(-1),I1-3.0
    if mode == "shear":
        I1=3.0+gap.square();I2=I1
        _,w1,w2=energy.invariant_quantities_all(I1-3.0,I2-3.0)
        one=torch.ones_like(gap);zero=torch.zeros_like(gap)
        C=torch.stack((torch.stack((one,gap,zero),-1),
                       torch.stack((gap,one+gap.square(),zero),-1),
                       torch.stack((zero,zero,one),-1)),-2)
        C2=C@C
        M=2.0*w1[...,None,None]*C+2.0*w2[...,None,None]*(I1[...,None,None]*C-C2)
        identity=torch.eye(3,dtype=gap.dtype,device=gap.device)
        X=M-torch.diagonal(M,dim1=-2,dim2=-1).sum(-1)[...,None,None]*identity/3.0
        return 2.0*(w1+w2)*gap,X[...,0,1],X.square().sum((-2,-1)),I1-3.0
    raise ValueError(mode)


def reduced_response(material, strain, internal, mode):
    potential = material.dissipation_potential
    energy = material.nonequilibrium_energy
    coordinate = _coordinate(strain, mode)
    gap = coordinate[...,None]-internal
    branch_stress, force_component, force_norm2, elastic_i1 = _branch_invariants_and_mandel(
        energy, gap, mode)
    if mode == "uniaxial":
        branch_stress = branch_stress/(1.0+strain)[...,None]
        viscous = torch.exp(internal)
        viscous_i1 = viscous.square()+2.0/viscous-3.0
        hardening = internal.abs()
        velocity_multiplier = 1.0
    else:
        viscous_i1 = internal.square()
        hardening = 2.0/np.sqrt(3.0)*torch.asinh(internal/2.0).abs()
        velocity_multiplier = 2.0
    total_i1 = ((1.0+strain).square()+2.0/(1.0+strain)-3.0
                if mode == "uniaxial" else strain.square())
    branches = torch.arange(7, device=internal.device).expand(internal.shape)
    features = {"total_I1_minus_3": total_i1[...,None].expand_as(internal),
                "elastic_I1_minus_3": elastic_i1,
                "viscous_I1_minus_3": viscous_i1,
                "hardening_variable": hardening}
    j = .5*force_norm2/potential.sigma_ref.square()
    _, slope = potential.potential_and_slope(j, branches, features)
    velocity = velocity_multiplier*slope*force_component/potential.sigma_ref
    dissipation = slope*force_norm2/potential.sigma_ref
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
    moduli = material.nonequilibrium_energy.effective_branch_moduli_MPa
    fallback = 2.0*moduli*response["slope"]/material.dissipation_potential.sigma_ref
    rate = torch.where(resolved, response["coordinate_velocity"]/
                       torch.where(resolved,gap,torch.ones_like(gap)), fallback)
    return rate.clamp(1e-8,1e10)


def advance_reduced(material, internal, strain0, strain1, dt, mode):
    x0,x1=_coordinate(strain0,mode),_coordinate(strain1,mode)
    midpoint=.5*(x0+x1)
    initial=_effective_rate(material,midpoint,internal,mode)
    predicted=x0[...,None]-(x0[...,None]-internal)*torch.exp(-.5*dt[...,None]*initial)
    corrected=_effective_rate(material,midpoint,predicted,mode)
    rate=torch.sqrt(initial*corrected)
    z=(dt[...,None]*rate).clamp(1e-12,1e10)
    decay=torch.exp(-z);ramp=-torch.expm1(-z)/z
    elastic_gap=decay*(x0[...,None]-internal)+ramp*(x1-x0)[...,None]
    return x1[...,None]-elastic_gap


def relaxation_rollout(material, conditions, differentiable=True):
    if not conditions: return []
    indexed={id(item):None for item in conditions}
    for mode in ("uniaxial","shear"):
        group=[item for item in conditions if item.mode==mode]
        if not group: continue
        lengths=[len(item.time_s) for item in group];maximum=max(lengths)
        pad=lambda value,count:torch.cat((value,value[-1:].repeat(count-len(value))))
        time=torch.stack([pad(item.time_s,maximum) for item in group])
        strain=torch.stack([pad(item.strain,maximum) for item in group])
        internal=torch.zeros((len(group),7),dtype=time.dtype,device=time.device)
        stresses=[]; dissipations=[]; states=[]
        for index in range(maximum):
            if index:
                dt=(time[:,index]-time[:,index-1]).clamp_min(0.0)
                advanced=advance_reduced(material,internal,strain[:,index-1],strain[:,index],dt,mode)
                active=torch.tensor([index<length for length in lengths],dtype=torch.bool,device=time.device)
                internal=torch.where(active[:,None],advanced,internal)
            response=reduced_response(material,strain[:,index],internal,mode)
            stresses.append(response["stress"]);dissipations.append(response["dissipation"]);states.append(internal)
            if not differentiable: internal=internal.detach()
        stresses=torch.stack(stresses,1);dissipations=torch.stack(dissipations,1);states=torch.stack(states,1)
        for row,item in enumerate(group):
            count=lengths[row]
            indexed[id(item)]={"condition":item,"stress":stresses[row,:count],
                               "internal":states[row,:count],"dissipation":dissipations[row,:count]}
    return [indexed[id(item)] for item in conditions]


def relaxation_terms(predictions):
    q_losses=[];ordering=[];rneq=[];decade=[];total=[];loading=[];rows=[]
    by_mode={"uniaxial":[],"shear":[]}
    for item in predictions:
        c=item["condition"];peak=c.loading_end_index;eq=item["stress"].new_tensor(c.equilibrium_MPa)
        obs=c.stress_MPa;obs_neq=(obs[peak]-eq).clamp_min(1e-10);pred_neq=(item["stress"][peak]-eq).clamp_min(1e-10)
        q_obs=obs_neq/eq.abs().clamp_min(1e-10);q_pred=pred_neq/eq.abs().clamp_min(1e-10)
        q_losses.append(torch.log(q_pred/q_obs).square());by_mode[c.mode].append((float(c.strain[peak]),q_obs,q_pred))
        obs_shape=(obs[peak:]-eq)/obs_neq;pred_shape=(item["stress"][peak:]-eq)/pred_neq
        error=pred_shape-obs_shape;rneq.append(error.square().mean())
        elapsed=c.time_s[peak:]-c.time_s[peak];bins=[]
        for left,right in zip(DECADE_EDGES[:-1],DECADE_EDGES[1:]):
            mask=(elapsed>=left)&(elapsed<right)
            if bool(mask.any()):bins.append(error[mask].square().mean())
        decade.append(torch.stack(bins).mean())
        obs_R=obs[peak:]/obs[peak];pred_R=item["stress"][peak:]/item["stress"][peak]
        total.append((pred_R-obs_R).square().mean())
        loading.append((((item["stress"][:peak+1]-obs[:peak+1])/eq.abs().clamp_min(.05)).square().mean()))
        rows.append({"condition":c.key,"mode":c.mode,"split":c.split,"label":c.label,
                     "experimental_q":float(q_obs.detach()),"predicted_q":float(q_pred.detach()),
                     "Rneq_rmse":float(torch.sqrt(error.square().mean()).detach()),
                     "R_rmse":float(torch.sqrt((pred_R-obs_R).square().mean()).detach()),
                     "stress_rmse_MPa":float(torch.sqrt((item["stress"]-obs).square().mean()).detach()),
                     "minimum_dissipation":float(item["dissipation"].min().detach())})
    for values in by_mode.values():
        values=sorted(values,key=lambda value:value[0])
        for left,right in zip(values[:-1],values[1:]):
            gap=(right[1]-left[1]).detach().clamp_min(1e-8)
            ordering.append((torch.relu(.2*gap-(right[2]-left[2]))/gap).square())
    mean=lambda values:torch.stack(values).mean()
    return {"q_log":mean(q_losses),"q_ordering":mean(ordering) if ordering else mean(q_losses)*0,
            "Rneq":mean(rneq),"Rneq_decade":mean(decade),"R":mean(total),"loading":mean(loading),
            "rows":rows}


def dma_rollout(material, conditions, *, cycles=3, points_per_cycle=24,
                differentiable=True):
    if not conditions:return None
    dtype=material.tau_s.dtype;device=material.tau_s.device
    frequency=torch.tensor([item.frequency_hz for item in conditions],dtype=dtype,device=device)
    amplitude=torch.tensor([item.amplitude_fraction for item in conditions],dtype=dtype,device=device)
    dt=1.0/(frequency*points_per_cycle);internal=torch.zeros((len(conditions),7),dtype=dtype,device=device)
    phase_step=2.0*torch.pi/points_per_cycle;saved=[];minimum=torch.tensor(float("inf"),dtype=dtype,device=device)
    for step in range(1,cycles*points_per_cycle+1):
        s0=amplitude*torch.sin(torch.as_tensor((step-1)*phase_step,dtype=dtype,device=device))
        s1=amplitude*torch.sin(torch.as_tensor(step*phase_step,dtype=dtype,device=device))
        internal=advance_reduced(material,internal,s0,s1,dt,"shear")
        response=reduced_response(material,s1,internal,"shear");minimum=torch.minimum(minimum,response["dissipation"].min())
        if step>(cycles-2)*points_per_cycle:saved.append(response["stress"])
        if not differentiable:internal=internal.detach()
    values=torch.stack(saved,1);current=values[:,-points_per_cycle:];previous=values[:,:points_per_cycle]
    phase=phase_step*torch.arange(1,points_per_cycle+1,dtype=dtype,device=device)
    storage=2.0*torch.mean(current*torch.sin(phase),1)/amplitude
    loss=2.0*torch.mean(current*torch.cos(phase),1)/amplitude
    change=torch.amax(torch.abs(current-previous),1)/torch.clamp(torch.amax(torch.abs(current),1),min=.025)
    return {"storage":storage,"loss":loss,"cycle_change":change,"minimum_dissipation":minimum}


def dma_terms(material,conditions,prediction):
    dtype=prediction["storage"].dtype;device=prediction["storage"].device
    observed_storage=torch.tensor([x.storage_MPa for x in conditions],dtype=dtype,device=device)
    observed_loss=torch.tensor([x.loss_MPa for x in conditions],dtype=dtype,device=device)
    geq=equilibrium_shear_modulus(material.equilibrium_energy_parameters)
    storage_error=(prediction["storage"]-observed_storage)/(observed_storage-geq).abs().clamp_min(.10)
    loss_error=torch.log(prediction["loss"].clamp_min(1e-7)/observed_loss.clamp_min(1e-7))
    rows=[{"frequency_hz":item.frequency_hz,"split":item.split,
           "experimental_storage_MPa":item.storage_MPa,"predicted_storage_MPa":float(prediction["storage"][i].detach()),
           "experimental_loss_MPa":item.loss_MPa,"predicted_loss_MPa":float(prediction["loss"][i].detach()),
           "cycle_change":float(prediction["cycle_change"][i].detach())} for i,item in enumerate(conditions)]
    return {"storage":storage_error.square().mean(),"loss":loss_error.square().mean(),"rows":rows}


def objective(material,relaxation,frequency,split,weights,*,cycles=3,points_per_cycle=24,
              differentiable=True):
    r=[item for item in relaxation if item.split==split];f=[item for item in frequency if item.split==split]
    rpred=relaxation_rollout(material,r,differentiable);rt=relaxation_terms(rpred)
    fpred=dma_rollout(material,f,cycles=cycles,points_per_cycle=points_per_cycle,differentiable=differentiable)
    ft=dma_terms(material,f,fpred)
    total=(weights["q_log"]*rt["q_log"]+weights["q_ordering"]*rt["q_ordering"]+
           weights["Rneq"]*rt["Rneq"]+weights["Rneq_decade"]*rt["Rneq_decade"]+
           weights["R"]*rt["R"]+weights["loading"]*rt["loading"]+
           weights["storage"]*ft["storage"]+weights["loss"]*ft["loss"])
    return {"loss":total,"relaxation":rt,"frequency":ft,
            "relaxation_predictions":rpred,"frequency_prediction":fpred}


def snapshot_energy(energy):
    return {name:value.detach().cpu().clone() for name,value in energy.state_dict().items()}


def clone_material(material):
    return copy.deepcopy(material)
