"""Low-parameter nonequilibrium-energy release for Chapter-3 Stage 4."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from dual_potential.constitutive import MaterialParameters
from dual_potential.energies import LowParameterOgdenEnergy, GroupedLowParameterOgdenEnergy
from stage2_rollout import load_stage2_material
from stage2_training import predict, weighted_mean
from stage3_real import real_objective


def build_stage4_material(config: dict, device="cpu", *, grouped=False,
                          physical: dict | None = None):
    material, saved = load_stage2_material(config["stage3_checkpoint"],
        config["teacher_parameter_file"], device)
    parameters = MaterialParameters.from_json(config["teacher_parameter_file"])
    if physical is None:
        physical = {"Gneq_MPa":parameters.Gneq_MPa,
                    "shares":list(parameters.shares),
                    "exponent":parameters.a_ogden,
                    "fast_exponent":parameters.a_ogden,
                    "slow_exponent":parameters.a_ogden}
    bounds=tuple(config["parameter_bounds"]["ogden_exponent"])
    if grouped:
        energy=GroupedLowParameterOgdenEnergy(physical["Gneq_MPa"],physical["shares"],
            physical.get("fast_exponent",physical.get("exponent",parameters.a_ogden)),
            physical.get("slow_exponent",physical.get("exponent",parameters.a_ogden)),
            parameters.tau_s,config["parameter_bounds"]["grouped_transition_tau_s"],
            bounds,dtype=torch.float64,device=device)
    else:
        energy=LowParameterOgdenEnergy(physical["Gneq_MPa"],physical["shares"],
            physical.get("exponent",parameters.a_ogden),bounds,
            dtype=torch.float64,device=device)
    material.nonequilibrium_energy=energy
    material.equilibrium_energy_parameters=parameters.zhang_li
    sync_branch_scale(material)
    return material,saved,parameters


def physical_parameters(material) -> dict:
    energy=material.nonequilibrium_energy
    moduli=energy.branch_moduli_MPa
    shares=moduli/moduli.sum()
    result={"Gneq_MPa":float(moduli.sum().detach()),
            "shares":[float(x) for x in shares.detach().cpu()],
            "branch_moduli_MPa":[float(x) for x in moduli.detach().cpu()]}
    exponent=energy.exponent
    if exponent.ndim==0:
        result["energy_family"]="shared_one_term_ogden"
        result["exponent"]=float(exponent.detach())
    else:
        result["energy_family"]="grouped_fast_slow_one_term_ogden"
        result["fast_exponent"]=float(energy.fast_exponent.detach())
        result["slow_exponent"]=float(energy.slow_exponent.detach())
        result["branch_exponents"]=[float(x) for x in exponent.detach().cpu()]
    return result


@torch.no_grad()
def sync_branch_scale(material):
    """Keep normalized force coordinates tied to the released branch moduli."""
    scale=material.nonequilibrium_energy.branch_moduli_MPa/material.dissipation_potential.sigma_ref
    material.dissipation_potential.branch_scale.copy_(scale.detach())


def scaled_anchor_targets(base_material, data):
    return predict(base_material.dissipation_potential,data)[1].detach()


def scaled_anchor_loss(material,data,target_D,index,base_moduli):
    current_moduli=material.nonequilibrium_energy.branch_moduli_MPa
    ratio=current_moduli/base_moduli
    branch=data.branch[index]
    X=data.X[index]*ratio[branch,None,None]
    j=.5*torch.einsum("...ij,...ij->...",X,X)/material.dissipation_potential.sigma_ref.square()
    features={name:value[index] for name,value in data.features.items()}
    _,slope=material.dissipation_potential.potential_and_slope(j,branch,features)
    D=slope[...,None,None]*X/material.dissipation_potential.sigma_ref
    target=target_D[index]
    norm2=torch.einsum("...ij,...ij->...",X,X).clamp_min(1e-30)
    target_slope=material.dissipation_potential.sigma_ref*torch.einsum("...ij,...ij->...",X,target)/norm2
    log=torch.mean((torch.log(slope.clamp_min(1e-16))-torch.log(target_slope.clamp_min(1e-16))).square())
    tau=data.tau[index,None,None]
    transformed=torch.mean((torch.asinh(tau*D/.02)-torch.asinh(tau*target/.02)).square())
    return log+.5*transformed


@torch.no_grad()
def scaled_teacher_nrmse(material,data,target_D,base_moduli):
    current_moduli=material.nonequilibrium_energy.branch_moduli_MPa
    ratio=current_moduli/base_moduli
    X=data.X*ratio[data.branch,None,None]
    j=.5*torch.einsum("...ij,...ij->...",X,X)/material.dissipation_potential.sigma_ref.square()
    _,slope=material.dissipation_potential.potential_and_slope(j,data.branch,data.features)
    D=slope[...,None,None]*X/material.dissipation_potential.sigma_ref
    error=torch.linalg.matrix_norm(data.tau[:,None,None]*(D-target_D))
    target=torch.linalg.matrix_norm(data.tau[:,None,None]*target_D)
    return float(torch.sqrt(weighted_mean(error.square(),data.weight))/
                 torch.sqrt(weighted_mean(target.square(),data.weight)).clamp_min(1e-16))


def platform_terms(predictions):
    errors=[];ordering=[];rows=[]
    by_mode={"uniaxial":[],"shear":[]}
    for item in predictions:
        c=item["condition"];peak=c.loading_end_index
        observed=float(c.stress_MPa[-1]/c.stress_MPa[peak])
        predicted=item["stress"][-1]/item["stress"][peak]
        errors.append((predicted-predicted.new_tensor(observed)).square())
        level=c.strain[peak]
        by_mode[c.mode].append((level,predicted,predicted.new_tensor(observed),c.key))
        rows.append({"condition":c.key,"experimental_platform_R":observed,
                     "predicted_platform_R":float(predicted.detach())})
    for values in by_mode.values():
        values=sorted(values,key=lambda x:float(x[0].detach()))
        for left,right in zip(values[:-1],values[1:]):
            observed_gap=(left[2]-right[2]).clamp_min(1e-6)
            predicted_gap=left[1]-right[1]
            ordering.append(torch.relu(.2*observed_gap-predicted_gap).square()/observed_gap.square())
    zero=predictions[0]["stress"].new_tensor(0.0)
    return {"mse":torch.stack(errors).mean() if errors else zero,
            "ordering":torch.stack(ordering).mean() if ordering else zero,"rows":rows}


def regularization(material,parameters,config):
    weights=config["regularization"];energy=material.nonequilibrium_energy
    moduli=energy.branch_moduli_MPa;shares=moduli/moduli.sum()
    base_shares=torch.tensor(parameters.shares,dtype=shares.dtype,device=shares.device)
    base_shares=base_shares/base_shares.sum()
    logtau=torch.log10(material.tau_s)
    centroid=torch.sum(shares*logtau);base_centroid=torch.sum(base_shares*logtau)
    variance=torch.sum(shares*(logtau-centroid).square())
    base_variance=torch.sum(base_shares*(logtau-base_centroid).square())
    exponent=energy.exponent
    exponent_log=torch.mean(torch.log(exponent/parameters.a_ogden).square())
    pieces={
      "total_modulus_log":torch.log(moduli.sum()/parameters.Gneq_MPa).square(),
      "shares_hellinger":torch.sum((torch.sqrt(shares.clamp_min(1e-16))-torch.sqrt(base_shares.clamp_min(1e-16))).square()),
      "log_time_centroid":(centroid-base_centroid).square(),
      "log_time_variance":(variance-base_variance).square(),
      "exponent_log":exponent_log}
    if hasattr(energy,"fast_exponent"):
        pieces["grouped_exponent_separation"]=torch.log(energy.fast_exponent/energy.slow_exponent).square()
    total=sum(weights.get(name,0.0)*value for name,value in pieces.items())
    return total,pieces


def stage4_objective(material,relaxation,frequency,split,phase,parameters,config,
                     *,anchor_data=None,anchor_target=None,base_moduli=None,anchor_index=None):
    opt=config["optimization"]
    real=real_objective(material,relaxation,frequency,split,
        cycles=opt["training_cycles"],points_per_cycle=opt["training_points_per_cycle"],
        differentiable=phase!="evaluation")
    platform=platform_terms(real["relaxation_predictions"])
    terms=real["relaxation"]["terms"]
    storage2=real["frequency"]["storage_rmse_scaled"].square()
    losslog2=real["frequency"]["loss_log_rmse"].square()
    shape_limit=float(config["regularization"]["Rneq_guard_rmse"])
    shape_scale=shape_limit**2
    shape_normalized=.5*(terms["Rneq"]+terms["Rneq_decade"])/shape_scale
    shape_guard=torch.relu(terms["Rneq"]/shape_scale-1.0).square()
    reg,reg_terms=regularization(material,parameters,config)
    if phase=="energy":
        loss=(.35*terms["q_log"]+.10*terms["q_ordering"]+.25*storage2+
              .05*losslog2+.05*terms["Rneq"]+.05*terms["R"]+
              .10*platform["mse"]+.05*platform["ordering"]+reg+
              config["regularization"]["Rneq_guard_weight"]*shape_guard)
        anchor=loss.new_tensor(0.0)
    elif phase=="dissipation":
        anchor=scaled_anchor_loss(material,anchor_data,anchor_target,anchor_index,base_moduli)
        loss=.50*real["loss"]+.50*shape_normalized+opt["anchor_weight"]*anchor
    elif phase=="joint":
        anchor=scaled_anchor_loss(material,anchor_data,anchor_target,anchor_index,base_moduli)
        focus=.35*terms["q_log"]+.30*storage2+.15*terms["Rneq"]+.10*terms["R"]+.10*platform["mse"]
        loss=(.45*real["loss"]+.20*focus+.35*shape_normalized+
              opt["anchor_weight"]*anchor+reg+
              config["regularization"]["Rneq_guard_weight"]*shape_guard)
    elif phase=="evaluation":
        anchor=real["loss"].new_tensor(0.0);loss=real["loss"]
    else:raise ValueError(phase)
    return {"loss":loss,"real":real,"platform":platform,"regularization":reg,
            "regularization_terms":reg_terms,"anchor":anchor}


def summary_metrics(value):
    real=value["real"];rows=real["relaxation"]["rows"]
    q_obs=np.array([row["experimental_q"] for row in rows]);q_pred=np.array([row["predicted_q"] for row in rows])
    q_order=[]
    for mode in ("uniaxial","shear"):
        selected=[row for row in rows if row["mode"]==mode]
        selected=sorted(selected,key=lambda row:row["experimental_q"])
        q_order += [right["predicted_q"]>left["predicted_q"] for left,right in zip(selected[:-1],selected[1:])]
    platforms=value["platform"]["rows"];platform_order=[]
    for mode in ("uniaxial","shear"):
        selected=[row for row in platforms if row["condition"].startswith(mode)]
        selected=sorted(selected,key=lambda row:float(row["condition"].split(":")[1].replace("%","")))
        platform_order += [right["predicted_platform_R"]<left["predicted_platform_R"] for left,right in zip(selected[:-1],selected[1:])]
    return {"real_objective":float(real["loss"].detach()),
      "q_log_rmse":float(np.sqrt(np.mean(np.log(np.maximum(q_pred,1e-16)/np.maximum(q_obs,1e-16))**2))),
      "q_ordering_accuracy":float(np.mean(q_order)) if q_order else math.nan,
      "Rneq_rmse":float(torch.sqrt(real["relaxation"]["terms"]["Rneq"]).detach()),
      "R_rmse":float(torch.sqrt(real["relaxation"]["terms"]["R"]).detach()),
      "platform_R_rmse":float(torch.sqrt(value["platform"]["mse"]).detach()),
      "platform_ordering_accuracy":float(np.mean(platform_order)) if platform_order else math.nan,
      "frequency_storage_rmse_scaled":float(real["frequency"]["storage_rmse_scaled"].detach()),
      "frequency_loss_log_rmse":float(real["frequency"]["loss_log_rmse"].detach()),
      "minimum_dissipation":float(real["frequency_prediction"]["minimum_dissipation"].detach()),
      "relaxation_rows":rows,"platform_rows":platforms,
      "frequency_rows":real["frequency"]["rows"]}


def clamp_energy(material,config):
    energy=material.nonequilibrium_energy
    low,high=config["parameter_bounds"]["Gneq_MPa"]
    with torch.no_grad():
        energy.raw_total.clamp_(math.log(math.expm1(low)),math.log(math.expm1(high)))


def snapshot(material):
    return {"dissipation":{k:v.detach().cpu().clone() for k,v in material.dissipation_potential.state_dict().items()},
            "energy":{k:v.detach().cpu().clone() for k,v in material.nonequilibrium_energy.state_dict().items()},
            "physical":physical_parameters(material)}


def restore(material,state):
    material.dissipation_potential.load_state_dict(state["dissipation"])
    material.nonequilibrium_energy.load_state_dict(state["energy"])
    sync_branch_scale(material)


def load_stage4_checkpoint(checkpoint_path,config,device="cpu"):
    saved=torch.load(checkpoint_path,map_location=device,weights_only=False)
    material,_,parameters=build_stage4_material(config,device,
        grouped=saved["energy_family"]=="grouped_fast_slow_one_term_ogden",
        physical=saved["physical_parameters"])
    material.dissipation_potential.load_state_dict(saved["dissipation_state_dict"])
    material.nonequilibrium_energy.load_state_dict(saved["energy_state_dict"])
    sync_branch_scale(material);material.eval()
    return material,saved,parameters
