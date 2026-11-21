"""Stage-6 ED-PICNN construction, losses and block-coordinate utilities."""

from __future__ import annotations

from project_paths import load_config

import copy
from pathlib import Path

import numpy as np
import torch

from dual_potential.dissipation import Stage6GatedDissipationPotential
from stage2_training import predict, weighted_mean
from stage5b_energy import (dma_rollout, dma_terms, load_stage5b_material,
                            objective, relaxation_rollout, relaxation_terms)


def load_stage6_material(config: dict, device="cpu"):
    stage5b_config = load_config(
        Path(config["stage5b_config"]).read_text(encoding="utf-8"))
    material, meta, parameters = load_stage5b_material(
        stage5b_config, "hierarchical_fast_slow_energy_network", device,
        synchronize_branch_scale=True)
    saved = torch.load(config["stage5b_energy_checkpoint"], map_location=device,
                       weights_only=False)
    material.nonequilibrium_energy.load_state_dict(saved["energy_state_dict"], strict=False)
    moduli = material.nonequilibrium_energy.effective_branch_moduli_MPa.detach()
    base = material.dissipation_potential
    gate = config["dissipation_gate"]
    material.dissipation_potential = Stage6GatedDissipationPotential(
        base, moduli, material.tau_s,
        transition_tau_s=gate["transition_tau_s"],
        state_scales=gate["state_scales"],
        initial_gate=gate["initial_gate"],
        initial_nonlinear=gate["initial_nonlinear"])
    synchronize_dissipation_spectrum(material)
    return material, {**meta, "stage5b": saved}, parameters


@torch.no_grad()
def synchronize_dissipation_spectrum(material):
    """Synchronize deterministic D normalization after a bounded share update."""
    potential = material.dissipation_potential
    moduli = material.nonequilibrium_energy.effective_branch_moduli_MPa.detach()
    potential.branch_moduli_MPa.copy_(moduli)
    potential.branch_scale.copy_(moduli/potential.sigma_ref)
    if hasattr(potential.base, "branch_scale"):
        potential.base.branch_scale.copy_(moduli/potential.sigma_ref)
    potential.target_reference_slope.copy_(
        potential.sigma_ref/(2.0*moduli*potential.tau_s))
    zero = torch.zeros(7, dtype=moduli.dtype, device=moduli.device)
    branches = torch.arange(7, dtype=torch.long, device=moduli.device)
    features = {name: zero for name in potential.FEATURE_NAMES}
    _, base_slope = potential.base.potential_and_slope(zero, branches, features)
    potential.reference_calibration.copy_(
        potential.target_reference_slope/base_slope.clamp_min(1e-30))


def ordering_accuracy(rows):
    correct = []
    for mode in ("uniaxial", "shear"):
        values = sorted((row for row in rows if row["mode"] == mode),
                        key=lambda row: float(row["label"].replace("%", "")))
        for left, right in zip(values[:-1], values[1:]):
            correct.append(right["predicted_q"] > left["predicted_q"])
    return float(np.mean(correct)) if correct else 1.0


def summary_metrics(value):
    r, f = value["relaxation"], value["frequency"]
    rows = r["rows"]
    return {
        "objective": float(value["loss"].detach()),
        "q_log_RMSE": float(torch.sqrt(r["q_log"]).detach()),
        "q_ordering_accuracy": ordering_accuracy(rows),
        "Rneq_RMSE": float(torch.sqrt(r["Rneq"]).detach()),
        "Rneq_decade_RMSE": float(torch.sqrt(r["Rneq_decade"]).detach()),
        "R_RMSE": float(torch.sqrt(r["R"]).detach()),
        "loading_RMSE_scaled": float(torch.sqrt(r["loading"]).detach()),
        "storage_RMSE_scaled": float(torch.sqrt(f["storage"]).detach()),
        "loss_log_RMSE": float(torch.sqrt(f["loss"]).detach()),
        "maximum_cycle_change": float(value["frequency_prediction"]["cycle_change"].max().detach()),
        "minimum_dissipation": float(min(row["minimum_dissipation"] for row in rows)),
    }


def evaluate(material, relaxation, frequency, split, weights, config):
    synchronize_dissipation_spectrum(material)
    with torch.enable_grad():
        value = objective(material, relaxation, frequency, split, weights,
            cycles=config["optimization"]["training_cycles"],
            points_per_cycle=config["optimization"]["training_points_per_cycle"],
            differentiable=False)
    return value, summary_metrics(value)


def differentiable_objective(material, relaxation, frequency, weights, config):
    synchronize_dissipation_spectrum(material)
    return objective(material, relaxation, frequency, "training", weights,
        cycles=config["optimization"]["training_cycles"],
        points_per_cycle=config["optimization"]["training_points_per_cycle"],
        differentiable=True)


def capture_teacher_anchor(potential, datasets):
    """Capture the calibrated high/medium D response before real-data release."""
    result = {}
    with torch.no_grad():
        for name, data in datasets.items():
            slope, D = predict(potential, data)
            result[name] = {"slope": slope.detach().clone(), "D": D.detach().clone()}
    return result


def teacher_anchor_loss(potential, data, targets, index):
    slope, D = predict(potential, data, index)
    target_slope, target_D = targets["slope"][index], targets["D"][index]
    weight = data.weight[index]
    log_error = torch.log(slope.clamp_min(1e-16))-torch.log(target_slope.clamp_min(1e-16))
    log_loss = weighted_mean(log_error.square(), weight)
    scaled = torch.asinh(data.tau[index, None, None]*D/0.02)
    target_scaled = torch.asinh(data.tau[index, None, None]*target_D/0.02)
    tensor = weighted_mean(torch.sum((scaled-target_scaled).square(), (-2, -1))/3.0, weight)
    return log_loss+0.5*tensor


@torch.no_grad()
def teacher_anchor_nrmse(potential, data, targets, mask=None):
    if mask is None:
        mask = torch.ones(data.count, dtype=torch.bool, device=data.X.device)
    index = torch.where(mask)[0]
    _, D = predict(potential, data, index)
    target_D = targets["D"][index]
    error = torch.linalg.matrix_norm(data.tau[index, None, None]*(D-target_D))
    target = torch.linalg.matrix_norm(data.tau[index, None, None]*target_D)
    return float(torch.sqrt(weighted_mean(error.square(), data.weight[index]))/
                 torch.sqrt(weighted_mean(target.square(), data.weight[index])).clamp_min(1e-16))


def parameter_anchor(module, anchor):
    terms = []
    for name, value in module.named_parameters():
        if name in anchor:
            scale = 1.0+anchor[name].abs()
            terms.append(((value-anchor[name])/scale).square().mean())
    prototype = next(module.parameters())
    return torch.stack(terms).mean() if terms else prototype.square().sum()*0.0


def branch_share_regularization(energy):
    base = energy.branch_moduli_MPa/energy.branch_moduli_MPa.sum()
    current = energy.effective_branch_moduli_MPa/energy.effective_branch_moduli_MPa.sum()
    return torch.mean(torch.log(current/base).square())


def set_trainable(material, phase: str):
    for parameter in material.parameters():
        parameter.requires_grad_(False)
    energy = material.nonequilibrium_energy
    dissipation = material.dissipation_potential
    if phase == "energy":
        for name, parameter in energy.named_parameters():
            parameter.requires_grad_(name != "raw_share_delta")
    elif phase == "dissipation":
        dissipation.raw_state_gate.requires_grad_(True)
        dissipation.raw_nonlinear.requires_grad_(True)
        for parameter in dissipation.state_network.parameters():
            parameter.requires_grad_(True)
    elif phase == "shares":
        energy.raw_share_delta.requires_grad_(True)
    elif phase == "joint":
        for name, parameter in energy.named_parameters():
            parameter.requires_grad_(name != "raw_share_delta")
        dissipation.raw_state_gate.requires_grad_(True)
        dissipation.raw_nonlinear.requires_grad_(True)
        for parameter in dissipation.state_network.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError(phase)
    selected = [parameter for parameter in material.parameters() if parameter.requires_grad]
    if not selected:
        raise RuntimeError(f"empty Stage-6 block: {phase}")
    return selected


def snapshot(material):
    return {
        "energy": {name: value.detach().cpu().clone()
                   for name, value in material.nonequilibrium_energy.state_dict().items()},
        "dissipation": {name: value.detach().cpu().clone()
                        for name, value in material.dissipation_potential.state_dict().items()},
    }


def restore(material, state):
    material.nonequilibrium_energy.load_state_dict(state["energy"], strict=False)
    material.dissipation_potential.load_state_dict(state["dissipation"], strict=False)
    synchronize_dissipation_spectrum(material)


def clone_material(material):
    return copy.deepcopy(material)


def curve_rows(value, seed, block, split):
    rows = []
    for item in value["relaxation_predictions"]:
        c = item["condition"]; peak = c.loading_end_index; eq = c.equilibrium_MPa
        obs_peak = float(c.stress_MPa[peak]); pred_peak = float(item["stress"][peak].detach())
        obs_neq = max(obs_peak-eq, 1e-12); pred_neq = max(pred_peak-eq, 1e-12)
        for index in range(len(c.time_s)):
            obs = float(c.stress_MPa[index]); pred = float(item["stress"][index].detach())
            rows.append({"seed": seed, "block": block, "split": split,
                "condition": c.key, "mode": c.mode, "label": c.label,
                "time_s": float(c.time_s[index]),
                "hold_time_s": max(float(c.time_s[index]-c.time_s[peak]), 0.0),
                "strain": float(c.strain[index]),
                "experimental_stress_MPa": obs, "predicted_stress_MPa": pred,
                "experimental_R": obs/obs_peak, "predicted_R": pred/pred_peak,
                "experimental_Rneq": (obs-eq)/obs_neq,
                "predicted_Rneq": (pred-eq)/pred_neq})
    return rows
