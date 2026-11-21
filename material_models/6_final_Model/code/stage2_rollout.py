"""Checkpoint loading and complete-history stage-2 rollout diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from dual_potential.constitutive import DualPotentialMaterial, MaterialParameters
from dual_potential.dissipation import Stage2DissipationPICNN, T2GroupedPowerPotential
from dual_potential.energies import FixedOgdenEnergy, ZhangLiEquilibriumEnergy
from dual_potential.integrator import ImplicitExponentialIntegrator
from stage2_training import NORMALIZATION


def load_stage2_material(checkpoint_path, parameter_file, device="cpu"):
    p = MaterialParameters.from_json(parameter_file)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network = Stage2DissipationPICNN(saved["architecture"], p.tau_s,
        p.branch_moduli_MPa, 0.11565279376482657, NORMALIZATION,
        dtype=torch.float64, device=device)
    network.load_state_dict(saved["state_dict"]); network.eval()
    material = DualPotentialMaterial(ZhangLiEquilibriumEnergy(p.zhang_li, device=device),
        FixedOgdenEnergy(p.branch_moduli_MPa, p.a_ogden, device=device), network,
        p.tau_s, device=device)
    return material, saved


def load_trajectory(cache_path, metadata_path, trajectory_id, device="cpu"):
    raw = np.load(cache_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    item = next(row for row in metadata if row["trajectory_id"] == trajectory_id)
    selected = raw["trajectory_trajectory_index"] == item["trajectory_index"]
    convert = lambda name: torch.as_tensor(raw[name][selected], dtype=torch.float64, device=device)
    return item, {"time": convert("trajectory_time_s"), "F": convert("trajectory_F"),
                  "Fv": convert("trajectory_Fv"), "stress": convert("trajectory_teacher_stress_MPa")}


def _errors(states, stresses, teacher):
    state_error = torch.linalg.matrix_norm(states-teacher["Fv"], dim=(-2,-1))
    state_reference = torch.linalg.matrix_norm(teacher["Fv"]-torch.eye(3, dtype=states.dtype, device=states.device), dim=(-2,-1))
    state_nrmse = torch.sqrt(torch.mean(state_error.square()))/torch.sqrt(torch.mean(state_reference.square())).clamp_min(1e-12)
    stress_rmse = torch.sqrt(torch.mean((stresses-teacher["stress"]).square()))
    stress_nrmse = stress_rmse/torch.sqrt(torch.mean(teacher["stress"].square())).clamp_min(1e-12)
    return {"state_nrmse": float(state_nrmse.detach()), "stress_rmse_MPa": float(stress_rmse.detach()),
            "stress_nrmse": float(stress_nrmse.detach()),
            "maximum_detFv_error": float((torch.linalg.det(states)-1).abs().max().detach())}


def rollout(material, item, trajectory, method="implicit"):
    start = perf_counter(); state = trajectory["Fv"][0].clone()
    states, stresses, minimum_dissipation = [state], [], float("inf")
    response = material.response(trajectory["F"][0], state, item["mode"], create_graph=False)
    stresses.append(response["measured_stress_MPa"].detach())
    minimum_dissipation = min(minimum_dissipation, float(response["branch_dissipation_MPa_per_s"].min().detach()))
    integrator = ImplicitExponentialIntegrator(material)
    failures = 0
    for index in range(1, len(trajectory["time"])):
        dt = float(trajectory["time"][index]-trajectory["time"][index-1])
        F0, F1 = trajectory["F"][index-1], trajectory["F"][index]
        try:
            if method == "explicit":
                next_state = []
                for branch in range(7):
                    D = material.branch_flow(F1, state[branch], branch, create_graph=False)["flow_rate_per_s"]
                    next_state.append(torch.matrix_exp(dt*D)@state[branch])
                state = torch.stack(next_state).detach()
            elif method == "local_implicit":
                state, _ = integrator.advance(state, F0, F1, dt)
            elif method == "full_implicit":
                midpoint = integrator._midpoint(F0, F1)
                state, _ = integrator.advance(state, F0, midpoint, 0.5*dt)
                state, _ = integrator.advance(state, midpoint, F1, 0.5*dt)
            else: raise ValueError(method)
        except (RuntimeError, torch.linalg.LinAlgError, FloatingPointError):
            failures += 1; break
        if not torch.isfinite(state).all():
            failures += 1; break
        states.append(state)
        response = material.response(F1, state, item["mode"], create_graph=False)
        if (not torch.isfinite(response["measured_stress_MPa"]).all()
                or not torch.isfinite(response["branch_dissipation_MPa_per_s"]).all()):
            failures += 1; break
        stresses.append(response["measured_stress_MPa"].detach())
        minimum_dissipation = min(minimum_dissipation, float(response["branch_dissipation_MPa_per_s"].min().detach()))
    if failures:
        return {"trajectory_id": item["trajectory_id"], "method": method, "failed": True,
                "completed_points": len(states), "wall_time_s": perf_counter()-start}
    result = {"trajectory_id": item["trajectory_id"], "split": item["split"],
              "confidence": item["confidence"], "path_family": item["path_family"],
              "mode": item["mode"], "method": method, "failed": False,
              "point_count": len(states), "wall_time_s": perf_counter()-start,
              "minimum_dissipation_MPa_per_s": minimum_dissipation}
    result.update(_errors(torch.stack(states), torch.stack(stresses), trajectory))
    return result


def differentiable_complete_history_loss(material, item, trajectory):
    """Closed-loop implicit rollout loss over one complete loading history."""
    material.stop_rollout_physics_gradient = True
    state = trajectory["Fv"][0].clone()
    states, stresses = [state], []
    integrator = ImplicitExponentialIntegrator(material, jacobian_mode="finite_difference")
    stresses.append(material.response(trajectory["F"][0], state, item["mode"],
                                      create_graph=False)["measured_stress_MPa"].detach())
    for index in range(1, len(trajectory["time"])):
        dt = float(trajectory["time"][index]-trajectory["time"][index-1])
        state, _ = integrator.advance(state, trajectory["F"][index-1],
                                      trajectory["F"][index], dt,
                                      differentiable=True)
        states.append(state)
        stresses.append(material.response(trajectory["F"][index], state,
                                          item["mode"], create_graph=False)["measured_stress_MPa"].detach())
    states, stresses = torch.stack(states), torch.stack(stresses)
    identity = torch.eye(3, dtype=states.dtype, device=states.device)
    state_scale = torch.mean((trajectory["Fv"]-identity).square()).clamp_min(1e-10)
    stress_scale = torch.mean(trajectory["stress"].square()).clamp_min(1e-8)
    state_loss = torch.mean((states-trajectory["Fv"]).square())/state_scale
    stress_loss = torch.mean((stresses-trajectory["stress"]).square())/stress_scale
    material.stop_rollout_physics_gradient = False
    return state_loss, {"state_loss": state_loss, "stress_loss": stress_loss}
