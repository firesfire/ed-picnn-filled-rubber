"""Stage-7 joint objective and ED-PICNN checkpoint utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from stage5b_energy import objective as stage6_objective
from stage6_joint import (load_stage6_material, ordering_accuracy,
                          synchronize_dissipation_spectrum)


def load_stage7_material(config: dict, device="cpu", *, load_checkpoint=True):
    material, meta, parameters = load_stage6_material(config, device)
    if load_checkpoint:
        saved = torch.load(config["stage6_checkpoint"], map_location=device,
                           weights_only=False)
        material.nonequilibrium_energy.load_state_dict(
            saved["energy_state_dict"], strict=False)
        material.dissipation_potential.load_state_dict(
            saved["dissipation_state_dict"], strict=False)
        synchronize_dissipation_spectrum(material)
        meta = {**meta, "stage6": saved}
    return material, meta, parameters


def _mean(values, prototype):
    return torch.stack(values).mean() if values else prototype*0.0


def additional_terms(value):
    """Differentiable q-slope, early/mid/late and tan-delta terms."""
    predictions = value["relaxation_predictions"]
    prototype = predictions[0]["stress"].sum()*0.0
    by_mode = {"uniaxial": [], "shear": []}
    stage_errors = []
    for item in predictions:
        c = item["condition"]; peak = c.loading_end_index
        eq = item["stress"].new_tensor(c.equilibrium_MPa)
        obs_neq = (c.stress_MPa[peak]-eq).clamp_min(1e-10)
        pred_neq = (item["stress"][peak]-eq).clamp_min(1e-10)
        q_obs = obs_neq/eq.abs().clamp_min(1e-10)
        q_pred = pred_neq/eq.abs().clamp_min(1e-10)
        level = float(c.strain[peak])
        by_mode[c.mode].append((level, q_obs, q_pred))
        obs = (c.stress_MPa[peak:]-eq)/obs_neq
        pred = (item["stress"][peak:]-eq)/pred_neq
        elapsed = c.time_s[peak:]-c.time_s[peak]
        condition_stages = []
        for left, right in ((0.0, 1.0), (1.0, 100.0), (100.0, float("inf"))):
            mask = (elapsed >= left) & (elapsed < right)
            if bool(mask.any()):
                condition_stages.append((pred[mask]-obs[mask]).square().mean())
        stage_errors.append(_mean(condition_stages, prototype))
    slope_errors = []
    for values in by_mode.values():
        values = sorted(values, key=lambda row: row[0])
        for left, right in zip(values[:-1], values[1:]):
            level_gap = max(np.log1p(right[0])-np.log1p(left[0]), 1e-8)
            obs_slope = (torch.log(right[1])-torch.log(left[1]))/level_gap
            pred_slope = (torch.log(right[2])-torch.log(left[2]))/level_gap
            slope_errors.append((pred_slope-obs_slope).square())
    fp = value["frequency_prediction"]
    observed_storage = torch.as_tensor(
        [row["experimental_storage_MPa"] for row in value["frequency"]["rows"]],
        dtype=fp["storage"].dtype, device=fp["storage"].device)
    observed_loss = torch.as_tensor(
        [row["experimental_loss_MPa"] for row in value["frequency"]["rows"]],
        dtype=fp["loss"].dtype, device=fp["loss"].device)
    observed_tan = observed_loss/observed_storage.clamp_min(1e-10)
    predicted_tan = fp["loss"].clamp_min(1e-10)/fp["storage"].clamp_min(1e-10)
    tan_delta = torch.log(predicted_tan/observed_tan.clamp_min(1e-10)).square().mean()
    return {"q_slope": _mean(slope_errors, prototype),
            "Rneq_stage": _mean(stage_errors, prototype),
            "tan_delta": tan_delta}


def objective(material, relaxation, frequency, split, weights, config,
              *, differentiable=True):
    base = stage6_objective(material, relaxation, frequency, split, weights,
        cycles=config["optimization"]["training_cycles"],
        points_per_cycle=config["optimization"]["training_points_per_cycle"],
        differentiable=differentiable)
    extra = additional_terms(base)
    base["loss"] = (base["loss"] + weights["q_slope"]*extra["q_slope"] +
                    weights["Rneq_stage"]*extra["Rneq_stage"] +
                    weights["tan_delta"]*extra["tan_delta"])
    base["additional"] = extra
    return base


def summary_metrics(value):
    r, f, a = value["relaxation"], value["frequency"], value["additional"]
    return {
        "objective": float(value["loss"].detach()),
        "q_log_RMSE": float(torch.sqrt(r["q_log"]).detach()),
        "q_ordering_accuracy": ordering_accuracy(r["rows"]),
        "q_slope_RMSE": float(torch.sqrt(a["q_slope"]).detach()),
        "Rneq_RMSE": float(torch.sqrt(r["Rneq"]).detach()),
        "Rneq_decade_RMSE": float(torch.sqrt(r["Rneq_decade"]).detach()),
        "Rneq_stage_RMSE": float(torch.sqrt(a["Rneq_stage"]).detach()),
        "R_RMSE": float(torch.sqrt(r["R"]).detach()),
        "loading_RMSE_scaled": float(torch.sqrt(r["loading"]).detach()),
        "storage_RMSE_scaled": float(torch.sqrt(f["storage"]).detach()),
        "loss_log_RMSE": float(torch.sqrt(f["loss"]).detach()),
        "tan_delta_log_RMSE": float(torch.sqrt(a["tan_delta"]).detach()),
        "maximum_cycle_change": float(value["frequency_prediction"]["cycle_change"].max().detach()),
        "minimum_dissipation": float(min(row["minimum_dissipation"] for row in r["rows"])),
    }


def evaluate(material, relaxation, frequency, split, weights, config):
    synchronize_dissipation_spectrum(material)
    with torch.enable_grad():
        value = objective(material, relaxation, frequency, split, weights,
                          config, differentiable=False)
    return value, summary_metrics(value)

