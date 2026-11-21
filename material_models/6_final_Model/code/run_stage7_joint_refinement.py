"""Stage 7: guarded joint refinement and multi-objective Pareto selection."""

from __future__ import annotations

from project_paths import load_config

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from stage2_training import load_datasets
from stage3_real import load_real_data
from stage6_joint import (branch_share_regularization, capture_teacher_anchor,
    curve_rows, parameter_anchor, restore, set_trainable, snapshot,
    synchronize_dissipation_spectrum, teacher_anchor_loss, teacher_anchor_nrmse)
from stage7_joint import evaluate, load_stage7_material, objective

CONFIG = ROOT / "configs" / "stage7_joint_consistency.yaml"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
        encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def focus(metrics, phase):
    relax = (0.25*metrics["q_log_RMSE"]**2 +
             0.10*metrics["q_slope_RMSE"]**2 +
             0.25*metrics["Rneq_RMSE"]**2 +
             0.20*metrics["Rneq_decade_RMSE"]**2 +
             0.20*metrics["Rneq_stage_RMSE"]**2)
    dma = (0.45*metrics["storage_RMSE_scaled"]**2 +
           0.35*metrics["loss_log_RMSE"]**2 +
           0.20*metrics["tan_delta_log_RMSE"]**2)
    if phase == "energy":
        return relax + 0.65*dma
    if phase == "dissipation":
        return 1.10*relax + dma
    return relax + dma


def combined_focus(training, validation, phase):
    return 0.75*focus(training, phase) + 0.25*focus(validation, phase)


def data_guard(training, validation, config):
    g = config["cross_guards"]
    return bool(
        validation["q_log_RMSE"] <= g["maximum_validation_q_log_RMSE"] and
        validation["Rneq_RMSE"] <= g["maximum_validation_Rneq_RMSE"] and
        training["Rneq_RMSE"] <= g["maximum_training_Rneq_RMSE"] and
        validation["storage_RMSE_scaled"] <= g["maximum_validation_storage_scaled_RMSE"] and
        validation["loss_log_RMSE"] <= g["maximum_validation_loss_log_RMSE"] and
        training["q_ordering_accuracy"] >= g["minimum_training_q_ordering_accuracy"] and
        min(training["minimum_dissipation"], validation["minimum_dissipation"]) >= -1e-12)


def optimizer_for(material, phase, config):
    trainable = set_trainable(material, phase)
    o = config["optimization"]
    if phase == "joint":
        energy = [p for p in material.nonequilibrium_energy.parameters() if p.requires_grad]
        dissipation = [p for p in material.dissipation_potential.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": energy, "lr": o["joint_energy_learning_rate"]},
            {"params": dissipation, "lr": o["joint_dissipation_learning_rate"]}],
            weight_decay=o["weight_decay"])
        learning_rate = max(o["joint_energy_learning_rate"], o["joint_dissipation_learning_rate"])
    else:
        key = "share_learning_rate" if phase == "shares" else f"{phase}_learning_rate"
        learning_rate = o[key]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate,
                                      weight_decay=o["weight_decay"])
    return trainable, optimizer, learning_rate


def train_endpoint(material, relaxation, frequency, datasets, teacher_targets,
                   config, seed, history):
    initial = snapshot(material)
    energy_anchor = {n: p.detach().clone()
                     for n, p in material.nonequilibrium_energy.named_parameters()}
    dissipation_anchor = {n: p.detach().clone()
                          for n, p in material.dissipation_potential.named_parameters()}
    train_data = datasets["train"]
    started = time.perf_counter()
    for cycle in range(1, config["optimization"]["alternations"] + 1):
        for phase, epochs_key in (("energy", "energy_epochs"),
                                  ("dissipation", "dissipation_epochs"),
                                  ("shares", "share_epochs"),
                                  ("joint", "joint_epochs")):
            epochs = config["optimization"][epochs_key]
            trainable, optimizer, learning_rate = optimizer_for(material, phase, config)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, epochs,
                eta_min=learning_rate*config["optimization"]["minimum_learning_rate_fraction"])
            start_state = snapshot(material)
            _, start_train = evaluate(material, relaxation, frequency, "training",
                config["loss_weights"]["joint"], config)
            _, start_val = evaluate(material, relaxation, frequency, "validation",
                config["loss_weights"]["joint"], config)
            best_state = start_state; best_score = combined_focus(start_train, start_val, phase)
            best_epoch = 0
            for epoch in range(1, epochs + 1):
                optimizer.zero_grad(set_to_none=True)
                value = objective(material, relaxation, frequency, "training",
                    config["loss_weights"][phase if phase in ("energy", "dissipation") else "joint"],
                    config, differentiable=True)
                loss = value["loss"]
                if phase in ("energy", "joint"):
                    loss = loss + config["regularization"]["energy_stage6_anchor"]*parameter_anchor(
                        material.nonequilibrium_energy, energy_anchor)
                if phase in ("dissipation", "joint"):
                    count = min(config["dissipation_gate"]["teacher_anchor_batch"], train_data.count)
                    index = torch.randint(train_data.count, (count,), device=train_data.X.device)
                    loss = loss + config["regularization"]["high_medium_teacher_anchor"]*teacher_anchor_loss(
                        material.dissipation_potential, train_data, teacher_targets["train"], index)
                    loss = loss + config["regularization"]["dissipation_stage6_anchor"]*parameter_anchor(
                        material.dissipation_potential, dissipation_anchor)
                if phase == "shares":
                    loss = loss + config["regularization"]["branch_share_log_anchor"]*branch_share_regularization(
                        material.nonequilibrium_energy)
                if not bool(torch.isfinite(loss)):
                    restore(material, start_state)
                    break
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(
                    trainable, config["optimization"]["gradient_clip"])
                optimizer.step(); synchronize_dissipation_spectrum(material); scheduler.step()
                if epoch == 1 or epoch % config["optimization"]["evaluation_interval"] == 0 or epoch == epochs:
                    _, train_metrics = evaluate(material, relaxation, frequency, "training",
                        config["loss_weights"]["joint"], config)
                    _, validation_metrics = evaluate(material, relaxation, frequency, "validation",
                        config["loss_weights"]["joint"], config)
                    score = combined_focus(train_metrics, validation_metrics, phase)
                    feasible = data_guard(train_metrics, validation_metrics, config)
                    history.append({"seed": seed, "cycle": cycle, "phase": phase,
                        "epoch": epoch, "training_loss": float(loss.detach()),
                        "gradient_norm": float(gradient.detach()), "focus_score": score,
                        "data_guard_passed": feasible, "elapsed_s": time.perf_counter()-started,
                        **{f"training_{k}": v for k, v in train_metrics.items()},
                        **{f"validation_{k}": v for k, v in validation_metrics.items()}})
                    print(f"stage7 seed={seed} cycle={cycle} {phase} {epoch:02d} "
                          f"q={validation_metrics['q_log_RMSE']:.3f} "
                          f"Rn={train_metrics['Rneq_RMSE']:.3f}/{validation_metrics['Rneq_RMSE']:.3f} "
                          f"Gp={validation_metrics['storage_RMSE_scaled']:.3f} "
                          f"Gpp={validation_metrics['loss_log_RMSE']:.3f} guard={feasible}", flush=True)
                    if feasible and score < best_score - 1e-10:
                        best_state = snapshot(material); best_score = score; best_epoch = epoch
            restore(material, best_state)
            print(f"stage7 seed={seed} cycle={cycle} {phase} accepted_epoch={best_epoch}", flush=True)
    return initial, snapshot(material), time.perf_counter()-started


def interpolate_state(initial, endpoint, alpha):
    result = {}
    for group in ("energy", "dissipation"):
        result[group] = {}
        for key, left in initial[group].items():
            right = endpoint[group][key]
            if left.is_floating_point():
                result[group][key] = (1.0-alpha)*left + alpha*right
            else:
                result[group][key] = left.clone()
    return result


def teacher_metrics(potential, datasets, targets):
    values = {}
    for name, data in datasets.items():
        values[name] = teacher_anchor_nrmse(potential, data, targets[name])
        mask = data.confidence == 0
        if bool(mask.any()):
            values[f"{name}_high"] = teacher_anchor_nrmse(
                potential, data, targets[name], mask)
    return values


def spectrum_change(material, base_moduli):
    current = material.nonequilibrium_energy.effective_branch_moduli_MPa.detach()
    total = base_moduli.sum().clamp_min(1e-12)
    return float(torch.max(torch.abs(current-base_moduli))/total)


def nondominated(frame, x, y):
    values = frame[[x, y]].to_numpy(float); result = []
    for i, point in enumerate(values):
        dominated = np.any(np.all(values <= point, axis=1) & np.any(values < point, axis=1))
        result.append(not dominated)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(); torch.set_default_dtype(torch.float64)
    config = load_config(Path(args.config).read_text(encoding="utf-8"))
    results = Path(config["outputs"]["results_dir"]); checkpoints = Path(config["outputs"]["checkpoints_dir"])
    results.mkdir(parents=True, exist_ok=True); checkpoints.mkdir(parents=True, exist_ok=True)
    base, _, parameters = load_stage7_material(config, "cpu", load_checkpoint=True)
    _, relaxation, frequency = load_real_data(config["real_training_cache"], parameters,
        points_per_decade=config["data"]["points_per_time_decade"],
        maximum_loading_points=config["data"]["maximum_loading_points"], device="cpu")
    datasets, _ = load_datasets(config["teacher_anchor_cache"], "cpu")
    teacher_reference, _, _ = load_stage7_material(config, "cpu", load_checkpoint=False)
    teacher_targets = capture_teacher_anchor(teacher_reference.dissipation_potential, datasets)
    base_moduli = base.nonequilibrium_energy.effective_branch_moduli_MPa.detach().clone()
    _, base_train = evaluate(base, relaxation, frequency, "training", config["loss_weights"]["joint"], config)
    _, base_val = evaluate(base, relaxation, frequency, "validation", config["loss_weights"]["joint"], config)

    history = []; endpoints = []
    for seed in config["optimization"]["seeds"]:
        torch.manual_seed(seed); np.random.seed(seed)
        material, _, _ = load_stage7_material(config, "cpu", load_checkpoint=True)
        initial, endpoint, wall = train_endpoint(material, relaxation, frequency, datasets,
                                                  teacher_targets, config, seed, history)
        endpoint_path = checkpoints / f"edpicnn_stage7_endpoint_seed{seed}.pt"
        torch.save({"schema_version": "2026-08-31-stage7-endpoint-v1", "seed": seed,
                    "energy_state_dict": endpoint["energy"],
                    "dissipation_state_dict": endpoint["dissipation"],
                    "source_stage6_checkpoint": config["stage6_checkpoint"],
                    "amplitude_labels_used_for_gradient": False}, endpoint_path)
        endpoints.append((seed, initial, endpoint, endpoint_path, wall))
    pd.DataFrame(history).to_csv(results / "stage7_block_coordinate_history.csv", index=False)

    rows = []; states = {}
    for seed, initial, endpoint, endpoint_path, wall in endpoints:
        material, _, _ = load_stage7_material(config, "cpu", load_checkpoint=True)
        for alpha in config["optimization"]["pareto_alpha_grid"]:
            state = interpolate_state(initial, endpoint, float(alpha)); restore(material, state)
            _, training = evaluate(material, relaxation, frequency, "training",
                config["loss_weights"]["joint"], config)
            _, validation = evaluate(material, relaxation, frequency, "validation",
                config["loss_weights"]["joint"], config)
            teacher = teacher_metrics(material.dissipation_potential, datasets, teacher_targets)
            high = max(teacher.get("validation_high_medium_high", np.inf),
                       teacher.get("test_high_medium_high", np.inf))
            spectrum = spectrum_change(material, base_moduli)
            feasible = (data_guard(training, validation, config) and
                        high <= config["cross_guards"]["maximum_teacher_high_confidence_flow_NRMSE"] and
                        spectrum <= config["cross_guards"]["maximum_reference_spectrum_relative_change"])
            relax_loss = combined_focus(training, validation, "dissipation")
            dma_loss = 0.5*(training["storage_RMSE_scaled"]**2 + validation["storage_RMSE_scaled"]**2 +
                            training["loss_log_RMSE"]**2 + validation["loss_log_RMSE"]**2)
            row = {"seed": seed, "alpha": float(alpha), "feasible": feasible,
                   "relaxation_loss": relax_loss, "DMA_loss": dma_loss,
                   "teacher_high_confidence_NRMSE": high,
                   "reference_spectrum_max_relative_change": spectrum,
                   "endpoint_wall_time_s": wall,
                   **{f"training_{k}": v for k, v in training.items()},
                   **{f"validation_{k}": v for k, v in validation.items()}}
            rows.append(row); states[(seed, float(alpha))] = state
    frame = pd.DataFrame(rows)
    frame["pareto_relax_DMA"] = nondominated(frame, "relaxation_loss", "DMA_loss")
    frame["pareto_q_Gp"] = nondominated(frame, "validation_q_log_RMSE", "validation_storage_RMSE_scaled")
    frame["pareto_Rneq_Gpp"] = nondominated(frame, "validation_Rneq_RMSE", "validation_loss_log_RMSE")
    frame.to_csv(results / "stage7_joint_pareto_candidates.csv", index=False)
    feasible = frame[frame["feasible"]].copy()
    if feasible.empty:
        raise RuntimeError("Stage-7 has no teacher-retained data-feasible candidate; Stage-6 must remain active")
    criteria = ["relaxation_loss", "DMA_loss", "validation_q_log_RMSE",
                "validation_Rneq_RMSE"]
    normalized = []
    for name in criteria:
        lo, hi = feasible[name].min(), feasible[name].max()
        normalized.append((feasible[name]-lo)/(hi-lo+1e-12))
    feasible["knee_score"] = sum(normalized)/len(normalized)
    selected_row = feasible.sort_values(["knee_score", "teacher_high_confidence_NRMSE"]).iloc[0]
    selected_key = (int(selected_row["seed"]), float(selected_row["alpha"]))
    selected_state = states[selected_key]
    selected_material, _, _ = load_stage7_material(config, "cpu", load_checkpoint=True)
    restore(selected_material, selected_state)
    train_value, train_metrics = evaluate(selected_material, relaxation, frequency, "training",
        config["loss_weights"]["joint"], config)
    val_value, val_metrics = evaluate(selected_material, relaxation, frequency, "validation",
        config["loss_weights"]["joint"], config)
    teacher = teacher_metrics(selected_material.dissipation_potential, datasets, teacher_targets)
    selected_path = checkpoints / "edpicnn_stage7_selected.pt"
    checkpoint = {"schema_version": "2026-08-31-stage7-selected-v1",
        "seed": selected_key[0], "alpha": selected_key[1],
        "energy_state_dict": selected_state["energy"],
        "dissipation_state_dict": selected_state["dissipation"],
        "metrics": {"training": train_metrics, "validation": val_metrics,
                    "teacher_anchor_nrmse": teacher},
        "source_stage6_checkpoint": config["stage6_checkpoint"],
        "source_stage6_sha256": sha256(config["stage6_checkpoint"]),
        "amplitude_labels_used_for_gradient": False}
    torch.save(checkpoint, selected_path)
    curves = curve_rows(train_value, selected_key[0], "stage7_selected", "training")
    curves += curve_rows(val_value, selected_key[0], "stage7_selected", "validation")
    pd.DataFrame(curves).to_csv(results / "stage7_reduced_relaxation_curves.csv", index=False)
    frequency_rows = []
    for split, value in (("training", train_value), ("validation", val_value)):
        for row in value["frequency"]["rows"]:
            frequency_rows.append({"split": split, **row})
    pd.DataFrame(frequency_rows).to_csv(results / "stage7_reduced_frequency_dma.csv", index=False)
    summary = {"schema_version": "2026-08-31-stage7-training-summary-v1",
        "stage6_baseline": {"training": base_train, "validation": base_val},
        "seed_count": len(endpoints), "pareto_candidate_count": len(frame),
        "feasible_candidate_count": len(feasible),
        "selected_seed": selected_key[0], "selected_alpha": selected_key[1],
        "selected_checkpoint": str(selected_path), "selected_checkpoint_sha256": sha256(selected_path),
        "training": train_metrics, "validation": val_metrics,
        "teacher_anchor_nrmse": teacher,
        "amplitude_labels_used_for_gradient": False}
    save_json(results / "stage7_training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
