"""Stage 6: alternating ED-PICNN block-coordinate calibration."""

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
    curve_rows, differentiable_objective, evaluate, load_stage6_material,
    parameter_anchor, restore, set_trainable, snapshot, synchronize_dissipation_spectrum,
    teacher_anchor_loss, teacher_anchor_nrmse)

CONFIG = ROOT / "configs" / "stage6_edpicnn_block_coordinate.yaml"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perturb_energy(energy, seed, standard_deviation):
    generator = torch.Generator(device=energy.branch_moduli_MPa.device)
    generator.manual_seed(seed)
    with torch.no_grad():
        for name, parameter in energy.named_parameters():
            if name != "raw_share_delta":
                parameter.add_(standard_deviation*torch.randn(parameter.shape,
                    dtype=parameter.dtype, device=parameter.device, generator=generator))


def focus_score(metrics, phase):
    if phase == "energy":
        return metrics["q_log_RMSE"]**2 + 0.30*metrics["storage_RMSE_scaled"]**2
    if phase == "dissipation":
        return (metrics["Rneq_RMSE"]**2 + metrics["Rneq_decade_RMSE"]**2 +
                0.25*metrics["loss_log_RMSE"]**2)
    return metrics["objective"]


def combined_focus_score(training, validation, phase):
    """Eight-condition average: six fitted histories and two held-out histories."""
    return 0.75*focus_score(training, phase) + 0.25*focus_score(validation, phase)


def guard_pass(metrics, anchor, phase, config):
    guard = config["cross_guards"]
    if phase == "energy":
        return (metrics["Rneq_RMSE"] <= anchor["Rneq_RMSE"] + guard["Rneq_absolute_increase"] and
                metrics["loss_log_RMSE"] <= anchor["loss_log_RMSE"] + guard["loss_log_absolute_increase"])
    if phase == "dissipation":
        return (metrics["q_log_RMSE"] <= anchor["q_log_RMSE"] + guard["q_log_absolute_increase"] and
                metrics["storage_RMSE_scaled"] <= anchor["storage_RMSE_scaled"] +
                guard["storage_scaled_absolute_increase"])
    return (metrics["q_log_RMSE"] <= anchor["q_log_RMSE"] + guard["q_log_absolute_increase"] and
            metrics["Rneq_RMSE"] <= anchor["Rneq_RMSE"] + guard["Rneq_absolute_increase"] and
            metrics["storage_RMSE_scaled"] <= anchor["storage_RMSE_scaled"] +
            guard["storage_scaled_absolute_increase"] and
            metrics["loss_log_RMSE"] <= anchor["loss_log_RMSE"] + guard["loss_log_absolute_increase"])


def optimize_block(material, relaxation, frequency, datasets, anchor_targets,
                   energy_anchor, dissipation_anchor, config, seed, cycle, phase,
                   epochs, history):
    trainable = set_trainable(material, phase)
    synchronize_dissipation_spectrum(material)
    _, anchor_validation = evaluate(material, relaxation, frequency, "validation",
        config["loss_weights"]["joint"], config)
    _, anchor_training = evaluate(material, relaxation, frequency, "training",
        config["loss_weights"]["joint"], config)
    if phase == "joint":
        energy_parameters = [p for p in material.nonequilibrium_energy.parameters() if p.requires_grad]
        gate_parameters = [material.dissipation_potential.raw_state_gate,
                           material.dissipation_potential.raw_nonlinear,
                           *material.dissipation_potential.state_network.parameters()]
        base_parameters = [p for p in material.dissipation_potential.base.parameters() if p.requires_grad]
        groups = [
            {"params": energy_parameters, "lr": config["optimization"]["joint_energy_learning_rate"]},
            {"params": gate_parameters, "lr": config["optimization"]["joint_dissipation_learning_rate"]}]
        if base_parameters:
            groups.append({"params": base_parameters, "lr": config["optimization"]["joint_base_dpicnn_learning_rate"]})
        optimizer = torch.optim.AdamW(groups,
            weight_decay=config["optimization"]["weight_decay"])
        learning_rate = max(config["optimization"]["joint_energy_learning_rate"],
                            config["optimization"]["joint_dissipation_learning_rate"])
    elif phase == "dissipation":
        learning_rate = config["optimization"]["dissipation_learning_rate"]
        gate_parameters = [material.dissipation_potential.raw_state_gate,
                           material.dissipation_potential.raw_nonlinear,
                           *material.dissipation_potential.state_network.parameters()]
        base_parameters = [p for p in material.dissipation_potential.base.parameters() if p.requires_grad]
        groups = [{"params": gate_parameters, "lr": learning_rate}]
        if base_parameters:
            groups.append({"params": base_parameters, "lr": config["optimization"]["base_dpicnn_learning_rate"]})
        optimizer = torch.optim.AdamW(groups,
            weight_decay=config["optimization"]["weight_decay"])
    else:
        learning_rate_key = "share_learning_rate" if phase == "shares" else f"{phase}_learning_rate"
        learning_rate = config["optimization"][learning_rate_key]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate,
            weight_decay=config["optimization"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs,
        eta_min=learning_rate*config["optimization"]["minimum_learning_rate_fraction"])
    best_state = snapshot(material); best_metrics = dict(anchor_validation)
    best_score = combined_focus_score(anchor_training, anchor_validation, phase); best_epoch = 0
    start = time.perf_counter(); training_data = datasets["train"]
    for epoch in range(1, epochs+1):
        optimizer.zero_grad(set_to_none=True)
        value = differentiable_objective(material, relaxation, frequency,
            config["loss_weights"][phase if phase in ("energy", "dissipation") else "joint"], config)
        loss = value["loss"]
        if phase in ("energy", "joint"):
            loss = loss + config["regularization"]["energy_anchor"]*parameter_anchor(
                material.nonequilibrium_energy, energy_anchor)
        if phase in ("dissipation", "joint"):
            count = min(config["dissipation_gate"]["teacher_anchor_batch"], training_data.count)
            index = torch.randint(training_data.count, (count,), device=training_data.X.device)
            loss = loss + config["regularization"]["high_medium_teacher_anchor"]*teacher_anchor_loss(
                material.dissipation_potential, training_data, anchor_targets["train"], index)
            loss = loss + config["regularization"]["dissipation_parameter_anchor"]*parameter_anchor(
                material.dissipation_potential, dissipation_anchor)
        if phase == "shares":
            loss = loss + config["regularization"]["branch_share_log_anchor"]*branch_share_regularization(
                material.nonequilibrium_energy)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite Stage-6 {phase} loss")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, config["optimization"]["gradient_clip"])
        optimizer.step(); synchronize_dissipation_spectrum(material); scheduler.step()
        if epoch == 1 or epoch % config["optimization"]["evaluation_interval"] == 0 or epoch == epochs:
            _, train_metrics = evaluate(material, relaxation, frequency, "training",
                config["loss_weights"]["joint"], config)
            _, validation_metrics = evaluate(material, relaxation, frequency, "validation",
                config["loss_weights"]["joint"], config)
            score = combined_focus_score(train_metrics, validation_metrics, phase)
            feasible = guard_pass(validation_metrics, anchor_validation, phase, config)
            row = {"seed": seed, "cycle": cycle, "phase": phase, "epoch": epoch,
                "training_loss": float(loss.detach()), "gradient_norm": float(gradient.detach()),
                "validation_focus_score": score, "cross_guard_passed": feasible,
                "elapsed_s": time.perf_counter()-start,
                **{f"training_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()}}
            history.append(row)
            print(f"stage6 seed={seed} cycle={cycle} {phase} epoch={epoch:02d} "
                  f"q={validation_metrics['q_log_RMSE']:.3f} "
                  f"Rn(train/val)={train_metrics['Rneq_RMSE']:.3f}/{validation_metrics['Rneq_RMSE']:.3f} "
                  f"Gp={validation_metrics['storage_RMSE_scaled']:.3f} "
                  f"Gpp={validation_metrics['loss_log_RMSE']:.3f} guard={feasible}", flush=True)
            if feasible and score < best_score-1e-10:
                best_score = score; best_state = snapshot(material)
                best_metrics = dict(validation_metrics); best_epoch = epoch
    restore(material, best_state)
    return {"phase": phase, "best_epoch": best_epoch, "best_score": best_score,
            "validation": best_metrics, "wall_time_s": time.perf_counter()-start}


def acceptance(training, validation, teacher, config):
    g = config["cross_guards"]
    return {
        "validation_q": validation["q_log_RMSE"] <= g["maximum_validation_q_log_RMSE"],
        "validation_Rneq": validation["Rneq_RMSE"] <= g["maximum_validation_Rneq_RMSE"],
        "training_Rneq": training["Rneq_RMSE"] <= g["maximum_training_Rneq_RMSE"],
        "storage": validation["storage_RMSE_scaled"] <= g["maximum_validation_storage_scaled_RMSE"],
        "loss": validation["loss_log_RMSE"] <= g["maximum_validation_loss_log_RMSE"],
        "validation_ordering": validation["q_ordering_accuracy"] >= 1.0-1e-12,
        "training_ordering": training["q_ordering_accuracy"] >=
            g["minimum_training_q_ordering_accuracy"],
        "teacher_high_confidence": max(teacher["validation_high_medium_high"],
            teacher["test_high_medium_high"]) <= g["maximum_teacher_high_medium_flow_NRMSE"],
        "dissipation": min(training["minimum_dissipation"],
                           validation["minimum_dissipation"]) >= -1e-12,
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args(); torch.set_default_dtype(torch.float64)
    config = load_config(Path(args.config).read_text(encoding="utf-8"))
    results = Path(config["outputs"]["results_dir"]); checkpoints = Path(config["outputs"]["checkpoints_dir"])
    results.mkdir(parents=True, exist_ok=True); checkpoints.mkdir(parents=True, exist_ok=True)
    base_material, _, parameters = load_stage6_material(config, "cpu")
    raw, relaxation, frequency = load_real_data(config["real_training_cache"], parameters,
        points_per_decade=config["data"]["points_per_time_decade"],
        maximum_loading_points=config["data"]["maximum_loading_points"], device="cpu")
    datasets, _ = load_datasets(config["teacher_anchor_cache"], "cpu")

    history = []; seed_rows = []; seed_summaries = []; curve_data = []; dma_data = []
    for seed in config["optimization"]["seeds"]:
        torch.manual_seed(seed); np.random.seed(seed)
        material, meta, _ = load_stage6_material(config, "cpu")
        perturb_energy(material.nonequilibrium_energy, seed,
                       config["optimization"]["energy_perturbation_std"])
        synchronize_dissipation_spectrum(material)
        energy_anchor = {name: value.detach().clone()
                         for name, value in material.nonequilibrium_energy.named_parameters()}
        dissipation_anchor = {name: value.detach().clone()
                              for name, value in material.dissipation_potential.named_parameters()}
        teacher_targets = capture_teacher_anchor(material.dissipation_potential, datasets)
        initial_train_value, initial_train = evaluate(material, relaxation, frequency, "training",
            config["loss_weights"]["joint"], config)
        initial_validation_value, initial_validation = evaluate(material, relaxation, frequency, "validation",
            config["loss_weights"]["joint"], config)
        blocks = []; started = time.perf_counter()
        for cycle in range(1, config["optimization"]["alternations"]+1):
            for phase, key in (("energy", "energy_epochs"), ("dissipation", "dissipation_epochs"),
                               ("shares", "share_epochs"), ("joint", "joint_epochs")):
                blocks.append(optimize_block(material, relaxation, frequency, datasets,
                    teacher_targets, energy_anchor, dissipation_anchor, config, seed, cycle,
                    phase, config["optimization"][key], history))
        train_value, train_metrics = evaluate(material, relaxation, frequency, "training",
            config["loss_weights"]["joint"], config)
        validation_value, validation_metrics = evaluate(material, relaxation, frequency, "validation",
            config["loss_weights"]["joint"], config)
        teacher_metrics = {}
        for name, data in datasets.items():
            teacher_metrics[name] = teacher_anchor_nrmse(material.dissipation_potential, data,
                                                          teacher_targets[name])
            for confidence, label in ((0, "high"), (1, "medium")):
                mask = data.confidence == confidence
                if bool(mask.any()):
                    teacher_metrics[f"{name}_{label}"] = teacher_anchor_nrmse(
                        material.dissipation_potential, data, teacher_targets[name], mask)
        gates = acceptance(train_metrics, validation_metrics, teacher_metrics, config)
        passed = all(gates.values())
        path = checkpoints / f"edpicnn_stage6_seed{seed}.pt"
        checkpoint = {"schema_version": "2026-08-31-stage6-edpicnn-v1", "seed": seed,
            "energy_state_dict": snapshot(material)["energy"],
            "dissipation_state_dict": snapshot(material)["dissipation"],
            "energy_physical_parameters": material.nonequilibrium_energy.physical_parameters(),
            "dissipation_physical_parameters": material.dissipation_potential.physical_parameters(),
            "metrics": {"initial_training": initial_train, "initial_validation": initial_validation,
                        "training": train_metrics, "validation": validation_metrics,
                        "teacher_anchor_nrmse": teacher_metrics},
            "acceptance": gates, "passed": passed, "blocks": blocks,
            "stage5b_energy_checkpoint": config["stage5b_energy_checkpoint"],
            "stage5b_energy_checkpoint_sha256": sha256(config["stage5b_energy_checkpoint"]),
            "amplitude_labels_used_for_gradient": False}
        torch.save(checkpoint, path)
        summary = {"seed": seed, "checkpoint": str(path), "passed": passed,
            "wall_time_s": time.perf_counter()-started,
            **{f"initial_validation_{key}": value for key, value in initial_validation.items()},
            **{f"training_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **{f"teacher_{key}_flow_NRMSE": value for key, value in teacher_metrics.items()}}
        seed_rows.append(summary); seed_summaries.append(checkpoint)
        for split, value in (("training", train_value), ("validation", validation_value)):
            curve_data.extend(curve_rows(value, seed, "final", split))
            for row in value["frequency"]["rows"]:
                dma_data.append({"seed": seed, "split": split, **row})

    frame = pd.DataFrame(seed_rows); frame.to_csv(results / "stage6_seed_results.csv", index=False)
    pd.DataFrame(history).to_csv(results / "stage6_block_coordinate_history.csv", index=False)
    pd.DataFrame(curve_data).to_csv(results / "stage6_reduced_relaxation_curves.csv", index=False)
    pd.DataFrame(dma_data).to_csv(results / "stage6_reduced_frequency_dma.csv", index=False)
    feasible = [item for item in seed_summaries if item["passed"]]
    pool = feasible or seed_summaries
    selected = min(pool, key=lambda item: (item["metrics"]["validation"]["objective"],
        item["metrics"]["validation"]["Rneq_RMSE"], item["metrics"]["validation"]["q_log_RMSE"]))
    source = checkpoints / f"edpicnn_stage6_seed{selected['seed']}.pt"
    selected_path = checkpoints / ("edpicnn_stage6_selected.pt" if feasible else
                                   "edpicnn_stage6_best_diagnostic.pt")
    selected_path.write_bytes(source.read_bytes())
    output = {"schema_version": "2026-08-31-stage6-training-summary-v1",
        "seed_count": len(seed_summaries), "feasible_seed_count": len(feasible),
        "selected_seed": selected["seed"], "selected_checkpoint": str(selected_path),
        "selected_passed": bool(feasible), "selected_metrics": selected["metrics"],
        "selected_acceptance": selected["acceptance"],
        "amplitude_labels_used_for_gradient": False,
        "config_sha256": sha256(args.config)}
    save_json(results / "stage6_training_summary.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
