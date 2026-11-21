"""Teacher-retention Pareto interpolation for the Stage-6 state gate."""

from __future__ import annotations

from project_paths import load_config

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from stage2_training import load_datasets
from stage3_real import load_real_data
from stage6_joint import (capture_teacher_anchor, evaluate, load_stage6_material,
                          restore, snapshot, synchronize_dissipation_spectrum,
                          teacher_anchor_nrmse)

CONFIG = ROOT / "configs" / "stage6_edpicnn_block_coordinate.yaml"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    torch.set_default_dtype(torch.float64)
    config = load_config(CONFIG.read_text(encoding="utf-8"))
    results = Path(config["outputs"]["results_dir"]); checkpoints = Path(config["outputs"]["checkpoints_dir"])
    diagnostic_path = checkpoints / "edpicnn_stage6_best_diagnostic.pt"
    diagnostic = torch.load(diagnostic_path, map_location="cpu", weights_only=False)
    material, _, parameters = load_stage6_material(config, "cpu")
    material.nonequilibrium_energy.load_state_dict(diagnostic["energy_state_dict"], strict=False)
    synchronize_dissipation_spectrum(material)
    initial_d = snapshot(material)["dissipation"]
    datasets, _ = load_datasets(config["teacher_anchor_cache"], "cpu")
    teacher_targets = capture_teacher_anchor(material.dissipation_potential, datasets)
    raw, relaxation, frequency = load_real_data(config["real_training_cache"], parameters,
        points_per_decade=config["data"]["points_per_time_decade"],
        maximum_loading_points=config["data"]["maximum_loading_points"], device="cpu")
    final_d = diagnostic["dissipation_state_dict"]
    trainable = ["raw_state_gate", "raw_nonlinear", "state_network.0.weight",
                 "state_network.0.bias", "state_network.2.weight", "state_network.2.bias"]
    rows = []; candidates = []
    for alpha in np.linspace(0.0, 1.0, 21):
        state = {name: value.clone() for name, value in initial_d.items()}
        for name in trainable:
            state[name] = (1.0-alpha)*initial_d[name] + alpha*final_d[name]
        material.dissipation_potential.load_state_dict(state, strict=False)
        synchronize_dissipation_spectrum(material)
        _, training = evaluate(material, relaxation, frequency, "training",
            config["loss_weights"]["joint"], config)
        _, validation = evaluate(material, relaxation, frequency, "validation",
            config["loss_weights"]["joint"], config)
        teacher = {}
        for name, data in datasets.items():
            teacher[name] = teacher_anchor_nrmse(material.dissipation_potential, data,
                                                  teacher_targets[name])
            for confidence, label in ((0, "high"), (1, "medium")):
                mask = data.confidence == confidence
                if bool(mask.any()):
                    teacher[f"{name}_{label}"] = teacher_anchor_nrmse(
                        material.dissipation_potential, data, teacher_targets[name], mask)
        max_teacher = max(teacher["validation_high_medium_high"],
                          teacher["test_high_medium_high"])
        feasible = (max_teacher <= config["cross_guards"]["maximum_teacher_high_medium_flow_NRMSE"] and
                    training["Rneq_RMSE"] <= config["cross_guards"]["maximum_training_Rneq_RMSE"] and
                    validation["Rneq_RMSE"] <= config["cross_guards"]["maximum_validation_Rneq_RMSE"] and
                    validation["storage_RMSE_scaled"] <= config["cross_guards"]["maximum_validation_storage_scaled_RMSE"] and
                    validation["loss_log_RMSE"] <= config["cross_guards"]["maximum_validation_loss_log_RMSE"])
        row = {"alpha": alpha, "feasible": feasible, "teacher_high_medium_NRMSE": max_teacher,
               **{f"training_{key}": value for key, value in training.items()},
               **{f"validation_{key}": value for key, value in validation.items()}}
        rows.append(row); candidates.append((row, state, training, validation, teacher))
        print(f"alpha={alpha:.2f} teacher={max_teacher:.3f} "
              f"Rneq={training['Rneq_RMSE']:.3f}/{validation['Rneq_RMSE']:.3f} "
              f"q={validation['q_log_RMSE']:.3f} feasible={feasible}", flush=True)
    pd.DataFrame(rows).to_csv(results / "stage6_teacher_retention_pareto.csv", index=False)
    feasible = [item for item in candidates if item[0]["feasible"]]
    pool = feasible or candidates
    selected = min(pool, key=lambda item: (item[2]["Rneq_RMSE"]+item[3]["Rneq_RMSE"],
                                            item[3]["q_log_RMSE"]))
    row, state, training, validation, teacher = selected
    material.dissipation_potential.load_state_dict(state, strict=False)
    synchronize_dissipation_spectrum(material)
    checkpoint = dict(diagnostic)
    checkpoint.update({"schema_version": "2026-08-31-stage6-teacher-pareto-v1",
        "dissipation_state_dict": snapshot(material)["dissipation"],
        "dissipation_physical_parameters": material.dissipation_potential.physical_parameters(),
        "metrics": {**diagnostic.get("metrics", {}), "training": training,
                    "validation": validation, "teacher_anchor_nrmse": teacher},
        "teacher_pareto_alpha": row["alpha"], "teacher_pareto_feasible": bool(feasible),
        "passed": bool(feasible), "amplitude_labels_used_for_gradient": False})
    name = "edpicnn_stage6_selected.pt" if feasible else "edpicnn_stage6_teacher_pareto_diagnostic.pt"
    path = checkpoints / name; torch.save(checkpoint, path)
    summary = {"schema_version": "2026-08-31-stage6-teacher-pareto-summary-v1",
        "selected_alpha": row["alpha"], "feasible_count": len(feasible),
        "selected_checkpoint": str(path), "passed": bool(feasible),
        "training": training, "validation": validation, "teacher_anchor_nrmse": teacher}
    save_json(results / "stage6_teacher_pareto_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
