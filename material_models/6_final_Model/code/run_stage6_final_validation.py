"""Independent full-tensor validation of the teacher-retained Stage-6 ED-PICNN."""

from __future__ import annotations

from project_paths import load_config

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from dual_potential.solvers import MaterialPointSolver
from run_stage2_physics_validation import convexity_checks, cost_checks, physics_checks, time_step_checks
from run_stage3_final_validation import curve_rows, full_tensor_dma_subset, summarize_relaxation
from run_stage5b_final_validation import stability_screen
from stage2_training import load_datasets
from stage3_real import load_real_data
from stage6_joint import load_stage6_material, synchronize_dissipation_spectrum

CONFIG = ROOT / "configs" / "stage6_edpicnn_block_coordinate.yaml"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_default_dtype(torch.float64)
    config = load_config(CONFIG.read_text(encoding="utf-8"))
    output = Path(config["outputs"]["results_dir"]); output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["outputs"]["checkpoints_dir"]) / "edpicnn_stage6_selected.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    material, _, parameters = load_stage6_material(config, "cpu")
    material.nonequilibrium_energy.load_state_dict(saved["energy_state_dict"], strict=False)
    material.dissipation_potential.load_state_dict(saved["dissipation_state_dict"], strict=False)
    synchronize_dissipation_spectrum(material); material.eval()
    _, relaxation, frequency = load_real_data(config["real_training_cache"], parameters,
        points_per_decade=config["data"]["points_per_time_decade"],
        maximum_loading_points=config["data"]["maximum_loading_points"])

    start = perf_counter()
    rows, relaxation_seconds = curve_rows(material, relaxation, "Stage6-ED-PICNN",
                                           "full_tensor_local_implicit")
    curves = pd.DataFrame(rows); summary = summarize_relaxation(curves)
    curves.to_csv(output / "stage6_full_tensor_relaxation_curves.csv", index=False)
    summary.to_csv(output / "stage6_full_tensor_relaxation_metrics.csv", index=False)

    frequencies = torch.tensor([item.frequency_hz for item in frequency], dtype=torch.float64)
    linear_start = perf_counter()
    storage, loss = MaterialPointSolver(material).local_linear_shear_moduli(frequencies)
    linear_seconds = perf_counter()-linear_start
    frequency_rows = []
    for index, item in enumerate(frequency):
        frequency_rows.append({"frequency_hz": item.frequency_hz, "split": item.split,
            "amplitude_fraction": item.amplitude_fraction,
            "experimental_storage_MPa": item.storage_MPa,
            "predicted_storage_MPa": float(storage[index]),
            "experimental_loss_MPa": item.loss_MPa,
            "predicted_loss_MPa": float(loss[index])})
    frequency_frame = pd.DataFrame(frequency_rows)
    frequency_frame.to_csv(output / "stage6_local_linear_frequency_dma.csv", index=False)

    amplitude = json.loads(Path(config["amplitude_diagnostic_file"]).read_text(encoding="utf-8"))
    periodic = full_tensor_dma_subset({"Stage6-ED-PICNN": material}, amplitude["rows"])
    periodic.to_csv(output / "stage6_full_tensor_amplitude_dma_subset.csv", index=False)

    datasets, _ = load_datasets(config["teacher_anchor_cache"], "cpu")
    physics = physics_checks(material); convexity = convexity_checks(material, datasets)
    convergence = time_step_checks(material); cost = cost_checks(material, datasets)
    stability = stability_screen(material)
    validation = summary[summary.split == "validation"]
    q_log = float(np.sqrt(np.mean(np.log(validation.predicted_q/validation.experimental_q)**2)))
    rneq = float(np.sqrt(np.mean(validation.Rneq_RMSE**2)))
    training = summary[summary.split == "training"]
    training_rneq = float(np.sqrt(np.mean(training.Rneq_RMSE**2)))
    frequency_frame["storage_relative_error"] = ((frequency_frame.predicted_storage_MPa-
        frequency_frame.experimental_storage_MPa)/frequency_frame.experimental_storage_MPa)
    frequency_frame["loss_relative_error"] = ((frequency_frame.predicted_loss_MPa-
        frequency_frame.experimental_loss_MPa)/frequency_frame.experimental_loss_MPa)
    frequency_metrics = {
        "storage_relative_RMSE": float(np.sqrt(np.mean(frequency_frame.storage_relative_error**2))),
        "loss_relative_RMSE": float(np.sqrt(np.mean(frequency_frame.loss_relative_error**2)))}
    pareto = json.loads((output / "stage6_teacher_pareto_summary.json").read_text(encoding="utf-8"))
    high_teacher = max(pareto["teacher_anchor_nrmse"]["validation_high_medium_high"],
                       pareto["teacher_anchor_nrmse"]["test_high_medium_high"])
    acceptance = {
        "reference_state": max(physics["reference_energy_abs_MPa"], physics["reference_stress_abs_MPa"],
            physics["reference_force_max_MPa"], physics["reference_flow_max_per_s"]) <= 1e-10,
        "objectivity": max(physics["objectivity_energy_relative_error"],
                            physics["objectivity_stress_relative_error"]) <= 1e-8,
        "isotropy": max(physics["isotropy_energy_relative_error"],
                         physics["isotropy_flow_max_abs_error_per_s"]) <= 1e-8,
        "nonnegative_dissipation": physics["minimum_branch_dissipation_MPa_per_s"] >= -1e-12 and
            summary.minimum_dissipation_MPa_per_s.min() >= -1e-12,
        "isochoric_internal": max(row["maximum_detFv_error"] for row in convergence) <= 1e-8,
        "convex_dissipation": convexity["monotone_slope"] and convexity["positive_slope"],
        "newton_convergence": summary.maximum_newton_residual.max() <=
            config["full_validation"]["newton_residual_maximum"],
        "time_step_convergence": convergence[1]["terminal_stress_relative_to_dt_quarter"] <= 0.01,
        "energy_stability_screen": stability["passed"],
        "periodic_no_failure": int(periodic.failed.sum()) == 0,
        "validation_q": q_log <= config["cross_guards"]["maximum_validation_q_log_RMSE"],
        "validation_Rneq": rneq <= config["cross_guards"]["maximum_validation_Rneq_RMSE"],
        "training_Rneq": training_rneq <= config["cross_guards"]["maximum_training_Rneq_RMSE"],
        "teacher_high_confidence": high_teacher <=
            config["cross_guards"]["maximum_teacher_high_medium_flow_NRMSE"],
    }
    result = {"schema_version": "2026-08-31-stage6-final-validation-v1",
        "model_role": "teacher-retained ED-PICNN Stage-6 candidate",
        "source_checkpoint": str(checkpoint_path), "source_checkpoint_sha256": sha256(checkpoint_path),
        "timings": {"full_tensor_relaxation_s": relaxation_seconds,
                    "local_linear_dma_s": linear_seconds,
                    "total_before_checks_s": perf_counter()-start},
        "full_tensor": {"training_Rneq_RMSE": training_rneq,
                        "validation_q_log_RMSE": q_log, "validation_Rneq_RMSE": rneq},
        "frequency_metrics": frequency_metrics, "teacher_high_confidence_NRMSE": high_teacher,
        "physics": physics, "convexity": convexity, "time_step_convergence": convergence,
        "cost": cost, "stability": stability,
        "periodic": {"case_count": len(periodic), "failure_count": int(periodic.failed.sum()),
                     "steady_count": int(periodic.loc[~periodic.failed, "steady"].sum())},
        "acceptance": acceptance, "passed": bool(all(acceptance.values())),
        "amplitude_labels_used_for_gradient": False}
    (output / "stage6_final_validation_summary.json").write_text(json.dumps(result,
        ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
        encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "full_tensor": result["full_tensor"],
        "frequency_metrics": frequency_metrics, "acceptance": acceptance,
        "timings": result["timings"]}, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


if __name__ == "__main__":
    main()
