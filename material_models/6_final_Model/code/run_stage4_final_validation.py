"""Independent final validation for Chapter-3 Stage 4 GD-PICNN."""

from __future__ import annotations

from project_paths import load_config

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from run_stage2_physics_validation import convexity_checks, cost_checks, physics_checks, time_step_checks
from run_stage3_final_validation import curve_rows, dma_dense, full_tensor_dma_subset, summarize_relaxation
from stage2_training import load_datasets
from stage3_real import load_real_data
from stage4_energy import build_stage4_material, load_stage4_checkpoint, physical_parameters

CONFIG = ROOT / "configs" / "stage4_gdpicnn_energy_release.yaml"
STAGE3 = ROOT / "checkpoints" / "stage3" / "dpicnn_stage3_selected.pt"
STAGE4 = ROOT / "checkpoints" / "stage4" / "gdpicnn_stage4_selected.pt"
OUTPUT = ROOT / "results" / "stage4"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-dense", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG.read_text(encoding="utf-8"))

    stage3, _, parameters = build_stage4_material(config, "cpu")
    stage3.eval()
    stage4, saved, _ = load_stage4_checkpoint(STAGE4, config, "cpu")
    materials = {"Stage3-fixed-energy": stage3, "Stage4-GD-PICNN": stage4}

    raw, relaxation, frequency = load_real_data(
        config["real_training_cache"], parameters,
        points_per_decade=config["data"]["points_per_time_decade"],
        maximum_loading_points=config["data"]["maximum_loading_points"], device="cpu")
    amplitude = json.loads(Path(config["amplitude_external_file"]).read_text(encoding="utf-8"))
    if amplitude["used_for_optimization"] or amplitude["source_workbook_sha256"] != raw["source_workbook_sha256"]:
        raise RuntimeError("amplitude diagnostic provenance violation")

    dense_paths = [OUTPUT / "final_relaxation_curves.csv", OUTPUT / "final_relaxation_metrics.csv",
                   OUTPUT / "final_frequency_dma.csv", OUTPUT / "final_amplitude_dma_external.csv"]
    timings = {}
    if args.reuse_dense and all(path.exists() for path in dense_paths):
        curves, summary, frequency_frame, amplitude_frame = [pd.read_csv(p) for p in dense_paths]
        timings["dense_results_reused"] = True
    else:
        rows = []
        for model, material in materials.items():
            for solver_kind in ("reduced",):
                part, seconds = curve_rows(material, relaxation, model, solver_kind)
                rows.extend(part)
                timings[f"{model}:{solver_kind}:all_relaxation_s"] = seconds
        # The Stage-3 full-tensor result was already independently validated.
        # Reuse that immutable evidence and recompute only the new Stage-4 model.
        prior = pd.read_csv(ROOT / "results" / "stage3" / "final_relaxation_curves.csv")
        prior = prior[(prior.model == "Stage3-real-finetuned") &
                      (prior.solver == "full_tensor_local_implicit")].copy()
        prior["model"] = "Stage3-fixed-energy"
        rows.extend(prior.to_dict("records"))
        part, seconds = curve_rows(stage4, relaxation, "Stage4-GD-PICNN", "full_tensor_local_implicit")
        rows.extend(part)
        timings["Stage4-GD-PICNN:full_tensor_local_implicit:all_relaxation_s"] = seconds
        curves = pd.DataFrame(rows)
        summary = summarize_relaxation(curves)
        frequency_frame, amplitude_frame = dma_dense(materials, frequency, amplitude["rows"])
        curves.to_csv(dense_paths[0], index=False)
        summary.to_csv(dense_paths[1], index=False)
        frequency_frame.to_csv(dense_paths[2], index=False)
        amplitude_frame.to_csv(dense_paths[3], index=False)

    full_dma_path = OUTPUT / "final_full_tensor_dma_subset.csv"
    if args.reuse_dense and full_dma_path.exists():
        full_dma = pd.read_csv(full_dma_path)
    else:
        prior_dma = pd.read_csv(ROOT / "results" / "stage3" / "final_full_tensor_dma_subset.csv")
        prior_dma = prior_dma[prior_dma.model == "Stage3-real-finetuned"].copy()
        prior_dma["model"] = "Stage3-fixed-energy"
        new_dma = full_tensor_dma_subset({"Stage4-GD-PICNN": stage4}, amplitude["rows"])
        full_dma = pd.concat((prior_dma, new_dma), ignore_index=True)
        full_dma.to_csv(full_dma_path, index=False)

    datasets, _ = load_datasets(config["synthetic_anchor_cache"], "cpu")
    physics = physics_checks(stage4)
    convexity = convexity_checks(stage4, datasets)
    convergence = time_step_checks(stage4)
    cost = cost_checks(stage4, datasets)
    full_relax = summary[summary.solver == "full_tensor_local_implicit"]
    valid_dma = full_dma[~full_dma.failed]
    finite_curve_columns = ["time_s", "strain", "experimental_stress_MPa", "predicted_stress_MPa",
                            "experimental_R", "predicted_R", "experimental_Rneq", "predicted_Rneq"]
    acceptance = {
        "reference_state": max(physics["reference_energy_abs_MPa"], physics["reference_stress_abs_MPa"],
                               physics["reference_force_max_MPa"], physics["reference_flow_max_per_s"]) <= 1e-10,
        "objectivity": max(physics["objectivity_energy_relative_error"],
                           physics["objectivity_stress_relative_error"]) <= 1e-8,
        "isotropy_state_covariance": max(physics["isotropy_energy_relative_error"],
                                         physics["isotropy_flow_max_abs_error_per_s"]) <= 1e-8,
        "deviatoric_force": physics["maximum_force_trace_MPa"] <= 1e-8,
        "nonnegative_dissipation": (physics["minimum_branch_dissipation_MPa_per_s"] >= -1e-12 and
                                    summary.minimum_dissipation_MPa_per_s.min() >= -1e-12 and
                                    valid_dma.minimum_dissipation_MPa_per_s.min() >= -1e-12),
        "isochoric_internal": max(row["maximum_detFv_error"] for row in convergence) <= 1e-8,
        "convex_monotone_potential": convexity["monotone_slope"] and convexity["positive_slope"],
        "newton_convergence": full_relax.maximum_newton_residual.max() <= 1.01e-8,
        "time_step_terminal_stress_below_one_percent": convergence[1]["terminal_stress_relative_to_dt_quarter"] <= .01,
        "all_dense_values_finite": (np.isfinite(curves[finite_curve_columns]).all().all() and
                                    np.isfinite(frequency_frame.select_dtypes("number")).all().all() and
                                    np.isfinite(amplitude_frame.select_dtypes("number")).all().all()),
        "full_tensor_dma_no_failure": int(full_dma.failed.sum()) == 0,
    }

    selected_parameters = physical_parameters(stage4)
    selected_parameters.update({"release_level": saved["release_level"],
                                "seed": saved["seed"], "tau_s": [float(x) for x in stage4.tau_s]})
    (OUTPUT / "gdpicnn_stage4_selected_parameters.json").write_text(
        json.dumps(selected_parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {
        "schema_version": "2026-08-30-chapter3-stage4-final-validation-v1",
        "release_level": saved["release_level"], "amplitude_labels_used_for_optimization": False,
        "source_workbook_sha256": raw["source_workbook_sha256"],
        "checkpoints": {"stage3": {"path": str(STAGE3), "sha256": sha256(STAGE3)},
                        "stage4": {"path": str(STAGE4), "sha256": sha256(STAGE4)}},
        "selected_parameters": selected_parameters, "timings": timings,
        "physics": physics, "convexity": convexity, "time_step_convergence": convergence,
        "cost": cost, "acceptance": acceptance, "passed": all(acceptance.values()),
        "full_tensor_dma": {"case_count": len(full_dma), "failure_count": int(full_dma.failed.sum()),
                            "steady_count": int(valid_dma.steady.sum())},
    }
    encoder = lambda value: value.item() if isinstance(value, np.generic) else str(value)
    (OUTPUT / "stage4_final_validation_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=encoder), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "release_level": saved["release_level"],
                      "acceptance": acceptance, "full_tensor_dma": result["full_tensor_dma"],
                      "timings": timings}, ensure_ascii=False, indent=2, default=encoder))


if __name__ == "__main__":
    main()
