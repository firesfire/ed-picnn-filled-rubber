"""Stage-7 independent validation and four-route time-frequency audit."""

from __future__ import annotations

from project_paths import load_config

import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.integrate import quad
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from dual_potential.solvers import MaterialPointSolver
from run_stage2_physics_validation import (convexity_checks, cost_checks,
    physics_checks, time_step_checks)
from run_stage3_final_validation import curve_rows, summarize_relaxation
from run_stage5b_final_validation import stability_screen
from stage2_training import load_datasets
from stage3_real import equilibrium_shear_modulus, load_real_data
from stage6_joint import synchronize_dissipation_spectrum
from stage7_joint import load_stage7_material

CONFIG = ROOT / "configs" / "stage7_joint_consistency.yaml"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
        encoding="utf-8")


def state_max_difference(left, right):
    maximum = 0.0
    for group in ("energy_state_dict", "dissipation_state_dict"):
        for key, value in left[group].items():
            if value.is_floating_point():
                maximum = max(maximum, float(torch.max(torch.abs(value-right[group][key]))))
    return maximum


def full_tensor_relaxation(material, relaxation, config, output, equivalent):
    stage6 = ROOT / "results" / "stage6"
    target_curves = output / "stage7_full_tensor_relaxation_curves.csv"
    target_metrics = output / "stage7_full_tensor_relaxation_metrics.csv"
    if equivalent and (stage6 / "stage6_full_tensor_relaxation_curves.csv").exists():
        curves = pd.read_csv(stage6 / "stage6_full_tensor_relaxation_curves.csv")
        curves["model"] = "Stage7-ED-PICNN"
        metrics = pd.read_csv(stage6 / "stage6_full_tensor_relaxation_metrics.csv")
        metrics["model"] = "Stage7-ED-PICNN"
        curves.to_csv(target_curves, index=False); metrics.to_csv(target_metrics, index=False)
        return curves, metrics, 0.0, "reused_after_exact_state_equivalence"
    rows, seconds = curve_rows(material, relaxation, "Stage7-ED-PICNN",
                               "full_tensor_local_implicit")
    curves = pd.DataFrame(rows); metrics = summarize_relaxation(curves)
    curves.to_csv(target_curves, index=False); metrics.to_csv(target_metrics, index=False)
    return curves, metrics, seconds, "recomputed"


def numerical_transform(geq, moduli, tau, frequencies, tail_time):
    rows = []
    for frequency in frequencies:
        omega = 2.0*np.pi*frequency
        # Integrate branch-by-branch. A single multi-scale tail can make
        # QUADPACK miss the fastest decay at high frequency and bias G'.
        sin_integral = 0.0; cos_integral = 0.0
        for modulus, time_scale in zip(moduli, tau):
            upper = min(float(tail_time), 40.0*float(time_scale))
            decay = lambda t, m=float(modulus), ts=float(time_scale): m*np.exp(-t/ts)
            sin_integral += quad(decay, 0.0, upper, weight="sin", wvar=omega,
                                 epsabs=1e-12, epsrel=1e-10, limit=300)[0]
            cos_integral += quad(decay, 0.0, upper, weight="cos", wvar=omega,
                                 epsabs=1e-12, epsrel=1e-10, limit=300)[0]
        rows.append((geq+omega*sin_integral, omega*cos_integral))
    return np.asarray(rows)


def direct_relaxation(material, config, geq, moduli, tau):
    audit = config["four_route_validation"]
    amplitude = float(audit["relaxation_amplitude_fraction"])
    load_time = float(audit["loading_time_s"])
    hold = np.asarray(audit["relaxation_times_s"], float)
    positive = hold[hold > 0]
    per_decade = int(audit["direct_points_per_decade"])
    dense = np.logspace(np.log10(positive.min()), np.log10(positive.max()),
                        int(round((np.log10(positive.max())-np.log10(positive.min()))*per_decade))+1)
    integration_hold = np.unique(np.r_[positive, dense])
    time = torch.tensor(np.r_[0.0, load_time, load_time+integration_hold], dtype=torch.float64)
    primary = torch.tensor(np.r_[0.0, amplitude, np.full(len(integration_hold), amplitude)],
                           dtype=torch.float64)
    result = MaterialPointSolver(material).solve_history(primary, time, "shear")
    all_direct = result.measured_stress_MPa[1:].detach().numpy()/amplitude
    direct = np.r_[all_direct[0], [all_direct[1+np.where(np.isclose(integration_hold, t,
        rtol=1e-12, atol=1e-15))[0][0]] for t in positive]]
    analytical = geq + np.exp(-hold[:, None]/tau[None, :])@moduli
    rows = pd.DataFrame({"hold_time_s": hold, "direct_full_tensor_modulus_MPa": direct,
                         "analytical_local_spectrum_modulus_MPa": analytical,
                         "relative_error": (direct-analytical)/np.maximum(np.abs(analytical), 1e-12)})
    return rows, result


def route_audit(material, config, output):
    audit = config["four_route_validation"]
    frequencies = np.asarray(audit["frequencies_Hz"], float)
    tau = material.tau_s.detach().numpy()
    moduli = material.nonequilibrium_energy.effective_branch_moduli_MPa.detach().numpy()
    geq = float(equilibrium_shear_modulus(material.equilibrium_energy_parameters))
    direct, direct_history = direct_relaxation(material, config, geq, moduli, tau)
    direct.to_csv(output / "stage7_direct_small_strain_relaxation.csv", index=False)

    tangent_storage, tangent_loss = MaterialPointSolver(material).local_linear_shear_moduli(
        torch.tensor(frequencies, dtype=torch.float64))
    tangent = np.c_[tangent_storage.detach().numpy(), tangent_loss.detach().numpy()]
    transforms = {}
    for tail in (1000.0, 10000.0, float(audit["transform_tail_time_s"])):
        transforms[tail] = numerical_transform(geq, moduli, tau, frequencies, tail)
    selected_transform = transforms[float(audit["transform_tail_time_s"])]

    solver = MaterialPointSolver(material)
    periodic_rows = []
    for frequency in frequencies:
        started = perf_counter()
        history, dynamic = solver.periodic_shear(float(frequency), 0.001,
            points_per_cycle=int(audit["periodic_points_per_cycle"]),
            maximum_cycles=int(audit["periodic_maximum_cycles"]),
            steady_tolerance=float(audit["periodic_steady_tolerance"]))
        periodic_rows.append({"frequency_hz": frequency,
            "periodic_storage_MPa": dynamic.storage_MPa,
            "periodic_loss_MPa": dynamic.loss_MPa,
            "cycle_change": dynamic.cycle_change, "steady": dynamic.steady,
            "third_harmonic_ratio": dynamic.harmonic_ratios[3],
            "minimum_dissipation_MPa_per_s": float(history.branch_dissipation_MPa_per_s.min().detach()),
            "maximum_detFv_error": float((torch.linalg.det(history.Fv)-1.0).abs().max()),
            "wall_time_s": perf_counter()-started})
    periodic = pd.DataFrame(periodic_rows)

    rows = []
    for i, frequency in enumerate(frequencies):
        row = {"frequency_hz": frequency, "amplitude_fraction": 0.001,
            "route2_transform_storage_MPa": selected_transform[i, 0],
            "route2_transform_loss_MPa": selected_transform[i, 1],
            "route3_tangent_storage_MPa": tangent[i, 0],
            "route3_tangent_loss_MPa": tangent[i, 1],
            **periodic.iloc[i].to_dict()}
        for tail, values in transforms.items():
            row[f"transform_storage_T{tail:g}s_MPa"] = values[i, 0]
            row[f"transform_loss_T{tail:g}s_MPa"] = values[i, 1]
        row["transform_vs_tangent_storage_relative_error"] = abs(selected_transform[i, 0]-tangent[i, 0])/abs(tangent[i, 0])
        row["transform_vs_tangent_loss_relative_error"] = abs(selected_transform[i, 1]-tangent[i, 1])/max(abs(tangent[i, 1]), 1e-12)
        row["periodic_vs_tangent_storage_relative_error"] = abs(row["periodic_storage_MPa"]-tangent[i, 0])/abs(tangent[i, 0])
        row["periodic_vs_tangent_loss_relative_error"] = abs(row["periodic_loss_MPa"]-tangent[i, 1])/max(abs(tangent[i, 1]), 1e-12)
        rows.append(row)
    routes = pd.DataFrame(rows)
    routes.to_csv(output / "stage7_four_route_frequency_consistency.csv", index=False)
    direct_max = float(np.max(np.abs(direct.relative_error)))
    metrics = {"direct_relaxation_vs_spectrum_max_relative_error": direct_max,
        "transform_vs_tangent_storage_max_relative_error": float(routes.transform_vs_tangent_storage_relative_error.max()),
        "transform_vs_tangent_loss_max_relative_error": float(routes.transform_vs_tangent_loss_relative_error.max()),
        "periodic_vs_tangent_storage_max_relative_error": float(routes.periodic_vs_tangent_storage_relative_error.max()),
        "periodic_vs_tangent_loss_max_relative_error": float(routes.periodic_vs_tangent_loss_relative_error.max()),
        "periodic_steady_count": int(routes.steady.sum()),
        "periodic_case_count": len(routes),
        "transform_tail_times_s": [1000.0, 10000.0, float(audit["transform_tail_time_s"])],
        "minimum_periodic_dissipation_MPa_per_s": float(routes.minimum_dissipation_MPa_per_s.min()),
        "maximum_periodic_detFv_error": float(routes.maximum_detFv_error.max())}
    return routes, direct, metrics


def main():
    torch.set_default_dtype(torch.float64)
    config = load_config(CONFIG.read_text(encoding="utf-8"))
    output = Path(config["outputs"]["results_dir"]); output.mkdir(parents=True, exist_ok=True)
    selected_path = Path(config["outputs"]["checkpoints_dir"]) / "edpicnn_stage7_selected.pt"
    saved = torch.load(selected_path, map_location="cpu", weights_only=False)
    material, _, parameters = load_stage7_material(config, "cpu", load_checkpoint=False)
    material.nonequilibrium_energy.load_state_dict(saved["energy_state_dict"], strict=False)
    material.dissipation_potential.load_state_dict(saved["dissipation_state_dict"], strict=False)
    synchronize_dissipation_spectrum(material); material.eval()
    stage6_saved = torch.load(config["stage6_checkpoint"], map_location="cpu", weights_only=False)
    state_difference = state_max_difference(saved, stage6_saved)
    exact_equivalence = state_difference == 0.0
    _, relaxation, frequency = load_real_data(config["real_training_cache"], parameters,
        points_per_decade=config["data"]["points_per_time_decade"],
        maximum_loading_points=config["data"]["maximum_loading_points"])
    curves, relaxation_metrics, relaxation_seconds, relaxation_mode = full_tensor_relaxation(
        material, relaxation, config, output, exact_equivalence)

    route_start = perf_counter(); routes, direct, route_metrics = route_audit(material, config, output)
    route_seconds = perf_counter()-route_start
    all_frequencies = torch.tensor([item.frequency_hz for item in frequency], dtype=torch.float64)
    storage, loss = MaterialPointSolver(material).local_linear_shear_moduli(all_frequencies)
    frequency_rows = []
    for i, item in enumerate(frequency):
        frequency_rows.append({"frequency_hz": item.frequency_hz, "split": item.split,
            "experimental_storage_MPa": item.storage_MPa, "predicted_storage_MPa": float(storage[i]),
            "experimental_loss_MPa": item.loss_MPa, "predicted_loss_MPa": float(loss[i])})
    frequency_frame = pd.DataFrame(frequency_rows)
    frequency_frame.to_csv(output / "stage7_local_linear_frequency_dma.csv", index=False)
    storage_rmse = float(np.sqrt(np.mean(((frequency_frame.predicted_storage_MPa-frequency_frame.experimental_storage_MPa)/frequency_frame.experimental_storage_MPa)**2)))
    loss_rmse = float(np.sqrt(np.mean(((frequency_frame.predicted_loss_MPa-frequency_frame.experimental_loss_MPa)/frequency_frame.experimental_loss_MPa)**2)))

    datasets, _ = load_datasets(config["teacher_anchor_cache"], "cpu")
    physics = physics_checks(material); convexity = convexity_checks(material, datasets)
    convergence = time_step_checks(material); cost = cost_checks(material, datasets)
    stability = stability_screen(material)
    training = relaxation_metrics[relaxation_metrics.split == "training"]
    validation = relaxation_metrics[relaxation_metrics.split == "validation"]
    train_rneq = float(np.sqrt(np.mean(training.Rneq_RMSE**2)))
    val_rneq = float(np.sqrt(np.mean(validation.Rneq_RMSE**2)))
    val_q = float(np.sqrt(np.mean(np.log(validation.predicted_q/validation.experimental_q)**2)))
    training_summary = json.loads((output / "stage7_training_summary.json").read_text(encoding="utf-8"))
    teacher_high = max(training_summary["teacher_anchor_nrmse"]["validation_high_medium_high"],
                       training_summary["teacher_anchor_nrmse"]["test_high_medium_high"])
    a = config["four_route_validation"]
    acceptance = {
        "same_checkpoint_all_routes": True,
        "exact_stage6_equivalence_when_alpha_zero": exact_equivalence,
        "direct_relaxation_consistency": route_metrics["direct_relaxation_vs_spectrum_max_relative_error"] <= 0.02,
        "transform_storage_consistency": route_metrics["transform_vs_tangent_storage_max_relative_error"] <= a["maximum_storage_route_relative_error"],
        "transform_loss_consistency": route_metrics["transform_vs_tangent_loss_max_relative_error"] <= a["maximum_loss_route_relative_error"],
        "periodic_storage_consistency": route_metrics["periodic_vs_tangent_storage_max_relative_error"] <= a["maximum_storage_route_relative_error"],
        "periodic_loss_consistency": route_metrics["periodic_vs_tangent_loss_max_relative_error"] <= a["maximum_loss_route_relative_error"],
        "periodic_steady": route_metrics["periodic_steady_count"] == route_metrics["periodic_case_count"],
        "reference_state": max(physics["reference_energy_abs_MPa"], physics["reference_stress_abs_MPa"], physics["reference_force_max_MPa"], physics["reference_flow_max_per_s"]) <= 1e-10,
        "objectivity": max(physics["objectivity_energy_relative_error"], physics["objectivity_stress_relative_error"]) <= 1e-8,
        "isotropy": max(physics["isotropy_energy_relative_error"], physics["isotropy_flow_max_abs_error_per_s"]) <= 1e-8,
        "nonnegative_dissipation": physics["minimum_branch_dissipation_MPa_per_s"] >= -1e-12 and route_metrics["minimum_periodic_dissipation_MPa_per_s"] >= -1e-12,
        "isochoric_internal": max(route_metrics["maximum_periodic_detFv_error"], max(row["maximum_detFv_error"] for row in convergence)) <= 1e-8,
        "convex_dissipation": convexity["monotone_slope"] and convexity["positive_slope"],
        "time_step_convergence": convergence[1]["terminal_stress_relative_to_dt_quarter"] <= 0.01,
        "energy_stability": stability["passed"],
        "validation_q": val_q <= config["cross_guards"]["maximum_validation_q_log_RMSE"],
        "validation_Rneq": val_rneq <= config["cross_guards"]["maximum_validation_Rneq_RMSE"],
        "training_Rneq": train_rneq <= config["cross_guards"]["maximum_training_Rneq_RMSE"],
        "teacher_high_confidence": teacher_high <= config["cross_guards"]["maximum_teacher_high_confidence_flow_NRMSE"],
    }
    result = {"schema_version": "2026-08-31-stage7-consistency-validation-v1",
        "source_checkpoint": str(selected_path), "source_checkpoint_sha256": sha256(selected_path),
        "selected_alpha": saved["alpha"], "stage6_state_max_absolute_difference": state_difference,
        "full_tensor_relaxation_mode": relaxation_mode,
        "full_tensor": {"training_Rneq_RMSE": train_rneq,
                        "validation_q_log_RMSE": val_q, "validation_Rneq_RMSE": val_rneq},
        "experimental_frequency": {"storage_relative_RMSE": storage_rmse,
                                   "loss_relative_RMSE": loss_rmse},
        "four_route": route_metrics, "physics": physics, "convexity": convexity,
        "time_step_convergence": convergence, "cost": cost, "stability": stability,
        "teacher_high_confidence_NRMSE": teacher_high,
        "timings": {"full_tensor_relaxation_s": relaxation_seconds,
                    "four_route_audit_s": route_seconds},
        "acceptance": acceptance, "passed": bool(all(acceptance.values())),
        "amplitude_labels_used_for_gradient": False,
        "experimental_same_amplitude_relaxation_available": False}
    save_json(output / "stage7_consistency_validation_summary.json", result)
    print(json.dumps({"passed": result["passed"], "four_route": route_metrics,
        "full_tensor": result["full_tensor"], "experimental_frequency": result["experimental_frequency"],
        "acceptance": acceptance, "timings": result["timings"]},
        ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


if __name__ == "__main__":
    main()
