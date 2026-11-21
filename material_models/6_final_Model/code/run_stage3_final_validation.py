"""Final independent validation for Chapter-3 Stage 3.

The reduced scalar charts are used for dense curves.  Every relaxation
history is also recomputed with the production full-tensor local-implicit
exponential-map solver.  Sweep-amplitude labels are loaded only here, after
the checkpoint has been selected.
"""

from __future__ import annotations

from project_paths import load_config

import hashlib
import argparse
import json
import math
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
from stage2_rollout import load_stage2_material
from stage3_real import (DMACondition, dma_rollout, equilibrium_stress,
                         load_real_data, relaxation_rollout)
from run_stage2_physics_validation import (convexity_checks, cost_checks,
                                           physics_checks, time_step_checks)
from stage2_training import load_datasets

CONFIG = ROOT / "configs" / "stage3_dpicnn_real_finetuning.yaml"
STAGE2 = ROOT / "checkpoints" / "stage2" / "dpicnn_stage2_final.pt"
STAGE3 = ROOT / "checkpoints" / "stage3" / "dpicnn_stage3_selected.pt"
AMPLITUDE = ROOT / "audit" / "stage3" / "stage3_amplitude_external_diagnostic.json"
OUTPUT = ROOT / "results" / "stage3"


def number(value):
    return float(torch.as_tensor(value).detach().cpu())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve_rows(material, conditions, model, solver_kind):
    rows = []
    start = perf_counter()
    if solver_kind == "reduced":
        values = relaxation_rollout(material, conditions, differentiable=False)
        predictions = [(item["stress"], item["dissipation"].min(), None) for item in values]
    else:
        predictions = []
        solver = MaterialPointSolver(material)
        for condition in conditions:
            history = solver.solve_history(condition.strain, condition.time_s, condition.mode)
            residual = max((d.residual_norm for step in history.diagnostics for d in step), default=0.0)
            predictions.append((history.measured_stress_MPa,
                                history.branch_dissipation_MPa_per_s.min(), residual))
    for condition, (stress, minimum_dissipation, residual) in zip(conditions, predictions):
        peak = condition.loading_end_index
        equilibrium = float(condition.equilibrium_MPa)
        exp_peak = number(condition.stress_MPa[peak]); pred_peak = number(stress[peak])
        exp_neq_peak = max(exp_peak-equilibrium, 1e-12)
        pred_neq_peak = max(pred_peak-equilibrium, 1e-12)
        for index in range(len(condition.time_s)):
            exp = number(condition.stress_MPa[index]); pred = number(stress[index])
            rows.append({
                "model": model, "solver": solver_kind, "condition": condition.key,
                "split": condition.split, "mode": condition.mode, "label": condition.label,
                "time_s": number(condition.time_s[index]), "strain": number(condition.strain[index]),
                "experimental_stress_MPa": exp, "predicted_stress_MPa": pred,
                "experimental_relaxation_modulus_MPa": exp/max(number(condition.strain[peak]), 1e-12),
                "predicted_relaxation_modulus_MPa": pred/max(number(condition.strain[peak]), 1e-12),
                "experimental_R": exp/exp_peak, "predicted_R": pred/pred_peak,
                "experimental_Rneq": (exp-equilibrium)/exp_neq_peak,
                "predicted_Rneq": (pred-equilibrium)/pred_neq_peak,
                "equilibrium_stress_MPa": equilibrium, "loading_end_index": peak,
                "minimum_dissipation_MPa_per_s": number(minimum_dissipation),
                "maximum_newton_residual": residual,
            })
    return rows, perf_counter()-start


def summarize_relaxation(frame):
    rows = []
    for keys, group in frame.groupby(["model", "solver", "condition", "split", "mode", "label"]):
        peak = int(group.loading_end_index.iloc[0]); hold = group.iloc[peak:]
        rmse = lambda column: float(np.sqrt(np.mean(np.square(hold[column]))))
        rows.append(dict(zip(["model","solver","condition","split","mode","label"], keys)) | {
            "stress_RMSE_MPa": rmse("predicted_stress_MPa") if False else float(np.sqrt(np.mean(np.square(hold.predicted_stress_MPa-hold.experimental_stress_MPa)))),
            "R_RMSE": float(np.sqrt(np.mean(np.square(hold.predicted_R-hold.experimental_R)))),
            "Rneq_RMSE": float(np.sqrt(np.mean(np.square(hold.predicted_Rneq-hold.experimental_Rneq)))),
            "experimental_q": float((group.experimental_stress_MPa.iloc[peak]-group.equilibrium_stress_MPa.iloc[0])/max(abs(group.equilibrium_stress_MPa.iloc[0]),1e-12)),
            "predicted_q": float((group.predicted_stress_MPa.iloc[peak]-group.equilibrium_stress_MPa.iloc[0])/max(abs(group.equilibrium_stress_MPa.iloc[0]),1e-12)),
            "minimum_dissipation_MPa_per_s": float(group.minimum_dissipation_MPa_per_s.min()),
            "maximum_newton_residual": float(group.maximum_newton_residual.dropna().max()) if group.maximum_newton_residual.notna().any() else math.nan,
        })
    return pd.DataFrame(rows)


def dma_dense(materials, frequency_conditions, amplitude_rows):
    frequency_rows, amplitude_output = [], []
    frequencies = torch.tensor([c.frequency_hz for c in frequency_conditions], dtype=torch.float64)
    for model, material in materials.items():
        reduced = dma_rollout(material, frequency_conditions, cycles=8, points_per_cycle=96, differentiable=False)
        local_storage, local_loss = MaterialPointSolver(material).local_linear_shear_moduli(frequencies)
        for index, condition in enumerate(frequency_conditions):
            frequency_rows.append({
                "model": model, "frequency_hz": condition.frequency_hz, "split": condition.split,
                "amplitude_fraction": condition.amplitude_fraction,
                "experimental_storage_MPa": condition.storage_MPa,
                "experimental_loss_MPa": condition.loss_MPa,
                "reduced_cycle_storage_MPa": number(reduced["storage"][index]),
                "reduced_cycle_loss_MPa": number(reduced["loss"][index]),
                "reduced_cycle_change": number(reduced["cycle_change"][index]),
                "reduced_third_harmonic_ratio": number(reduced["third_harmonic"][index]),
                "local_linear_storage_MPa": number(local_storage[index]),
                "local_linear_loss_MPa": number(local_loss[index]),
            })
        amplitude_conditions = tuple(DMACondition(1.0, row["amplitude_fraction"],
            row["storage_modulus_MPa"], row["loss_modulus_MPa"], "external_diagnostic")
            for row in amplitude_rows)
        reduced_amp = dma_rollout(material, amplitude_conditions, cycles=8,
                                  points_per_cycle=96, differentiable=False)
        for index, row in enumerate(amplitude_rows):
            amplitude_output.append({
                "model": model, **row,
                "predicted_storage_MPa": number(reduced_amp["storage"][index]),
                "predicted_loss_MPa": number(reduced_amp["loss"][index]),
                "cycle_change": number(reduced_amp["cycle_change"][index]),
                "third_harmonic_ratio": number(reduced_amp["third_harmonic"][index]),
            })
    return pd.DataFrame(frequency_rows), pd.DataFrame(amplitude_output)


def full_tensor_dma_subset(materials, amplitude_rows):
    rows = []
    # Representative endpoints and centres cross-check the dense reduced charts.
    cases = [("frequency", f, 0.001) for f in (0.1, 10.0, 100.0)]
    selected = [0, min(range(len(amplitude_rows)), key=lambda i: abs(amplitude_rows[i]["amplitude_percent"]-1.0)), len(amplitude_rows)-1]
    cases += [("amplitude", 1.0, amplitude_rows[i]["amplitude_fraction"]) for i in selected]
    for model, material in materials.items():
        solver = MaterialPointSolver(material)
        for family, frequency, amplitude in cases:
            start = perf_counter()
            try:
                history, moduli = solver.periodic_shear(frequency, amplitude,
                    points_per_cycle=16, maximum_cycles=3, steady_tolerance=3e-3)
                rows.append({"model":model,"family":family,"frequency_hz":frequency,
                    "amplitude_fraction":amplitude,"storage_MPa":moduli.storage_MPa,
                    "loss_MPa":moduli.loss_MPa,"cycle_change":moduli.cycle_change,
                    "steady":moduli.steady,"third_harmonic_ratio":moduli.harmonic_ratios[3],
                    "minimum_dissipation_MPa_per_s":number(history.branch_dissipation_MPa_per_s.min()),
                    "maximum_detFv_error":number((torch.linalg.det(history.Fv)-1).abs().max()),
                    "wall_time_s":perf_counter()-start,"failed":False})
            except Exception as error:
                rows.append({"model":model,"family":family,"frequency_hz":frequency,
                    "amplitude_fraction":amplitude,"wall_time_s":perf_counter()-start,
                    "failed":True,"error":f"{type(error).__name__}: {error}"})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--reuse-dense",action="store_true")
    args=parser.parse_args()
    torch.set_default_dtype(torch.float64); OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG.read_text(encoding="utf-8"))
    parameters = __import__("dual_potential.constitutive", fromlist=["MaterialParameters"]).MaterialParameters.from_json(config["teacher_parameter_file"])
    materials = {
        "Stage2-pretrained": load_stage2_material(STAGE2, config["teacher_parameter_file"], "cpu")[0],
        "Stage3-real-finetuned": load_stage2_material(STAGE3, config["teacher_parameter_file"], "cpu")[0],
    }
    for material in materials.values():
        material.equilibrium_energy_parameters = parameters.zhang_li
    raw, relaxation, frequency = load_real_data(config["real_training_cache"], parameters,
        points_per_decade=config["data"]["points_per_time_decade"],
        maximum_loading_points=config["data"]["maximum_loading_points"], device="cpu")
    amplitude = json.loads(AMPLITUDE.read_text(encoding="utf-8"))
    if amplitude["used_for_optimization"] or amplitude["source_workbook_sha256"] != raw["source_workbook_sha256"]:
        raise RuntimeError("amplitude diagnostic provenance violation")

    timings={}
    dense_paths=[OUTPUT/"final_relaxation_curves.csv",OUTPUT/"final_relaxation_metrics.csv",
                 OUTPUT/"final_frequency_dma.csv",OUTPUT/"final_amplitude_dma_external.csv"]
    if args.reuse_dense and all(path.exists() for path in dense_paths):
        curves=pd.read_csv(dense_paths[0]);summary=pd.read_csv(dense_paths[1])
        frequency_frame=pd.read_csv(dense_paths[2]);amplitude_frame=pd.read_csv(dense_paths[3])
        timings["dense_results_reused"]=True
    else:
        all_curve_rows=[]
        for model, material in materials.items():
            for solver_kind in ("reduced", "full_tensor_local_implicit"):
                rows, seconds=curve_rows(material, relaxation, model, solver_kind)
                all_curve_rows.extend(rows);timings[f"{model}:{solver_kind}:all_relaxation_s"]=seconds
        curves=pd.DataFrame(all_curve_rows);curves.to_csv(dense_paths[0],index=False)
        summary=summarize_relaxation(curves);summary.to_csv(dense_paths[1],index=False)
        frequency_frame, amplitude_frame=dma_dense(materials,frequency,amplitude["rows"])
        frequency_frame.to_csv(dense_paths[2],index=False)
        amplitude_frame.to_csv(dense_paths[3],index=False)
    full_dma_path=OUTPUT/"final_full_tensor_dma_subset.csv"
    if args.reuse_dense and full_dma_path.exists():
        full_dma=pd.read_csv(full_dma_path)
    else:
        full_dma=full_tensor_dma_subset(materials,amplitude["rows"])
        full_dma.to_csv(full_dma_path,index=False)

    datasets,_=load_datasets(config["synthetic_anchor_cache"],"cpu")
    selected=materials["Stage3-real-finetuned"]
    physics=physics_checks(selected);convexity=convexity_checks(selected,datasets)
    convergence=time_step_checks(selected);cost=cost_checks(selected,datasets)
    acceptance={
        "reference_state":max(physics["reference_energy_abs_MPa"],physics["reference_stress_abs_MPa"],physics["reference_force_max_MPa"],physics["reference_flow_max_per_s"])<=1e-10,
        "objectivity":max(physics["objectivity_energy_relative_error"],physics["objectivity_stress_relative_error"])<=1e-8,
        "isotropy_state_covariance":max(physics["isotropy_energy_relative_error"],physics["isotropy_flow_max_abs_error_per_s"])<=1e-8,
        "deviatoric_force":physics["maximum_force_trace_MPa"]<=1e-8,
        "nonnegative_dissipation":physics["minimum_branch_dissipation_MPa_per_s"]>=-1e-12 and summary.minimum_dissipation_MPa_per_s.min()>=-1e-12 and full_dma.loc[~full_dma.failed,"minimum_dissipation_MPa_per_s"].min()>=-1e-12,
        "isochoric_internal":max(row["maximum_detFv_error"] for row in convergence)<=1e-8,
        "convex_monotone_potential":convexity["monotone_slope"] and convexity["positive_slope"],
        "newton_convergence":summary.loc[summary.solver=="full_tensor_local_implicit","maximum_newton_residual"].max()<=1.01e-8,
        "time_step_terminal_stress_below_one_percent":convergence[1]["terminal_stress_relative_to_dt_quarter"]<=.01,
        "all_dense_values_finite":np.isfinite(curves[["time_s","strain","experimental_stress_MPa","predicted_stress_MPa","experimental_R","predicted_R","experimental_Rneq","predicted_Rneq"]]).all().all() and np.isfinite(frequency_frame.select_dtypes("number")).all().all() and np.isfinite(amplitude_frame.select_dtypes("number")).all().all(),
    }
    result={"schema_version":"2026-08-30-chapter3-stage3-final-validation-v1",
        "fixed_energy":True,"amplitude_labels_used_for_optimization":False,
        "source_workbook_sha256":raw["source_workbook_sha256"],
        "checkpoints":{"stage2":{"path":str(STAGE2),"sha256":sha256(STAGE2)},
                       "stage3":{"path":str(STAGE3),"sha256":sha256(STAGE3)}},
        "timings":timings,"physics":physics,"convexity":convexity,
        "time_step_convergence":convergence,"cost":cost,"acceptance":acceptance,
        "passed":all(acceptance.values()),
        "full_tensor_dma":{"case_count":len(full_dma),"failure_count":int(full_dma.failed.sum()),
            "steady_count":int(full_dma.loc[~full_dma.failed,"steady"].sum())}}
    encoder=lambda value: value.item() if isinstance(value,np.generic) else str(value)
    (OUTPUT/"stage3_final_validation_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,default=encoder),encoding="utf-8")
    print(json.dumps({"passed":result["passed"],"acceptance":acceptance,
        "full_tensor_dma":result["full_tensor_dma"],"timings":timings},ensure_ascii=False,indent=2,default=encoder))


if __name__=="__main__":main()
