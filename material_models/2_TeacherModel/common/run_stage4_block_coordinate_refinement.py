"""Two-cycle block-coordinate refinement of T1/T2 one-term Ogden models.

Order per cycle: Ogden energy -> flow -> softmax branch spectrum -> narrow
joint cloud.  Every accepted checkpoint is replayed by the production implicit
integrator.  The held-out histories are reported but never enter selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
ROOT = TEACHER.parent
for path in (HERE, TEACHER / "3_PRF" / "code", TEACHER / "4_GroupedPRF" / "code",
             TEACHER / "5_T1_OgdenPRF" / "code", TEACHER / "6_T2_GroupedOgdenPRF" / "code"):
    sys.path.insert(0, str(path))

from comparison_core import ZhangLi, load_config, load_relaxation
from run_stage3_ogden_flow_calibration import (
    BOUNDS, DECADE_EDGES, PHASES, build_model, clean_json, fast_history,
    file_record, flow_to_vector, flows_from_vector, make_records,
    objective_components, save_figure, scenario_from_stress, search_bounds,
    segment_mse, write_json,
)
from teacher_model.solvers import simulate_history


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 8,
                     "axes.spines.right": False, "axes.spines.top": False,
                     "legend.frameon": False})

G_BOUNDS = (0.01, 1.0)
A_BOUNDS = (0.5, 8.0)
SPECTRUM_LOG_RADIUS = (1.0, 1.35)
SPECTRUM_KL_WEIGHT = 0.02
SPECTRUM_LOG_WEIGHT = 0.002


def stage3_path(model_name: str) -> Path:
    return (TEACHER / "results_ogden_teacher_models" / "stage3_flow_refit" /
            "parameters" / f"{model_name}_Ogden_1_stage3_flow_parameters.json")


def softmax(logits):
    z = np.asarray(logits, float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def full_ordering(records, q_values):
    rows = []
    correct = total = 0
    increasing = True
    q_values = np.asarray(q_values, float)
    for mode in ("uniaxial", "shear"):
        chosen = sorted([i for i, record in enumerate(records) if record["mode"] == mode],
                        key=lambda i: records[i]["strain"])
        differences = np.diff(q_values[chosen])
        increasing = increasing and bool(np.all(differences > 0.0))
        correct += int(np.sum(differences > 0.0))
        total += len(differences)
        rows.append({"mode": mode, "strictly_increasing": bool(np.all(differences > 0.0))})
    return {"full_ordering_accuracy": correct / max(total, 1),
            "full_strictly_increasing_both": increasing, "mode_ordering": rows}


def total_relaxation_mse(records, stresses, split="training"):
    values = []
    for record, stress in zip(records, stresses):
        if record["split"] != split:
            continue
        peak = record["peak_index"]
        model_hold = np.asarray(stress, float)[peak:]
        exp_hold = record["experimental_stress_MPa"]
        model_R = model_hold / max(abs(model_hold[0]), 1e-14)
        exp_R = exp_hold / max(abs(exp_hold[0]), 1e-14)
        values.append(float(np.mean((model_R - exp_R) ** 2)))
    return float(np.mean(values)) if values else 0.0


def refinement_score(components, total_R_mse=0.0, spectrum_penalty=0.0):
    violation = 1.0 - components["q_ordering_accuracy"]
    return (0.40 * components["Rneq_balanced_mse"] +
            0.35 * components["q_log_mse"] +
            0.10 * components["q_trend_mse"] +
            0.10 * components["q_ordering_loss"] +
            0.05 * total_R_mse + 0.50 * violation + spectrum_penalty)


def config_hash(payload):
    raw = json.dumps(clean_json(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _record_run(task):
    model_name, payload, record, strict = task
    try:
        model = build_model(model_name, payload, payload["flows"])
        if strict:
            output = simulate_history(model, record["history_time_s"],
                                      record["history_strain"], record["mode"],
                                      max_scalar_increment=0.025, return_states=True)
            stress = output["nominal_stress_MPa"]
            diagnostics = output["diagnostics"]
            max_det = max((abs(np.linalg.det(Fv) - 1.0)
                           for state in output["states"] for Fv in state.Fv), default=0.0)
            min_dissipation = min((float(d["dissipation_MPa_per_s"])
                                   for d in diagnostics), default=0.0)
            max_residual = max((float(d["residual"]) for d in diagnostics), default=0.0)
        else:
            stress, state, _ = fast_history(model, record["history_time_s"],
                                            record["history_strain"], record["mode"])
            max_det = max((abs(np.linalg.det(Fv) - 1.0) for Fv in state.Fv), default=0.0)
            min_dissipation = 0.0
            max_residual = 0.0
        q, shape = scenario_from_stress(record, stress)
        return {"success": True, "q": q, "shape": shape, "stress": stress,
                "max_detFv_error": max_det, "min_dissipation": min_dissipation,
                "max_residual": max_residual}
    except Exception as error:
        return {"success": False, "error": repr(error)}


def evaluate_many(model_name, payloads, records, strict, workers):
    unique = []
    seen = set()
    for payload in payloads:
        key = config_hash(payload)
        if key not in seen:
            unique.append(payload); seen.add(key)
    tasks = []
    for config_index, payload in enumerate(unique):
        for record_index, record in enumerate(records):
            tasks.append((config_index, record_index,
                          (model_name, payload, record, strict)))
    collected = [[None] * len(records) for _ in unique]
    if strict:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(_record_run, task): (ci, ri)
                          for ci, ri, task in tasks}
            for future in as_completed(future_map):
                ci, ri = future_map[future]
                collected[ci][ri] = future.result()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(_record_run, task): (ci, ri)
                          for ci, ri, task in tasks}
            for future in as_completed(future_map):
                ci, ri = future_map[future]
                collected[ci][ri] = future.result()

    results = []
    for payload, rows in zip(unique, collected):
        if any(row is None or not row.get("success", False) for row in rows):
            results.append({"success": False, "score": 1e6, "payload": payload,
                            "error": "; ".join(row.get("error", "missing")
                                                for row in rows if row is None or not row.get("success", False))})
            continue
        q = [row["q"] for row in rows]
        shapes = [row["shape"] for row in rows]
        stresses = [row["stress"] for row in rows]
        components = objective_components(records, q, shapes)
        total_R = total_relaxation_mse(records, stresses)
        ordering = full_ordering(records, q)
        results.append({"success": True, "payload": payload, "q": q,
                        "shapes": shapes, "stresses": stresses,
                        "components": components, "total_R_mse": total_R,
                        "score": refinement_score(components, total_R), **ordering,
                        "max_detFv_error": max(row["max_detFv_error"] for row in rows),
                        "min_dissipation": min(row["min_dissipation"] for row in rows),
                        "max_residual": max(row["max_residual"] for row in rows)})
    return results


def flatten_result(result, model_name, cycle, block, accepted, runtime_s):
    row = {"model": model_name, "cycle": cycle, "block": block,
           "accepted": accepted, "runtime_s": runtime_s,
           "score": result["score"], "Gneq_MPa": result["payload"]["Gneq_MPa"],
           "a_ogden": result["payload"]["a_ogden"],
           "full_ordering_accuracy": result.get("full_ordering_accuracy"),
           "full_strictly_increasing_both": result.get("full_strictly_increasing_both"),
           "total_R_mse": result.get("total_R_mse"),
           "max_detFv_error": result.get("max_detFv_error"),
           "min_dissipation": result.get("min_dissipation"),
           "max_residual": result.get("max_residual")}
    row.update(result.get("components", {}))
    for index, value in enumerate(result["payload"]["p_alpha"]):
        row[f"p_{index+1}"] = value
    for group, flow in result["payload"]["flows"].items():
        for name in ("nonlinear_multiplier", "s0_MPa", "m", "c", "h0"):
            row[f"{group}_{name}"] = flow[name]
    return row


def choose_strict(current, candidates, model_name, records, workers, block):
    started = time.time()
    strict = evaluate_many(model_name, [current["payload"]] + candidates,
                           records, True, workers)
    valid = [row for row in strict if row.get("success", False)]
    if not valid:
        return current, False, time.time() - started
    baseline = next((row for row in valid
                     if config_hash(row["payload"]) == config_hash(current["payload"])), current)
    admissible = [row for row in valid
                  if row["components"]["q_ordering_accuracy"] >=
                     baseline["components"]["q_ordering_accuracy"]]
    best = min(admissible or [baseline], key=lambda row: row["score"])
    accepted = best["score"] < baseline["score"] - 1e-8
    return (best if accepted else baseline), accepted, time.time() - started


def energy_candidates(current, model_name, records, cycle, workers):
    payloads = []
    deltas = (-0.8, -0.4, 0.0, 0.4, 0.8) if cycle == 1 else (-0.3, -0.15, 0.0, 0.15, 0.3)
    for delta in deltas:
        payload = copy.deepcopy(current["payload"])
        payload["a_ogden"] = float(np.clip(payload["a_ogden"] + delta, *A_BOUNDS))
        payloads.append(payload)
    screened = evaluate_many(model_name, payloads, records, False, workers)
    adjusted = []
    training = [i for i, record in enumerate(records) if record["split"] == "training"]
    for row in screened:
        if not row.get("success", False):
            continue
        q_model = np.asarray(row["q"])[training]
        q_exp = np.asarray([records[i]["experimental_q"] for i in training])
        scale = float(np.exp(np.mean(np.log(q_exp) - np.log(q_model))))
        payload = copy.deepcopy(row["payload"])
        payload["Gneq_MPa"] = float(np.clip(payload["Gneq_MPa"] * scale, *G_BOUNDS))
        adjusted.append(payload)
    rescored = evaluate_many(model_name, adjusted, records, False, workers)
    rescored = sorted([row for row in rescored if row.get("success", False)],
                      key=lambda row: row["score"])
    return [row["payload"] for row in rescored[:3]]


def flow_vector(payload, model_name):
    if model_name == "T1":
        return flow_to_vector(payload["flows"]["shared"])
    return np.r_[flow_to_vector(payload["flows"]["fast"]),
                 flow_to_vector(payload["flows"]["slow"])]


def flow_candidates(current, model_name, records, cycle, workers, seed):
    rng = np.random.default_rng(seed + cycle)
    groups = 1 if model_name == "T1" else 2
    lo, hi = search_bounds(groups)
    center = flow_vector(current["payload"], model_name)
    count = (40 if model_name == "T1" else 64) if cycle == 1 else (28 if model_name == "T1" else 44)
    radius = 0.055 if cycle == 1 else 0.028
    vectors = [center]
    vectors.extend(np.clip(center + rng.normal(size=len(center)) * radius * (hi - lo), lo, hi)
                   for _ in range(count))
    payloads = []
    for vector in vectors:
        payload = copy.deepcopy(current["payload"])
        payload["flows"] = flows_from_vector(model_name, vector)
        payloads.append(payload)
    screened = evaluate_many(model_name, payloads, records, False, workers)
    screened = sorted([row for row in screened if row.get("success", False)],
                      key=lambda row: row["score"])
    return [row["payload"] for row in screened[:4]]


def _branch_basis_worker(task):
    model_name, payload, record = task
    model = build_model(model_name, payload, payload["flows"])
    output = simulate_history(model, record["history_time_s"], record["history_strain"],
                              record["mode"], max_scalar_increment=0.025,
                              return_states=True)
    peak = record["peak_index"]
    contributions = []
    for i in range(len(record["hold_time_s"])):
        index = peak + i
        F = model.deformation(record["history_strain"][index], record["mode"])
        response = model.response(F, output["states"][index])
        row = []
        for sigma in response["branch_cauchy"]:
            if record["mode"] == "uniaxial":
                row.append(float((sigma[0, 0] - 0.5 * (sigma[1, 1] + sigma[2, 2])) / F[0, 0]))
            else:
                row.append(float(sigma[0, 1]))
        contributions.append(row)
    return np.asarray(contributions, float)


def spectrum_candidates(current, model_name, records, cycle, workers, seed):
    payload = current["payload"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        basis = list(pool.map(_branch_basis_worker,
                              [(model_name, payload, record) for record in records]))
    p_current = np.asarray(payload["p_alpha"], float)
    p_old = np.asarray(payload.get("stage4_reference_p", payload["p_alpha"]), float)
    p_old = p_old / np.sum(p_old)
    center_old = np.log(np.maximum(p_old, 1e-14))
    radius = SPECTRUM_LOG_RADIUS[cycle - 1]
    bounds = [(value - radius, value + radius) for value in center_old]

    def evaluate_logits(logits, return_parts=False):
        p = softmax(logits)
        q_values, shapes, stresses = [], [], []
        ratios = p / p_current
        for record, matrix in zip(records, basis):
            noneq = matrix @ ratios
            hold = record["equilibrium_MPa"] + noneq
            q = float(noneq[0] / record["equilibrium_MPa"])
            if q <= 0 or not np.all(np.isfinite(hold)):
                return (1e6, None) if return_parts else 1e6
            q_values.append(q)
            shapes.append(noneq / noneq[0])
            full = np.r_[np.full(record["peak_index"], np.nan), hold]
            stresses.append(full)
        components = objective_components(records, q_values, shapes)
        # R(t) can be calculated directly on the hold data here.
        r_losses = []
        for record, matrix in zip(records, basis):
            if record["split"] != "training":
                continue
            hold = record["equilibrium_MPa"] + matrix @ ratios
            model_R = hold / hold[0]
            exp_R = record["experimental_stress_MPa"] / record["experimental_stress_MPa"][0]
            r_losses.append(float(np.mean((model_R - exp_R) ** 2)))
        total_R = float(np.mean(r_losses))
        kl = float(np.sum(p * np.log(np.maximum(p, 1e-300) / np.maximum(p_old, 1e-300))))
        log_penalty = float(np.mean((np.log(np.maximum(p, 1e-14)) - center_old) ** 2))
        penalty = SPECTRUM_KL_WEIGHT * kl + SPECTRUM_LOG_WEIGHT * log_penalty
        score = refinement_score(components, total_R, penalty)
        parts = {"p": p, "components": components, "total_R_mse": total_R,
                 "kl": kl, "log_penalty": log_penalty, "score": score}
        return (score, parts) if return_parts else score

    rng = np.random.default_rng(seed + 100 + cycle)
    starts = [np.log(np.maximum(p_current, 1e-14))]
    starts.extend(np.clip(starts[0] + rng.normal(0, 0.15, len(p_current)),
                          np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1]) for _ in range(2))
    optimized = []
    for start in starts:
        fit = minimize(evaluate_logits, start, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 120, "ftol": 1e-11, "maxls": 25})
        _, parts = evaluate_logits(fit.x, return_parts=True)
        candidate = copy.deepcopy(payload)
        candidate["p_alpha"] = parts["p"].tolist()
        candidate["old_weights"] = parts["p"].tolist()
        candidate["stage4_spectrum_KL"] = parts["kl"]
        candidate["stage4_spectrum_log_penalty"] = parts["log_penalty"]
        optimized.append((parts["score"], candidate))
    optimized.sort(key=lambda item: item[0])
    return [candidate for _, candidate in optimized[:2]]


def joint_candidates(current, model_name, records, cycle, workers, seed):
    rng = np.random.default_rng(seed + 1000 + cycle)
    payload = current["payload"]
    groups = 1 if model_name == "T1" else 2
    lo, hi = search_bounds(groups)
    center_flow = flow_vector(payload, model_name)
    p_old = np.asarray(payload.get("stage4_reference_p", payload["p_alpha"]), float)
    old_logits = np.log(np.maximum(p_old, 1e-14))
    count = 28 if model_name == "T1" else 40
    radius = 0.018 if cycle == 1 else 0.010
    candidates = [copy.deepcopy(payload)]
    for _ in range(count):
        candidate = copy.deepcopy(payload)
        candidate["Gneq_MPa"] = float(np.clip(payload["Gneq_MPa"] * np.exp(rng.normal(0, 0.06)), *G_BOUNDS))
        candidate["a_ogden"] = float(np.clip(payload["a_ogden"] + rng.normal(0, 0.12), *A_BOUNDS))
        vector = np.clip(center_flow + rng.normal(size=len(center_flow)) * radius * (hi - lo), lo, hi)
        candidate["flows"] = flows_from_vector(model_name, vector)
        logits = np.log(np.maximum(payload["p_alpha"], 1e-14)) + rng.normal(0, 0.07, 7)
        logits = np.clip(logits, old_logits - SPECTRUM_LOG_RADIUS[cycle - 1],
                         old_logits + SPECTRUM_LOG_RADIUS[cycle - 1])
        shares = softmax(logits)
        candidate["p_alpha"] = shares.tolist(); candidate["old_weights"] = shares.tolist()
        candidates.append(candidate)
    screened = evaluate_many(model_name, candidates, records, False, workers)
    screened = sorted([row for row in screened if row.get("success", False)],
                      key=lambda row: row["score"])
    return [row["payload"] for row in screened[:4]]


def final_tables(model_name, result, records):
    metrics, histories = [], []
    for record, q, shape, stress in zip(records, result["q"], result["shapes"], result["stresses"]):
        peak = record["peak_index"]
        model_hold = np.asarray(stress)[peak:]
        exp_R = record["experimental_stress_MPa"] / record["experimental_stress_MPa"][0]
        model_R = model_hold / model_hold[0]
        error = shape - record["experimental_Rneq"]
        decade = segment_mse(record["hold_time_s"], error,
                             [(DECADE_EDGES[j], DECADE_EDGES[j + 1])
                              for j in range(len(DECADE_EDGES) - 1)])
        phase = segment_mse(record["hold_time_s"], error, PHASES)
        metrics.append({"model": f"{model_name}-Ogden-1", "mode": record["mode"],
                        "condition": record["condition"], "split": record["split"],
                        "strain": record["strain"], "experimental_q": record["experimental_q"],
                        "model_q": q, "q_log_error": np.log(q / record["experimental_q"]),
                        "Rneq_RMSE": np.sqrt(np.mean(error ** 2)),
                        "Rneq_decade_equal_RMSE": np.sqrt(decade),
                        "Rneq_phase_equal_RMSE": np.sqrt(phase),
                        "R_RMSE": np.sqrt(np.mean((model_R - exp_R) ** 2))})
        for i, hold_time in enumerate(record["hold_time_s"]):
            histories.append({"model": f"{model_name}-Ogden-1", "mode": record["mode"],
                              "condition": record["condition"], "split": record["split"],
                              "hold_time_s": hold_time,
                              "experimental_stress_MPa": record["experimental_stress_MPa"][i],
                              "model_stress_MPa": model_hold[i],
                              "experimental_R": exp_R[i], "model_R": model_R[i],
                              "experimental_Rneq": record["experimental_Rneq"][i],
                              "model_Rneq": shape[i]})
    return pd.DataFrame(metrics), pd.DataFrame(histories)


def plot_outputs(metrics, histories, spectra, block_history, figures):
    colors = ["#0F4D92", "#42949E", "#E28E2C", "#B64342"]
    for model in histories["model"].unique():
        for mode in ("uniaxial", "shear"):
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            data = histories[(histories["model"] == model) & (histories["mode"] == mode)]
            conditions = sorted(data["condition"].unique(), key=lambda x: float(str(x).replace("%", "")))
            for color, condition in zip(colors, conditions):
                row = data[data["condition"] == condition]
                ax.plot(row["hold_time_s"], row["experimental_Rneq"], "o", mfc="none", mew=.7,
                        ms=3, color=color, label=f"Experiment {condition}")
                ax.plot(row["hold_time_s"], row["model_Rneq"], "-", lw=1.5, color=color,
                        label=f"Model {condition}")
            ax.set(xlabel="Hold time (s)", ylabel=r"Normalized nonequilibrium relaxation $R_{neq}(t)$",
                   title=f"{model} · {'Uniaxial' if mode == 'uniaxial' else 'Simple shear'}")
            ax.set_xlim(left=0); ax.grid(alpha=.18); ax.legend(ncol=2, fontsize=6.5)
            save_figure(fig, figures / f"{model}_{mode}_stage4_Rneq")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), sharex="col")
    for row_index, model in enumerate(("T1-Ogden-1", "T2-Ogden-1")):
        for col_index, mode in enumerate(("uniaxial", "shear")):
            ax = axes[row_index, col_index]
            data = metrics[(metrics["model"] == model) & (metrics["mode"] == mode)].sort_values("strain")
            ax.plot(data["strain"], data["experimental_q"], "o-", color="#272727", label="Experiment")
            ax.plot(data["strain"], data["model_q"], "s-", color="#B64342", label="Stage 4")
            held = data["split"] == "held_out"
            ax.scatter(data.loc[held, "strain"], data.loc[held, "model_q"], s=60,
                       facecolors="none", edgecolors="#0F4D92", label="Held out")
            ax.set(xlabel="Engineering strain" if mode == "uniaxial" else "Shear strain",
                   ylabel=r"Normalized nonequilibrium peak $q$",
                   title=f"{model.split('-')[0]} · {'Uniaxial' if mode == 'uniaxial' else 'Simple shear'}")
            ax.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.01))
    save_figure(fig, figures / "stage4_peak_trend", rect=(0, 0, 1, .93))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for model, marker in (("T1", "o"), ("T2", "s")):
        row = spectra[spectra["model"] == model]
        ax.plot(row["tau_s"], row["p_initial"], "--", marker=marker, lw=1.2, label=f"{model} Initial")
        ax.plot(row["tau_s"], row["p_final"], "-", marker=marker, lw=1.5, label=f"{model} Stage 4")
    ax.set_xscale("log"); ax.set(xlabel=r"Branch time scale $\tau_\alpha$ / s", ylabel=r"Branch fraction $p_\alpha$")
    ax.grid(alpha=.18); ax.legend(ncol=2)
    save_figure(fig, figures / "stage4_branch_spectra")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for model, data in block_history.groupby("model"):
        ax.plot(np.arange(len(data)), data.score, "o-", label=model)
    ax.set(xlabel="Checkpoint index", ylabel="Joint objective", title="Two block-coordinate refinement rounds")
    ax.grid(alpha=.18); ax.legend()
    save_figure(fig, figures / "stage4_refinement_path")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--results-root", type=Path,
                        default=TEACHER / "results_ogden_teacher_models" / "stage4_block_coordinate")
    parser.add_argument("--figures-root", type=Path,
                        default=TEACHER / "figures_ogden_teacher_models" / "stage4_block_coordinate")
    args = parser.parse_args()
    if args.cycles < 2:
        raise ValueError("the protocol requires at least two cycles")
    results = args.results_root.resolve(); figures = args.figures_root.resolve()
    tables = results / "tables"; parameters = results / "parameters"
    for path in (tables, parameters, figures): path.mkdir(parents=True, exist_ok=True)

    config, config_path = load_config(HERE / "model_comparison_config.json")
    split_path = HERE / "model_comparison_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    records = make_records(load_relaxation(config["data"]["viscoelastic"]),
                           ZhangLi(**config["zhang_li"]),
                           set(split["relaxation"]["held_out"]))
    all_block_rows, final_results, spectra_rows = [], {}, []
    started_all = time.time()

    for model_name in ("T1", "T2"):
        payload = json.loads(stage3_path(model_name).read_text(encoding="utf-8"))
        payload["stage4_reference_p"] = list(payload["p_alpha"])
        current = evaluate_many(model_name, [payload], records, True, args.workers)[0]
        all_block_rows.append(flatten_result(current, model_name, 0, "stage3_start", True, 0.0))
        print(model_name, "start", current["score"], current["components"], flush=True)

        for cycle in range(1, args.cycles + 1):
            for block, generator in (
                ("energy", lambda: energy_candidates(current, model_name, records, cycle, args.workers)),
                ("flow", lambda: flow_candidates(current, model_name, records, cycle, args.workers,
                                                   args.seed + (0 if model_name == "T1" else 10000))),
                ("spectrum", lambda: spectrum_candidates(current, model_name, records, cycle,
                                                           args.workers, args.seed)),
                ("joint", lambda: joint_candidates(current, model_name, records, cycle, args.workers,
                                                     args.seed + (0 if model_name == "T1" else 10000)))):
                block_started = time.time()
                candidates = generator()
                chosen, accepted, strict_runtime = choose_strict(
                    current, candidates, model_name, records, args.workers, block)
                current = chosen
                runtime = time.time() - block_started
                all_block_rows.append(flatten_result(current, model_name, cycle, block,
                                                     accepted, runtime))
                print(model_name, "cycle", cycle, block, "accepted", accepted,
                      "score", current["score"], "G", current["payload"]["Gneq_MPa"],
                      "a", current["payload"]["a_ogden"], "strict_s", strict_runtime,
                      flush=True)
        final_results[model_name] = current
        p0 = np.asarray(payload["stage4_reference_p"], float)
        pf = np.asarray(current["payload"]["p_alpha"], float)
        for tau, initial, final in zip(current["payload"]["tau_s"], p0, pf):
            spectra_rows.append({"model": model_name, "tau_s": tau,
                                 "p_initial": initial, "p_final": final,
                                 "log_ratio": np.log(final / initial)})

    metric_frames, history_frames = [], []
    for model_name, result in final_results.items():
        metrics, histories = final_tables(model_name, result, records)
        metric_frames.append(metrics); history_frames.append(histories)
        payload = copy.deepcopy(result["payload"])
        final_p = np.asarray(payload["p_alpha"], float)
        reference_p = np.asarray(payload["stage4_reference_p"], float)
        final_log_ratio = np.log(final_p / reference_p)
        payload.update({"schema_version": "2026-08-27-stage4-block-coordinate-v1",
                        "stage": "two_cycle_block_coordinate_refinement",
                        "stage4_cycles": args.cycles, "stage4_final_score": result["score"],
                        "energy_refitted": True, "flow_refitted": True,
                        "branch_shares_refitted": True,
                        "branch_share_parameterization": "softmax",
                        "stage4_spectrum_KL": float(np.sum(final_p * final_log_ratio)),
                        "stage4_spectrum_log_penalty": float(np.mean(final_log_ratio ** 2)),
                        "stage4_spectrum_max_abs_log_ratio": float(np.max(np.abs(final_log_ratio))),
                        "spectrum_regularization": {"KL_weight": SPECTRUM_KL_WEIGHT,
                                                    "log_weight": SPECTRUM_LOG_WEIGHT,
                                                    "log_radius_by_cycle": SPECTRUM_LOG_RADIUS},
                        "held_out_conditions": list(split["relaxation"]["held_out"])})
        filename = f"{model_name}_Ogden_1_stage4_block_parameters.json"
        write_json(parameters / filename, payload)
        destination = TEACHER / ("5_T1_OgdenPRF" if model_name == "T1" else
                                 "6_T2_GroupedOgdenPRF") / "results" / "parameters"
        write_json(destination / filename, payload)

    metrics = pd.concat(metric_frames, ignore_index=True)
    histories = pd.concat(history_frames, ignore_index=True)
    block_history = pd.DataFrame(all_block_rows)
    spectra = pd.DataFrame(spectra_rows)
    metrics.to_csv(tables / "stage4_condition_metrics.csv", index=False, encoding="utf-8-sig")
    histories.to_csv(tables / "stage4_relaxation_histories.csv", index=False, encoding="utf-8-sig")
    block_history.to_csv(tables / "stage4_block_history.csv", index=False, encoding="utf-8-sig")
    spectra.to_csv(tables / "stage4_spectrum.csv", index=False, encoding="utf-8-sig")
    plot_outputs(metrics, histories, spectra, block_history, figures)

    aggregate = []
    for (model, split_name), data in metrics.groupby(["model", "split"]):
        aggregate.append({"model": model, "split": split_name,
                          "q_log_RMSE": float(np.sqrt(np.mean(data.q_log_error ** 2))),
                          "Rneq_RMSE": float(np.sqrt(np.mean(data.Rneq_RMSE ** 2))),
                          "Rneq_decade_equal_RMSE": float(np.sqrt(np.mean(data.Rneq_decade_equal_RMSE ** 2))),
                          "Rneq_phase_equal_RMSE": float(np.sqrt(np.mean(data.Rneq_phase_equal_RMSE ** 2))),
                          "R_RMSE": float(np.sqrt(np.mean(data.R_RMSE ** 2)))})
    summary = {"stage": "stage4_two_cycle_block_coordinate", "cycles": args.cycles,
               "elapsed_s": time.time() - started_all, "aggregate": aggregate,
               "final": {model: {"score": result["score"],
                                  "Gneq_MPa": result["payload"]["Gneq_MPa"],
                                  "a_ogden": result["payload"]["a_ogden"],
                                  "p_alpha": result["payload"]["p_alpha"],
                                  "flows": result["payload"]["flows"],
                                  "components": result["components"],
                                  "full_ordering_accuracy": result["full_ordering_accuracy"],
                                  "full_strictly_increasing_both": result["full_strictly_increasing_both"],
                                  "max_detFv_error": result["max_detFv_error"],
                                  "min_dissipation": result["min_dissipation"],
                                  "max_residual": result["max_residual"]}
                         for model, result in final_results.items()}}
    write_json(results / "stage4_summary.json", summary)
    manifest = {"stage": summary["stage"], "seed": args.seed,
                "inputs": [file_record(path) for path in
                           (Path(config["data"]["viscoelastic"]), config_path, split_path,
                            stage3_path("T1"), stage3_path("T2"), HERE / "ogden_prf_core.py",
                            Path(__file__))],
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "numpy": np.__version__, "pandas": pd.__version__,
                                "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
                "outputs": {"tables": sorted(str(path) for path in tables.glob("*.csv")),
                            "parameters": sorted(str(path) for path in parameters.glob("*.json")),
                            "figures": sorted(str(path) for path in figures.glob("*.*"))}}
    write_json(results / "stage4_reproducibility_manifest.json", manifest)
    print(json.dumps(clean_json(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
