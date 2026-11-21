"""Stage-3 flow calibration with fixed stage-2 Ogden energy and branch spectrum.

T1 retains one shared regularized hardening power-law flow. T2 retains the
frozen fast/slow grouping.  The primary objective balances Rneq by complete
condition, logarithmic time decade, and early/middle/late relaxation phase.
Peak q and its ordering are protective secondary terms.
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
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import qmc, spearmanr

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
ROOT = TEACHER.parent
for path in (HERE, TEACHER / "3_PRF" / "code", TEACHER / "4_GroupedPRF" / "code",
             TEACHER / "5_T1_OgdenPRF" / "code", TEACHER / "6_T2_GroupedOgdenPRF" / "code"):
    sys.path.insert(0, str(path))

from comparison_core import ZhangLi, equilibrium_stress, load_config, load_relaxation
from t1_ogden.model import build_model as build_t1
from t2_ogden.model import build_model as build_t2
from teacher_model.model import ModelState
from teacher_model.solvers import simulate_history


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 8,
                     "axes.spines.right": False, "axes.spines.top": False,
                     "legend.frameon": False})

DECADE_EDGES = np.array([0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, np.inf])
PHASES = ((0.0, 1.0, "early"), (1.0, 100.0, "middle"),
          (100.0, np.inf, "late"))
FLOW_NAMES = ("nonlinear_multiplier", "s0_MPa", "m", "c", "h0")
BOUNDS = {
    "nonlinear_multiplier": (0.0, 50.0),
    "s0_MPa": (0.01, 2.0),
    "m": (1.05, 5.0),
    # Strong accumulated-strain hardening is retained as an admissible screen;
    # c<=0 keeps the multiplier nonnegative and nonincreasing with h.
    "c": (-20.0, 0.0),
    "h0": (0.005, 1.0),
}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2),
                    encoding="utf-8")


def file_record(path: Path) -> dict:
    path = path.resolve(); stat = path.stat()
    return {"absolute_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": stat.st_size, "modified_time_ns": stat.st_mtime_ns}


def save_figure(fig, base: Path, rect=None):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.0, rect=rect)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def stage2_path(model_name: str) -> Path:
    directory = TEACHER / ("5_T1_OgdenPRF" if model_name == "T1" else
                           "6_T2_GroupedOgdenPRF") / "results" / "parameters"
    return directory / f"{model_name}_Ogden_1_stage2_parameters.json"


def flow_to_vector(flow: dict) -> np.ndarray:
    return np.array([np.log1p(flow["nonlinear_multiplier"]), np.log(flow["s0_MPa"]),
                     np.log(flow["m"]-1.0), flow["c"], np.log(flow["h0"])], float)


def vector_to_flow(vector) -> dict:
    x = np.asarray(vector, float)
    return {"gamma0_per_s": 0.0,
            "nonlinear_multiplier": float(np.expm1(x[0])),
            "s0_MPa": float(np.exp(x[1])),
            "m": float(1.0 + np.exp(x[2])),
            "c": float(x[3]), "h0": float(np.exp(x[4])),
            "q0_MPa": 1e-6, "flow_form": "timescale_preserving",
            "tau_transition_s": 1.0, "tau_gate_exponent": 4.0}


def search_bounds(groups=1):
    lo = np.array([np.log1p(BOUNDS["nonlinear_multiplier"][0]),
                   np.log(BOUNDS["s0_MPa"][0]), np.log(BOUNDS["m"][0]-1.0),
                   BOUNDS["c"][0], np.log(BOUNDS["h0"][0])])
    hi = np.array([np.log1p(BOUNDS["nonlinear_multiplier"][1]),
                   np.log(BOUNDS["s0_MPa"][1]), np.log(BOUNDS["m"][1]-1.0),
                   BOUNDS["c"][1], np.log(BOUNDS["h0"][1])])
    return np.tile(lo, groups), np.tile(hi, groups)


def flows_from_vector(model_name: str, vector) -> dict:
    x = np.asarray(vector, float)
    if model_name == "T1":
        return {"shared": vector_to_flow(x)}
    return {"fast": vector_to_flow(x[:5]), "slow": vector_to_flow(x[5:])}


def vector_from_flat_row(model_name: str, row) -> np.ndarray:
    groups = ("shared",) if model_name == "T1" else ("fast", "slow")
    values = []
    for group in groups:
        values.extend(flow_to_vector({
            "nonlinear_multiplier": float(row[f"{group}_nonlinear_multiplier"]),
            "s0_MPa": float(row[f"{group}_s0_MPa"]),
            "m": float(row[f"{group}_m"]), "c": float(row[f"{group}_c"]),
            "h0": float(row[f"{group}_h0"])}))
    return np.asarray(values)


def build_model(model_name: str, stage2: dict, flows: dict):
    common = (stage2["zhang_li"], stage2["tau_s"], stage2["p_alpha"],
              stage2["Gneq_MPa"], stage2["a_ogden"])
    if model_name == "T1":
        return build_t1(common[0], flows["shared"], common[1], common[2],
                        common[3], common[4])
    return build_t2(common[0], flows["fast"], flows["slow"], common[1], common[2],
                    common[3], common[4], stage2.get("transition_tau_s", 1.0))


def hold_landmarks(duration: float, per_decade=5) -> np.ndarray:
    positive = np.logspace(-2, np.log10(max(duration, 1e-2)),
                           max(2, int(np.ceil((np.log10(max(duration, 1e-2))+2)*per_decade))+1))
    return np.unique(np.r_[0.0, positive[positive <= duration], duration])


def make_records(curves, zhang_li: ZhangLi, held_out: set[str], per_decade=5):
    rows = []
    for curve in curves:
        peak = int(np.argmax(curve.strain)); peak_time = float(curve.time_s[peak])
        hold_raw = curve.time_s[peak:] - peak_time
        marks = hold_landmarks(float(hold_raw[-1]), per_decade)
        stress = np.interp(marks, hold_raw, curve.nominal_stress_MPa[peak:])
        equilibrium = float(equilibrium_stress(curve.mode, curve.strain[peak], zhang_li))
        q = float((stress[0]-equilibrium)/equilibrium)
        Rneq = (stress-equilibrium)/max(abs(stress[0]-equilibrium), 1e-12)
        history_t = np.r_[curve.time_s[:peak+1], peak_time+marks[1:]]
        history_x = np.r_[curve.strain[:peak+1], np.full(len(marks)-1, curve.strain[peak])]
        rows.append({"mode": curve.mode, "condition": curve.label,
                     "key": f"{curve.mode}:{curve.label}",
                     "split": "held_out" if f"{curve.mode}:{curve.label}" in held_out else "training",
                     "strain": float(curve.strain[peak]), "peak_index": peak,
                     "equilibrium_MPa": equilibrium, "experimental_q": q,
                     "hold_time_s": marks, "experimental_stress_MPa": stress,
                     "experimental_Rneq": Rneq, "history_time_s": history_t,
                     "history_strain": history_x})
    return rows


def symmetric_exponential(A: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5*(A+A.T))
    return (vectors*np.exp(values))@vectors.T


def fast_history(model, time_s, scalar_history, mode, increment_cap=0.18):
    """Capped midpoint exponential proxy; det(Fv)=1 by construction.

    This intentionally inexpensive map is used only to rank a broad candidate
    set.  A cap avoids thousands of explicit substeps for manifestly excessive
    old-flow candidates.  It is never used for parameter acceptance or reports.
    """
    t = np.asarray(time_s, float); x = np.asarray(scalar_history, float)
    state = ModelState.identity(len(model.parameters.branches))
    stress = [model.measured_stress(model.deformation(x[0], mode), state, mode)]
    capped = 0; total_updates = 0
    for k in range(1, len(t)):
        total_dt = float(t[k]-t[k-1])
        if total_dt <= 0:
            raise ValueError("non-increasing time")
        subdivisions = max(1, int(np.ceil(abs(x[k]-x[k-1])/0.02)))
        for j in range(subdivisions):
            h = total_dt/subdivisions
            fraction = (j+.5)/subdivisions
            xm = x[k-1]+(x[k]-x[k-1])*fraction
            Fm = model.deformation(xm, mode)
            rates = [model._branch(Fm, Fv, branch)[0]
                     for Fv, branch in zip(state.Fv, model.parameters.branches)]
            half = []
            for D, Fv in zip(rates, state.Fv):
                Z = .5*h*D; norm = float(np.linalg.norm(Z)); total_updates += 1
                if norm > .5*increment_cap:
                    Z *= (.5*increment_cap/norm); capped += 1
                half.append(symmetric_exponential(Z)@Fv)
            middle = [model._branch(Fm, Fv, branch)[0]
                      for Fv, branch in zip(half, model.parameters.branches)]
            updated = []
            for D, Fv in zip(middle, state.Fv):
                Z = h*D; norm = float(np.linalg.norm(Z)); total_updates += 1
                if norm > increment_cap:
                    Z *= increment_cap/norm; capped += 1
                updated.append(symmetric_exponential(Z)@Fv)
            state = ModelState(updated)
        stress.append(model.measured_stress(model.deformation(x[k], mode), state, mode))
    return np.asarray(stress), state, capped/max(total_updates, 1)


def scenario_from_stress(record: dict, stress: np.ndarray):
    peak = record["peak_index"]
    hold = stress[peak:]
    noneq_peak = float(hold[0]-record["equilibrium_MPa"])
    if noneq_peak <= 1e-10 or not np.all(np.isfinite(hold)):
        raise FloatingPointError("nonpositive/nonfinite nonequilibrium peak")
    return noneq_peak/record["equilibrium_MPa"], (
        hold-record["equilibrium_MPa"])/noneq_peak


def segment_mse(time_s, error, edges):
    values = []
    for left, right, *_ in edges:
        mask = (time_s >= left) & (time_s < right)
        if np.any(mask):
            values.append(float(np.mean(error[mask]**2)))
    return float(np.mean(values)) if values else 0.0


def objective_components(records, q_model, shapes, include_split="training"):
    idx = [i for i, row in enumerate(records) if row["split"] == include_split]
    curve_losses, decade_losses, phase_losses = [], [], []
    for i in idx:
        error = shapes[i]-records[i]["experimental_Rneq"]
        curve_losses.append(float(np.mean(error**2)))
        decade_losses.append(segment_mse(records[i]["hold_time_s"], error,
                                         [(DECADE_EDGES[j], DECADE_EDGES[j+1])
                                          for j in range(len(DECADE_EDGES)-1)]))
        phase_losses.append(segment_mse(records[i]["hold_time_s"], error, PHASES))
    q_exp = np.asarray([row["experimental_q"] for row in records])
    q_model = np.asarray(q_model)
    q_log = float(np.mean((np.log(q_model[idx])-np.log(q_exp[idx]))**2))
    trend, ordering = [], []
    ordering_pairs = 0; ordering_correct = 0; mode_increasing = []
    for mode in ("uniaxial", "shear"):
        chosen = sorted([i for i in idx if records[i]["mode"] == mode],
                        key=lambda i: records[i]["strain"])
        for left, right in zip(chosen[:-1], chosen[1:]):
            dm = np.log(q_model[right])-np.log(q_model[left])
            de = np.log(q_exp[right])-np.log(q_exp[left])
            trend.append((dm-de)**2)
            ordering.append(max(0.0, 0.02-dm)**2)
            ordering_pairs += 1
            ordering_correct += int(dm > 0)
        mode_increasing.append(bool(np.all(np.diff(q_model[chosen]) > 0)))
    shape = (float(np.mean(curve_losses))+float(np.mean(decade_losses))+
             float(np.mean(phase_losses)))/3.0
    components = {"Rneq_condition_mse": float(np.mean(curve_losses)),
                  "Rneq_decade_equal_mse": float(np.mean(decade_losses)),
                  "Rneq_phase_equal_mse": float(np.mean(phase_losses)),
                  "Rneq_balanced_mse": shape, "q_log_mse": q_log,
                  "q_trend_mse": float(np.mean(trend)),
                  "q_ordering_loss": float(np.mean(ordering)),
                  "q_ordering_accuracy": ordering_correct/max(ordering_pairs, 1),
                  "q_strictly_increasing_both": bool(all(mode_increasing))}
    violation = 1.0-components["q_ordering_accuracy"]
    components["objective"] = (0.60*shape + 0.15*q_log +
                               0.10*components["q_trend_mse"] +
                               0.15*components["q_ordering_loss"] +
                               0.30*violation)
    return components


def evaluate(model_name: str, vector, records, stage2, strict=False):
    started = time.time(); result = {"model": model_name, "success": False}
    try:
        flows = flows_from_vector(model_name, vector)
        model = build_model(model_name, stage2, flows)
        q_values, shapes, states, cap_fractions = [], [], [], []
        for record in records:
            if strict:
                output = simulate_history(model, record["history_time_s"],
                                          record["history_strain"], record["mode"],
                                          max_scalar_increment=0.025, return_states=True)
                stress = output["nominal_stress_MPa"]; state = output["states"][-1]
                cap_fraction = 0.0
            else:
                stress, state, cap_fraction = fast_history(
                    model, record["history_time_s"], record["history_strain"], record["mode"])
            q, shape = scenario_from_stress(record, stress)
            q_values.append(q); shapes.append(shape); states.append(state)
            cap_fractions.append(cap_fraction)
        components = objective_components(records, q_values, shapes)
        result.update({"success": True, **components, "q": q_values,
                       "shapes": shapes, "flows": flows,
                       "surrogate_cap_fraction": float(np.mean(cap_fractions)),
                       "max_detFv_error": float(max(abs(np.linalg.det(Fv)-1.0)
                           for state in states for Fv in state.Fv))})
    except Exception as error:
        result.update({"objective": 1e6, "error": repr(error)})
    result["vector"] = np.asarray(vector, float)
    result["elapsed_s"] = time.time()-started
    return result


def _worker(task):
    return evaluate(*task)


def anchors(model_name: str, stage2: dict):
    old = stage2["flows"]
    if model_name == "T1":
        base = flow_to_vector(old["shared"])
        result = [base]
        for multiplier in (0.0, 1.0, 5.0, 15.0):
            for c in (-2.0, -8.0, -16.0):
                flow = copy.deepcopy(old["shared"])
                flow["nonlinear_multiplier"] = multiplier; flow["c"] = c
                result.append(flow_to_vector(flow))
        return result
    base = np.r_[flow_to_vector(old["fast"]), flow_to_vector(old["slow"])]
    result = [base]
    for fast_A, slow_A, c in ((0, 0, -2), (1, 5, -8), (5, 15, -12),
                              (10, 30, -16), (2, 25, -20)):
        fast = copy.deepcopy(old["fast"]); slow = copy.deepcopy(old["slow"])
        fast.update(nonlinear_multiplier=fast_A, c=c)
        slow.update(nonlinear_multiplier=slow_A, c=c)
        result.append(np.r_[flow_to_vector(fast), flow_to_vector(slow)])
    return result


def screen(model_name, records, stage2, count, seeds, workers):
    groups = 1 if model_name == "T1" else 2
    lo, hi = search_bounds(groups)
    candidates = anchors(model_name, stage2)
    for seed in seeds:
        unit = qmc.LatinHypercube(d=len(lo), seed=seed).random(count)
        candidates.extend(qmc.scale(unit, lo, hi))
    tasks = [(model_name, np.clip(x, lo, hi), records, stage2, False) for x in candidates]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result(); rows.append(row)
            if i % max(1, len(futures)//8) == 0:
                best = min(rows, key=lambda item: item["objective"])
                print(model_name, i, len(futures), "best", best["objective"], flush=True)
    rows.sort(key=lambda item: item["objective"])
    return rows


def local_cloud_search(model_name, center, records, stage2, count, seed, workers,
                       radius=0.08):
    groups = 1 if model_name == "T1" else 2
    lo, hi = search_bounds(groups); rng = np.random.default_rng(seed)
    candidates = np.clip(np.asarray(center)[None, :]+rng.normal(size=(count, len(lo)))*
                         radius*(hi-lo)[None, :], lo, hi)
    tasks = [(model_name, x, records, stage2, False) for x in candidates]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(_worker, task) for task in tasks]):
            rows.append(future.result())
    rows.sort(key=lambda item: item["objective"])
    return rows


def strict_validation(model_name, candidates, records, stage2):
    rows = []
    for rank, candidate in enumerate(candidates, 1):
        result = evaluate(model_name, candidate["vector"], records, stage2, strict=True)
        result["surrogate_rank"] = rank
        result["surrogate_objective"] = candidate["objective"]
        rows.append(result)
        print(model_name, "strict", rank, result["objective"], flush=True)
    rows.sort(key=lambda item: (not item.get("q_strictly_increasing_both", False),
                                item["objective"]))
    return rows


def scalar_row(row: dict, tag: str, rank: int):
    output = {"tag": tag, "rank": rank}
    for key, value in row.items():
        if key not in ("q", "shapes", "flows", "vector"):
            output[key] = value
    if "flows" in row:
        for group, flow in row["flows"].items():
            for name in FLOW_NAMES:
                output[f"{group}_{name}"] = flow[name]
    return output


def exact_histories(model_name, selected, records, stage2):
    model = build_model(model_name, stage2, selected["flows"])
    histories, metrics, diagnostics = [], [], []
    for record in records:
        output = simulate_history(model, record["history_time_s"], record["history_strain"],
                                  record["mode"], max_scalar_increment=0.025,
                                  return_states=True)
        stress = output["nominal_stress_MPa"]; peak = record["peak_index"]
        q, shape = scenario_from_stress(record, stress)
        error = shape-record["experimental_Rneq"]
        decade = segment_mse(record["hold_time_s"], error,
                             [(DECADE_EDGES[j], DECADE_EDGES[j+1])
                              for j in range(len(DECADE_EDGES)-1)])
        phase = segment_mse(record["hold_time_s"], error, PHASES)
        rho = float(spearmanr(record["hold_time_s"], shape).statistic)
        metrics.append({"model": f"{model_name}-Ogden-1", "mode": record["mode"],
                        "condition": record["condition"], "split": record["split"],
                        "strain": record["strain"], "experimental_q": record["experimental_q"],
                        "model_q": q, "q_log_error": np.log(q/record["experimental_q"]),
                        "Rneq_RMSE": np.sqrt(np.mean(error**2)),
                        "Rneq_decade_equal_RMSE": np.sqrt(decade),
                        "Rneq_phase_equal_RMSE": np.sqrt(phase),
                        "Rneq_time_spearman": rho})
        for i, time_value in enumerate(record["hold_time_s"]):
            histories.append({"model": f"{model_name}-Ogden-1", "mode": record["mode"],
                              "condition": record["condition"], "split": record["split"],
                              "hold_time_s": time_value,
                              "experimental_stress_MPa": record["experimental_stress_MPa"][i],
                              "model_stress_MPa": stress[peak+i],
                              "experimental_Rneq": record["experimental_Rneq"][i],
                              "model_Rneq": shape[i]})
        diagnostics.append({"model": model_name, "mode": record["mode"],
                            "condition": record["condition"],
                            "max_detFv_error": max(abs(np.linalg.det(Fv)-1)
                                                   for state in output["states"] for Fv in state.Fv),
                            "min_dissipation": min(item["dissipation_MPa_per_s"]
                                                   for item in output["diagnostics"]),
                            "max_residual": max(item["residual"] for item in output["diagnostics"])})
    return pd.DataFrame(metrics), pd.DataFrame(histories), pd.DataFrame(diagnostics)


def plot_rneq(histories, figures):
    colors = ["#0F4D92", "#42949E", "#E28E2C", "#B64342"]
    for model in histories["model"].unique():
        for mode in ("uniaxial", "shear"):
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            data = histories[(histories["model"] == model) & (histories["mode"] == mode)]
            conditions = sorted(data["condition"].unique(),
                                key=lambda value: float(str(value).replace("%", "")))
            for color, condition in zip(colors, conditions):
                row = data[data["condition"] == condition]
                ax.plot(row["hold_time_s"], row["experimental_Rneq"], "o", mfc="none",
                        mew=.7, ms=3, color=color, label=f"Experiment {condition}")
                ax.plot(row["hold_time_s"], row["model_Rneq"], "-", lw=1.5,
                        color=color, label=f"Model {condition}")
            ax.set(xlabel="Hold time (s)", ylabel=r"Normalized nonequilibrium relaxation $R_{neq}(t)$",
                   title=f"{model} · {'Uniaxial' if mode == 'uniaxial' else 'Simple shear'}")
            ax.set_xlim(left=0); ax.grid(alpha=.18); ax.legend(ncol=2, fontsize=6.5)
            save_figure(fig, figures/f"{model}_{mode}_stage3_Rneq")


def plot_q(metrics, figures):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), sharex="col")
    for row_index, model in enumerate(("T1-Ogden-1", "T2-Ogden-1")):
        for col_index, mode in enumerate(("uniaxial", "shear")):
            ax = axes[row_index, col_index]
            data = metrics[(metrics["model"] == model) & (metrics["mode"] == mode)].sort_values("strain")
            ax.plot(data["strain"], data["experimental_q"], "o-", color="#272727",
                    lw=1.5, ms=4, label="Experiment")
            ax.plot(data["strain"], data["model_q"], "s-", color="#B64342",
                    lw=1.5, ms=4, label="Stage 3 model")
            held = data["split"] == "held_out"
            ax.scatter(data.loc[held, "strain"], data.loc[held, "model_q"], s=60,
                       facecolors="none", edgecolors="#0F4D92", lw=1.1, label="Held-out condition")
            ax.set(xlabel="Engineering strain" if mode == "uniaxial" else "Shear strain",
                   ylabel=r"Normalized nonequilibrium peak $q$",
                   title=f"{model.split('-')[0]} · {'Uniaxial' if mode == 'uniaxial' else 'Simple shear'}")
            ax.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.01))
    save_figure(fig, figures/"stage3_peak_trend", rect=(0, 0, 1, .93))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-t1", type=int, default=24)
    parser.add_argument("--global-t2", type=int, default=48)
    parser.add_argument("--local", type=int, default=32)
    parser.add_argument("--strict-top", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260827, 20260828, 20260829])
    parser.add_argument("--results-root", type=Path,
                        default=TEACHER/"results_ogden_teacher_models"/"stage3_flow_refit")
    parser.add_argument("--figures-root", type=Path,
                        default=TEACHER/"figures_ogden_teacher_models"/"stage3_flow_refit")
    parser.add_argument("--finalize-from-pareto", type=Path, default=None,
                        help="Skip search and select stable q-increasing candidates from this strict table.")
    args = parser.parse_args()
    results = args.results_root.resolve(); figures = args.figures_root.resolve()
    tables = results/"tables"; parameters = results/"parameters"
    tables.mkdir(parents=True, exist_ok=True); parameters.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    config, config_path = load_config(HERE/"model_comparison_config.json")
    split_path = HERE/"model_comparison_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    records = make_records(load_relaxation(config["data"]["viscoelastic"]),
                           ZhangLi(**config["zhang_li"]),
                           set(split["relaxation"]["held_out"]))
    all_search, all_strict, selected = [], [], {}
    started = time.time()
    if args.finalize_from_pareto is not None:
        pareto = pd.read_csv(args.finalize_from_pareto)
        for model_name in ("T1", "T2"):
            admissible = pareto[(pareto["model"] == model_name) &
                                pareto["q_strictly_increasing_both"].astype(bool) &
                                np.isfinite(pareto["Rneq_balanced_mse"])]
            best = admissible.sort_values(["Rneq_balanced_mse", "q_log_mse"]).iloc[0]
            stage2 = json.loads(stage2_path(model_name).read_text(encoding="utf-8"))
            result = evaluate(model_name, vector_from_flat_row(model_name, best),
                              records, stage2, strict=True)
            result["selection_source"] = str(args.finalize_from_pareto.resolve())
            result["surrogate_rank"] = int(best["surrogate_rank"])
            selected[model_name] = result
            print(model_name, "Pareto selected", result["surrogate_rank"],
                  result["Rneq_balanced_mse"], result["objective"], flush=True)
    else:
        for model_name, count in (("T1", args.global_t1), ("T2", args.global_t2)):
            stage2 = json.loads(stage2_path(model_name).read_text(encoding="utf-8"))
            global_rows = screen(model_name, records, stage2, count, args.seeds, args.workers)
            local_rows = local_cloud_search(model_name, global_rows[0]["vector"], records,
                                            stage2, args.local, args.seeds[0]+100,
                                            args.workers)
            combined = sorted(global_rows+local_rows, key=lambda item: item["objective"])
            strict_rows = strict_validation(model_name, combined[:args.strict_top], records, stage2)
            selected[model_name] = strict_rows[0]
            all_search.extend([scalar_row(row, "surrogate", rank+1)
                               for rank, row in enumerate(combined)])
            all_strict.extend([scalar_row(row, "strict", rank+1)
                               for rank, row in enumerate(strict_rows)])
            print(model_name, "selected", strict_rows[0]["objective"], strict_rows[0]["flows"],
                  flush=True)
        pd.DataFrame(all_search).to_csv(tables/"stage3_surrogate_search.csv", index=False,
                                        encoding="utf-8-sig")
        pd.DataFrame(all_strict).to_csv(tables/"stage3_strict_candidates.csv", index=False,
                                        encoding="utf-8-sig")
    metric_frames, history_frames, diagnostic_frames = [], [], []
    for model_name in ("T1", "T2"):
        stage2 = json.loads(stage2_path(model_name).read_text(encoding="utf-8"))
        metrics, histories, diagnostics = exact_histories(model_name, selected[model_name],
                                                          records, stage2)
        metric_frames.append(metrics); history_frames.append(histories)
        diagnostic_frames.append(diagnostics)
        payload = copy.deepcopy(stage2)
        payload.update({"schema_version": "2026-08-27-stage3-flow-refit-v1",
                        "stage": "fixed_ogden_flow_refit", "flows": selected[model_name]["flows"],
                        "stage3_objective": selected[model_name]["objective"],
                        "loss_weights": {"Rneq_balanced": .60, "q_log": .15,
                                         "q_trend": .10, "q_ordering": .15,
                                         "incorrect_pair_penalty": .30},
                        "selection_rule": "q strictly increasing in both modes, then minimum strict Rneq-balanced MSE",
                        "pareto_source": selected[model_name].get("selection_source"),
                        "energy_refitted": False, "branch_shares_refitted": False,
                        "flow_refitted": True})
        filename = f"{model_name}_Ogden_1_stage3_flow_parameters.json"
        write_json(parameters/filename, payload)
        destination = TEACHER/("5_T1_OgdenPRF" if model_name == "T1" else
                               "6_T2_GroupedOgdenPRF")/"results"/"parameters"
        write_json(destination/filename, payload)
    metrics = pd.concat(metric_frames, ignore_index=True)
    histories = pd.concat(history_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    metrics.to_csv(tables/"stage3_condition_metrics.csv", index=False, encoding="utf-8-sig")
    histories.to_csv(tables/"stage3_relaxation_histories.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(tables/"stage3_numerical_checks.csv", index=False, encoding="utf-8-sig")
    plot_rneq(histories, figures); plot_q(metrics, figures)

    aggregate = []
    for (model, split_name), data in metrics.groupby(["model", "split"]):
        aggregate.append({"model": model, "split": split_name,
                          "q_log_RMSE": float(np.sqrt(np.mean(data["q_log_error"]**2))),
                          "Rneq_RMSE": float(np.sqrt(np.mean(data["Rneq_RMSE"]**2))),
                          "Rneq_decade_equal_RMSE": float(np.sqrt(np.mean(data["Rneq_decade_equal_RMSE"]**2))),
                          "Rneq_phase_equal_RMSE": float(np.sqrt(np.mean(data["Rneq_phase_equal_RMSE"]**2)))})
    summary = {"stage": "stage3_fixed_ogden_flow_refit", "elapsed_s": time.time()-started,
               "selected": {model: {"objective": row["objective"], "flows": row["flows"],
                                     "max_detFv_error": row["max_detFv_error"],
                                     "surrogate_objective": row.get("surrogate_objective"),
                                     "Rneq_balanced_mse": row.get("Rneq_balanced_mse"),
                                     "q_log_mse": row.get("q_log_mse"),
                                     "q_ordering_accuracy": row.get("q_ordering_accuracy"),
                                     "surrogate_rank": row.get("surrogate_rank"),
                                     "selection_source": row.get("selection_source")}
                            for model, row in selected.items()},
               "aggregate": aggregate,
               "numerical_checks": {"max_detFv_error": float(diagnostics["max_detFv_error"].max()),
                                     "min_dissipation": float(diagnostics["min_dissipation"].min()),
                                     "max_newton_residual": float(diagnostics["max_residual"].max())}}
    write_json(results/"stage3_summary.json", summary)
    sources = [Path(config["data"]["viscoelastic"]), config_path, split_path,
               stage2_path("T1"), stage2_path("T2"), HERE/"ogden_prf_core.py", Path(__file__)]
    manifest = {"stage": summary["stage"], "inputs": [file_record(path) for path in sources],
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "numpy": np.__version__, "pandas": pd.__version__,
                                "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
                "outputs": {"tables": sorted(str(path) for path in tables.glob("*.csv")),
                            "parameters": sorted(str(path) for path in parameters.glob("*.json")),
                            "figures": sorted(str(path) for path in figures.glob("*.*"))}}
    write_json(results/"stage3_reproducibility_manifest.json", manifest)
    print(json.dumps(clean_json(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
