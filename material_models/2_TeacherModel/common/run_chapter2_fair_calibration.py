"""Unified T0--T1-Ogden--T2-Ogden chapter-2 fair calibration.

The runner implements the frozen stage-1--3 protocol.  Search and selection use
training conditions only.  Held-out relaxation histories and the 5--20 Hz band
are evaluated only after a representative has been selected.  Although the
shared workbook loader reads the amplitude-DMA sheet, those arrays are never
referenced by the search, selection, or validation calculations in this runner.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
for directory in (HERE, TEACHER / "2_QLV" / "code_paper",
                  TEACHER / "3_PRF" / "code", TEACHER / "4_GroupedPRF" / "code",
                  TEACHER / "5_T1_OgdenPRF" / "code",
                  TEACHER / "6_T2_GroupedOgdenPRF" / "code"):
    sys.path.insert(0, str(directory))

from comparison_core import (ZhangLi, equilibrium_shear_modulus,
                             equilibrium_stress, file_record, frequency_masks,
                             load_config, load_dma, load_relaxation,
                             qlv_piecewise_linear_basis)
from run_stage3_ogden_flow_calibration import (build_model, fast_history,
                                               make_records,
                                               scenario_from_stress)
from run_stage4_block_coordinate_refinement import softmax
from teacher_model.dma import simulate_dma
from teacher_model.solvers import simulate_history


DECADE_EDGES = np.array([0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, np.inf])
MODEL_ORDER = ("T0", "T1", "T2")


def scalarization_schedule(count: int, values: list[float]) -> list[float]:
    """Allocate exactly ``count`` starts across the frozen scalarizations."""
    return [float(values[index % len(values)]) for index in range(int(count))]


def physical_parameter_vector(model_name: str, row: dict) -> tuple[list[str], np.ndarray]:
    if model_name == "T0":
        values = np.asarray(row["weights"], float)
        return [f"w_{index + 1}" for index in range(len(values))], values
    payload = row["payload"]
    names = ["Gneq_MPa", "a_ogden"]
    values = [payload["Gneq_MPa"], payload["a_ogden"]]
    groups = ["shared"] if model_name == "T1" else ["fast", "slow"]
    for group in groups:
        flow = payload["flows"][group]
        for key in ("nonlinear_multiplier", "s0_MPa", "m", "c", "h0"):
            names.append(f"{group}_{key}")
            values.append(flow[key])
    for index, share in enumerate(payload["p_alpha"]):
        names.append(f"p_{index + 1}")
        values.append(share)
    return names, np.asarray(values, float)


def stability_summary(model_name: str, selected: dict, candidates: list[dict],
                      config: dict) -> dict:
    representatives = selected.get("seed_representatives", [])
    names, matrix = None, []
    seed_rows = []
    for row in representatives:
        current_names, vector = physical_parameter_vector(model_name, row)
        names = current_names if names is None else names
        matrix.append(vector)
        seed_rows.append({"seed": int(row["seed"]), "candidate_id": row["candidate_id"],
                          "L_time": row["L_time"], "L_frequency": row["L_frequency"],
                          "joint_score": joint_score(row),
                          **{name: value for name, value in zip(current_names, vector)}})
    matrix = np.asarray(matrix, float)
    mean = np.mean(matrix, axis=0)
    standard_deviation = np.std(matrix, axis=0, ddof=1) if len(matrix) > 1 else np.zeros(matrix.shape[1])
    cv = standard_deviation / np.maximum(np.abs(mean), 1e-12)
    if model_name == "T0":
        lower = np.full(7, config["joint"]["weight_lower_bound"])
        upper = np.full(7, config["joint"]["weight_upper_bound"])
        raw = matrix
    else:
        lower, upper = payload_bounds(model_name, config)
        raw = np.asarray([row["vector"] for row in representatives], float)
    span = np.maximum(upper-lower, 1e-14)
    boundary_hits = np.sum((raw-lower <= 0.01*span) | (upper-raw <= 0.01*span), axis=0)
    effective_counts = []
    for row in representatives:
        shares = (np.asarray(row["weights"], float) if model_name == "T0"
                  else np.asarray(row["payload"]["p_alpha"], float))
        shares = shares / np.sum(shares)
        effective_counts.append(float(np.exp(-np.sum(shares*np.log(np.maximum(shares, 1e-300))))))
    return {
        "model": model_name,
        "completed_independent_seeds": len(representatives),
        "parameter_names": names,
        "parameter_mean": mean,
        "parameter_standard_deviation": standard_deviation,
        "parameter_CV": cv,
        "raw_search_coordinate_boundary_hits": boundary_hits,
        "effective_branch_count_mean": float(np.mean(effective_counts)),
        "effective_branch_count_standard_deviation": float(np.std(effective_counts, ddof=1)) if len(effective_counts) > 1 else 0.0,
        "failed_candidates": int(sum(not row.get("success", False) for row in candidates)),
        "total_candidates": len(candidates),
        "seed_rows": seed_rows,
    }


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2),
                    encoding="utf-8")


def object_hash(value) -> str:
    raw = json.dumps(clean(value), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def segment_mse(time_s, error) -> float:
    values = []
    for left, right in zip(DECADE_EDGES[:-1], DECADE_EDGES[1:]):
        mask = (time_s >= left) & (time_s < right)
        if np.any(mask):
            values.append(float(np.mean(np.asarray(error)[mask] ** 2)))
    return float(np.mean(values)) if values else 0.0


def fair_time_metrics(records: list[dict], stresses: list[np.ndarray],
                      split: str | None, weights: dict) -> dict:
    indices = [index for index, record in enumerate(records)
               if split is None or record["split"] == split]
    if not indices:
        raise ValueError(f"no records for split={split}")
    q_model, q_exp = [], []
    curve_mse, decade_mse, total_mse, stress_mse, curve_rows = [], [], [], [], []
    for index in indices:
        record = records[index]
        stress = np.asarray(stresses[index], float)
        hold = stress[int(record["peak_index"]):]
        experimental = np.asarray(record["experimental_stress_MPa"], float)
        if len(hold) != len(experimental) or not np.all(np.isfinite(hold)):
            raise FloatingPointError(f"invalid predicted history for {record['key']}")
        equilibrium = float(record["equilibrium_MPa"])
        nonequilibrium_peak = float(hold[0] - equilibrium)
        if nonequilibrium_peak <= 1e-12 or equilibrium <= 0.0:
            raise FloatingPointError(f"nonpositive nonequilibrium peak for {record['key']}")
        q_value = nonequilibrium_peak / equilibrium
        q_model.append(q_value); q_exp.append(float(record["experimental_q"]))
        rneq_model = (hold - equilibrium) / nonequilibrium_peak
        rneq_exp = np.asarray(record["experimental_Rneq"], float)
        rneq_error = rneq_model - rneq_exp
        r_model = hold / max(abs(float(hold[0])), 1e-14)
        r_exp = experimental / max(abs(float(experimental[0])), 1e-14)
        curve_value = float(np.mean(rneq_error ** 2))
        decade_value = segment_mse(np.asarray(record["hold_time_s"], float), rneq_error)
        total_value = float(np.mean((r_model - r_exp) ** 2))
        stress_value = float(np.mean((hold - experimental) ** 2))
        curve_mse.append(curve_value); decade_mse.append(decade_value)
        total_mse.append(total_value); stress_mse.append(stress_value)
        curve_rows.append({
            "mode": record["mode"], "condition": record["condition"],
            "split": record["split"], "experimental_q": record["experimental_q"],
            "model_q": q_value, "Rneq_RMSE": np.sqrt(curve_value),
            "Rneq_decade_RMSE": np.sqrt(decade_value),
            "R_RMSE": np.sqrt(total_value), "stress_RMSE_MPa": np.sqrt(stress_value),
        })
    q_model = np.asarray(q_model, float); q_exp = np.asarray(q_exp, float)
    q_log_mse = float(np.mean((np.log(q_model) - np.log(q_exp)) ** 2))
    correct = total_pairs = 0
    for mode in ("uniaxial", "shear"):
        chosen = [position for position, index in enumerate(indices)
                  if records[index]["mode"] == mode]
        for left in range(len(chosen)):
            for right in range(left + 1, len(chosen)):
                i, j = chosen[left], chosen[right]
                experimental_difference = q_exp[j] - q_exp[i]
                if abs(experimental_difference) <= 1e-14:
                    continue
                model_difference = q_model[j] - q_model[i]
                correct += int(model_difference * experimental_difference > 0.0)
                total_pairs += 1
    ordering_accuracy = correct / max(total_pairs, 1)
    ordering_inversion_fraction = 1.0 - ordering_accuracy if total_pairs else 0.0
    components = {
        "q_log_mse": q_log_mse,
        "q_log_RMSE": float(np.sqrt(q_log_mse)),
        "q_ordering_accuracy": float(ordering_accuracy),
        "q_ordering_inversion_fraction": float(ordering_inversion_fraction),
        "Rneq_condition_mse": float(np.mean(curve_mse)),
        "Rneq_condition_RMSE": float(np.sqrt(np.mean(curve_mse))),
        "Rneq_decade_mse": float(np.mean(decade_mse)),
        "Rneq_decade_RMSE": float(np.sqrt(np.mean(decade_mse))),
        "R_condition_mse": float(np.mean(total_mse)),
        "R_condition_RMSE": float(np.sqrt(np.mean(total_mse))),
        "stress_mse_MPa2": float(np.mean(stress_mse)),
        "stress_RMSE_MPa": float(np.sqrt(np.mean(stress_mse))),
        "curve_metrics": curve_rows,
        "conditions": len(indices),
    }
    loss_squared = (
        float(weights["q_log_RMSE_weight"]) * components["q_log_mse"]
        + float(weights["q_ordering_inversion_weight"]) * ordering_inversion_fraction ** 2
        + float(weights["Rneq_condition_balanced_RMSE_weight"]) * components["Rneq_condition_mse"]
        + float(weights["Rneq_decade_balanced_RMSE_weight"]) * components["Rneq_decade_mse"]
        + float(weights["R_condition_balanced_RMSE_weight"]) * components["R_condition_mse"]
    )
    components["L_time"] = float(np.sqrt(max(loss_squared, 0.0)))
    return components


def frequency_scales(dma, train_mask, config) -> tuple[float, float]:
    floors = config["dma"]["normalization_floor_MPa"]
    storage = max(float(np.ptp(dma.frequency_storage_MPa[train_mask])),
                  float(floors["storage"]))
    loss = max(float(np.ptp(dma.frequency_loss_MPa[train_mask])),
               float(floors["loss"]))
    return storage, loss


def fair_frequency_metrics(dma, mask, storage, loss, scales) -> dict:
    storage_error = np.asarray(storage, float)[mask] - dma.frequency_storage_MPa[mask]
    loss_error = np.asarray(loss, float)[mask] - dma.frequency_loss_MPa[mask]
    gp_rmse = float(np.sqrt(np.mean(storage_error ** 2)))
    gpp_rmse = float(np.sqrt(np.mean(loss_error ** 2)))
    gp_nrmse, gpp_nrmse = gp_rmse / scales[0], gpp_rmse / scales[1]
    experimental_tan = dma.frequency_loss_MPa[mask] / dma.frequency_storage_MPa[mask]
    predicted_tan = np.asarray(loss, float)[mask] / np.maximum(np.asarray(storage, float)[mask], 1e-14)
    return {
        "storage_RMSE_MPa": gp_rmse,
        "loss_RMSE_MPa": gpp_rmse,
        "storage_NRMSE": float(gp_nrmse),
        "loss_NRMSE": float(gpp_nrmse),
        "tan_delta_RMSE": float(np.sqrt(np.mean((predicted_tan - experimental_tan) ** 2))),
        "L_frequency": float(np.sqrt(0.5 * gp_nrmse ** 2 + 0.5 * gpp_nrmse ** 2)),
        "points": int(np.sum(mask)),
    }


def joint_score(row: dict, time_weight=0.5) -> float:
    return float(np.sqrt(time_weight * float(row["L_time"]) ** 2
                         + (1.0 - time_weight) * float(row["L_frequency"]) ** 2))


def nondominated(rows: list[dict]) -> list[dict]:
    valid = [row for row in rows if row.get("success", False)
             and np.isfinite(row.get("L_time", np.inf))
             and np.isfinite(row.get("L_frequency", np.inf))]
    front = []
    for index, row in enumerate(valid):
        dominated = False
        for other_index, other in enumerate(valid):
            if index == other_index:
                continue
            if (other["L_time"] <= row["L_time"] and
                    other["L_frequency"] <= row["L_frequency"] and
                    (other["L_time"] < row["L_time"] or
                     other["L_frequency"] < row["L_frequency"])):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda item: item["L_time"])


def representative(rows: list[dict], time_weight=0.5) -> dict:
    front = nondominated(rows)
    if not front:
        raise RuntimeError("empty admissible Pareto front")
    return min(front, key=lambda row: (
        joint_score(row, time_weight), row.get("q_log_RMSE", np.inf),
        row.get("Rneq_condition_RMSE", np.inf), row.get("parameter_count", np.inf),
        row.get("runtime_s", np.inf)))


def build_records(config, split):
    curves = load_relaxation(config["data"]["viscoelastic"])
    zhang_li = ZhangLi(**config["zhang_li"])
    return make_records(curves, zhang_li, set(split["relaxation"]["held_out"]),
                        config["time_loss"]["points_per_decade"])


def build_t0_bases(records, zhang_li, tau):
    bases = []
    for record in records:
        equilibrium_history = equilibrium_stress(record["mode"],
                                                 record["history_strain"], zhang_li)
        bases.append(qlv_piecewise_linear_basis(record["history_time_s"],
                                                equilibrium_history, tau))
    return bases


def t0_histories(records, bases, weights, zhang_li):
    result = []
    for record, basis in zip(records, bases):
        equilibrium_history = equilibrium_stress(record["mode"],
                                                 record["history_strain"], zhang_li)
        result.append(np.asarray(equilibrium_history) + np.asarray(basis) @ weights)
    return result


def t0_frequency(dma, tau, weights, zhang_li):
    omega_tau = 2.0 * np.pi * dma.frequency_hz[:, None] * tau[None, :]
    g0 = equilibrium_shear_modulus(zhang_li)
    storage = g0 * (1.0 + (omega_tau ** 2 / (1.0 + omega_tau ** 2)) @ weights)
    loss = g0 * ((omega_tau / (1.0 + omega_tau ** 2)) @ weights)
    return storage, loss


def t0_fit(initial, lam, records, bases, dma, tau, zhang_li, train_mask,
           scales, time_weights, bounds, maxiter=160):
    log_bounds = [(np.log(bounds[0]), np.log(bounds[1]))] * 7

    def objective(log_weights):
        weights = np.exp(log_weights)
        stresses = t0_histories(records, bases, weights, zhang_li)
        time_metrics = fair_time_metrics(records, stresses, "training", time_weights)
        storage, loss = t0_frequency(dma, tau, weights, zhang_li)
        frequency_metrics = fair_frequency_metrics(dma, train_mask, storage, loss, scales)
        return lam * time_metrics["L_time"] ** 2 + (1.0-lam) * frequency_metrics["L_frequency"] ** 2

    fit = minimize(objective, np.log(np.clip(initial, *bounds)), method="L-BFGS-B",
                   bounds=log_bounds, options={"maxiter": maxiter, "ftol": 1e-12,
                                                "maxls": 30})
    weights = np.exp(fit.x)
    stresses = t0_histories(records, bases, weights, zhang_li)
    time_metrics = fair_time_metrics(records, stresses, "training", time_weights)
    storage, loss = t0_frequency(dma, tau, weights, zhang_li)
    frequency_metrics = fair_frequency_metrics(dma, train_mask, storage, loss, scales)
    return weights, fit, time_metrics, frequency_metrics


def _flow_search_bounds(config, groups):
    bounds = config["flow_bounds"]
    lo_one = np.array([np.log1p(bounds["nonlinear_multiplier"][0]),
                       np.log(bounds["s0_MPa"][0]), np.log(bounds["m"][0]-1.0),
                       bounds["c"][0], np.log(bounds["h0"][0])])
    hi_one = np.array([np.log1p(bounds["nonlinear_multiplier"][1]),
                       np.log(bounds["s0_MPa"][1]), np.log(bounds["m"][1]-1.0),
                       bounds["c"][1], np.log(bounds["h0"][1])])
    return np.tile(lo_one, groups), np.tile(hi_one, groups)


def _flow_to_vector(flow):
    return np.array([np.log1p(flow["nonlinear_multiplier"]), np.log(flow["s0_MPa"]),
                     np.log(flow["m"]-1.0), flow["c"], np.log(flow["h0"])], float)


def _vector_to_flow(vector):
    return {
        "gamma0_per_s": 0.0,
        "nonlinear_multiplier": float(np.expm1(vector[0])),
        "s0_MPa": float(np.exp(vector[1])),
        "m": float(1.0 + np.exp(vector[2])),
        "c": float(vector[3]),
        "h0": float(np.exp(vector[4])),
        "q0_MPa": 1e-6,
        "flow_form": "timescale_preserving",
        "tau_transition_s": 1.0,
        "tau_gate_exponent": 4.0,
    }


def payload_bounds(model_name, config):
    groups = 1 if model_name == "T1" else 2
    flow_lo, flow_hi = _flow_search_bounds(config, groups)
    G = config["ogden_bounds"]["Gneq_MPa"]
    exponent = config["ogden_bounds"]["a_ogden"]
    lower = np.r_[np.log(G[0]), exponent[0], flow_lo, np.full(6, -20.0)]
    upper = np.r_[np.log(G[1]), exponent[1], flow_hi, np.full(6, 20.0)]
    return lower, upper


def payload_to_vector(model_name, payload):
    flow = (_flow_to_vector(payload["flows"]["shared"])
            if model_name == "T1" else
            np.r_[_flow_to_vector(payload["flows"]["fast"]),
                  _flow_to_vector(payload["flows"]["slow"])])
    shares = np.asarray(payload["p_alpha"], float)
    alr = np.log(np.maximum(shares[:-1], 1e-300) / max(float(shares[-1]), 1e-300))
    return np.r_[np.log(payload["Gneq_MPa"]), payload["a_ogden"], flow, alr]


def vector_to_payload(model_name, vector, anchor, config):
    vector = np.asarray(vector, float)
    groups = 1 if model_name == "T1" else 2
    flow_vector = vector[2:2 + 5 * groups]
    logits = np.r_[vector[2 + 5 * groups:], 0.0]
    payload = {
        "schema_version": config["schema_version"],
        "model": f"{model_name}-Ogden-1",
        "zhang_li": copy.deepcopy(config["zhang_li"]),
        "tau_s": list(config["spectrum"]["tau_s"]),
        "p_alpha": softmax(logits).tolist(),
        "Gneq_MPa": float(np.exp(vector[0])),
        "a_ogden": float(vector[1]),
        "transition_tau_s": 1.0,
    }
    if model_name == "T1":
        payload["flows"] = {"shared": _vector_to_flow(flow_vector)}
    else:
        payload["flows"] = {
            "fast": _vector_to_flow(flow_vector[:5]),
            "slow": _vector_to_flow(flow_vector[5:]),
        }
    return payload


def local_complex_modulus(model_name, payload, frequency_hz, equilibrium_G0):
    omega = 2.0 * np.pi * np.asarray(frequency_hz, float)
    storage = np.full_like(omega, equilibrium_G0)
    loss = np.zeros_like(omega)
    for tau, share in zip(payload["tau_s"], payload["p_alpha"]):
        group = "shared" if model_name == "T1" else ("fast" if tau <= 1.0 else "slow")
        flow = payload["flows"][group]
        ratio = float(flow["q0_MPa"]) / max(float(flow["s0_MPa"]), 1e-14)
        multiplier = 1.0 + float(flow["nonlinear_multiplier"]) * ratio ** (float(flow["m"])-1.0)
        effective_tau = float(tau) / multiplier
        z = omega * effective_tau
        modulus = float(payload["Gneq_MPa"]) * float(share)
        storage += modulus * z * z / (1.0 + z * z)
        loss += modulus * z / (1.0 + z * z)
    return storage, loss


def _screen_worker(task):
    model_name, candidate, records, dma_payload, train_mask, scales, time_weights, equilibrium_G0 = task
    started = time.time()
    result = {key: candidate[key] for key in
              ("candidate_id", "seed", "stage", "budget_group", "vector", "payload")}
    result.update({"model": model_name, "success": False})
    try:
        payload = candidate["payload"]
        model = build_model(model_name, payload, payload["flows"])
        stresses = []
        cap_fractions = []
        for record in records:
            stress, _state, cap = fast_history(model, record["history_time_s"],
                                               record["history_strain"], record["mode"])
            stresses.append(stress); cap_fractions.append(cap)
        time_metrics = fair_time_metrics(records, stresses, "training", time_weights)
        class DMA:
            pass
        dma = DMA()
        for key, value in dma_payload.items():
            setattr(dma, key, np.asarray(value))
        storage, loss = local_complex_modulus(model_name, payload, dma.frequency_hz,
                                              equilibrium_G0)
        frequency_metrics = fair_frequency_metrics(dma, np.asarray(train_mask, bool),
                                                    storage, loss, scales)
        result.update({"success": True, **{k: v for k, v in time_metrics.items()
                                           if k != "curve_metrics"},
                       **frequency_metrics,
                       "surrogate_cap_fraction": float(np.mean(cap_fractions)),
                       "screen_joint_score": float(np.sqrt(0.5*time_metrics["L_time"]**2
                                                           +0.5*frequency_metrics["L_frequency"]**2))})
    except Exception as error:
        result["error"] = repr(error)
    result["runtime_s"] = float(time.time() - started)
    return result


def _strict_time_worker(task):
    model_name, row, records, time_weights = task
    started = time.time()
    result = {key: row[key] for key in ("candidate_id", "seed", "stage", "budget_group",
                                        "vector", "payload")}
    result.update({"model": model_name, "success": False})
    try:
        payload = row["payload"]
        model = build_model(model_name, payload, payload["flows"])
        stresses, minimum_dissipation = [], np.inf
        maximum_det, maximum_residual = 0.0, 0.0
        for record in records:
            output = simulate_history(model, record["history_time_s"],
                                      record["history_strain"], record["mode"],
                                      max_scalar_increment=0.025, return_states=True)
            stresses.append(output["nominal_stress_MPa"])
            maximum_det = max(maximum_det, max(
                (abs(np.linalg.det(Fv)-1.0) for state in output["states"] for Fv in state.Fv),
                default=0.0))
            minimum_dissipation = min(minimum_dissipation, min(
                (float(item["dissipation_MPa_per_s"]) for item in output["diagnostics"]),
                default=0.0))
            maximum_residual = max(maximum_residual, max(
                (float(item["residual"]) for item in output["diagnostics"]), default=0.0))
        train = fair_time_metrics(records, stresses, "training", time_weights)
        held = fair_time_metrics(records, stresses, "held_out", time_weights)
        full = fair_time_metrics(records, stresses, None, time_weights)
        result.update({"success": True,
                       **{k: v for k, v in train.items() if k != "curve_metrics"},
                       "training_time_metrics": train,
                       "held_out_time_metrics": held,
                       "all_time_metrics": full,
                       "stresses": stresses,
                       "max_detFv_error": float(maximum_det),
                       "min_dissipation_MPa_per_s": float(minimum_dissipation),
                       "max_residual": float(maximum_residual)})
    except Exception as error:
        result["error"] = repr(error)
    result["runtime_s"] = float(time.time() - started)
    return result


def _periodic_worker(task):
    model_name, candidate_id, payload, frequency, amplitude, ppc, min_cycles, max_cycles, tolerance = task
    started = time.time()
    try:
        model = build_model(model_name, payload, payload["flows"])
        output = simulate_dma(model, float(frequency), float(amplitude),
                              points_per_cycle=int(ppc), min_cycles=int(min_cycles),
                              max_cycles=int(max_cycles), steady_tolerance=float(tolerance))
        return {"success": True, "model": model_name, "candidate_id": candidate_id,
                "frequency_hz": float(frequency), "storage_MPa": output["storage_MPa"],
                "loss_MPa": output["loss_MPa"], "tan_delta": output["tan_delta"],
                "cycles": output["cycles"], "steady": output["steady"],
                "runtime_s": time.time()-started}
    except Exception as error:
        return {"success": False, "model": model_name, "candidate_id": candidate_id,
                "frequency_hz": float(frequency), "error": repr(error),
                "runtime_s": time.time()-started}


def parallel_map(function, tasks, workers, progress_label):
    results = []
    if not tasks:
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % max(1, len(tasks)//10) == 0 or index == len(tasks):
                print(f"{progress_label}: {index}/{len(tasks)}", flush=True)
    return results


def shortlist(rows, include_saturation, per_seed=2):
    allowed = [row for row in rows if row.get("success", False) and
               (include_saturation or row["budget_group"] in ("base", "anchor"))]
    chosen = {}
    def add(row):
        chosen[row["candidate_id"]] = row
    front = nondominated(allowed)
    for lam in (0.15, 0.30, 0.50, 0.70, 0.85):
        for row in sorted(front, key=lambda item: joint_score(item, lam))[:2]:
            add(row)
    for seed in sorted({row["seed"] for row in allowed if int(row["seed"]) >= 0}):
        seed_rows = [row for row in allowed if int(row["seed"]) == int(seed)]
        for row in sorted(seed_rows, key=lambda item: joint_score(item))[:per_seed]:
            add(row)
    if allowed:
        add(min(allowed, key=lambda row: row["L_time"]))
        add(min(allowed, key=lambda row: row["L_frequency"]))
    return list(chosen.values())


def load_anchor(model_name, registry):
    path = (HERE / registry["main_models"][model_name]["initial_checkpoint"]).resolve()
    return json.loads(path.read_text(encoding="utf-8")), path


def generate_global_candidates(model_name, anchor, config, seed, count,
                               stage, budget_group):
    lower, upper = payload_bounds(model_name, config)
    design = qmc.LatinHypercube(d=len(lower), seed=seed).random(count)
    vectors = qmc.scale(design, lower, upper)
    result = []
    for index, vector in enumerate(vectors):
        payload = vector_to_payload(model_name, vector, anchor, config)
        result.append({"candidate_id": f"{model_name}:{seed}:{stage}:{index}",
                       "seed": int(seed), "stage": stage, "budget_group": budget_group,
                       "vector": vector, "payload": payload})
    return result


def generate_local_candidates(model_name, anchor, config, seed, count,
                              center, stage, budget_group):
    lower, upper = payload_bounds(model_name, config)
    rng = np.random.default_rng(seed)
    vectors = np.asarray(center)[None, :] + rng.normal(size=(count, len(lower))) * (
        0.08 * (upper-lower))[None, :]
    vectors = np.clip(vectors, lower, upper)
    result = []
    for index, vector in enumerate(vectors):
        result.append({"candidate_id": f"{model_name}:{seed}:{stage}:{index}",
                       "seed": int(seed), "stage": stage, "budget_group": budget_group,
                       "vector": vector,
                       "payload": vector_to_payload(model_name, vector, anchor, config)})
    return result


def update_screen_checkpoint(path, metadata, rows):
    write_json(path, {"metadata": metadata, "rows": rows})


def read_screen_checkpoint(path, metadata):
    if not path.exists():
        return []
    old = json.loads(path.read_text(encoding="utf-8"))
    if old.get("metadata") != clean(metadata):
        return []
    return old.get("rows", [])


def run_t0(output, config, protocol, split, records, dma, train_mask,
           held_mask, scales, smoke):
    model_dir = output / "T0"; model_dir.mkdir(parents=True, exist_ok=True)
    zhang_li = ZhangLi(**config["zhang_li"])
    tau = np.asarray(config["spectrum"]["tau_s"], float)
    bases = build_t0_bases(records, zhang_li, tau)
    time_weights = protocol["time_objective"]
    seeds = protocol["optimization"]["seeds"][:1] if smoke else protocol["optimization"]["seeds"]
    base_global = 2 if smoke else int(protocol["optimization"]["T0"]["global_per_seed"])
    base_local = 1 if smoke else int(protocol["optimization"]["T0"]["local_per_seed"])
    lower = float(config["joint"]["weight_lower_bound"])
    upper = float(config["joint"]["weight_upper_bound"])
    candidates = []
    started = time.time()
    for seed in seeds:
        for stage, count, extra_seed, group in (
                ("base_global", base_global, seed, "base"),
                ("saturation_global", max(1, base_global//2), seed+100000, "saturation")):
            design = qmc.LatinHypercube(d=7, seed=extra_seed).random(count)
            initials = np.exp(qmc.scale(design, np.full(7, np.log(lower)),
                                         np.full(7, np.log(upper))))
            schedule = scalarization_schedule(
                count, protocol["pareto_and_selection"]["reported_scalarizations"])
            for index, (initial, lam) in enumerate(zip(initials, schedule)):
                tick = time.time()
                weights, fit, tm, fm = t0_fit(initial, lam, records, bases, dma,
                                              tau, zhang_li, train_mask, scales,
                                              time_weights, (lower, upper),
                                              maxiter=40 if smoke else 160)
                candidates.append({"candidate_id": f"T0:{seed}:{stage}:{lam}:{index}",
                                   "model": "T0", "seed": seed, "stage": stage,
                                   "budget_group": group, "success": bool(np.isfinite(fit.fun)),
                                   "time_weight": lam, "weights": weights,
                                   **{k: v for k, v in tm.items() if k != "curve_metrics"},
                                   **fm, "runtime_s": time.time()-tick,
                                   "optimizer_success": bool(fit.success),
                                   "optimizer_message": str(fit.message)})
        for group, count, seed_offset, allowed_groups in (
                ("base", base_local, 200000, ("base",)),
                ("saturation", max(1, base_local//2), 300000, ("base", "saturation"))):
            pool = [row for row in candidates if row["seed"] == seed and
                    row["budget_group"] in allowed_groups]
            schedule = scalarization_schedule(
                count, protocol["pareto_and_selection"]["reported_scalarizations"])
            rng = np.random.default_rng(seed+seed_offset)
            for index, lam in enumerate(schedule):
                center = min(pool, key=lambda row: joint_score(row, lam))["weights"]
                initial = np.exp(np.log(center) + rng.normal(size=7)*0.20)
                initial = np.clip(initial, lower, upper)
                tick = time.time()
                weights, fit, tm, fm = t0_fit(initial, lam, records, bases, dma,
                                              tau, zhang_li, train_mask, scales,
                                              time_weights, (lower, upper),
                                              maxiter=40 if smoke else 160)
                candidates.append({"candidate_id": f"T0:{seed}:{group}_local:{lam}:{index}",
                                   "model": "T0", "seed": seed, "stage": f"{group}_local",
                                   "budget_group": group, "success": bool(np.isfinite(fit.fun)),
                                   "time_weight": lam, "weights": weights,
                                   **{k: v for k, v in tm.items() if k != "curve_metrics"},
                                   **fm, "runtime_s": time.time()-tick,
                                   "optimizer_success": bool(fit.success),
                                   "optimizer_message": str(fit.message)})
        print(f"T0 seed {seed} complete", flush=True)
    base = representative([row for row in candidates if row["budget_group"] == "base"])
    selected = representative(candidates)
    weights = np.asarray(selected["weights"], float)
    stresses = t0_histories(records, bases, weights, zhang_li)
    storage, loss = t0_frequency(dma, tau, weights, zhang_li)
    selected["training_time_metrics"] = fair_time_metrics(records, stresses, "training", time_weights)
    selected["held_out_time_metrics"] = fair_time_metrics(records, stresses, "held_out", time_weights)
    selected["all_time_metrics"] = fair_time_metrics(records, stresses, None, time_weights)
    selected["training_frequency_metrics"] = fair_frequency_metrics(dma, train_mask, storage, loss, scales)
    selected["held_out_frequency_metrics"] = fair_frequency_metrics(dma, held_mask, storage, loss, scales)
    selected["frequency_prediction"] = {"frequency_hz": dma.frequency_hz,
                                        "storage_MPa": storage, "loss_MPa": loss,
                                        "method": "exact analytic QLV complex modulus"}
    change = abs(joint_score(selected)-joint_score(base))/max(joint_score(base), 1e-14)
    selected["budget_saturation"] = {"base_score": joint_score(base),
                                      "saturated_score": joint_score(selected),
                                      "relative_change": change,
                                      "passed": bool(change <= protocol["optimization"]["budget_saturation_relative_loss_tolerance"]),
                                      "comparison": "base_to_1.5x",
                                      "saturation_rounds_completed": 1}
    selected["runtime_total_s"] = time.time()-started
    selected["parameter_count"] = 7
    seed_representatives = [copy.deepcopy(representative(
        [row for row in candidates if row["seed"] == seed])) for seed in seeds]
    selected["completed_independent_seed_representatives"] = len(seed_representatives)
    selected["seed_representatives"] = seed_representatives
    pd.DataFrame([{k: clean(v) for k, v in row.items()
                   if k not in ("weights",)} | {**{f"weight_{i+1}": value
                                                    for i, value in enumerate(row["weights"])}}
                  for row in candidates]).to_csv(model_dir/"screen_candidates.csv", index=False,
                                                  encoding="utf-8-sig")
    write_json(model_dir/"screen_candidates.json", candidates)
    return selected, candidates


def run_ogden(model_name, output, config, protocol, registry, records, dma,
              train_mask, held_mask, scales, workers, smoke,
              extra_saturation_rounds=0):
    model_dir = output / model_name; model_dir.mkdir(parents=True, exist_ok=True)
    anchor, anchor_path = load_anchor(model_name, registry)
    time_weights = protocol["time_objective"]
    seeds = protocol["optimization"]["seeds"][:1] if smoke else protocol["optimization"]["seeds"]
    base_global = 2 if smoke else int(protocol["optimization"][model_name]["global_per_seed"])
    base_local = 1 if smoke else int(protocol["optimization"][model_name]["local_per_seed"])
    metadata = {"model": model_name, "config_hash": object_hash(config),
                "protocol_hash": object_hash(protocol), "smoke": bool(smoke),
                "seeds": seeds, "base_global": base_global, "base_local": base_local}
    checkpoint = model_dir / "screen_checkpoint.json"
    rows = read_screen_checkpoint(checkpoint, metadata)
    known = {row["candidate_id"] for row in rows}
    dma_payload = {"frequency_hz": dma.frequency_hz,
                   "frequency_storage_MPa": dma.frequency_storage_MPa,
                   "frequency_loss_MPa": dma.frequency_loss_MPa}
    equilibrium_G0 = equilibrium_shear_modulus(ZhangLi(**config["zhang_li"]))
    started = time.time()

    anchor_vector = payload_to_vector(model_name, anchor)
    anchor_candidate = {"candidate_id": f"{model_name}:-1:anchor:0", "seed": -1,
                        "stage": "anchor", "budget_group": "anchor",
                        "vector": anchor_vector,
                        "payload": vector_to_payload(model_name, anchor_vector, anchor, config)}
    waves = [("anchor", [anchor_candidate])]
    for seed in seeds:
        waves.append((f"seed{seed}_base_global", generate_global_candidates(
            model_name, anchor, config, seed, base_global, "base_global", "base")))
    for label, candidates in waves:
        missing = [candidate for candidate in candidates if candidate["candidate_id"] not in known]
        tasks = [(model_name, candidate, records, dma_payload, train_mask, scales,
                  time_weights, equilibrium_G0) for candidate in missing]
        new_rows = parallel_map(_screen_worker, tasks, workers, f"{model_name} {label}")
        rows.extend(new_rows); known.update(row["candidate_id"] for row in new_rows)
        update_screen_checkpoint(checkpoint, metadata, rows)

    for seed in seeds:
        seed_global = [row for row in rows if row.get("seed") == seed and
                       row.get("stage") == "base_global" and row.get("success")]
        center = np.asarray(min(seed_global, key=joint_score)["vector"])
        candidates = generate_local_candidates(model_name, anchor, config, seed+200000,
                                               base_local, center, "base_local", "base")
        for candidate in candidates:
            candidate["seed"] = seed
            candidate["candidate_id"] = candidate["candidate_id"].replace(str(seed+200000), str(seed))
        missing = [candidate for candidate in candidates if candidate["candidate_id"] not in known]
        tasks = [(model_name, candidate, records, dma_payload, train_mask, scales,
                  time_weights, equilibrium_G0) for candidate in missing]
        new_rows = parallel_map(_screen_worker, tasks, workers, f"{model_name} seed{seed} base local")
        rows.extend(new_rows); known.update(row["candidate_id"] for row in new_rows)
        update_screen_checkpoint(checkpoint, metadata, rows)

    for seed in seeds:
        candidates = generate_global_candidates(model_name, anchor, config, seed+100000,
                                                max(1, base_global//2),
                                                "saturation_global", "saturation")
        for candidate in candidates:
            candidate["seed"] = seed
            candidate["candidate_id"] = candidate["candidate_id"].replace(str(seed+100000), str(seed))
        missing = [candidate for candidate in candidates if candidate["candidate_id"] not in known]
        tasks = [(model_name, candidate, records, dma_payload, train_mask, scales,
                  time_weights, equilibrium_G0) for candidate in missing]
        new_rows = parallel_map(_screen_worker, tasks, workers, f"{model_name} seed{seed} saturation global")
        rows.extend(new_rows); known.update(row["candidate_id"] for row in new_rows)
        update_screen_checkpoint(checkpoint, metadata, rows)
        combined_global = [row for row in rows if row.get("seed") == seed and
                           "global" in row.get("stage", "") and row.get("success")]
        center = np.asarray(min(combined_global, key=joint_score)["vector"])
        local = generate_local_candidates(model_name, anchor, config, seed+300000,
                                          max(1, base_local//2), center,
                                          "saturation_local", "saturation")
        for candidate in local:
            candidate["seed"] = seed
            candidate["candidate_id"] = candidate["candidate_id"].replace(str(seed+300000), str(seed))
        missing = [candidate for candidate in local if candidate["candidate_id"] not in known]
        tasks = [(model_name, candidate, records, dma_payload, train_mask, scales,
                  time_weights, equilibrium_G0) for candidate in missing]
        new_rows = parallel_map(_screen_worker, tasks, workers, f"{model_name} seed{seed} saturation local")
        rows.extend(new_rows); known.update(row["candidate_id"] for row in new_rows)
        update_screen_checkpoint(checkpoint, metadata, rows)

    # A failed 1.5x saturation gate triggers additional 0.5x rounds without
    # changing the frozen base budget.  Each round is independently seeded and
    # its local cloud is centred after all preceding rounds.
    for round_index in range(2, 2 + int(extra_saturation_rounds)):
        group = f"saturation_{round_index}"
        for seed in seeds:
            offset = (round_index + 2) * 100000
            candidates = generate_global_candidates(
                model_name, anchor, config, seed + offset,
                max(1, base_global//2), f"saturation{round_index}_global", group)
            for candidate in candidates:
                candidate["seed"] = seed
                candidate["candidate_id"] = candidate["candidate_id"].replace(
                    str(seed + offset), str(seed))
            missing = [candidate for candidate in candidates
                       if candidate["candidate_id"] not in known]
            tasks = [(model_name, candidate, records, dma_payload, train_mask, scales,
                      time_weights, equilibrium_G0) for candidate in missing]
            new_rows = parallel_map(
                _screen_worker, tasks, workers,
                f"{model_name} seed{seed} saturation round {round_index} global")
            rows.extend(new_rows); known.update(row["candidate_id"] for row in new_rows)
            update_screen_checkpoint(checkpoint, metadata, rows)
            preceding = [row for row in rows if row.get("seed") == seed and
                         "global" in row.get("stage", "") and row.get("success")]
            center = np.asarray(min(preceding, key=joint_score)["vector"])
            local = generate_local_candidates(
                model_name, anchor, config, seed + offset + 100000,
                max(1, base_local//2), center, f"saturation{round_index}_local", group)
            for candidate in local:
                candidate["seed"] = seed
                candidate["candidate_id"] = candidate["candidate_id"].replace(
                    str(seed + offset + 100000), str(seed))
            missing = [candidate for candidate in local
                       if candidate["candidate_id"] not in known]
            tasks = [(model_name, candidate, records, dma_payload, train_mask, scales,
                      time_weights, equilibrium_G0) for candidate in missing]
            new_rows = parallel_map(
                _screen_worker, tasks, workers,
                f"{model_name} seed{seed} saturation round {round_index} local")
            rows.extend(new_rows); known.update(row["candidate_id"] for row in new_rows)
            update_screen_checkpoint(checkpoint, metadata, rows)

    strict_seed_rows = shortlist(rows, False) + shortlist(rows, True)
    strict_map = {row["candidate_id"]: row for row in strict_seed_rows}
    strict_checkpoint = model_dir / "strict_time_candidates.json"
    strict_rows = []
    if strict_checkpoint.exists():
        old = json.loads(strict_checkpoint.read_text(encoding="utf-8"))
        if old.get("metadata") == clean(metadata):
            strict_rows = old.get("rows", [])
    strict_known = {row["candidate_id"] for row in strict_rows}
    missing = [row for row in strict_map.values() if row["candidate_id"] not in strict_known]
    tasks = [(model_name, row, records, time_weights) for row in missing]
    strict_rows.extend(parallel_map(_strict_time_worker, tasks, workers,
                                    f"{model_name} strict time"))
    for row in strict_rows:
        if row.get("success"):
            storage, loss = local_complex_modulus(model_name, row["payload"],
                                                  dma.frequency_hz, equilibrium_G0)
            row.update(fair_frequency_metrics(dma, train_mask, storage, loss, scales))
    write_json(strict_checkpoint, {"metadata": metadata, "rows": strict_rows})

    periodic_candidates = shortlist(strict_rows, False, per_seed=2) + shortlist(strict_rows, True, per_seed=2)
    periodic_map = {row["candidate_id"]: row for row in periodic_candidates
                    if row.get("success")}
    periodic_checkpoint = model_dir / "periodic_predictions.json"
    periodic_rows = []
    if periodic_checkpoint.exists():
        old = json.loads(periodic_checkpoint.read_text(encoding="utf-8"))
        if old.get("metadata") == clean(metadata):
            periodic_rows = old.get("rows", [])
    periodic_keys = {(row["candidate_id"], float(row["frequency_hz"])) for row in periodic_rows}
    ppc = 32 if smoke else int(config["dma"]["strict_periodic_points_per_cycle"])
    frequencies = dma.frequency_hz[train_mask]
    if smoke:
        frequencies = np.unique(frequencies[[0, -1]])
    tasks = []
    for candidate_id, row in periodic_map.items():
        for frequency in frequencies:
            if (candidate_id, float(frequency)) not in periodic_keys:
                tasks.append((model_name, candidate_id, row["payload"], float(frequency),
                              config["dma"]["calibration_amplitude_fraction"], ppc,
                              config["dma"]["minimum_cycles"],
                              8 if smoke else config["dma"]["maximum_cycles"],
                              config["dma"]["steady_tolerance"]))
    periodic_rows.extend(parallel_map(_periodic_worker, tasks, workers,
                                      f"{model_name} strict periodic"))
    write_json(periodic_checkpoint, {"metadata": metadata, "rows": periodic_rows})

    final_rows = []
    frequency_lookup = {float(value): index for index, value in enumerate(dma.frequency_hz)}
    for candidate_id, row in periodic_map.items():
        values = [item for item in periodic_rows if item["candidate_id"] == candidate_id]
        if len(values) != len(frequencies) or any(not item.get("success", False)
                                                   or not item.get("steady", False)
                                                   for item in values):
            continue
        storage = np.full(len(dma.frequency_hz), np.nan)
        loss = np.full(len(dma.frequency_hz), np.nan)
        for item in values:
            index = frequency_lookup[float(item["frequency_hz"])]
            storage[index] = item["storage_MPa"]; loss[index] = item["loss_MPa"]
        if smoke:
            mask = np.isfinite(storage)
            class MiniDMA:
                pass
            mini = MiniDMA()
            mini.frequency_storage_MPa = dma.frequency_storage_MPa[mask]
            mini.frequency_loss_MPa = dma.frequency_loss_MPa[mask]
            mini.frequency_hz = dma.frequency_hz[mask]
            metric = fair_frequency_metrics(mini, np.ones(np.sum(mask), bool),
                                            storage[mask], loss[mask], scales)
        else:
            metric = fair_frequency_metrics(dma, train_mask, storage, loss, scales)
        final = copy.deepcopy(row); final.update(metric)
        final["periodic_training_prediction"] = {
            "frequency_hz": [item["frequency_hz"] for item in sorted(values, key=lambda x: x["frequency_hz"])],
            "storage_MPa": [item["storage_MPa"] for item in sorted(values, key=lambda x: x["frequency_hz"])],
            "loss_MPa": [item["loss_MPa"] for item in sorted(values, key=lambda x: x["frequency_hz"])],
            "points_per_cycle": ppc,
        }
        final_rows.append(final)
    if not final_rows:
        raise RuntimeError(f"{model_name}: no candidate completed strict periodic evaluation")
    preceding_groups = {"base", "anchor", "saturation"}
    preceding_groups.update(f"saturation_{index}" for index in range(2, 1 + int(extra_saturation_rounds)))
    base_pool = [row for row in final_rows if row["budget_group"] in preceding_groups]
    # Detach the selected records from ``final_rows`` before attaching
    # validation and seed-summary data.  Otherwise the winning row can later
    # contain itself through ``seed_representatives`` and become impossible to
    # serialise.
    base = copy.deepcopy(representative(base_pool))
    selected = copy.deepcopy(representative(final_rows))

    # Held-out periodic frequencies are evaluated only after selection.
    held_frequencies = dma.frequency_hz[held_mask]
    held_tasks = [(model_name, selected["candidate_id"], selected["payload"], float(frequency),
                   config["dma"]["calibration_amplitude_fraction"], ppc,
                   config["dma"]["minimum_cycles"],
                   8 if smoke else config["dma"]["maximum_cycles"],
                   config["dma"]["steady_tolerance"]) for frequency in held_frequencies]
    held_periodic = parallel_map(_periodic_worker, held_tasks, workers,
                                 f"{model_name} held-out periodic")
    if any(not row.get("success", False) or not row.get("steady", False)
           for row in held_periodic):
        raise RuntimeError(f"{model_name}: selected candidate failed held-out periodic evaluation")
    held_storage = np.array([row["storage_MPa"] for row in sorted(held_periodic,
                                                                   key=lambda x: x["frequency_hz"])])
    held_loss = np.array([row["loss_MPa"] for row in sorted(held_periodic,
                                                            key=lambda x: x["frequency_hz"])])
    class HeldDMA:
        pass
    held_dma = HeldDMA()
    held_dma.frequency_hz = dma.frequency_hz[held_mask]
    held_dma.frequency_storage_MPa = dma.frequency_storage_MPa[held_mask]
    held_dma.frequency_loss_MPa = dma.frequency_loss_MPa[held_mask]
    selected["held_out_frequency_metrics"] = fair_frequency_metrics(
        held_dma, np.ones(len(held_frequencies), bool), held_storage, held_loss, scales)
    selected["held_out_periodic_prediction"] = held_periodic
    change = abs(joint_score(selected)-joint_score(base))/max(joint_score(base), 1e-14)
    selected["budget_saturation"] = {"base_score": joint_score(base),
                                      "saturated_score": joint_score(selected),
                                      "relative_change": change,
                                      "passed": bool(change <= protocol["optimization"]["budget_saturation_relative_loss_tolerance"]),
                                      "comparison": ("base_to_1.5x" if not extra_saturation_rounds
                                                     else f"previous_round_to_{1.5 + 0.5*extra_saturation_rounds:.1f}x"),
                                      "saturation_rounds_completed": 1 + int(extra_saturation_rounds)}
    selected["runtime_total_s"] = time.time()-started
    selected["parameter_count"] = registry["main_models"][model_name]["independent_parameter_count"]
    selected["anchor_path"] = str(anchor_path)
    seed_representatives = []
    for seed in seeds:
        seed_rows = [row for row in final_rows if int(row["seed"]) == int(seed)]
        if seed_rows:
            seed_representatives.append(copy.deepcopy(representative(seed_rows)))
    selected["completed_independent_seed_representatives"] = len(seed_representatives)
    selected["seed_representatives"] = seed_representatives
    pd.DataFrame([{k: clean(v) for k, v in row.items()
                   if k not in ("payload", "vector", "curve_metrics")}
                  for row in rows]).to_csv(model_dir/"screen_candidates.csv", index=False,
                                           encoding="utf-8-sig")
    write_json(model_dir/"strict_final_candidates.json", final_rows)
    return selected, rows, final_rows


def parameter_payload(model_name, selected, config, protocol, registry, split_path):
    common = {
        "schema_version": "2026-08-28-chapter2-fair-joint-v1",
        "model": model_name,
        "role": "joint evaluation parameters selected from training conditions only",
        "configuration_sha256": file_record(HERE/"model_comparison_config.json")["sha256"],
        "protocol_sha256": file_record(HERE/"chapter2_fair_comparison_protocol.json")["sha256"],
        "registry_sha256": file_record(HERE/"chapter2_model_registry.json")["sha256"],
        "split_sha256": file_record(split_path)["sha256"],
        "data": file_record(config["data"]["viscoelastic"]),
        "selection_rule": protocol["pareto_and_selection"]["representative_score"],
        "selected_candidate_id": selected["candidate_id"],
        "training_L_time": selected["L_time"],
        "training_L_frequency": selected["L_frequency"],
        "training_joint_score": joint_score(selected),
        "held_out_time_metrics": selected["held_out_time_metrics"],
        "held_out_frequency_metrics": selected["held_out_frequency_metrics"],
        "budget_saturation": selected["budget_saturation"],
        "parameter_count": registry["main_models"][model_name]["independent_parameter_count"],
    }
    if model_name == "T0":
        common.update({"zhang_li": config["zhang_li"],
                       "tau_s": config["spectrum"]["tau_s"],
                       "weights": selected["weights"]})
    else:
        common.update(selected["payload"])
    return common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--extra-t2-saturation-rounds", type=int, default=0,
                        help="additional 0.5x T2 budget rounds after a failed 1.5x gate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    smoke = args.mode == "smoke"
    output = args.output or (TEACHER / ("results_chapter2_fair_comparison_smoke"
                                       if smoke else "results_chapter2_fair_comparison"))
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config, config_path = load_config(HERE/"model_comparison_config.json")
    protocol = json.loads((HERE/"chapter2_fair_comparison_protocol.json").read_text(encoding="utf-8"))
    registry = json.loads((HERE/"chapter2_model_registry.json").read_text(encoding="utf-8"))
    split_path = HERE/"model_comparison_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    records = build_records(config, split)
    dma = load_dma(config["data"]["viscoelastic"])
    train_mask, held_mask = frequency_masks(dma.frequency_hz,
                                             config["split"]["frequency_validation_hz"])
    scales = frequency_scales(dma, train_mask, config)
    selections, candidate_sets = {}, {}
    t0, t0_candidates = run_t0(output, config, protocol, split, records, dma, train_mask,
                               held_mask, scales, smoke)
    selections["T0"] = t0
    candidate_sets["T0"] = t0_candidates
    for model_name in ("T1", "T2"):
        selected, screen, _strict = run_ogden(
            model_name, output, config, protocol, registry, records, dma,
            train_mask, held_mask, scales, args.workers, smoke,
            args.extra_t2_saturation_rounds if model_name == "T2" else 0)
        selections[model_name] = selected
        candidate_sets[model_name] = screen
    parameters = output/"parameters"; parameters.mkdir(exist_ok=True)
    table_rows = []
    accumulated_costs = {}
    for model_name in MODEL_ORDER:
        selected = selections[model_name]
        payload = parameter_payload(model_name, selected, config, protocol, registry, split_path)
        write_json(parameters/f"{model_name}_chapter2_joint_evaluation_parameters.json", payload)
        screen_cpu = float(sum(row.get("runtime_s", 0.0)
                               for row in candidate_sets[model_name]))
        strict_cpu = periodic_cpu = 0.0
        if model_name != "T0":
            strict_path = output/model_name/"strict_time_candidates.json"
            periodic_path = output/model_name/"periodic_predictions.json"
            strict_cpu = float(sum(row.get("runtime_s", 0.0)
                                   for row in json.loads(strict_path.read_text(encoding="utf-8"))["rows"]))
            periodic_cpu = float(sum(row.get("runtime_s", 0.0)
                                     for row in json.loads(periodic_path.read_text(encoding="utf-8"))["rows"]))
        accumulated_costs[model_name] = {
            "screen_candidate_cpu_s": screen_cpu,
            "strict_time_cpu_s": strict_cpu,
            "strict_periodic_cpu_s": periodic_cpu,
            "accumulated_candidate_cpu_s": screen_cpu + strict_cpu + periodic_cpu,
            "note": "sum of recorded candidate runtimes; differs from wall time under parallel execution and checkpoint reuse",
        }
        table_rows.append({
            "model": model_name,
            "training_L_time": selected["L_time"],
            "training_L_frequency": selected["L_frequency"],
            "training_joint_score": joint_score(selected),
            "training_q_log_RMSE": selected["q_log_RMSE"],
            "training_q_ordering_accuracy": selected["q_ordering_accuracy"],
            "training_Rneq_RMSE": selected["Rneq_condition_RMSE"],
            "held_out_L_time": selected["held_out_time_metrics"]["L_time"],
            "held_out_q_log_RMSE": selected["held_out_time_metrics"]["q_log_RMSE"],
            "held_out_Rneq_RMSE": selected["held_out_time_metrics"]["Rneq_condition_RMSE"],
            "held_out_L_frequency": selected["held_out_frequency_metrics"]["L_frequency"],
            "parameter_count": selected["parameter_count"],
            "budget_saturation_relative_change": selected["budget_saturation"]["relative_change"],
            "budget_saturation_passed": selected["budget_saturation"]["passed"],
            "completed_seed_representatives": selected.get("completed_independent_seed_representatives",
                                                               len(protocol["optimization"]["seeds"])),
            "runtime_total_s": selected["runtime_total_s"],
            "accumulated_candidate_cpu_s": accumulated_costs[model_name]["accumulated_candidate_cpu_s"],
        })
    tables = output/"tables"; tables.mkdir(exist_ok=True)
    pd.DataFrame(table_rows).to_csv(tables/"chapter2_fair_model_summary.csv",
                                    index=False, encoding="utf-8-sig")
    stability = {model: stability_summary(model, selections[model],
                                           candidate_sets[model], config)
                 for model in MODEL_ORDER}
    write_json(tables/"chapter2_parameter_stability.json", stability)
    seed_table = [row | {"model": model} for model in MODEL_ORDER
                  for row in stability[model]["seed_rows"]]
    pd.DataFrame(seed_table).to_csv(tables/"chapter2_seed_representatives.csv",
                                    index=False, encoding="utf-8-sig")
    summary = {
        "schema_version": "2026-08-28-chapter2-fair-calibration-summary-v1",
        "mode": args.mode,
        "generated_at": datetime.now().astimezone().isoformat(),
        "elapsed_s": time.time()-started,
        "workers": args.workers,
        "extra_t2_saturation_rounds": args.extra_t2_saturation_rounds,
        "python": platform.python_version(), "scipy": scipy.__version__,
        "configuration": file_record(config_path),
        "protocol": file_record(HERE/"chapter2_fair_comparison_protocol.json"),
        "registry": file_record(HERE/"chapter2_model_registry.json"),
        "split": file_record(split_path),
        "data": file_record(config["data"]["viscoelastic"]),
        "frequency_scales_MPa": {"storage": scales[0], "loss": scales[1]},
        "models": {model: {
            "candidate_id": selections[model]["candidate_id"],
            "training_L_time": selections[model]["L_time"],
            "training_L_frequency": selections[model]["L_frequency"],
            "training_joint_score": joint_score(selections[model]),
            "held_out_L_time": selections[model]["held_out_time_metrics"]["L_time"],
            "held_out_L_frequency": selections[model]["held_out_frequency_metrics"]["L_frequency"],
            "budget_saturation": selections[model]["budget_saturation"],
            "completed_independent_seeds": stability[model]["completed_independent_seeds"],
            "effective_branch_count_mean": stability[model]["effective_branch_count_mean"],
            "failed_candidates": stability[model]["failed_candidates"],
            "computational_cost": accumulated_costs[model],
        } for model in MODEL_ORDER},
        "all_models_budget_saturated": all(
            selections[model]["budget_saturation"]["passed"] for model in MODEL_ORDER),
    }
    write_json(output/"chapter2_fair_calibration_summary.json", summary)
    print(json.dumps(clean(summary["models"]), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
