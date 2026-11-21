"""Joint relaxation--0.1% frequency-DMA refinement of stage-4 Ogden models.

Amplitude DMA is deliberately absent from every search, early-stopping and
selection calculation. Candidate screening uses the local small-strain complex
modulus; every strict Pareto candidate is then rescored by full-cycle periodic
integration before the representative model is selected.
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
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
for path in (HERE, TEACHER / "3_PRF" / "code", TEACHER / "4_GroupedPRF" / "code",
             TEACHER / "5_T1_OgdenPRF" / "code", TEACHER / "6_T2_GroupedOgdenPRF" / "code"):
    sys.path.insert(0, str(path))

from comparison_core import (ZhangLi, dma_metrics, frequency_masks, load_config,
                             load_dma, load_relaxation, nondominated)
from run_stage3_ogden_flow_calibration import (
    build_model, clean_json, file_record, flow_to_vector, flows_from_vector,
    make_records, save_figure, search_bounds, stage2_path, write_json,
)
from run_stage4_block_coordinate_refinement import (
    config_hash, evaluate_many, flatten_result, full_ordering, refinement_score,
    softmax,
)
from teacher_model.dma import simulate_dma


plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.spines.right": False, "axes.spines.top": False,
                     "axes.linewidth": 0.8, "legend.frameon": False,
                     "axes.unicode_minus": False, "font.size": 8})

FREQUENCY_AMPLITUDE = 0.001
TIME_WEIGHTS = (0.15, 0.30, 0.50, 0.70, 0.85)
G_BOUNDS = (0.045, 1.10)
A_BOUNDS = (2.5, 6.0)
SPECTRUM_LOG_RADIUS = 1.0


def stage4_path(model_name):
    return (TEACHER / "results_ogden_teacher_models" / "stage4_block_coordinate" /
            "parameters" / f"{model_name}_Ogden_1_stage4_block_parameters.json")


def flow_vector(payload, model_name):
    if model_name == "T1":
        return flow_to_vector(payload["flows"]["shared"])
    return np.r_[flow_to_vector(payload["flows"]["fast"]),
                 flow_to_vector(payload["flows"]["slow"])]


def zero_force_multiplier(flow):
    ratio = float(flow["q0_MPa"]) / max(float(flow["s0_MPa"]), 1e-14)
    return 1.0 + float(flow["nonlinear_multiplier"]) * ratio ** (float(flow["m"]) - 1.0)


def local_complex_modulus(model_name, payload, frequency_hz):
    model = build_model(model_name, payload, payload["flows"])
    omega = 2.0 * np.pi * np.asarray(frequency_hz, float)
    gp = np.full_like(omega, model.G0, dtype=float)
    gpp = np.zeros_like(omega, dtype=float)
    for tau, share in zip(payload["tau_s"], payload["p_alpha"]):
        group = "shared" if model_name == "T1" else ("fast" if tau <= payload.get("transition_tau_s", 1.0) else "slow")
        multiplier = zero_force_multiplier(payload["flows"][group])
        effective_tau = float(tau) / multiplier
        z = omega * effective_tau
        modulus = float(payload["Gneq_MPa"]) * float(share)
        gp += modulus * z * z / (1.0 + z * z)
        gpp += modulus * z / (1.0 + z * z)
    return gp, gpp


def spectrum_regularization(payload, reference_p):
    p = np.asarray(payload["p_alpha"], float)
    reference = np.asarray(reference_p, float)
    ratio = np.log(np.maximum(p, 1e-300) / np.maximum(reference, 1e-300))
    return {"spectrum_KL": float(np.sum(p * ratio)),
            "spectrum_log_mse": float(np.mean(ratio ** 2)),
            "spectrum_max_abs_log_ratio": float(np.max(np.abs(ratio)))}


def add_frequency_metrics(row, model_name, dma, train_mask, held_mask, reference_p):
    gp, gpp = local_complex_modulus(model_name, row["payload"], dma.frequency_hz)
    row["local_storage_MPa"] = gp
    row["local_loss_MPa"] = gpp
    row["frequency_training"] = dma_metrics(
        dma.frequency_storage_MPa[train_mask], dma.frequency_loss_MPa[train_mask],
        gp[train_mask], gpp[train_mask])
    row["frequency_held_out"] = dma_metrics(
        dma.frequency_storage_MPa[held_mask], dma.frequency_loss_MPa[held_mask],
        gp[held_mask], gpp[held_mask])
    row.update(spectrum_regularization(row["payload"], reference_p))
    row["time_score"] = float(np.sqrt(max(row["score"], 0.0)))
    row["frequency_score_local"] = row["frequency_training"]["balanced_nrmse"]
    return row


def generate_candidates(model_name, base, old, seeds, per_seed):
    groups = 1 if model_name == "T1" else 2
    lo, hi = search_bounds(groups)
    base_flow = flow_vector(base, model_name)
    old_flow = flow_vector(old, model_name)
    reference_logits = np.log(np.maximum(base["p_alpha"], 1e-14))
    candidates = []
    metadata = {}

    def add(payload, seed, family):
        payload = copy.deepcopy(payload)
        payload["p_alpha"] = softmax(np.log(np.maximum(payload["p_alpha"], 1e-14))).tolist()
        payload["old_weights"] = list(payload["p_alpha"])
        key = config_hash(payload)
        if key not in metadata:
            candidates.append(payload)
            metadata[key] = {"seed": seed, "family": family}

    add(base, -1, "stage4_anchor")
    for G in (0.06, 0.08, 0.12, 0.20, 0.35, 0.55, 0.80, 1.00):
        for fraction in (0.0, 0.5, 1.0):
            payload = copy.deepcopy(base)
            payload["Gneq_MPa"] = G
            payload["flows"] = flows_from_vector(
                model_name, np.clip((1.0 - fraction) * base_flow + fraction * old_flow, lo, hi))
            add(payload, -1, f"G_grid_flow_{fraction:g}")

    dimension = 4 + len(base_flow) + 7
    for seed in seeds:
        design = qmc.LatinHypercube(d=dimension, seed=seed).random(per_seed)
        normal_rng = np.random.default_rng(seed + 5000)
        for unit in design:
            payload = copy.deepcopy(base)
            payload["Gneq_MPa"] = float(np.exp(np.log(G_BOUNDS[0]) + unit[0] *
                                                  (np.log(G_BOUNDS[1]) - np.log(G_BOUNDS[0]))))
            payload["a_ogden"] = float(A_BOUNDS[0] + unit[1] * (A_BOUNDS[1] - A_BOUNDS[0]))
            interpolation = -0.25 + 1.5 * unit[2]
            vector = ((1.0 - interpolation) * base_flow + interpolation * old_flow +
                      normal_rng.normal(size=len(base_flow)) * 0.035 * (hi - lo))
            payload["flows"] = flows_from_vector(model_name, np.clip(vector, lo, hi))
            logits = reference_logits + normal_rng.normal(0.0, 0.32, 7)
            logits = np.clip(logits, reference_logits - SPECTRUM_LOG_RADIUS,
                             reference_logits + SPECTRUM_LOG_RADIUS)
            payload["p_alpha"] = softmax(logits).tolist()
            add(payload, seed, "lhs_joint")
    return candidates, metadata


def scalar_screen_row(row, metadata, model_name):
    payload = row["payload"]
    output = {"model": model_name, **metadata[config_hash(payload)],
              "config_hash": config_hash(payload), "time_score": row["time_score"],
              "frequency_score_local": row["frequency_score_local"],
              "frequency_storage_nrmse": row["frequency_training"]["storage_nrmse"],
              "frequency_loss_nrmse": row["frequency_training"]["loss_nrmse"],
              "q_log_mse": row["components"]["q_log_mse"],
              "Rneq_balanced_mse": row["components"]["Rneq_balanced_mse"],
              "q_ordering_accuracy": row["components"]["q_ordering_accuracy"],
              "Gneq_MPa": payload["Gneq_MPa"], "a_ogden": payload["a_ogden"],
              "spectrum_KL": row["spectrum_KL"],
              "spectrum_max_abs_log_ratio": row["spectrum_max_abs_log_ratio"]}
    return output


def shortlist(screened):
    valid = [row for row in screened if row.get("success", False)]
    t = np.asarray([row["time_score"] for row in valid])
    f = np.asarray([row["frequency_score_local"] for row in valid])
    tn = (t - t.min()) / max(float(t.max() - t.min()), 1e-14)
    fn = (f - f.min()) / max(float(f.max() - f.min()), 1e-14)
    chosen = []
    for weight in TIME_WEIGHTS:
        score = weight * tn + (1.0 - weight) * fn + 0.01 * np.asarray(
            [row["spectrum_KL"] for row in valid])
        score += np.asarray([0.5 if row["components"]["q_ordering_accuracy"] < 1.0 else 0.0
                             for row in valid])
        for index in np.argsort(score)[:2]:
            chosen.append(valid[int(index)]["payload"])
    chosen.extend([valid[int(np.argmin(t))]["payload"], valid[int(np.argmin(f))]["payload"]])
    result, seen = [], set()
    for payload in chosen:
        key = config_hash(payload)
        if key not in seen:
            result.append(payload); seen.add(key)
    return result


def _periodic_worker(task):
    model_name, payload, candidate_id, frequency, points, min_cycles, max_cycles, tolerance = task
    try:
        model = build_model(model_name, payload, payload["flows"])
        result = simulate_dma(model, float(frequency), FREQUENCY_AMPLITUDE,
                              points_per_cycle=points, min_cycles=min_cycles,
                              max_cycles=max_cycles, steady_tolerance=tolerance)
        return {"success": True, "candidate_id": candidate_id, "frequency_hz": frequency,
                "points_per_cycle": points, **{key: value for key, value in result.items()
                if np.isscalar(value)}}
    except Exception as error:
        return {"success": False, "candidate_id": candidate_id,
                "frequency_hz": frequency, "points_per_cycle": points,
                "error": repr(error)}


def periodic_many(model_name, payloads, frequencies, points, workers,
                  min_cycles=4, max_cycles=12, tolerance=1e-3):
    tasks = []
    for candidate_id, payload in enumerate(payloads):
        for frequency in frequencies:
            tasks.append((model_name, payload, candidate_id, float(frequency), points,
                          min_cycles, max_cycles, tolerance))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(_periodic_worker, task) for task in tasks]):
            rows.append(future.result())
    return rows


def add_periodic_scores(strict_rows, periodic_rows, dma, train_mask):
    frequency_to_exp = {float(f): (float(gp), float(gpp)) for f, gp, gpp in zip(
        dma.frequency_hz, dma.frequency_storage_MPa, dma.frequency_loss_MPa)}
    training_frequencies = np.asarray(dma.frequency_hz)[train_mask]
    for candidate_id, row in enumerate(strict_rows):
        selected = sorted([item for item in periodic_rows
                           if item["candidate_id"] == candidate_id and item.get("success", False)],
                          key=lambda item: item["frequency_hz"])
        if len(selected) != len(training_frequencies):
            row["frequency_score_periodic"] = 1e6
            row["periodic_all_steady"] = False
            continue
        exp = np.asarray([frequency_to_exp[item["frequency_hz"]] for item in selected])
        pred = np.asarray([[item["storage_MPa"], item["loss_MPa"]] for item in selected])
        metric = dma_metrics(exp[:, 0], exp[:, 1], pred[:, 0], pred[:, 1])
        row["frequency_training_periodic"] = metric
        row["frequency_score_periodic"] = metric["balanced_nrmse"]
        row["periodic_all_steady"] = bool(all(item["steady"] for item in selected))
    return strict_rows


def choose_representative(rows):
    admissible = [row for row in rows if row.get("frequency_score_periodic", 1e6) < 1e5
                  and row["components"]["q_ordering_accuracy"] == 1.0]
    if not admissible:
        admissible = [row for row in rows if row.get("frequency_score_periodic", 1e6) < 1e5]
    front = nondominated(admissible, "time_score", "frequency_score_periodic")
    t = np.asarray([row["time_score"] for row in front])
    f = np.asarray([row["frequency_score_periodic"] for row in front])
    tn = (t - t.min()) / max(float(t.max() - t.min()), 1e-14)
    fn = (f - f.min()) / max(float(f.max() - f.min()), 1e-14)
    selected = front[int(np.argmin(np.hypot(tn, fn)))] if len(front) > 1 else front[0]
    return selected, front


def strict_table_row(row, model_name, rank, selected=False):
    payload = row["payload"]
    return {"model": model_name, "rank": rank, "selected": selected,
            "config_hash": config_hash(payload), "time_score": row["time_score"],
            "frequency_score_local": row["frequency_score_local"],
            "frequency_score_periodic": row.get("frequency_score_periodic"),
            "periodic_all_steady": row.get("periodic_all_steady"),
            "q_log_RMSE": float(np.sqrt(row["components"]["q_log_mse"])),
            "Rneq_balanced_RMSE": float(np.sqrt(row["components"]["Rneq_balanced_mse"])),
            "q_ordering_accuracy": row["components"]["q_ordering_accuracy"],
            "full_ordering_accuracy": row["full_ordering_accuracy"],
            "Gneq_MPa": payload["Gneq_MPa"], "a_ogden": payload["a_ogden"],
            "spectrum_KL": row["spectrum_KL"],
            "max_detFv_error": row["max_detFv_error"],
            "min_dissipation": row["min_dissipation"],
            "max_residual": row["max_residual"]}


def final_periodic(model_name, payload, dma, workers):
    rows = periodic_many(model_name, [payload], dma.frequency_hz, 64, workers,
                         min_cycles=5, max_cycles=18, tolerance=7e-4)
    rows = sorted(rows, key=lambda row: row["frequency_hz"])
    convergence = periodic_many(model_name, [payload], [0.1, 10.0, 100.0], 128,
                                workers, min_cycles=5, max_cycles=18, tolerance=5e-4)
    return rows, convergence


def plot_pareto(all_rows, figures):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, model_name in zip(axes, ("T1", "T2")):
        data = all_rows[all_rows.model == model_name]
        ax.scatter(data.time_score, data.frequency_score_periodic, s=22,
                   facecolors="none", edgecolors="#7C8794", label="Full-evaluation candidates")
        front = data[data.on_pareto.astype(bool)].sort_values("time_score")
        ax.plot(front.time_score, front.frequency_score_periodic, "o-",
                color="#1F5A94", lw=1.5, ms=4, label="Pareto front")
        selected = data[data.selected.astype(bool)]
        ax.scatter(selected.time_score, selected.frequency_score_periodic,
                   marker="*", s=120, color="#B64342", zorder=5, label="Selected model")
        ax.set(title=f"{model_name}-Ogden-1", xlabel="Time-domain error",
               ylabel="Frequency-sweep balanced NRMSE")
        ax.grid(alpha=.18); ax.legend(fontsize=7)
    save_figure(fig, figures / "stage5_time_frequency_pareto_front")


def plot_frequency(final_rows, dma, train_mask, held_mask, figures):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharex=True)
    for ax, channel, ylabel in zip(axes, ("storage_MPa", "loss_MPa"),
                                   (r"Storage modulus $G'$ / MPa", r"Loss modulus $G''$ / MPa")):
        experimental = dma.frequency_storage_MPa if channel == "storage_MPa" else dma.frequency_loss_MPa
        ax.scatter(dma.frequency_hz[train_mask], experimental[train_mask], s=28,
                   facecolors="white", edgecolors="#202020", label="Experiment (calibration)")
        ax.scatter(dma.frequency_hz[held_mask], experimental[held_mask], s=34,
                   facecolors="white", edgecolors="#B64342", marker="D", label="Experiment (held out)")
        for model_name, color in (("T1", "#4C78A8"), ("T2", "#E07A3F")):
            row = final_rows[final_rows.model == model_name].sort_values("frequency_hz")
            ax.plot(row.frequency_hz, row[channel], color=color, lw=1.7,
                    label=f"{model_name}-Ogden-1")
        ax.set_xscale("log"); ax.set(xlabel="Frequency / Hz", ylabel=ylabel)
        ax.grid(alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.05))
    save_figure(fig, figures / "stage5_frequency_sweep_0p1percent", rect=(0, 0, 1, .91))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20260828, 20260829, 20260830])
    parser.add_argument("--t1-per-seed", type=int, default=48)
    parser.add_argument("--t2-per-seed", type=int, default=64)
    parser.add_argument("--results-root", type=Path,
                        default=TEACHER / "results_ogden_teacher_models" / "stage5_joint_time_frequency")
    parser.add_argument("--figures-root", type=Path,
                        default=TEACHER / "figures_ogden_teacher_models" / "stage5_joint_time_frequency")
    args = parser.parse_args()
    results = args.results_root.resolve(); figures = args.figures_root.resolve()
    tables = results / "tables"; parameters = results / "parameters"
    for path in (tables, parameters, figures): path.mkdir(parents=True, exist_ok=True)
    config, config_path = load_config(HERE / "model_comparison_config.json")
    split_path = HERE / "model_comparison_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    records = make_records(load_relaxation(config["data"]["viscoelastic"]),
                           ZhangLi(**config["zhang_li"]),
                           set(split["relaxation"]["held_out"]))
    dma = load_dma(config["data"]["viscoelastic"])
    train_mask, held_mask = frequency_masks(dma.frequency_hz, (5.0, 20.0))
    started = time.time(); screen_tables = []; strict_tables = []
    selected_models = {}; pareto_records = []; final_frequency_frames = []

    for model_name, per_seed in (("T1", args.t1_per_seed), ("T2", args.t2_per_seed)):
        base = json.loads(stage4_path(model_name).read_text(encoding="utf-8"))
        old = json.loads(stage2_path(model_name).read_text(encoding="utf-8"))
        candidates, metadata = generate_candidates(model_name, base, old,
                                                    args.seeds, per_seed)
        print(model_name, "screen candidates", len(candidates), flush=True)
        screened = evaluate_many(model_name, candidates, records, False, args.workers)
        screened = [add_frequency_metrics(row, model_name, dma, train_mask, held_mask,
                                          base["p_alpha"])
                    for row in screened if row.get("success", False)]
        screen_tables.extend(scalar_screen_row(row, metadata, model_name) for row in screened)
        candidates_strict = shortlist(screened)
        print(model_name, "strict candidates", len(candidates_strict), flush=True)
        strict = evaluate_many(model_name, candidates_strict, records, True, args.workers)
        strict = [add_frequency_metrics(row, model_name, dma, train_mask, held_mask,
                                        base["p_alpha"])
                  for row in strict if row.get("success", False)]
        periodic = periodic_many(model_name, [row["payload"] for row in strict],
                                 dma.frequency_hz[train_mask], 32, args.workers,
                                 min_cycles=4, max_cycles=12, tolerance=1e-3)
        strict = add_periodic_scores(strict, periodic, dma, train_mask)
        selected, front = choose_representative(strict)
        selected_models[model_name] = selected
        front_hashes = {config_hash(row["payload"]) for row in front}
        for rank, row in enumerate(sorted(strict, key=lambda item: item["time_score"]), 1):
            table_row = strict_table_row(row, model_name, rank,
                                         config_hash(row["payload"]) == config_hash(selected["payload"]))
            table_row["on_pareto"] = config_hash(row["payload"]) in front_hashes
            strict_tables.append(table_row); pareto_records.append(table_row)
        print(model_name, "selected", selected["time_score"],
              selected["frequency_score_periodic"], selected["payload"]["Gneq_MPa"],
              selected["payload"]["a_ogden"], flush=True)

        payload = copy.deepcopy(selected["payload"])
        payload.update({"schema_version": "2026-08-28-stage5-joint-time-frequency-v1",
                        "stage": "joint_relaxation_frequency_dma",
                        "frequency_dma_amplitude_fraction": FREQUENCY_AMPLITUDE,
                        "frequency_dma_amplitude_percent": 0.1,
                        "frequency_training_rule": split["frequency_dma"]["training_rule"],
                        "frequency_held_out_rule": split["frequency_dma"]["held_out_rule"],
                        "amplitude_dma_role": split["amplitude_dma"]["role"],
                        "stage5_time_score": selected["time_score"],
                        "stage5_frequency_score_periodic": selected["frequency_score_periodic"],
                        "selection_rule": "nearest normalized ideal point on strict periodic Pareto front",
                        "search_seeds": args.seeds})
        filename = f"{model_name}_Ogden_1_stage5_joint_parameters.json"
        write_json(parameters / filename, payload)
        destination = TEACHER / ("5_T1_OgdenPRF" if model_name == "T1" else
                                 "6_T2_GroupedOgdenPRF") / "results" / "parameters"
        write_json(destination / filename, payload)
        final_rows, convergence = final_periodic(model_name, payload, dma, args.workers)
        for row in final_rows:
            row["model"] = model_name
            index = int(np.argmin(abs(dma.frequency_hz - row["frequency_hz"])))
            row["split"] = "training" if train_mask[index] else "held_out"
            row["experimental_storage_MPa"] = float(dma.frequency_storage_MPa[index])
            row["experimental_loss_MPa"] = float(dma.frequency_loss_MPa[index])
        final_frequency_frames.append(pd.DataFrame(final_rows))
        pd.DataFrame(convergence).to_csv(tables / f"{model_name}_stage5_frequency_128ppc.csv",
                                         index=False, encoding="utf-8-sig")

    screen_frame = pd.DataFrame(screen_tables)
    strict_frame = pd.DataFrame(strict_tables)
    final_frequency = pd.concat(final_frequency_frames, ignore_index=True)
    screen_frame.to_csv(tables / "stage5_screen_candidates.csv", index=False, encoding="utf-8-sig")
    strict_frame.to_csv(tables / "stage5_strict_pareto_candidates.csv", index=False, encoding="utf-8-sig")
    final_frequency.to_csv(tables / "stage5_frequency_periodic_predictions.csv",
                           index=False, encoding="utf-8-sig")
    plot_pareto(strict_frame, figures)
    plot_frequency(final_frequency, dma, train_mask, held_mask, figures)

    final_metrics = []
    for model_name, data in final_frequency.groupby("model"):
        for split_name in ("training", "held_out"):
            row = data[data.split == split_name]
            metric = dma_metrics(row.experimental_storage_MPa, row.experimental_loss_MPa,
                                 row.storage_MPa, row.loss_MPa)
            final_metrics.append({"model": model_name, "split": split_name, **metric,
                                  "all_steady": bool(row.steady.all()),
                                  "max_third_to_first": float(row.third_to_first.max())})
    summary = {"stage": "stage5_joint_time_frequency", "elapsed_s": time.time() - started,
               "frequency_amplitude_fraction": FREQUENCY_AMPLITUDE,
               "amplitude_dma_used": False, "frequency_metrics": final_metrics,
               "selected": {model: {"time_score": row["time_score"],
                                     "frequency_score_periodic_screen": row["frequency_score_periodic"],
                                     "Gneq_MPa": row["payload"]["Gneq_MPa"],
                                     "a_ogden": row["payload"]["a_ogden"],
                                     "p_alpha": row["payload"]["p_alpha"],
                                     "flows": row["payload"]["flows"],
                                     "components": row["components"],
                                     "full_ordering_accuracy": row["full_ordering_accuracy"],
                                     "max_detFv_error": row["max_detFv_error"],
                                     "min_dissipation": row["min_dissipation"]}
                            for model, row in selected_models.items()}}
    write_json(results / "stage5_summary.json", summary)
    manifest = {"stage": summary["stage"], "inputs": [file_record(path) for path in
                (Path(config["data"]["viscoelastic"]), config_path, split_path,
                 stage4_path("T1"), stage4_path("T2"), HERE / "ogden_prf_core.py", Path(__file__))],
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "numpy": np.__version__, "pandas": pd.__version__,
                                "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
                "outputs": {"tables": sorted(str(path) for path in tables.glob("*.csv")),
                            "parameters": sorted(str(path) for path in parameters.glob("*.json")),
                            "figures": sorted(str(path) for path in figures.glob("*.*"))}}
    write_json(results / "stage5_reproducibility_manifest.json", manifest)
    print(json.dumps(clean_json(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
