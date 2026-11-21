"""Guarded incremental T1/T2 optimization for the chapter-2 baseline.

The old guarded T1/T2 checkpoints are immutable baselines.  T0 is deliberately
not recalibrated.  A new T1/T2 candidate can become active only if strict
time-domain relaxation and strict periodic frequency-DMA losses both improve,
while the common q and physical/numerical guardrails remain satisfied.
"""
from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from run_chapter2_fair_calibration import (
    HERE, TEACHER, ZhangLi, _periodic_worker, _screen_worker,
    _strict_time_worker, build_records, clean, equilibrium_shear_modulus,
    fair_frequency_metrics, file_record, frequency_masks, frequency_scales,
    load_config, load_dma, object_hash, parallel_map, payload_bounds,
    payload_to_vector, vector_to_payload, write_json,
)


PROTOCOL_PATH = HERE / "chapter2_guarded_incremental_protocol.json"
REGISTRY_PATH = HERE / "chapter2_model_registry.json"
FAIR_PROTOCOL_PATH = HERE / "chapter2_fair_comparison_protocol.json"
CONFIG_PATH = HERE / "model_comparison_config.json"
SPLIT_PATH = HERE / "model_comparison_split.json"
ACTIVE_RESULTS = TEACHER / "results_chapter2_fair_comparison"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scientific_payload(raw: dict) -> dict:
    """Keep only fields consumed by the constitutive implementation."""
    return {key: copy.deepcopy(raw[key]) for key in
            ("zhang_li", "tau_s", "p_alpha", "Gneq_MPa", "a_ogden", "flows",
             "transition_tau_s") if key in raw}


def candidate(model: str, cid: str, seed: int, stage: str, vector,
              anchor: dict, config: dict) -> dict:
    lo, hi = payload_bounds(model, config)
    vector = np.clip(np.asarray(vector, float), lo, hi)
    return {"candidate_id": cid, "seed": int(seed), "stage": stage,
            "budget_group": "guarded_incremental", "vector": vector,
            "payload": vector_to_payload(model, vector, anchor, config)}


def archived_payloads(model: str) -> list[tuple[str, dict]]:
    roots = [
        TEACHER / ("5_T1_OgdenPRF" if model == "T1" else "6_T2_GroupedOgdenPRF") / "results" / "parameters",
        TEACHER / "results_ogden_teacher_models" / "stage4_block_coordinate" / "parameters",
        TEACHER / "results_ogden_teacher_models" / "stage5_joint_time_frequency" / "parameters",
    ]
    found, seen = [], set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob(f"{model}*parameters.json")):
            try:
                raw = load_json(path)
                payload = scientific_payload(raw)
                key = sha256(json.dumps(clean(payload), sort_keys=True).encode()).hexdigest()
                if key not in seen and all(name in payload for name in
                                            ("p_alpha", "Gneq_MPa", "a_ogden", "flows")):
                    found.append((path.stem, payload)); seen.add(key)
            except Exception:
                continue
    active = ACTIVE_RESULTS / "parameters" / f"{model}_chapter2_joint_evaluation_parameters.json"
    if active.exists():
        raw = scientific_payload(load_json(active))
        key = sha256(json.dumps(clean(raw), sort_keys=True).encode()).hexdigest()
        if key not in seen:
            found.append(("equal_weight_active_archive", raw))
    return found


def local_cloud(model: str, anchor: dict, config: dict, protocol: dict) -> list[dict]:
    anchor_v = payload_to_vector(model, anchor)
    lo, hi = payload_bounds(model, config)
    groups = 1 if model == "T1" else 2
    flow_end = 2 + 5 * groups
    output = [candidate(model, f"{model}:baseline", -1, "old_guarded_baseline",
                        anchor_v, anchor, config)]
    sources = archived_payloads(model)
    for source_index, (label, payload) in enumerate(sources):
        try:
            source_v = payload_to_vector(model, payload)
        except Exception:
            continue
        output.append(candidate(model, f"{model}:archive:{source_index}", -2,
                                f"archive_{label}", source_v, anchor, config))
        for alpha_index, alpha in enumerate((0.04, 0.08, 0.15, 0.25, 0.40, 0.60)):
            vector = (1.0-alpha)*anchor_v + alpha*source_v
            output.append(candidate(model, f"{model}:interp:{source_index}:{alpha_index}",
                                    -2, f"anchor_interpolation_{alpha:.2f}", vector,
                                    anchor, config))

    seeds = protocol["incremental_search"]["seeds"]
    count = int(protocol["incremental_search"][f"{model}_local_candidates_per_seed"])
    for seed in seeds:
        rng = np.random.default_rng(seed + (0 if model == "T1" else 10000))
        for index in range(count):
            block = index % 4
            vector = anchor_v.copy()
            if block == 0:  # Ogden energy
                vector[0] += rng.normal(0.0, 0.10)
                vector[1] += rng.normal(0.0, 0.18)
                stage = "energy"
            elif block == 1:  # flow
                vector[2:flow_end] += rng.normal(0.0, 0.035, flow_end-2) * (hi-lo)[2:flow_end]
                stage = "flow"
            elif block == 2:  # softmax shares
                vector[flow_end:] += rng.normal(0.0, 0.22, len(vector)-flow_end)
                stage = "spectrum"
            else:  # small joint move
                vector[0] += rng.normal(0.0, 0.06)
                vector[1] += rng.normal(0.0, 0.10)
                vector[2:flow_end] += rng.normal(0.0, 0.018, flow_end-2) * (hi-lo)[2:flow_end]
                vector[flow_end:] += rng.normal(0.0, 0.12, len(vector)-flow_end)
                stage = "joint"
            output.append(candidate(model, f"{model}:{seed}:{stage}:{index}", seed,
                                    stage, vector, anchor, config))
    unique = {}
    for row in output:
        key = sha256(np.asarray(row["vector"], float).tobytes()).hexdigest()
        unique.setdefault(key, row)
    return list(unique.values())


def q_guard(row: dict, protocol: dict) -> bool:
    gate = protocol["training_guardrails"]
    return (row.get("success", False)
            and row.get("q_ordering_accuracy", -np.inf) >= gate["minimum_q_ordering_accuracy"]-1e-12
            and row.get("q_log_RMSE", np.inf) <= gate["maximum_q_log_RMSE"])


def physical_guard(row: dict, protocol: dict) -> bool:
    gate = protocol["training_guardrails"]
    return (q_guard(row, protocol)
            and np.isfinite(row.get("L_time", np.inf))
            and row.get("max_detFv_error", np.inf) <= gate["maximum_detFv_error"]
            and row.get("min_dissipation_MPa_per_s", -np.inf) >= gate["minimum_dissipation_MPa_per_s"])


def choose_screen_shortlist(rows: list[dict], baseline: dict, protocol: dict) -> list[dict]:
    eligible = [row for row in rows if q_guard(row, protocol)]
    if not eligible:
        return [baseline]
    selected = {baseline["candidate_id"]: baseline}
    def add(items):
        for row in items:
            selected[row["candidate_id"]] = row
    ratio = lambda row: max(row["L_time"]/baseline["L_time"],
                            row["L_frequency"]/baseline["L_frequency"])
    add(sorted(eligible, key=ratio)[:10])
    add(sorted([r for r in eligible if r["L_frequency"] <= 1.15*baseline["L_frequency"]],
               key=lambda r: r["L_time"])[:4])
    add(sorted([r for r in eligible if r["L_time"] <= 1.15*baseline["L_time"]],
               key=lambda r: r["L_frequency"])[:4])
    for seed in protocol["incremental_search"]["seeds"]:
        pool = [row for row in eligible if row["seed"] == seed]
        if pool:
            add([min(pool, key=ratio)])
    limit = int(protocol["incremental_search"]["strict_time_shortlist_per_model"])
    rest = [row for cid, row in selected.items() if cid != baseline["candidate_id"]]
    return [baseline] + sorted(rest, key=ratio)[:max(0, limit-1)]


def choose_periodic_shortlist(rows: list[dict], baseline: dict,
                              protocol: dict) -> list[dict]:
    eligible = [row for row in rows if physical_guard(row, protocol)]
    selected = {baseline["candidate_id"]: baseline}
    # Local-linear frequency estimates are used only to allocate expensive
    # periodic evaluations.  Final replacement uses full-cycle metrics.
    ratio = lambda row: max(row["L_time"]/baseline["L_time"],
                            row["L_frequency"]/baseline["L_frequency"])
    for row in sorted(eligible, key=ratio):
        if row["candidate_id"] != baseline["candidate_id"]:
            selected[row["candidate_id"]] = row
    limit = int(protocol["incremental_search"]["strict_periodic_shortlist_per_model"])
    return list(selected.values())[:limit]


def periodic_metrics(model: str, rows: list[dict], dma, mask, scales,
                     config: dict, workers: int, label: str) -> tuple[list[dict], list[dict]]:
    frequencies = np.asarray(dma.frequency_hz[mask], float)
    ppc = int(config["dma"]["strict_periodic_points_per_cycle"])
    tasks = []
    for row in rows:
        for frequency in frequencies:
            tasks.append((model, row["candidate_id"], row["payload"], float(frequency),
                          config["dma"]["calibration_amplitude_fraction"], ppc,
                          config["dma"]["minimum_cycles"], config["dma"]["maximum_cycles"],
                          config["dma"]["steady_tolerance"]))
    periodic = parallel_map(_periodic_worker, tasks, workers, label)
    enriched = []
    for row in rows:
        values = sorted([item for item in periodic
                         if item["candidate_id"] == row["candidate_id"]],
                        key=lambda item: item["frequency_hz"])
        if len(values) != len(frequencies) or any(not item.get("success", False)
                                                   or not item.get("steady", False)
                                                   for item in values):
            continue
        class MiniDMA: pass
        mini = MiniDMA()
        mini.frequency_hz = frequencies
        mini.frequency_storage_MPa = np.asarray(dma.frequency_storage_MPa[mask], float)
        mini.frequency_loss_MPa = np.asarray(dma.frequency_loss_MPa[mask], float)
        storage = np.asarray([item["storage_MPa"] for item in values], float)
        loss = np.asarray([item["loss_MPa"] for item in values], float)
        enriched_row = copy.deepcopy(row)
        enriched_row.update(fair_frequency_metrics(mini, np.ones(len(frequencies), bool),
                                                    storage, loss, scales))
        enriched_row["periodic_prediction"] = {
            "frequency_hz": frequencies, "storage_MPa": storage,
            "loss_MPa": loss, "points_per_cycle": ppc,
            "steady": True,
        }
        enriched.append(enriched_row)
    return enriched, periodic


def dominates(new: dict, old: dict, protocol: dict) -> bool:
    eps = float(protocol["nondegradation_selection"]["minimum_relative_improvement_each_axis"])
    return (new["L_time"] <= old["L_time"]*(1.0-eps)
            and new["L_frequency"] <= old["L_frequency"]*(1.0-eps))


def active_parameter(model: str, selected: dict, baseline: dict, protocol: dict,
                     config: dict, split_path: Path, baseline_path: Path) -> dict:
    payload = scientific_payload(selected["payload"])
    payload.update({
        "schema_version": protocol["schema_version"],
        "model": f"{model}-Ogden-1",
        "role": "active chapter-2 guarded joint-evaluation parameter set",
        "selected_candidate_id": selected["candidate_id"],
        "selection_rule": "replace old guarded baseline only if strict relaxation and strict periodic DMA both improve",
        "baseline_path": str(baseline_path),
        "baseline_candidate_id": baseline["candidate_id"],
        "baseline_retained": selected["candidate_id"] == baseline["candidate_id"],
        "training_L_time": selected["L_time"],
        "training_L_frequency": selected["L_frequency"],
        "training_joint_score": float(np.sqrt(0.5*selected["L_time"]**2 + 0.5*selected["L_frequency"]**2)),
        "training_time_metrics": selected["training_time_metrics"],
        "held_out_time_metrics": selected["held_out_time_metrics"],
        "held_out_frequency_metrics": selected["held_out_frequency_metrics"],
        "guardrails": protocol["training_guardrails"],
        "configuration_sha256": file_record(CONFIG_PATH)["sha256"],
        "protocol_sha256": file_record(PROTOCOL_PATH)["sha256"],
        "split_sha256": file_record(split_path)["sha256"],
        "data": file_record(config["data"]["viscoelastic"]),
        "frequency_dma_amplitude_fraction": config["dma"]["calibration_amplitude_fraction"],
        "amplitude_dma_role": "external diagnostic only",
    })
    return payload


def update_active_outputs(model: str, selected: dict, baseline: dict, protocol: dict,
                          config: dict, baseline_path: Path, output: Path):
    active_path = ACTIVE_RESULTS / "parameters" / f"{model}_chapter2_joint_evaluation_parameters.json"
    archive_dir = ACTIVE_RESULTS / "parameters" / "archive_equal_weight_selection"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / active_path.name
    if active_path.exists() and not archive.exists():
        shutil.copy2(active_path, archive)
    payload = active_parameter(model, selected, baseline, protocol, config,
                               SPLIT_PATH, baseline_path)
    write_json(active_path, payload)
    model_dir = TEACHER / ("5_T1_OgdenPRF" if model == "T1" else "6_T2_GroupedOgdenPRF")
    write_json(model_dir / "results" / "parameters" /
               f"{model}_Ogden_1_active_chapter2_parameters.json", payload)
    write_json(output / "parameters" / f"{model}_guarded_incremental_active.json", payload)

    strict_path = ACTIVE_RESULTS / model / "strict_final_candidates.json"
    strict_rows = load_json(strict_path) if strict_path.exists() else []
    strict_rows = [row for row in strict_rows if row.get("candidate_id") != selected["candidate_id"]]
    strict_rows.append(selected)
    write_json(strict_path, strict_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=TEACHER / "results_chapter2_guarded_incremental")
    args = parser.parse_args()
    output = args.output.resolve(); (output / "parameters").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    protocol = load_json(PROTOCOL_PATH)
    fair_protocol = load_json(FAIR_PROTOCOL_PATH)
    registry = load_json(REGISTRY_PATH)
    config, _ = load_config(CONFIG_PATH)
    split = load_json(SPLIT_PATH)
    records = build_records(config, split)
    dma = load_dma(config["data"]["viscoelastic"])
    train_mask, held_mask = frequency_masks(dma.frequency_hz,
                                             config["split"]["frequency_validation_hz"])
    scales = frequency_scales(dma, train_mask, config)
    equilibrium_G0 = equilibrium_shear_modulus(ZhangLi(**config["zhang_li"]))
    dma_payload = {"frequency_hz": dma.frequency_hz,
                   "frequency_storage_MPa": dma.frequency_storage_MPa,
                   "frequency_loss_MPa": dma.frequency_loss_MPa}
    summaries, candidate_tables = {}, []
    started_all = time.time()

    for model in ("T1", "T2"):
        (output / model).mkdir(parents=True, exist_ok=True)
        baseline_path = (HERE / protocol["baseline_policy"][model]).resolve()
        anchor = scientific_payload(load_json(baseline_path))
        cloud = local_cloud(model, anchor, config, protocol)
        tasks = [(model, item, records, dma_payload, train_mask, scales,
                  fair_protocol["time_objective"], equilibrium_G0) for item in cloud]
        screen = parallel_map(_screen_worker, tasks, args.workers, f"{model} guarded screen")
        baseline_screen = next(row for row in screen if row["candidate_id"] == f"{model}:baseline")
        shortlist = choose_screen_shortlist(screen, baseline_screen, protocol)
        strict_tasks = [(model, row, records, fair_protocol["time_objective"])
                        for row in shortlist]
        strict = parallel_map(_strict_time_worker, strict_tasks, args.workers,
                              f"{model} strict relaxation")
        for row in strict:
            if row.get("success"):
                from run_chapter2_fair_calibration import local_complex_modulus
                storage, loss = local_complex_modulus(model, row["payload"],
                                                      dma.frequency_hz, equilibrium_G0)
                row.update(fair_frequency_metrics(dma, train_mask, storage, loss, scales))
        baseline_strict = next(row for row in strict if row["candidate_id"] == f"{model}:baseline")
        periodic_shortlist = choose_periodic_shortlist(strict, baseline_strict, protocol)
        strict_periodic, periodic_rows = periodic_metrics(
            model, periodic_shortlist, dma, train_mask, scales, config, args.workers,
            f"{model} strict periodic training DMA")
        baseline_final = next(row for row in strict_periodic
                              if row["candidate_id"] == f"{model}:baseline")
        feasible = [row for row in strict_periodic
                    if physical_guard(row, protocol) and dominates(row, baseline_final, protocol)]
        if feasible:
            selected = min(feasible, key=lambda row: (
                max(row["L_time"]/baseline_final["L_time"],
                    row["L_frequency"]/baseline_final["L_frequency"]),
                row["Rneq_condition_RMSE"], row["q_log_RMSE"]))
            decision = "new candidate dominates the old baseline on both strict axes"
        else:
            selected = copy.deepcopy(baseline_final)
            decision = "no candidate improved both strict axes; old guarded baseline retained"

        # Held-out evaluation is deliberately delayed until after the training-only decision.
        selected_time = selected["held_out_time_metrics"]
        held_eval, held_periodic = periodic_metrics(
            model, [selected], dma, held_mask, scales, config, args.workers,
            f"{model} held-out periodic DMA")
        selected = held_eval[0]
        selected["held_out_time_metrics"] = selected_time
        selected["held_out_frequency_metrics"] = {
            key: selected[key] for key in ("storage_RMSE_MPa", "loss_RMSE_MPa",
                                           "storage_NRMSE", "loss_NRMSE",
                                           "tan_delta_RMSE", "L_frequency", "points")}
        # Restore training periodic metrics overwritten by held-out enrichment.
        training_selected = next(row for row in strict_periodic
                                 if row["candidate_id"] == selected["candidate_id"])
        for key in ("storage_RMSE_MPa", "loss_RMSE_MPa", "storage_NRMSE",
                    "loss_NRMSE", "tan_delta_RMSE", "L_frequency", "points",
                    "periodic_prediction"):
            selected[key] = copy.deepcopy(training_selected[key])
        selected["selection_decision"] = decision
        selected["baseline_metrics"] = {
            "training_L_time": baseline_final["L_time"],
            "training_L_frequency": baseline_final["L_frequency"],
            "q_log_RMSE": baseline_final["q_log_RMSE"],
            "q_ordering_accuracy": baseline_final["q_ordering_accuracy"],
            "Rneq_condition_RMSE": baseline_final["Rneq_condition_RMSE"],
        }
        update_active_outputs(model, selected, baseline_final, protocol, config,
                              baseline_path, output)
        write_json(output / model / "screen_candidates.json", screen)
        write_json(output / model / "strict_time_candidates.json", strict)
        write_json(output / model / "strict_periodic_candidates.json", strict_periodic)
        write_json(output / model / "periodic_training_raw.json", periodic_rows)
        write_json(output / model / "periodic_held_out_raw.json", held_periodic)
        summaries[model] = {
            "decision": decision,
            "selected_candidate_id": selected["candidate_id"],
            "baseline_candidate_id": baseline_final["candidate_id"],
            "baseline_training_L_time": baseline_final["L_time"],
            "selected_training_L_time": selected["L_time"],
            "baseline_training_L_frequency": baseline_final["L_frequency"],
            "selected_training_L_frequency": selected["L_frequency"],
            "selected_q_log_RMSE": selected["q_log_RMSE"],
            "selected_q_ordering_accuracy": selected["q_ordering_accuracy"],
            "selected_Rneq_condition_RMSE": selected["Rneq_condition_RMSE"],
            "selected_held_out_L_time": selected["held_out_time_metrics"]["L_time"],
            "selected_held_out_L_frequency": selected["held_out_frequency_metrics"]["L_frequency"],
            "screen_candidates": len(screen),
            "strict_time_candidates": len(strict),
            "strict_periodic_candidates": len(strict_periodic),
        }
        for row in strict_periodic:
            candidate_tables.append({"model": model, "candidate_id": row["candidate_id"],
                                     "stage": row["stage"], "L_time": row["L_time"],
                                     "L_frequency": row["L_frequency"],
                                     "q_log_RMSE": row["q_log_RMSE"],
                                     "q_ordering_accuracy": row["q_ordering_accuracy"],
                                     "Rneq_condition_RMSE": row["Rneq_condition_RMSE"],
                                     "dominates_baseline": dominates(row, baseline_final, protocol),
                                     "selected": row["candidate_id"] == selected["candidate_id"]})
        print(model, decision, summaries[model], flush=True)

    # T0 is copied by reference only; its file and results are intentionally untouched.
    summary = {
        "schema_version": protocol["schema_version"],
        "T0_action": "not rerun; existing qualified T0 remains byte-for-byte unchanged",
        "models": summaries,
        "elapsed_s": time.time()-started_all,
        "input_hashes": {path.name: file_record(path)["sha256"] for path in
                         (CONFIG_PATH, SPLIT_PATH, FAIR_PROTOCOL_PATH, PROTOCOL_PATH, REGISTRY_PATH)},
    }
    write_json(output / "guarded_incremental_summary.json", summary)
    pd.DataFrame(candidate_tables).to_csv(output / "tables" /
                                           "strict_periodic_candidate_comparison.csv",
                                           index=False, encoding="utf-8-sig")
    print(json.dumps(clean(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
