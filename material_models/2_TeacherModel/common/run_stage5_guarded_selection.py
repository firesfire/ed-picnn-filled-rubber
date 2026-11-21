"""Repair stage-5 selection with an explicit nonequilibrium-peak guardrail.

The original normalized-ideal selector can choose a DMA-favourable candidate
whose q magnitude is physically unusable.  This script preserves that result
as an audit trail, reconstructs deterministic screened candidates, strictly
replays a q-admissible shortlist, performs full-cycle 0.1% DMA, and writes a
guarded joint checkpoint.  Amplitude DMA is never read here.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
for path in (HERE, TEACHER / "3_PRF" / "code", TEACHER / "4_GroupedPRF" / "code",
             TEACHER / "5_T1_OgdenPRF" / "code", TEACHER / "6_T2_GroupedOgdenPRF" / "code"):
    sys.path.insert(0, str(path))

from comparison_core import ZhangLi, load_config, load_dma, load_relaxation, nondominated
from run_stage3_ogden_flow_calibration import make_records, write_json
from run_stage4_block_coordinate_refinement import config_hash, evaluate_many
from run_stage5_joint_time_frequency import (
    add_frequency_metrics, add_periodic_scores, generate_candidates, periodic_many,
    stage4_path, strict_table_row,
)

SEEDS = [20260828, 20260829, 20260830]
COUNTS = {"T1": 48, "T2": 64}
Q_LOG_RMSE_LIMIT = 0.75


def shortlist_from_screen(frame: pd.DataFrame, payload_map: dict[str, dict]) -> list[dict]:
    data = frame.copy()
    data["q_log_RMSE"] = np.sqrt(data["q_log_mse"].astype(float))
    admissible = data[(data["q_log_RMSE"] <= Q_LOG_RMSE_LIMIT) &
                      (data["q_ordering_accuracy"].astype(float) == 1.0)].copy()
    if admissible.empty:
        raise RuntimeError("No candidate passes q guardrail")
    t = admissible["time_score"].to_numpy(float)
    f = admissible["frequency_score_local"].to_numpy(float)
    tn = (t-t.min()) / max(float(t.max()-t.min()), 1e-14)
    fn = (f-f.min()) / max(float(f.max()-f.min()), 1e-14)
    chosen: list[str] = []
    for weight in (0.25, 0.40, 0.55, 0.70, 0.85):
        score = weight*tn + (1-weight)*fn + 0.01*admissible["spectrum_KL"].to_numpy(float)
        chosen.extend(admissible.iloc[np.argsort(score)[:2]]["config_hash"].tolist())
    chosen.extend(admissible.sort_values("frequency_score_local").head(2)["config_hash"].tolist())
    chosen.extend(admissible.sort_values("time_score").head(2)["config_hash"].tolist())
    result, seen = [], set()
    for key in chosen:
        if key not in seen and key in payload_map:
            result.append(payload_map[key]); seen.add(key)
    return result


def choose_guarded(rows: list[dict]):
    admissible = [row for row in rows
                  if np.sqrt(row["components"]["q_log_mse"]) <= Q_LOG_RMSE_LIMIT
                  and row["components"]["q_ordering_accuracy"] == 1.0
                  and row.get("periodic_all_steady", False)]
    if not admissible:
        raise RuntimeError("No strict periodic candidate passes q guardrail")
    front = nondominated(admissible, "time_score", "frequency_score_periodic")
    t = np.array([r["time_score"] for r in front]); f = np.array([r["frequency_score_periodic"] for r in front])
    tn = (t-t.min()) / max(float(t.max()-t.min()), 1e-14)
    fn = (f-f.min()) / max(float(f.max()-f.min()), 1e-14)
    # q amplitude is a primary physical observable, so it remains in selection
    # even after the hard guardrail.  Spectrum drift is a weak tie-breaker.
    qn = np.array([np.sqrt(r["components"]["q_log_mse"]) for r in front]) / Q_LOG_RMSE_LIMIT
    kl = np.array([r["spectrum_KL"] for r in front])
    score = np.hypot(tn, fn) + 0.20*qn + 0.02*kl
    return front[int(np.argmin(score))], front


def main():
    root = TEACHER / "results_ogden_teacher_models" / "stage5_joint_time_frequency"
    out = root / "guarded_selection"; (out / "parameters").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    config, _ = load_config(HERE / "model_comparison_config.json")
    split = json.loads((HERE / "model_comparison_split.json").read_text(encoding="utf-8"))
    records = make_records(load_relaxation(config["data"]["viscoelastic"]),
                           ZhangLi(**config["zhang_li"]), set(split["relaxation"]["held_out"]))
    dma = load_dma(config["data"]["viscoelastic"])
    train = (dma.frequency_hz < 5.0) | (dma.frequency_hz > 20.0)
    held = ~train
    screen = pd.read_csv(root / "tables" / "stage5_screen_candidates.csv")
    table_rows = []; summary = {}
    for model_name in ("T1", "T2"):
        base = json.loads(stage4_path(model_name).read_text(encoding="utf-8"))
        old_path = (TEACHER / ("3_PRF" if model_name == "T1" else "4_GroupedPRF") /
                    "results_paper_7branch" / "parameters" /
                    f"{model_name}_full_data_teacher_parameters.json")
        old_raw = json.loads(old_path.read_text(encoding="utf-8"))
        # generate_candidates expects p_alpha/Gneq naming; the stage-2 payload is
        # the canonical compatible old-flow anchor.
        from run_stage3_ogden_flow_calibration import stage2_path
        old = json.loads(stage2_path(model_name).read_text(encoding="utf-8"))
        generated, _ = generate_candidates(model_name, base, old, SEEDS, COUNTS[model_name])
        payload_map = {config_hash(p): p for p in generated}
        candidates = shortlist_from_screen(screen[screen.model == model_name], payload_map)
        # Always retain the time-optimal stage-4 anchor explicitly.
        if config_hash(base) not in {config_hash(p) for p in candidates}:
            candidates.append(base)
        strict = evaluate_many(model_name, candidates, records, True, workers=8)
        strict = [add_frequency_metrics(r, model_name, dma, train, held, base["p_alpha"])
                  for r in strict if r.get("success", False)]
        periodic = periodic_many(model_name, [r["payload"] for r in strict],
                                 dma.frequency_hz[train], 32, 8,
                                 min_cycles=4, max_cycles=14, tolerance=1e-3)
        strict = add_periodic_scores(strict, periodic, dma, train)
        selected, front = choose_guarded(strict)
        selected_key = config_hash(selected["payload"])
        front_keys = {config_hash(r["payload"]) for r in front}
        for rank, row in enumerate(sorted(strict, key=lambda r: r["time_score"]), 1):
            item = strict_table_row(row, model_name, rank, config_hash(row["payload"]) == selected_key)
            item["on_guarded_pareto"] = config_hash(row["payload"]) in front_keys
            table_rows.append(item)
        payload = copy.deepcopy(selected["payload"])
        payload.update({
            "schema_version": "2026-08-28-stage5-guarded-joint-v1",
            "stage": "joint_relaxation_frequency_dma_guarded",
            "frequency_dma_amplitude_fraction": 0.001,
            "frequency_dma_amplitude_percent": 0.1,
            "amplitude_dma_role": "external diagnostic only; never used for calibration or selection",
            "q_log_RMSE_guardrail": Q_LOG_RMSE_LIMIT,
            "selection_rule": "nearest joint Pareto compromise subject to q ordering and q-log-RMSE guardrail",
            "rejected_unguarded_checkpoint": str(root / "parameters" / f"{model_name}_Ogden_1_stage5_joint_parameters.json"),
            "stage5_time_score": selected["time_score"],
            "stage5_frequency_score_periodic": selected["frequency_score_periodic"],
        })
        name = f"{model_name}_Ogden_1_stage5_guarded_joint_parameters.json"
        write_json(out / "parameters" / name, payload)
        destination = TEACHER / ("5_T1_OgdenPRF" if model_name == "T1" else "6_T2_GroupedOgdenPRF") / "results" / "parameters"
        write_json(destination / name, payload)
        summary[model_name] = {
            "Gneq_MPa": payload["Gneq_MPa"], "a_ogden": payload["a_ogden"],
            "time_score": selected["time_score"],
            "frequency_score_periodic": selected["frequency_score_periodic"],
            "q_log_RMSE": float(np.sqrt(selected["components"]["q_log_mse"])),
            "Rneq_balanced_RMSE": float(np.sqrt(selected["components"]["Rneq_balanced_mse"])),
            "q_ordering_accuracy": selected["components"]["q_ordering_accuracy"],
            "full_ordering_accuracy": selected["full_ordering_accuracy"],
            "max_detFv_error": selected["max_detFv_error"],
            "min_dissipation": selected["min_dissipation"],
        }
        print(model_name, json.dumps(summary[model_name], ensure_ascii=False), flush=True)
    pd.DataFrame(table_rows).to_csv(out / "tables" / "stage5_guarded_strict_candidates.csv",
                                    index=False, encoding="utf-8-sig")
    write_json(out / "stage5_guarded_summary.json", {
        "frequency_amplitude_fraction": 0.001,
        "amplitude_dma_used": False,
        "q_log_RMSE_guardrail": Q_LOG_RMSE_LIMIT,
        "models": summary,
    })


if __name__ == "__main__":
    main()
