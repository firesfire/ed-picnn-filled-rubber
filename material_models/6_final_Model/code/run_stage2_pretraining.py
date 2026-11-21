"""Run fair multi-seed PICNN-A/B/C stage-2 teacher pretraining."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from stage2_training import load_datasets, train_model

CACHE = ROOT / "audit" / "stage2" / "stage2_teacher_cache.npz"
RESULTS = ROOT / "results" / "stage2"
CHECKPOINTS = ROOT / "checkpoints" / "stage2"
RESULTS.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
ARCHITECTURES = ("PICNN-A", "PICNN-B", "PICNN-C")
SEEDS = (20260828, 20260829, 20260830)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets, spectrum = load_datasets(CACHE, device)
    runs, histories = [], []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            print(f"training {architecture}, seed={seed}", flush=True)
            model, result = train_model(architecture, seed, datasets, spectrum,
                                        maximum_epochs=500, minimum_epochs=250,
                                        patience=80, batch_size=8192, learning_rate=0.003)
            checkpoint = CHECKPOINTS / f"{architecture.lower().replace('-', '_')}_seed_{seed}.pt"
            torch.save({
                "schema_version": "2026-08-30-chapter3-stage2-v1",
                "architecture": architecture, "seed": seed,
                "state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                "parameter_count": model.parameter_count,
                "teacher": "T2:20260829:spectrum:22",
                "fixed_nonequilibrium_energy": "T2-Ogden-1",
                "experimental_labels_accessed": False,
                "energy_supervision_used": False,
            }, checkpoint)
            result["checkpoint"] = str(checkpoint)
            runs.append(result)
            for row in result["history"]:
                histories.append({"architecture": architecture, "seed": seed, **row})
            summary = {"architecture": architecture, "seed": seed,
                       "best_epoch": result["best_epoch"],
                       "completed_epochs": result["completed_epochs"],
                       "training_time_s": result["training_time_s"],
                       "parameter_count": result["parameter_count"],
                       "checkpoint": str(checkpoint)}
            for name, metric in result["metrics"].items():
                for key in ("flow_nrmse", "dissipation_nrmse", "mean_direction_error_degree",
                            "maximum_direction_error_degree", "low_force_flow_nrmse",
                            "fast_flow_nrmse", "slow_flow_nrmse",
                            "minimum_predicted_dissipation_MPa_per_s"):
                    summary[f"{name}_{key}"] = metric[key]
            result["flat_summary"] = summary

    architecture_rows = []
    for architecture in ARCHITECTURES:
        selected = [run for run in runs if run["architecture"] == architecture]
        validation = [run["metrics"]["validation_high_medium"]["flow_nrmse"] for run in selected]
        testing = [run["metrics"]["test_high_medium"]["flow_nrmse"] for run in selected]
        architecture_rows.append({
            "architecture": architecture,
            "parameter_count": selected[0]["parameter_count"],
            "median_validation_high_medium_flow_nrmse": statistics.median(validation),
            "range_validation_high_medium_flow_nrmse": max(validation)-min(validation),
            "median_test_high_medium_flow_nrmse": statistics.median(testing),
            "median_training_time_s": statistics.median(run["training_time_s"] for run in selected),
        })
    architecture_rows.sort(key=lambda row: row["median_validation_high_medium_flow_nrmse"])
    selected_architecture = architecture_rows[0]["architecture"]
    selected_run = min((run for run in runs if run["architecture"] == selected_architecture),
                       key=lambda run: run["metrics"]["validation_high_medium"]["flow_nrmse"])
    selected_checkpoint = CHECKPOINTS / "dpicnn_stage2_selected.pt"
    selected_source = Path(selected_run["checkpoint"])
    selected_checkpoint.write_bytes(selected_source.read_bytes())

    with (RESULTS / "training_history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=histories[0].keys()); writer.writeheader(); writer.writerows(histories)
    flat = [run["flat_summary"] for run in runs]
    with (RESULTS / "seed_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat[0].keys()); writer.writeheader(); writer.writerows(flat)
    with (RESULTS / "architecture_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=architecture_rows[0].keys()); writer.writeheader(); writer.writerows(architecture_rows)
    detail = {"schema_version": "2026-08-30-chapter3-stage2-v1", "device": device,
              "architectures": list(ARCHITECTURES), "seeds": list(SEEDS),
              "training_branch_points": datasets["train"].count,
              "selected_architecture": selected_architecture,
              "selected_seed": selected_run["seed"],
              "selected_checkpoint": str(selected_checkpoint),
              "runs": [{k:v for k,v in run.items() if k not in ("history", "flat_summary")} for run in runs]}
    (RESULTS / "pretraining_summary.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected_architecture": selected_architecture,
                      "selected_seed": selected_run["seed"],
                      "architecture_comparison": architecture_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
