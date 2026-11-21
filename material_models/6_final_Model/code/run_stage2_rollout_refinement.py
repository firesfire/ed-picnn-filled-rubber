"""Guarded complete-history rollout refinement of the selected D-PICNN."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from stage2_rollout import load_stage2_material, load_trajectory, differentiable_complete_history_loss
from stage2_training import load_datasets, evaluate, loss_terms

PARAMETERS = ROOT.parent / "2_TeacherModel" / "results_chapter2_fair_comparison" / "parameters" / "T2_chapter2_joint_evaluation_parameters.json"
CACHE = ROOT / "audit" / "stage2" / "stage2_teacher_cache.npz"
METADATA = ROOT / "audit" / "stage2" / "stage2_trajectory_metadata.json"
SOURCE = ROOT / "checkpoints" / "stage2" / "dpicnn_stage2_selected.pt"
OUTPUT = ROOT / "checkpoints" / "stage2" / "dpicnn_stage2_rollout_refined.pt"
RESULT = ROOT / "results" / "stage2" / "rollout_refinement.json"


def main():
    device = "cpu"
    material, saved = load_stage2_material(SOURCE, PARAMETERS, device)
    datasets, _ = load_datasets(CACHE, device)
    baseline = evaluate(material.dissipation_potential, datasets["validation_high_medium"],
                        "validation_high_medium")
    optimizer = torch.optim.Adam(material.dissipation_potential.parameters(), lr=1e-6)
    # The high-confidence uniaxial candidate traj_0003 exceeded the frozen
    # substep limit during differentiable backpropagation and is retained as a
    # reported failed attempt. Two complete high-confidence shear histories
    # are therefore used without relaxing the numerical protocol.
    selected_ids = ("traj_0027",)
    history = []
    start = perf_counter()
    for epoch, identifier in enumerate(selected_ids, start=1):
        item, trajectory = load_trajectory(CACHE, METADATA, identifier, device)
        optimizer.zero_grad(set_to_none=True)
        rollout_loss, components = differentiable_complete_history_loss(material, item, trajectory)
        # A deterministic pointwise anchor limits drift away from the full
        # high/medium teacher state distribution.
        generator = torch.Generator().manual_seed(20260830 + epoch)
        index = torch.randperm(datasets["train"].count, generator=generator)[:8192]
        anchor = loss_terms(material.dissipation_potential, datasets["train"], index)["total"]
        total = rollout_loss + 0.02 * anchor
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(material.dissipation_potential.parameters(), 1.0)
        optimizer.step()
        history.append({"epoch": epoch, "trajectory_id": identifier,
                        "rollout_loss": float(rollout_loss.detach()),
                        "state_loss": float(components["state_loss"].detach()),
                        "stress_loss": float(components["stress_loss"].detach()),
                        "pointwise_anchor_loss": float(anchor.detach()),
                        "gradient_norm": float(gradient.detach()),
                        "elapsed_s": perf_counter()-start})
    refined = evaluate(material.dissipation_potential, datasets["validation_high_medium"],
                       "validation_high_medium")
    accepted = refined["flow_nrmse"] <= 1.05 * baseline["flow_nrmse"]
    if accepted:
        torch.save({**saved,
            "schema_version": "2026-08-30-chapter3-stage2-rollout-refined-v1",
            "state_dict": {k:v.detach().cpu() for k,v in material.dissipation_potential.state_dict().items()},
            "rollout_refinement_histories": list(selected_ids)}, OUTPUT)
    else:
        OUTPUT.write_bytes(SOURCE.read_bytes())
    result = {"accepted": accepted, "source_checkpoint": str(SOURCE),
              "output_checkpoint": str(OUTPUT), "histories": history,
              "baseline_validation_flow_nrmse": baseline["flow_nrmse"],
              "refined_validation_flow_nrmse": refined["flow_nrmse"],
              "guard_maximum_relative_pointwise_degradation": 0.05,
              "failed_rollout_refinement_attempts": [
                  {"trajectory_id": "traj_0003",
                   "reason": "differentiable implicit branch 0 exceeded maximum subdivision depth 10"},
                  {"trajectory_id": "traj_0033",
                   "reason": "after a 1e-5 rollout update, differentiable implicit branch 0 exceeded maximum subdivision depth 10"}],
              "wall_time_s": perf_counter()-start}
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
