"""Physics, convergence and inference-cost checks for the frozen stage-2 model."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from dual_potential.solvers import MaterialPointSolver
from dual_potential.tensor import deformation_gradient, rotation
from stage2_rollout import load_stage2_material
from stage2_training import load_datasets, predict

PARAMETERS = ROOT.parent / "2_TeacherModel" / "results_chapter2_fair_comparison" / "parameters" / "T2_chapter2_joint_evaluation_parameters.json"
CHECKPOINT = ROOT / "checkpoints" / "stage2" / "dpicnn_stage2_final.pt"
CACHE = ROOT / "audit" / "stage2" / "stage2_teacher_cache.npz"
OUTPUT = ROOT / "results" / "stage2"


def scalar(value):
    return float(torch.as_tensor(value).detach().cpu())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physics_checks(material):
    I = torch.eye(3, dtype=torch.float64)
    zero = material.response(I, material.identity_state(), "shear", create_graph=False)

    F = deformation_gradient(torch.tensor(0.28), "shear") @ deformation_gradient(torch.tensor(0.21), "uniaxial")
    Fv = material.identity_state()
    for branch in range(7):
        generator = torch.tensor([[0.01*(branch+1), 0.004*(branch+1), 0.0],
                                  [0.004*(branch+1), -0.006*(branch+1), 0.0],
                                  [0.0, 0.0, -0.004*(branch+1)]], dtype=torch.float64)
        Fv[branch] = torch.matrix_exp(generator)
    Q = rotation(torch.tensor([1.0, -2.0, 0.7]), 0.61)
    base = material.response(F, Fv, "shear", create_graph=False)
    objective = material.response(Q@F, Fv, "shear", create_graph=False)
    isotropic = material.response(F@Q, Fv@Q, "shear", create_graph=False)

    objective_stress = Q @ base["cauchy_unpressurized_MPa"] @ Q.T
    # The right-rotation pair keeps Fe=F Fv^{-1} unchanged, hence it is a
    # direct material-isotropy/state-covariance check for this representation.
    result = {
        "reference_energy_abs_MPa": abs(scalar(zero["total_energy_MPa"])),
        "reference_stress_abs_MPa": abs(scalar(zero["measured_stress_MPa"])),
        "reference_force_max_MPa": scalar(zero["driving_force_MPa"].abs().max()),
        "reference_flow_max_per_s": scalar(zero["flow_rate_per_s"].abs().max()),
        "objectivity_energy_relative_error": abs(scalar(objective["total_energy_MPa"]-base["total_energy_MPa"])) / max(abs(scalar(base["total_energy_MPa"])), 1e-14),
        "objectivity_stress_relative_error": scalar((objective["cauchy_unpressurized_MPa"]-objective_stress).abs().max()) / max(scalar(objective_stress.abs().max()), 1e-14),
        "isotropy_energy_relative_error": abs(scalar(isotropic["total_energy_MPa"]-base["total_energy_MPa"])) / max(abs(scalar(base["total_energy_MPa"])), 1e-14),
        "isotropy_flow_max_abs_error_per_s": scalar((isotropic["flow_rate_per_s"]-base["flow_rate_per_s"]).abs().max()),
        "maximum_force_trace_MPa": scalar(torch.diagonal(base["driving_force_MPa"], dim1=-2, dim2=-1).sum(-1).abs().max()),
        "minimum_branch_dissipation_MPa_per_s": scalar(base["branch_dissipation_MPa_per_s"].min()),
        "detF_error": abs(scalar(torch.linalg.det(F))-1.0),
        "maximum_initial_detFv_error": scalar((torch.linalg.det(Fv)-1.0).abs().max()),
    }
    return result


def convexity_checks(material, datasets):
    model = material.dissipation_potential
    data = datasets["validation_high_medium"]
    # Sample every available branch over objective states and sweep force magnitude.
    indices = torch.linspace(0, data.count-1, 256, dtype=torch.float64).round().long()
    branch = data.branch[indices]
    features = {name: value[indices] for name, value in data.features.items()}
    slopes = []
    for j_value in torch.logspace(-14, 4, 96, dtype=torch.float64):
        j = torch.full((len(indices),), j_value, dtype=torch.float64)
        slopes.append(model.potential_and_slope(j, branch, features)[1])
    slopes = torch.stack(slopes)
    differences = torch.diff(slopes, dim=0)
    return {
        "sample_count": len(indices), "force_grid_count": len(slopes),
        "minimum_slope": scalar(slopes.min()),
        "minimum_slope_increment": scalar(differences.min()),
        "monotone_slope": bool((differences >= -1e-12).all()),
        "positive_slope": bool((slopes > 0).all()),
    }


def time_step_checks(material):
    solver = MaterialPointSolver(material)
    rows = []
    states = {}
    for dt in (0.01, 0.005, 0.0025):
        start = perf_counter()
        history = solver.relaxation(0.03, 0.01, 0.04, "shear", time_step_s=dt)
        states[dt] = history.Fv[-1]
        rows.append({
            "dt_s": dt, "point_count": len(history.time_s),
            "terminal_stress_MPa": scalar(history.measured_stress_MPa[-1]),
            "maximum_detFv_error": scalar((torch.linalg.det(history.Fv)-1.0).abs().max()),
            "minimum_dissipation_MPa_per_s": scalar(history.branch_dissipation_MPa_per_s.min()),
            "maximum_newton_residual": max((d.residual_norm for step in history.diagnostics for d in step), default=0.0),
            "maximum_subdivision_depth": max((d.subdivisions for step in history.diagnostics for d in step), default=0),
            "wall_time_s": perf_counter()-start,
        })
    fine_stress = rows[-1]["terminal_stress_MPa"]
    fine_state = states[0.0025]
    scale = torch.linalg.matrix_norm(fine_state-torch.eye(3), dim=(-2,-1)).square().mean().sqrt().clamp_min(1e-14)
    for row in rows:
        row["terminal_stress_relative_to_dt_quarter"] = abs(row["terminal_stress_MPa"]-fine_stress)/max(abs(fine_stress), 1e-14)
        row["terminal_state_relative_to_dt_quarter"] = scalar(torch.linalg.matrix_norm(states[row["dt_s"]]-fine_state, dim=(-2,-1)).square().mean().sqrt()/scale)
    return rows


def cost_checks(material, datasets):
    model = material.dissipation_potential
    data = datasets["validation_high_medium"]
    count = min(4096, data.count)
    subset = data.select(torch.arange(count))
    for _ in range(5): predict(model, subset)
    start = perf_counter()
    repeats = 50
    for _ in range(repeats): predict(model, subset)
    point_seconds = (perf_counter()-start)/(repeats*count)

    F = deformation_gradient(torch.tensor(0.25), "shear")
    state = material.identity_state()
    for _ in range(3): material.response(F, state, "shear", create_graph=False)
    start = perf_counter(); repeats = 20
    for _ in range(repeats): material.response(F, state, "shear", create_graph=False)
    response_seconds = (perf_counter()-start)/repeats
    return {"pointwise_batch_size": count,
            "pointwise_potential_seconds_per_branch_point": point_seconds,
            "seven_branch_material_response_seconds": response_seconds}


def main():
    torch.set_default_dtype(torch.float64)
    material, saved = load_stage2_material(CHECKPOINT, PARAMETERS)
    datasets, _ = load_datasets(CACHE)
    physics = physics_checks(material)
    convexity = convexity_checks(material, datasets)
    convergence = time_step_checks(material)
    cost = cost_checks(material, datasets)
    pd.DataFrame(convergence).to_csv(OUTPUT / "time_step_convergence.csv", index=False)
    acceptance = {
        "reference_state": max(physics["reference_energy_abs_MPa"], physics["reference_stress_abs_MPa"], physics["reference_force_max_MPa"], physics["reference_flow_max_per_s"]) <= 1e-10,
        "objectivity": max(physics["objectivity_energy_relative_error"], physics["objectivity_stress_relative_error"]) <= 1e-8,
        "isotropy_state_covariance": max(physics["isotropy_energy_relative_error"], physics["isotropy_flow_max_abs_error_per_s"]) <= 1e-8,
        "deviatoric_force": physics["maximum_force_trace_MPa"] <= 1e-8,
        "nonnegative_dissipation": physics["minimum_branch_dissipation_MPa_per_s"] >= -1e-12 and min(row["minimum_dissipation_MPa_per_s"] for row in convergence) >= -1e-12,
        "isochoric_external_and_internal": max(physics["detF_error"], physics["maximum_initial_detFv_error"], max(row["maximum_detFv_error"] for row in convergence)) <= 1e-8,
        "convex_monotone_potential": convexity["monotone_slope"] and convexity["positive_slope"],
        "newton_convergence": max(row["maximum_newton_residual"] for row in convergence) <= 1.01e-8,
        "time_step_terminal_stress_below_one_percent": convergence[1]["terminal_stress_relative_to_dt_quarter"] <= 0.01,
        "all_finite": all(math.isfinite(value) for value in (*physics.values(), *convexity.values(), *(v for row in convergence for v in row.values()), *cost.values()) if not isinstance(value, bool)),
    }
    result = {
        "schema_version": "2026-08-30-chapter3-stage2-physics-v1",
        "architecture": saved["architecture"],
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
        "physics": physics, "convexity": convexity,
        "time_step_convergence": convergence, "cost": cost,
        "acceptance": acceptance, "passed": all(acceptance.values()),
    }
    (OUTPUT / "physics_validation_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "acceptance": acceptance,
                      "convergence": convergence, "cost": cost}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
