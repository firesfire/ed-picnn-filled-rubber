"""Independent full-tensor validation for the best diagnostic Stage-5B model."""

from __future__ import annotations

from project_paths import load_config

import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"code"))
from dual_potential.constitutive import MaterialParameters
from dual_potential.solvers import MaterialPointSolver
from run_stage2_physics_validation import convexity_checks, cost_checks, physics_checks, time_step_checks
from run_stage3_final_validation import curve_rows, full_tensor_dma_subset, summarize_relaxation
from run_stage5a_prescreen import deformation_path
from stage2_training import load_datasets
from stage3_real import load_real_data
from stage5b_energy import load_stage5b_material

CONFIG=ROOT/"configs"/"stage5b_epicnn_energy_finetuning.yaml"


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def total_energy(material,F):
    equilibrium=material.equilibrium_energy(F)
    nonequilibrium=torch.stack([material.nonequilibrium_energy(F,branch) for branch in range(7)]).sum()
    return equilibrium+nonequilibrium


def stability_screen(material):
    specifications={"uniaxial":(1.0,2.0,9),"shear":(0.0,.8,9),
                    "planar":(1.0,2.0,7),"equibiaxial":(1.0,1.5,7)}
    rng=np.random.default_rng(20260831);minimum_energy=float("inf");minimum_tangent=float("inf")
    minimum_acoustic=float("inf");finite=True;count=0
    for kind,(start,stop,size) in specifications.items():
        for value in np.linspace(start,stop,size):
            coordinate=torch.tensor(float(value),dtype=torch.float64,requires_grad=True)
            F=deformation_path(kind,coordinate);energy=total_energy(material,F)
            stress=torch.autograd.grad(energy,coordinate,create_graph=True)[0]
            tangent=torch.autograd.grad(stress,coordinate)[0]
            values=[float(x.detach()) for x in (energy,stress,tangent)]
            finite &= all(math.isfinite(x) for x in values);minimum_energy=min(minimum_energy,values[0])
            minimum_tangent=min(minimum_tangent,values[2]);count+=1
            flat=F.detach().reshape(-1).requires_grad_(True)
            Hessian=torch.autograd.functional.hessian(lambda x:total_energy(material,x.reshape(3,3)),flat).detach().numpy()
            Fnp=F.detach().numpy()
            for _ in range(12):
                n=rng.normal(size=3);n/=np.linalg.norm(n);normal=np.linalg.solve(Fnp.T,n);normal/=np.linalg.norm(normal)
                m=rng.normal(size=3);m-=np.dot(m,normal)*normal
                if np.linalg.norm(m)<1e-10:continue
                m/=np.linalg.norm(m);rank_one=np.outer(m,n).reshape(-1)
                minimum_acoustic=min(minimum_acoustic,float(rank_one@Hessian@rank_one))
    return {"state_count":count,"all_finite":finite,"minimum_total_energy_MPa":minimum_energy,
            "minimum_path_tangent_MPa":minimum_tangent,
            "minimum_projected_acoustic_modulus_MPa":minimum_acoustic,
            "passed":finite and minimum_energy>=-1e-9 and minimum_tangent>=-1e-5 and minimum_acoustic>=-1e-6}


def main():
    torch.set_default_dtype(torch.float64)
    config=load_config(CONFIG.read_text(encoding="utf-8"));output=Path(config["outputs"]["results_dir"])
    endpoint=pd.read_csv(output/"stage5b_pareto_endpoints.csv").sort_values("pareto_score").iloc[0]
    saved=torch.load(endpoint.checkpoint,map_location="cpu",weights_only=False)
    material,meta,parameters=load_stage5b_material(config,endpoint.candidate,"cpu",synchronize_branch_scale=True)
    material.nonequilibrium_energy.load_state_dict(saved["energy_state_dict"], strict=False);material.eval()
    _,relaxation,frequency=load_real_data(config["real_training_cache"],parameters,
        points_per_decade=config["data"]["points_per_time_decade"],
        maximum_loading_points=config["data"]["maximum_loading_points"])
    start=perf_counter();rows,relaxation_seconds=curve_rows(material,relaxation,"Stage5B-E-PICNN-diagnostic",
                                                            "full_tensor_local_implicit")
    curves=pd.DataFrame(rows);summary=summarize_relaxation(curves)
    curves.to_csv(output/"stage5b_full_tensor_relaxation_curves.csv",index=False)
    summary.to_csv(output/"stage5b_full_tensor_relaxation_metrics.csv",index=False)

    frequencies=torch.tensor([item.frequency_hz for item in frequency],dtype=torch.float64)
    linear_start=perf_counter();storage,loss=MaterialPointSolver(material).local_linear_shear_moduli(frequencies)
    linear_seconds=perf_counter()-linear_start
    frequency_rows=[]
    for index,item in enumerate(frequency):
        frequency_rows.append({"frequency_hz":item.frequency_hz,"split":item.split,
            "amplitude_fraction":item.amplitude_fraction,"experimental_storage_MPa":item.storage_MPa,
            "predicted_storage_MPa":float(storage[index]),"experimental_loss_MPa":item.loss_MPa,
            "predicted_loss_MPa":float(loss[index])})
    pd.DataFrame(frequency_rows).to_csv(output/"stage5b_local_linear_frequency_dma.csv",index=False)
    amplitude=json.loads(Path(config["amplitude_diagnostic_file"]).read_text(encoding="utf-8"))
    periodic=full_tensor_dma_subset({"Stage5B-E-PICNN-diagnostic":material},amplitude["rows"])
    periodic.to_csv(output/"stage5b_full_tensor_dma_subset.csv",index=False)

    datasets,_=load_datasets(load_config(Path(config["stage4_config"]).read_text(encoding="utf-8"))["synthetic_anchor_cache"],"cpu")
    physics=physics_checks(material);convexity=convexity_checks(material,datasets)
    convergence=time_step_checks(material);cost=cost_checks(material,datasets);stability=stability_screen(material)
    full=summary
    validation=full[full.split=="validation"]
    q_log=float(np.sqrt(np.mean(np.log(validation.predicted_q/validation.experimental_q)**2)))
    rneq=float(np.sqrt(np.mean(validation.Rneq_RMSE**2)))
    frequency_frame=pd.DataFrame(frequency_rows)
    frequency_frame["storage_relative_error"]=(frequency_frame.predicted_storage_MPa-frequency_frame.experimental_storage_MPa)/frequency_frame.experimental_storage_MPa
    frequency_frame["loss_relative_error"]=(frequency_frame.predicted_loss_MPa-frequency_frame.experimental_loss_MPa)/frequency_frame.experimental_loss_MPa
    frequency_metrics={"storage_relative_RMSE":float(np.sqrt(np.mean(frequency_frame.storage_relative_error**2))),
                       "loss_relative_RMSE":float(np.sqrt(np.mean(frequency_frame.loss_relative_error**2)))}
    acceptance={"reference_state":max(physics["reference_energy_abs_MPa"],physics["reference_stress_abs_MPa"],
        physics["reference_force_max_MPa"],physics["reference_flow_max_per_s"])<=1e-10,
        "objectivity":max(physics["objectivity_energy_relative_error"],physics["objectivity_stress_relative_error"])<=1e-8,
        "isotropy":max(physics["isotropy_energy_relative_error"],physics["isotropy_flow_max_abs_error_per_s"])<=1e-8,
        "nonnegative_dissipation":physics["minimum_branch_dissipation_MPa_per_s"]>=-1e-12 and full.minimum_dissipation_MPa_per_s.min()>=-1e-12,
        "isochoric_internal":max(row["maximum_detFv_error"] for row in convergence)<=1e-8,
        "convex_dissipation":convexity["monotone_slope"] and convexity["positive_slope"],
        "newton_convergence":full.maximum_newton_residual.max()<=1.01e-8,
        "time_step_convergence":convergence[1]["terminal_stress_relative_to_dt_quarter"]<=.01,
        "energy_stability_screen":stability["passed"],
        "periodic_no_failure":int(periodic.failed.sum())==0,
        "q_improves_stage4":q_log<json.loads((output/"stage5b_training_summary.json").read_text(encoding="utf-8"))["stage4_baseline"]["validation"]["q_log_RMSE"],
        "Rneq_guard":rneq<=1.1*json.loads((output/"stage5b_training_summary.json").read_text(encoding="utf-8"))["stage4_baseline"]["validation"]["Rneq_RMSE"]}
    result={"schema_version":"2026-08-31-chapter3-stage5b-final-validation-v1",
        "model_role":"best diagnostic checkpoint; not accepted for Stage 6 initialization unless all gates pass",
        "candidate":endpoint.candidate,"source_checkpoint":endpoint.checkpoint,"source_checkpoint_sha256":sha256(endpoint.checkpoint),
        "timings":{"full_tensor_relaxation_s":relaxation_seconds,"local_linear_dma_s":linear_seconds,
                   "total_before_checks_s":perf_counter()-start},
        "full_tensor_validation":{"q_log_RMSE":q_log,"Rneq_RMSE":rneq},"frequency_metrics":frequency_metrics,
        "physics":physics,"convexity":convexity,"time_step_convergence":convergence,"cost":cost,"stability":stability,
        "periodic":{"case_count":len(periodic),"failure_count":int(periodic.failed.sum()),
                    "steady_count":int(periodic.loc[~periodic.failed,"steady"].sum())},
        "acceptance":acceptance,"passed":all(acceptance.values()),
        "amplitude_labels_used_for_training":False}
    (output/"stage5b_final_validation_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding="utf-8")
    print(json.dumps({"passed":result["passed"],"full_tensor_validation":result["full_tensor_validation"],
        "frequency_metrics":frequency_metrics,"acceptance":acceptance,"timings":result["timings"]},ensure_ascii=False,indent=2,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))


if __name__=="__main__":main()
