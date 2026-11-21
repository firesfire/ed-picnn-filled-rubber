"""Targeted q--Rneq Pareto refinement of the best Stage-5B candidates."""

from __future__ import annotations

from project_paths import load_config

import json
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"code"))
from dual_potential.constitutive import MaterialParameters
from run_stage5b_epicnn import (differentiable_relaxation_objective,
                                evaluate_relaxation_only, snapshot_energy)
from stage3_real import load_real_data
from stage5b_energy import load_stage5b_material

CONFIG=ROOT/"configs"/"stage5b_epicnn_energy_finetuning.yaml"


def main():
    torch.set_default_dtype(torch.float64)
    config=load_config(CONFIG.read_text(encoding="utf-8"))
    results=Path(config["outputs"]["results_dir"]);checkpoints=Path(config["outputs"]["checkpoints_dir"])
    comparison=pd.read_csv(results/"stage5b_candidate_comparison.csv")
    stage4_config=load_config(Path(config["stage4_config"]).read_text(encoding="utf-8"))
    parameters=MaterialParameters.from_json(stage4_config["teacher_parameter_file"])
    _,relaxation,_=load_real_data(config["real_training_cache"],parameters,
        points_per_decade=config["data"]["points_per_time_decade"],
        maximum_loading_points=config["data"]["maximum_loading_points"])
    baseline=json.loads((results/"stage5b_training_summary.json").read_text(encoding="utf-8"))["stage4_baseline"]
    original=dict(config["loss_weights"])
    config["loss_weights"].update({"q_log":.18,"q_ordering":.35,"Rneq":.40,
        "Rneq_decade":.40,"R":.08,"loading":.04,"storage":0.0,"loss":0.0})
    rows=[];endpoints=[]
    for source in comparison.itertuples():
        saved=torch.load(source.checkpoint,map_location="cpu",weights_only=False)
        material,_,_=load_stage5b_material(config,source.candidate,"cpu",synchronize_branch_scale=True)
        material.nonequilibrium_energy.load_state_dict(saved["energy_state_dict"], strict=False)
        optimizer=torch.optim.Adam(material.nonequilibrium_energy.parameters(),lr=.0012)
        best=None
        for epoch in range(0,51):
            if epoch:
                optimizer.zero_grad();loss=differentiable_relaxation_objective(material,relaxation,config)
                loss.backward();torch.nn.utils.clip_grad_norm_(material.nonequilibrium_energy.parameters(),3.0)
                optimizer.step()
            if epoch%5==0:
                _,train=evaluate_relaxation_only(material,relaxation,"training",config)
                _,validation=evaluate_relaxation_only(material,relaxation,"validation",config)
                score=((validation["q_log_RMSE"]/baseline["validation"]["q_log_RMSE"])**2+
                       (validation["Rneq_RMSE"]/baseline["validation"]["Rneq_RMSE"])**2+
                       4.0*max(0.0,1.0-train["q_ordering_accuracy"])**2)
                row={"candidate":source.candidate,"source_seed":int(source.seed),"epoch":epoch,
                     "pareto_score":score,**{f"training_{k}":v for k,v in train.items()},
                     **{f"validation_{k}":v for k,v in validation.items()}}
                rows.append(row)
                if best is None or score<best[0]:best=(score,epoch,snapshot_energy(material.nonequilibrium_energy),row)
        material.nonequilibrium_energy.load_state_dict(best[2], strict=False)
        destination=checkpoints/f"{source.candidate}_pareto.pt"
        torch.save({"schema_version":"2026-08-31-stage5b-pareto-v1","candidate":source.candidate,
                    "source_seed":int(source.seed),"best_epoch":best[1],
                    "energy_state_dict":snapshot_energy(material.nonequilibrium_energy),
                    "energy_physical_parameters":material.nonequilibrium_energy.physical_parameters(),
                    "stage4_dissipation_state_dict":saved["stage4_dissipation_state_dict"],
                    "branch_scale_synchronized":True,"metrics":best[3]},destination)
        endpoints.append({**best[3],"checkpoint":str(destination),
            "q_improves_stage4":best[3]["validation_q_log_RMSE"]<baseline["validation"]["q_log_RMSE"],
            "Rneq_guard_pass":best[3]["validation_Rneq_RMSE"]<=1.1*baseline["validation"]["Rneq_RMSE"],
            "ordering_pass":best[3]["training_q_ordering_accuracy"]>=1.0-1e-12})
    frame=pd.DataFrame(rows);frame.to_csv(results/"stage5b_pareto_trajectory.csv",index=False)
    end=pd.DataFrame(endpoints);end["feasible"]=end.q_improves_stage4 & end.Rneq_guard_pass & end.ordering_pass
    end.to_csv(results/"stage5b_pareto_endpoints.csv",index=False)
    print(end[["candidate","validation_q_log_RMSE","validation_Rneq_RMSE",
               "training_q_ordering_accuracy","feasible"]].to_string(index=False))


if __name__=="__main__":main()
