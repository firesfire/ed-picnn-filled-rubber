"""Stage 3: fixed-energy D-PICNN fine-tuning on real relaxation and 0.1% DMA."""

from __future__ import annotations

from project_paths import load_config

import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))

from dual_potential.constitutive import MaterialParameters
from stage2_rollout import load_stage2_material
from stage2_training import evaluate as synthetic_evaluate, load_datasets, predict
from stage3_real import load_real_data, real_objective

CONFIG=ROOT/"configs"/"stage3_dpicnn_real_finetuning.yaml"
RESULTS=ROOT/"results"/"stage3"; CHECKPOINTS=ROOT/"checkpoints"/"stage3"


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def build_material(config,device):
    material,saved=load_stage2_material(config["stage2_checkpoint"],config["teacher_parameter_file"],device)
    parameters=MaterialParameters.from_json(config["teacher_parameter_file"])
    material.equilibrium_energy_parameters=parameters.zhang_li
    for parameter in material.equilibrium_energy.parameters(): parameter.requires_grad_(False)
    for parameter in material.nonequilibrium_energy.parameters(): parameter.requires_grad_(False)
    trainable=[name for name,p in material.named_parameters() if p.requires_grad]
    if not trainable or not all(name.startswith("dissipation_potential.") for name in trainable):
        raise RuntimeError("stage-3 trainable boundary is not dissipation-only")
    return material,saved,parameters


@torch.no_grad()
def make_anchor_targets(base, data):
    slope,D=predict(base.dissipation_potential,data)
    return slope.detach(),(data.tau[:,None,None]*D).detach()


def anchor_loss(material,data,targets,index):
    slope,D=predict(material.dissipation_potential,data,index)
    target_slope,target_scaled=targets[0][index],targets[1][index]
    current_scaled=data.tau[index,None,None]*D
    log=torch.mean((torch.log(slope.clamp_min(1e-16))-torch.log(target_slope.clamp_min(1e-16))).square())
    transformed=torch.mean((torch.asinh(current_scaled/.02)-torch.asinh(target_scaled/.02)).square())
    return log+.5*transformed


def metrics(values):
    return {"objective":float(values["loss"].detach()),
      "q_log_rmse":float(torch.sqrt(values["relaxation"]["terms"]["q_log"]).detach()),
      "q_ordering_rmse":float(torch.sqrt(values["relaxation"]["terms"]["q_ordering"]).detach()),
      "Rneq_rmse":float(torch.sqrt(values["relaxation"]["terms"]["Rneq"]).detach()),
      "Rneq_decade_rmse":float(torch.sqrt(values["relaxation"]["terms"]["Rneq_decade"]).detach()),
      "R_rmse":float(torch.sqrt(values["relaxation"]["terms"]["R"]).detach()),
      "loading_rmse":float(torch.sqrt(values["relaxation"]["terms"]["loading"]).detach()),
      "frequency_storage_rmse_scaled":float(values["frequency"]["storage_rmse_scaled"].detach()),
      "frequency_loss_log_rmse":float(values["frequency"]["loss_log_rmse"].detach()),
      "minimum_frequency_dissipation":float(values["frequency_prediction"]["minimum_dissipation"].detach())}


def state_snapshot(model):
    return {k:v.detach().cpu().clone() for k,v in model.dissipation_potential.state_dict().items()}


def parameter_drift(model,base_state):
    num=den=0.0
    for name,value in model.dissipation_potential.state_dict().items():
        if not torch.is_floating_point(value):continue
        delta=value.detach().cpu()-base_state[name]
        num+=float(torch.sum(delta.square()));den+=float(torch.sum(base_state[name].square()))
    return np.sqrt(num/max(den,1e-30))


def train_candidate(config,real_data,relaxation,frequency,datasets,anchor_targets,
                    anchor_weight,seed,device,epochs_override=None):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    material,saved,_=build_material(config,device);base_state=state_snapshot(material)
    optcfg=config["optimization"];epochs=epochs_override or optcfg["maximum_epochs"]
    optimizer=torch.optim.AdamW(material.dissipation_potential.parameters(),lr=optcfg["learning_rate"],weight_decay=optcfg["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=optcfg["minimum_learning_rate"])
    synthetic_limit=float(config["selection"]["synthetic_validation_flow_nrmse_maximum"])
    material.eval()
    with torch.no_grad():
        initial_validation=real_objective(material,relaxation,frequency,"validation",cycles=optcfg["training_cycles"],points_per_cycle=optcfg["training_points_per_cycle"],differentiable=False)
    initial_synthetic=synthetic_evaluate(material.dissipation_potential,datasets["validation_high_medium"],"validation_high_medium")
    best=float(metrics(initial_validation)["objective"]);best_state=state_snapshot(material);best_epoch=0
    best_synthetic=float(initial_synthetic["flow_nrmse"]);stale=0;history=[];start=perf_counter()
    train_data=datasets["train"]
    for epoch in range(1,epochs+1):
        material.train();optimizer.zero_grad(set_to_none=True)
        real=real_objective(material,relaxation,frequency,"training",cycles=optcfg["training_cycles"],points_per_cycle=optcfg["training_points_per_cycle"],differentiable=True)
        count=min(optcfg["synthetic_anchor_batch"],train_data.count)
        index=torch.randint(train_data.count,(count,),device=device)
        anchor=anchor_loss(material,train_data,anchor_targets,index)
        total=real["loss"]+anchor_weight*anchor
        if not bool(torch.isfinite(total)):raise FloatingPointError(f"nonfinite loss anchor={anchor_weight} seed={seed} epoch={epoch}")
        total.backward();gradient=torch.nn.utils.clip_grad_norm_(material.dissipation_potential.parameters(),optcfg["gradient_clip"])
        if not bool(torch.isfinite(gradient)):raise FloatingPointError("nonfinite gradient")
        optimizer.step();scheduler.step()
        if epoch==1 or epoch%optcfg["evaluation_interval"]==0 or epoch==epochs:
            material.eval()
            with torch.no_grad():
                validation=real_objective(material,relaxation,frequency,"validation",cycles=optcfg["training_cycles"],points_per_cycle=optcfg["training_points_per_cycle"],differentiable=False)
            vm=metrics(validation);tm=metrics(real)
            synthetic_validation=synthetic_evaluate(material.dissipation_potential,datasets["validation_high_medium"],"validation_high_medium")
            synthetic_flow=float(synthetic_validation["flow_nrmse"])
            feasible=synthetic_flow<=synthetic_limit
            row={"anchor_weight":anchor_weight,"seed":seed,"epoch":epoch,
                 "training_total":float(total.detach()),"training_real_objective":tm["objective"],
                 "validation_objective":vm["objective"],"validation_q_log_rmse":vm["q_log_rmse"],
                 "validation_Rneq_rmse":vm["Rneq_rmse"],"validation_frequency_storage_rmse_scaled":vm["frequency_storage_rmse_scaled"],
                 "validation_frequency_loss_log_rmse":vm["frequency_loss_log_rmse"],
                 "anchor_loss":float(anchor.detach()),"gradient_norm":float(gradient.detach()),
                 "synthetic_validation_flow_nrmse":synthetic_flow,"feasible":feasible,
                 "learning_rate":optimizer.param_groups[0]["lr"],"elapsed_s":perf_counter()-start}
            history.append(row)
            score=vm["objective"]
            if feasible and score<best-1e-7:
                best=score;best_state=state_snapshot(material);best_epoch=epoch;best_synthetic=synthetic_flow;stale=0
            else:stale+=optcfg["evaluation_interval"]
            print(f"stage3 anchor={anchor_weight:g} seed={seed} epoch={epoch:3d} val={score:.4g} syn={synthetic_flow:.3g} feasible={feasible} q={vm['q_log_rmse']:.3g} Rn={vm['Rneq_rmse']:.3g} Gp={vm['frequency_storage_rmse_scaled']:.3g} Gpp={vm['frequency_loss_log_rmse']:.3g}",flush=True)
            if epoch>=optcfg["minimum_epochs"] and stale>=optcfg["patience_epochs"]:break
    material.dissipation_potential.load_state_dict({k:v.to(device) for k,v in best_state.items()});material.eval()
    final_train=real_objective(material,relaxation,frequency,"training",cycles=optcfg["training_cycles"],points_per_cycle=optcfg["training_points_per_cycle"],differentiable=False)
    final_val=real_objective(material,relaxation,frequency,"validation",cycles=optcfg["training_cycles"],points_per_cycle=optcfg["training_points_per_cycle"],differentiable=False)
    synthetic={name:synthetic_evaluate(material.dissipation_potential,data,name) for name,data in datasets.items()}
    weight_tag=f"{anchor_weight:.3g}".replace(".","p")
    name=f"anchor_{weight_tag}_seed_{seed}.pt"
    checkpoint=CHECKPOINTS/name
    torch.save({"schema_version":"2026-08-30-stage3-candidate-v1","architecture":saved["architecture"],
      "seed":seed,"anchor_weight":anchor_weight,"best_epoch":best_epoch,"state_dict":best_state,
      "best_synthetic_validation_flow_nrmse":best_synthetic,
      "frozen_energy":True,"stage2_checkpoint":config["stage2_checkpoint"],
      "real_training_cache_sha256":real_data["source_workbook_sha256"]},checkpoint)
    return {"anchor_weight":anchor_weight,"seed":seed,"best_epoch":best_epoch,"completed_epochs":epoch,
      "training_time_s":perf_counter()-start,"checkpoint":str(checkpoint),
      "train":metrics(final_train),"validation":metrics(final_val),"synthetic":synthetic,
      "parameter_relative_drift":parameter_drift(material,base_state),"history":history}


def baseline(config,device):
    material,_,parameters=build_material(config,device)
    raw,relaxation,frequency=load_real_data(config["real_training_cache"],parameters,
      points_per_decade=config["data"]["points_per_time_decade"],maximum_loading_points=config["data"]["maximum_loading_points"],device=device)
    datasets,_=load_datasets(config["synthetic_anchor_cache"],device)
    result={}
    for split in ("training","validation"):
        value=real_objective(material,relaxation,frequency,split,cycles=config["optimization"]["training_cycles"],points_per_cycle=config["optimization"]["training_points_per_cycle"],differentiable=False)
        result[split]={"metrics":metrics(value),"relaxation_rows":value["relaxation"]["rows"],"frequency_rows":value["frequency"]["rows"]}
    result["synthetic"]={name:synthetic_evaluate(material.dissipation_potential,data,name) for name,data in datasets.items()}
    return result,raw,relaxation,frequency,datasets,material


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--baseline-only",action="store_true");parser.add_argument("--epochs",type=int);parser.add_argument("--guarded",action="store_true")
    args=parser.parse_args();config=load_config(CONFIG.read_text(encoding="utf-8"));RESULTS.mkdir(parents=True,exist_ok=True);CHECKPOINTS.mkdir(parents=True,exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu";torch.set_default_dtype(torch.float64)
    base,raw,relaxation,frequency,datasets,base_material=baseline(config,device)
    save_json(RESULTS/"stage3_baseline_reduced_metrics.json",base)
    print(json.dumps({"device":device,"baseline_training":base["training"]["metrics"],"baseline_validation":base["validation"]["metrics"]},indent=2))
    if args.baseline_only:return
    targets=make_anchor_targets(base_material,datasets["train"])
    candidates=[];all_history=[]
    anchor_weights=config["optimization"].get("guarded_anchor_weights",config["optimization"]["anchor_weights"]) if args.guarded else config["optimization"]["anchor_weights"]
    for weight in anchor_weights:
        for seed in config["optimization"]["seeds"]:
            candidate=train_candidate(config,raw,relaxation,frequency,datasets,targets,float(weight),int(seed),device,args.epochs)
            all_history.extend(candidate.pop("history"));candidates.append(candidate)
    suffix="_guarded" if args.guarded else ""
    with (RESULTS/f"training_history{suffix}.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=all_history[0].keys());writer.writeheader();writer.writerows(all_history)
    rows=[]
    for c in candidates:
        rows.append({"anchor_weight":c["anchor_weight"],"seed":c["seed"],"best_epoch":c["best_epoch"],
          "completed_epochs":c["completed_epochs"],"training_time_s":c["training_time_s"],"checkpoint":c["checkpoint"],
          "real_validation_objective":c["validation"]["objective"],"validation_q_log_rmse":c["validation"]["q_log_rmse"],
          "validation_Rneq_rmse":c["validation"]["Rneq_rmse"],"validation_frequency_storage_rmse_scaled":c["validation"]["frequency_storage_rmse_scaled"],
          "validation_frequency_loss_log_rmse":c["validation"]["frequency_loss_log_rmse"],
          "synthetic_validation_flow_nrmse":c["synthetic"]["validation_high_medium"]["flow_nrmse"],
          "synthetic_test_flow_nrmse":c["synthetic"]["test_high_medium"]["flow_nrmse"],
          "parameter_relative_drift":c["parameter_relative_drift"]})
    with (RESULTS/f"candidate_comparison{suffix}.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    feasible=[(r,c) for r,c in zip(rows,candidates) if r["synthetic_validation_flow_nrmse"]<=config["selection"]["synthetic_validation_flow_nrmse_maximum"]]
    if not feasible and config["selection"].get("require_feasible_checkpoint",True):
        raise RuntimeError("No stage-3 checkpoint satisfies the synthetic retention constraint")
    if not feasible:feasible=list(zip(rows,candidates))
    selected_row,selected=min(feasible,key=lambda pair:(pair[0]["real_validation_objective"],pair[0]["parameter_relative_drift"],pair[0]["anchor_weight"]))
    shutil.copy2(selected["checkpoint"],CHECKPOINTS/"dpicnn_stage3_selected.pt")
    summary={"schema_version":"2026-08-30-chapter3-stage3-training-v1","device":device,
      "baseline":base,"candidate_count":len(candidates),"candidates":candidates,
      "selection_rule":config["selection"],"selected":selected_row,
      "selected_checkpoint":str(CHECKPOINTS/"dpicnn_stage3_selected.pt"),
      "amplitude_labels_used_for_optimization":False,"energy_parameters_trainable":False}
    save_json(RESULTS/f"stage3_training_summary{suffix}.json",summary)
    print(json.dumps({"selected":selected_row},indent=2))


if __name__=="__main__":main()
