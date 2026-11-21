"""Block-coordinate low-parameter energy release for Chapter-3 Stage 4."""

from __future__ import annotations

from project_paths import load_config

import csv
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))

from stage2_training import load_datasets
from stage3_real import load_real_data
from stage4_energy import (build_stage4_material,clamp_energy,load_stage4_checkpoint,
    physical_parameters,restore,scaled_anchor_targets,scaled_teacher_nrmse,
    snapshot,stage4_objective,summary_metrics,sync_branch_scale)

CONFIG=ROOT/"configs"/"stage4_gdpicnn_energy_release.yaml"
RESULTS=ROOT/"results"/"stage4";CHECKPOINTS=ROOT/"checkpoints"/"stage4"


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def set_trainable(material,energy_names=(),dissipation=False):
    for parameter in material.parameters():parameter.requires_grad_(False)
    for name,parameter in material.nonequilibrium_energy.named_parameters():
        parameter.requires_grad_(name in energy_names)
    for parameter in material.dissipation_potential.parameters():
        parameter.requires_grad_(dissipation)
    selected=[parameter for parameter in material.parameters() if parameter.requires_grad]
    if not selected:raise RuntimeError("empty stage-4 optimization block")
    return selected


def evaluate(material,relaxation,frequency,parameters,config,split):
    material.eval();sync_branch_scale(material)
    with torch.no_grad():
        value=stage4_objective(material,relaxation,frequency,split,"evaluation",parameters,config)
    return summary_metrics(value)


def optimize_block(material,relaxation,frequency,datasets,target_D,base_moduli,
                   parameters,config,seed,label,phase,energy_names,epochs,learning_rate,
                   history):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    energy_parameters=set_trainable(material,energy_names,phase in ("dissipation","joint"))
    if phase=="joint":
        energy=[p for name,p in material.nonequilibrium_energy.named_parameters() if p.requires_grad]
        diss=[p for p in material.dissipation_potential.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW([
            {"params":energy,"lr":config["optimization"]["joint_energy_learning_rate"]},
            {"params":diss,"lr":config["optimization"]["joint_dissipation_learning_rate"]}],
            weight_decay=config["optimization"]["weight_decay"])
    else:
        optimizer=torch.optim.AdamW(energy_parameters,lr=learning_rate,
            weight_decay=config["optimization"]["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,
        eta_min=learning_rate*config["optimization"]["minimum_learning_rate_fraction"] if phase!="joint" else 1e-6)
    initial=evaluate(material,relaxation,frequency,parameters,config,"validation")
    if phase=="energy":
        with torch.no_grad():
            initial_focus=stage4_objective(material,relaxation,frequency,"training","energy",parameters,config)
        best_score=float(initial_focus["loss"])
    else:best_score=initial["real_objective"]
    best=snapshot(material);best_epoch=0
    anchor_data=datasets["train"];start=perf_counter()
    for epoch in range(1,epochs+1):
        material.train();sync_branch_scale(material);optimizer.zero_grad(set_to_none=True)
        count=min(config["optimization"]["synthetic_anchor_batch"],anchor_data.count)
        index=torch.randint(anchor_data.count,(count,),device=anchor_data.X.device)
        value=stage4_objective(material,relaxation,frequency,"training",phase,parameters,config,
            anchor_data=anchor_data,anchor_target=target_D["train"],base_moduli=base_moduli,
            anchor_index=index)
        if not bool(torch.isfinite(value["loss"])):raise FloatingPointError(f"nonfinite {label}")
        value["loss"].backward()
        gradient=torch.nn.utils.clip_grad_norm_(energy_parameters,config["optimization"]["gradient_clip"])
        if not bool(torch.isfinite(gradient)):raise FloatingPointError(f"nonfinite gradient {label}")
        optimizer.step();clamp_energy(material,config);sync_branch_scale(material);scheduler.step()
        if epoch==1 or epoch%config["optimization"]["evaluation_interval"]==0 or epoch==epochs:
            train=evaluate(material,relaxation,frequency,parameters,config,"training")
            validation=evaluate(material,relaxation,frequency,parameters,config,"validation")
            row={"seed":seed,"block":label,"phase":phase,"epoch":epoch,
                "loss":float(value["loss"].detach()),"gradient_norm":float(gradient.detach()),
                "training_real_objective":train["real_objective"],
                "validation_real_objective":validation["real_objective"],
                "training_q_log_rmse":train["q_log_rmse"],"validation_q_log_rmse":validation["q_log_rmse"],
                "training_Rneq_rmse":train["Rneq_rmse"],"validation_Rneq_rmse":validation["Rneq_rmse"],
                "validation_storage_rmse_scaled":validation["frequency_storage_rmse_scaled"],
                "validation_loss_log_rmse":validation["frequency_loss_log_rmse"],
                "Gneq_MPa":physical_parameters(material)["Gneq_MPa"],
                "elapsed_s":perf_counter()-start}
            if phase=="energy":
                with torch.no_grad():
                    focus=stage4_objective(material,relaxation,frequency,"training","energy",parameters,config)
                score=float(focus["loss"])
            else:score=validation["real_objective"]
            row["block_selection_score"]=score
            history.append(row)
            shape_feasible=validation["Rneq_rmse"]<=config["regularization"]["Rneq_guard_rmse"]
            row["shape_guard_feasible"]=shape_feasible
            if shape_feasible and score<best_score-1e-8:
                best_score=score;best=snapshot(material);best_epoch=epoch
            print(f"stage4 seed={seed} {label} epoch={epoch:3d} score={score:.4g} "
                  f"val={validation['real_objective']:.4g} feasible={shape_feasible} "
                  f"q={validation['q_log_rmse']:.3g} Rn={validation['Rneq_rmse']:.3g} "
                  f"Gp={validation['frequency_storage_rmse_scaled']:.3g} "
                  f"G={physical_parameters(material)['Gneq_MPa']:.3g}",flush=True)
    restore(material,best)
    return {"best_epoch":best_epoch,"best_validation_objective":best_score,
            "completed_epochs":epochs,"wall_time_s":perf_counter()-start}


def checkpoint_candidate(material,saved,config,seed,level,block_summaries,
                         relaxation,frequency,datasets,target_D,base_moduli,parameters):
    train=evaluate(material,relaxation,frequency,parameters,config,"training")
    validation=evaluate(material,relaxation,frequency,parameters,config,"validation")
    synthetic={name:scaled_teacher_nrmse(material,data,target_D[name],base_moduli)
               for name,data in datasets.items()}
    physical=physical_parameters(material)
    tag=f"seed_{seed}_{level}.pt";path=CHECKPOINTS/tag
    torch.save({"schema_version":"2026-08-30-chapter3-stage4-candidate-v1",
      "architecture":saved["architecture"],"seed":seed,"release_level":level,
      "energy_family":physical["energy_family"],"physical_parameters":physical,
      "energy_state_dict":{k:v.detach().cpu() for k,v in material.nonequilibrium_energy.state_dict().items()},
      "dissipation_state_dict":{k:v.detach().cpu() for k,v in material.dissipation_potential.state_dict().items()},
      "stage3_checkpoint":config["stage3_checkpoint"],"block_summaries":block_summaries,
      "amplitude_labels_used_for_optimization":False},path)
    return {"seed":seed,"release_level":level,"checkpoint":str(path),"physical":physical,
            "training":train,"validation":validation,"synthetic":synthetic,
            "block_summaries":block_summaries}


def prescreen(config,relaxation,frequency,parameters):
    rows=[]
    for value in config["prescreen"]["Gneq_grid_MPa"]:
        material,_,_=build_stage4_material(config,"cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():material.nonequilibrium_energy.raw_total.copy_(torch.tensor(math.log(math.expm1(value)),dtype=torch.float64,device=material.device))
        sync_branch_scale(material)
        for split in ("training","validation"):
            metric=evaluate(material,relaxation,frequency,parameters,config,split)
            rows.append({"Gneq_MPa":value,"split":split,**{k:v for k,v in metric.items() if not isinstance(v,list)}})
    return rows


def main():
    config=load_config(CONFIG.read_text(encoding="utf-8"));RESULTS.mkdir(parents=True,exist_ok=True);CHECKPOINTS.mkdir(parents=True,exist_ok=True)
    torch.set_default_dtype(torch.float64);device="cuda" if torch.cuda.is_available() else "cpu"
    base,saved,parameters=build_stage4_material(config,device)
    raw,relaxation,frequency=load_real_data(config["real_training_cache"],parameters,
        points_per_decade=config["data"]["points_per_time_decade"],
        maximum_loading_points=config["data"]["maximum_loading_points"],device=device)
    datasets,_=load_datasets(config["synthetic_anchor_cache"],device)
    base_moduli=torch.tensor(parameters.branch_moduli_MPa,dtype=torch.float64,device=device)
    target_D={name:scaled_anchor_targets(base,data) for name,data in datasets.items()}
    baseline={"training":evaluate(base,relaxation,frequency,parameters,config,"training"),
              "validation":evaluate(base,relaxation,frequency,parameters,config,"validation"),
              "physical":physical_parameters(base)}
    screen=prescreen(config,relaxation,frequency,parameters)
    with (RESULTS/"Gneq_prescreen.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=screen[0].keys());writer.writeheader();writer.writerows(screen)

    candidates=[];history=[]
    for seed in config["optimization"]["seeds"]:
        material,saved,_=build_stage4_material(config,device);blocks=[]
        blocks.append({"name":"G_energy",**optimize_block(material,relaxation,frequency,datasets,target_D,base_moduli,parameters,config,seed,
            "G_energy","energy",("raw_total",),config["optimization"]["G_only_epochs"],config["optimization"]["energy_learning_rate"],history)})
        blocks.append({"name":"G_dissipation",**optimize_block(material,relaxation,frequency,datasets,target_D,base_moduli,parameters,config,seed,
            "G_dissipation","dissipation",(),config["optimization"]["dissipation_refinement_epochs"],config["optimization"]["dissipation_learning_rate"],history)})
        candidates.append(checkpoint_candidate(material,saved,config,seed,"G",blocks.copy(),relaxation,frequency,datasets,target_D,base_moduli,parameters))

        blocks.append({"name":"shares_energy",**optimize_block(material,relaxation,frequency,datasets,target_D,base_moduli,parameters,config,seed,
            "shares_energy","energy",("raw_total","raw_shares"),config["optimization"]["shares_epochs"],config["optimization"]["shares_learning_rate"],history)})
        blocks.append({"name":"shares_dissipation",**optimize_block(material,relaxation,frequency,datasets,target_D,base_moduli,parameters,config,seed,
            "shares_dissipation","dissipation",(),config["optimization"]["dissipation_refinement_epochs"],config["optimization"]["dissipation_learning_rate"],history)})
        candidates.append(checkpoint_candidate(material,saved,config,seed,"G_shares",blocks.copy(),relaxation,frequency,datasets,target_D,base_moduli,parameters))

        blocks.append({"name":"exponent_energy",**optimize_block(material,relaxation,frequency,datasets,target_D,base_moduli,parameters,config,seed,
            "exponent_energy","energy",("raw_total","raw_shares","raw_exponent"),config["optimization"]["exponent_epochs"],config["optimization"]["exponent_learning_rate"],history)})
        blocks.append({"name":"exponent_dissipation",**optimize_block(material,relaxation,frequency,datasets,target_D,base_moduli,parameters,config,seed,
            "exponent_dissipation","dissipation",(),config["optimization"]["dissipation_refinement_epochs"],config["optimization"]["dissipation_learning_rate"],history)})
        blocks.append({"name":"shared_joint",**optimize_block(material,relaxation,frequency,datasets,target_D,base_moduli,parameters,config,seed,
            "shared_joint","joint",("raw_total","raw_shares","raw_exponent"),config["optimization"]["joint_refinement_epochs"],config["optimization"]["joint_energy_learning_rate"],history)})
        shared=checkpoint_candidate(material,saved,config,seed,"G_shares_exponent",blocks.copy(),relaxation,frequency,datasets,target_D,base_moduli,parameters)
        candidates.append(shared)

        trigger=(shared["validation"]["q_log_rmse"]>config["selection"]["grouped_trigger_validation_q_log_rmse"] or
                 shared["validation"]["frequency_storage_rmse_scaled"]>config["selection"]["grouped_trigger_validation_storage_rmse_scaled"])
        if trigger:
            current=physical_parameters(material);diss_state={k:v.detach().cpu().clone() for k,v in material.dissipation_potential.state_dict().items()}
            grouped,saved,_=build_stage4_material(config,device,grouped=True,physical=current)
            grouped.dissipation_potential.load_state_dict(diss_state);sync_branch_scale(grouped);material=grouped
            group_names=("raw_total","raw_shares","raw_fast_exponent","raw_slow_exponent")
            blocks.append({"name":"grouped_energy",**optimize_block(material,relaxation,frequency,datasets,target_D,base_moduli,parameters,config,seed,
                "grouped_energy","energy",group_names,config["optimization"]["grouped_epochs"],config["optimization"]["exponent_learning_rate"],history)})
            blocks.append({"name":"grouped_dissipation",**optimize_block(material,relaxation,frequency,datasets,target_D,base_moduli,parameters,config,seed,
                "grouped_dissipation","dissipation",(),config["optimization"]["dissipation_refinement_epochs"],config["optimization"]["dissipation_learning_rate"],history)})
            blocks.append({"name":"grouped_joint",**optimize_block(material,relaxation,frequency,datasets,target_D,base_moduli,parameters,config,seed,
                "grouped_joint","joint",group_names,config["optimization"]["joint_refinement_epochs"],config["optimization"]["joint_energy_learning_rate"],history)})
            candidates.append(checkpoint_candidate(material,saved,config,seed,"grouped",blocks.copy(),relaxation,frequency,datasets,target_D,base_moduli,parameters))

    with (RESULTS/"block_coordinate_history.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=history[0].keys());writer.writeheader();writer.writerows(history)
    rows=[]
    for c in candidates:
        row={"seed":c["seed"],"release_level":c["release_level"],"checkpoint":c["checkpoint"],
          "Gneq_MPa":c["physical"]["Gneq_MPa"],
          "exponent":c["physical"].get("exponent",math.nan),
          "fast_exponent":c["physical"].get("fast_exponent",math.nan),
          "slow_exponent":c["physical"].get("slow_exponent",math.nan),
          "training_real_objective":c["training"]["real_objective"],
          "validation_real_objective":c["validation"]["real_objective"],
          "training_q_log_rmse":c["training"]["q_log_rmse"],"validation_q_log_rmse":c["validation"]["q_log_rmse"],
          "training_q_ordering_accuracy":c["training"]["q_ordering_accuracy"],
          "training_platform_ordering_accuracy":c["training"]["platform_ordering_accuracy"],
          "validation_Rneq_rmse":c["validation"]["Rneq_rmse"],
          "validation_R_rmse":c["validation"]["R_rmse"],
          "validation_storage_rmse_scaled":c["validation"]["frequency_storage_rmse_scaled"],
          "validation_loss_log_rmse":c["validation"]["frequency_loss_log_rmse"],
          "synthetic_validation_scaled_flow_nrmse":c["synthetic"]["validation_high_medium"],
          "synthetic_test_scaled_flow_nrmse":c["synthetic"]["test_high_medium"],
          "minimum_dissipation":c["validation"]["minimum_dissipation"],
          "shares_json":json.dumps(c["physical"]["shares"])}
        rows.append(row)
    with (RESULTS/"candidate_comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    limit=config["selection"]["maximum_scaled_teacher_flow_nrmse"]
    rlimit=baseline["validation"]["Rneq_rmse"]*(1+config["selection"]["maximum_validation_Rneq_degradation_fraction"])
    feasible=[(row,c) for row,c in zip(rows,candidates) if row["synthetic_validation_scaled_flow_nrmse"]<=limit and
        row["validation_Rneq_rmse"]<=rlimit and row["minimum_dissipation"]>=-1e-12 and
        (not config["selection"]["require_q_ordering_training"] or row["training_q_ordering_accuracy"]>=1.0) and
        (not config["selection"]["require_platform_ordering_training"] or row["training_platform_ordering_accuracy"]>=1.0)]
    if not feasible:raise RuntimeError("no stage-4 candidate satisfies the frozen acceptance constraints")
    selected_row,selected=min(feasible,key=lambda pair:(pair[0]["validation_real_objective"],
        pair[0]["validation_q_log_rmse"]+pair[0]["validation_storage_rmse_scaled"]))
    source=Path(selected["checkpoint"]);destination=CHECKPOINTS/"gdpicnn_stage4_selected.pt";destination.write_bytes(source.read_bytes())
    summary={"schema_version":"2026-08-30-chapter3-stage4-training-v1","device":device,
      "baseline":baseline,"prescreen":screen,"candidate_count":len(candidates),"candidates":candidates,
      "selection_rule":config["selection"],"selected":selected_row,"selected_checkpoint":str(destination),
      "amplitude_labels_used_for_optimization":False,"grouped_candidate_triggered":any(c["release_level"]=="grouped" for c in candidates)}
    save_json(RESULTS/"stage4_training_summary.json",summary)
    print(json.dumps({"selected":selected_row,"baseline_validation":baseline["validation"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
