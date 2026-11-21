"""Stage 5B: dynamic E-PICNN calibration with frozen Stage-4 dissipation."""

from __future__ import annotations

from project_paths import load_config

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from dual_potential.constitutive import MaterialParameters
from stage3_real import load_real_data
from stage5b_energy import (dma_rollout, dma_terms, load_stage5b_material,
                            objective, relaxation_rollout, relaxation_terms,
                            snapshot_energy)

CONFIG = ROOT / "configs" / "stage5b_epicnn_energy_finetuning.yaml"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=lambda x: x.item() if isinstance(x,np.generic) else str(x)),
                    encoding="utf-8")


def ordering_accuracy(rows):
    correct=[]
    for mode in ("uniaxial","shear"):
        values=sorted((row for row in rows if row["mode"]==mode),
                      key=lambda row:float(row["label"].replace("%", "")) if "%" in row["label"] else
                      float(row["label"]))
        for left,right in zip(values[:-1],values[1:]):
            correct.append(right["predicted_q"]>left["predicted_q"])
    return float(np.mean(correct)) if correct else 1.0


def metrics(value):
    r=value["relaxation"];f=value["frequency"]
    rows=r["rows"]
    return {"objective":float(value["loss"].detach()),
            "q_log_RMSE":float(torch.sqrt(r["q_log"]).detach()),
            "q_ordering_accuracy":ordering_accuracy(rows),
            "Rneq_RMSE":float(torch.sqrt(r["Rneq"]).detach()),
            "Rneq_decade_RMSE":float(torch.sqrt(r["Rneq_decade"]).detach()),
            "R_RMSE":float(torch.sqrt(r["R"]).detach()),
            "loading_RMSE_scaled":float(torch.sqrt(r["loading"]).detach()),
            "storage_RMSE_scaled":float(torch.sqrt(f["storage"]).detach()),
            "loss_log_RMSE":float(torch.sqrt(f["loss"]).detach()),
            "minimum_dissipation":float(min(row["minimum_dissipation"] for row in rows))}


def evaluate(material,relaxation,frequency,split,config):
    with torch.enable_grad():
        value=objective(material,relaxation,frequency,split,config["loss_weights"],
                        cycles=config["optimization"]["training_cycles"],
                        points_per_cycle=config["optimization"]["training_points_per_cycle"],
                        differentiable=False)
    return value,metrics(value)


def evaluate_relaxation_only(material,relaxation,split,config):
    conditions=[item for item in relaxation if item.split==split]
    with torch.enable_grad():
        predictions=relaxation_rollout(material,conditions,differentiable=False)
        terms=relaxation_terms(predictions)
    weights=config["loss_weights"]
    loss=(weights["q_log"]*terms["q_log"]+weights["q_ordering"]*terms["q_ordering"]+
          weights["Rneq"]*terms["Rneq"]+weights["Rneq_decade"]*terms["Rneq_decade"]+
          weights["R"]*terms["R"]+weights["loading"]*terms["loading"])
    result={"objective":float(loss.detach()),"q_log_RMSE":float(torch.sqrt(terms["q_log"]).detach()),
            "q_ordering_accuracy":ordering_accuracy(terms["rows"]),
            "Rneq_RMSE":float(torch.sqrt(terms["Rneq"]).detach()),
            "Rneq_decade_RMSE":float(torch.sqrt(terms["Rneq_decade"]).detach()),
            "R_RMSE":float(torch.sqrt(terms["R"]).detach()),
            "loading_RMSE_scaled":float(torch.sqrt(terms["loading"]).detach()),
            "minimum_dissipation":float(min(row["minimum_dissipation"] for row in terms["rows"]))}
    return {"loss":loss,"relaxation":terms,"relaxation_predictions":predictions},result


def regularization(energy,anchor,config):
    terms=[]
    state=dict(energy.named_parameters())
    for name,value in state.items():
        scale=1.0+anchor[name].abs()
        weight=(config["regularization"]["hardening_log_anchor"] if "hardening" in name
                else config["regularization"]["stage5a_parameter_anchor"])
        terms.append(weight*((value-anchor[name])/scale).square().mean())
    entropy=[]
    for group in range(len(energy.raw_weight)):
        weights=torch.softmax(energy.raw_weight[group],0)
        entropy.append(torch.sum(weights*torch.log((weights*len(weights)).clamp_min(1e-12))))
    if entropy:
        terms.append(config["regularization"]["effective_ridge_entropy"]*torch.stack(entropy).mean())
    return torch.stack(terms).sum()


def differentiable_relaxation_objective(material, relaxation, config):
    conditions=[item for item in relaxation if item.split=="training"]
    prediction=relaxation_rollout(material,conditions,differentiable=True)
    terms=relaxation_terms(prediction);weights=config["loss_weights"]
    total=(weights["q_log"]*terms["q_log"]+weights["q_ordering"]*terms["q_ordering"]+
           weights["Rneq"]*terms["Rneq"]+weights["Rneq_decade"]*terms["Rneq_decade"]+
           weights["R"]*terms["R"]+weights["loading"]*terms["loading"])
    return total


def perturb_energy(energy,seed,std):
    generator=torch.Generator(device=energy.branch_moduli_MPa.device);generator.manual_seed(seed)
    with torch.no_grad():
        for parameter in energy.parameters():
            if parameter.requires_grad:
                parameter.add_(std*torch.randn(parameter.shape,dtype=parameter.dtype,
                                                device=parameter.device,generator=generator))


def normalization_diagnostic(config,frequency,output):
    rows=[]
    for candidate in config["energy"]["candidates"]:
        for synchronized in (False,True):
            material,_,_=load_stage5b_material(config,candidate,"cpu",
                synchronize_branch_scale=synchronized)
            prediction=dma_rollout(material,frequency,
                cycles=config["normalization_diagnostic"]["cycles"],
                points_per_cycle=config["normalization_diagnostic"]["points_per_cycle"],
                differentiable=False)
            terms=dma_terms(material,frequency,prediction)
            rows.append({"candidate":candidate,"branch_scale_synchronized":synchronized,
                         "storage_RMSE_scaled":float(torch.sqrt(terms["storage"]).detach()),
                         "loss_log_RMSE":float(torch.sqrt(terms["loss"]).detach()),
                         "maximum_cycle_change":float(prediction["cycle_change"].max().detach()),
                         "minimum_dissipation":float(prediction["minimum_dissipation"].detach())})
    frame=pd.DataFrame(rows);frame.to_csv(output/"dissipation_normalization_diagnostic.csv",index=False)
    grouped=frame.groupby("branch_scale_synchronized").storage_RMSE_scaled.mean()
    selected=bool(grouped.idxmin())
    return selected,frame


def curve_rows(predictions,candidate,seed,split):
    rows=[]
    for item in predictions:
        c=item["condition"];peak=c.loading_end_index;eq=c.equilibrium_MPa
        obs_peak=float(c.stress_MPa[peak]);pred_peak=float(item["stress"][peak].detach())
        obs_neq=max(obs_peak-eq,1e-12);pred_neq=max(pred_peak-eq,1e-12)
        for index in range(len(c.time_s)):
            elapsed=max(float(c.time_s[index]-c.time_s[peak]),0.0)
            obs=float(c.stress_MPa[index]);pred=float(item["stress"][index].detach())
            rows.append({"candidate":candidate,"seed":seed,"split":split,"condition":c.key,
                         "mode":c.mode,"label":c.label,"time_s":float(c.time_s[index]),
                         "hold_time_s":elapsed,"strain":float(c.strain[index]),
                         "experimental_stress_MPa":obs,"predicted_stress_MPa":pred,
                         "experimental_R":obs/obs_peak,"predicted_R":pred/pred_peak,
                         "experimental_Rneq":(obs-eq)/obs_neq,
                         "predicted_Rneq":(pred-eq)/pred_neq,
                         "minimum_dissipation":float(item["dissipation"][index].min().detach())})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default=str(CONFIG));args=parser.parse_args()
    torch.set_default_dtype(torch.float64)
    config=load_config(Path(args.config).read_text(encoding="utf-8"))
    output=Path(config["outputs"]["results_dir"]);checkpoints=Path(config["outputs"]["checkpoints_dir"])
    output.mkdir(parents=True,exist_ok=True);checkpoints.mkdir(parents=True,exist_ok=True)
    stage4_config=load_config(Path(config["stage4_config"]).read_text(encoding="utf-8"))
    parameters=MaterialParameters.from_json(stage4_config["teacher_parameter_file"])
    raw,relaxation,frequency=load_real_data(config["real_training_cache"],parameters,
        points_per_decade=config["data"]["points_per_time_decade"],
        maximum_loading_points=config["data"]["maximum_loading_points"],device="cpu")

    synchronize,normalization=normalization_diagnostic(config,frequency,output)
    stage4=pd.read_csv(ROOT/"results"/"stage4"/"final_relaxation_metrics.csv")
    stage4=stage4[(stage4.model=="Stage4-GD-PICNN") &
                  (stage4.solver=="full_tensor_local_implicit")]
    stage4_baseline={}
    for split in ("training","validation"):
        subset=stage4[stage4.split==split]
        stage4_baseline[split]={
            "q_log_RMSE":float(np.sqrt(np.mean(np.log(subset.predicted_q/subset.experimental_q)**2))),
            "Rneq_RMSE":float(np.sqrt(np.mean(subset["Rneq_RMSE"]**2))),
            "R_RMSE":float(np.sqrt(np.mean(subset["R_RMSE"]**2)))}

    seed_rows=[];all_curve_rows=[];all_dma_rows=[];candidate_best={}
    for candidate in config["energy"]["candidates"]:
        for seed in config["optimization"]["seeds"]:
            material,meta,_=load_stage5b_material(config,candidate,"cpu",
                synchronize_branch_scale=synchronize)
            anchor={name:value.detach().clone() for name,value in material.nonequilibrium_energy.named_parameters()}
            perturb_energy(material.nonequilibrium_energy,seed,config["optimization"]["perturbation_std"])
            optimizer=torch.optim.AdamW(material.nonequilibrium_energy.parameters(),
                lr=config["optimization"]["learning_rate"],weight_decay=config["optimization"]["weight_decay"])
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                config["optimization"]["epochs"],eta_min=config["optimization"]["learning_rate"]*
                config["optimization"]["minimum_learning_rate_fraction"])
            best=None;start=time.perf_counter()
            for epoch in range(1,config["optimization"]["epochs"]+1):
                optimizer.zero_grad()
                data_loss=differentiable_relaxation_objective(material,relaxation,config)
                reg=regularization(material.nonequilibrium_energy,anchor,config)
                total=data_loss+reg
                total.backward();gradient=torch.nn.utils.clip_grad_norm_(
                    material.nonequilibrium_energy.parameters(),config["optimization"]["gradient_clip"])
                optimizer.step();scheduler.step()
                if epoch==1 or epoch%config["optimization"]["evaluation_interval"]==0 or epoch==config["optimization"]["epochs"]:
                    train_value,train_metrics=evaluate_relaxation_only(material,relaxation,"training",config)
                    validation_value,validation_metrics=evaluate_relaxation_only(material,relaxation,"validation",config)
                    score=validation_metrics["objective"]
                    record={"epoch":epoch,"score":score,"train":train_metrics,"validation":validation_metrics,
                            "gradient_norm":float(gradient),"regularization":float(reg.detach())}
                    if best is None or score<best["score"]:
                        best={**record,"state":snapshot_energy(material.nonequilibrium_energy)}
                    if epoch%30==0:
                        print(f"stage5b {candidate} seed={seed} epoch={epoch} val={score:.4g} "
                              f"q={validation_metrics['q_log_RMSE']:.3g} Rneq={validation_metrics['Rneq_RMSE']:.3g}")
            material.nonequilibrium_energy.load_state_dict(best["state"])
            train_value,train_metrics=evaluate(material,relaxation,frequency,"training",config)
            validation_value,validation_metrics=evaluate(material,relaxation,frequency,"validation",config)
            wall=time.perf_counter()-start
            row={"candidate":candidate,"seed":seed,"best_epoch":best["epoch"],"wall_time_s":wall,
                 **{f"training_{key}":value for key,value in train_metrics.items()},
                 **{f"validation_{key}":value for key,value in validation_metrics.items()},
                 "parameter_count":sum(p.numel() for p in material.nonequilibrium_energy.parameters())}
            seed_rows.append(row)
            for split,value in (("training",train_value),("validation",validation_value)):
                all_curve_rows.extend(curve_rows(value["relaxation_predictions"],candidate,seed,split))
                for item in value["frequency"]["rows"]:all_dma_rows.append({"candidate":candidate,"seed":seed,**item})
            checkpoint={"schema_version":"2026-08-31-stage5b-candidate-v1","candidate":candidate,
                "seed":seed,"best_epoch":best["epoch"],"energy_state_dict":snapshot_energy(material.nonequilibrium_energy),
                "energy_physical_parameters":material.nonequilibrium_energy.physical_parameters(),
                "stage4_dissipation_state_dict":{key:value.detach().cpu() for key,value in material.dissipation_potential.state_dict().items()},
                "branch_scale_synchronized":synchronize,"metrics":{"training":train_metrics,"validation":validation_metrics},
                "stage4_checkpoint_sha256":sha256(config["stage4_checkpoint"]),
                "stage5a_spectrum_sha256":sha256(config["stage5a_spectrum_file"])}
            path=checkpoints/f"{candidate}_seed{seed}.pt";torch.save(checkpoint,path)
            selection=(validation_metrics["objective"],validation_metrics["q_log_RMSE"],row["parameter_count"])
            if candidate not in candidate_best or selection<candidate_best[candidate][0]:
                candidate_best[candidate]=(selection,path,row)

    seeds=pd.DataFrame(seed_rows);seeds.to_csv(output/"stage5b_seed_results.csv",index=False)
    pd.DataFrame(all_curve_rows).to_csv(output/"stage5b_reduced_relaxation_curves.csv",index=False)
    pd.DataFrame(all_dma_rows).to_csv(output/"stage5b_reduced_frequency_dma.csv",index=False)
    best_rows=[]
    for candidate,(_,path,row) in candidate_best.items():
        guard=(row["validation_Rneq_RMSE"] <= stage4_baseline["validation"]["Rneq_RMSE"]*
               (1+config["selection"]["maximum_validation_Rneq_degradation_fraction_from_stage4"]))
        feasible=(row["training_q_ordering_accuracy"]>=1.0-1e-12 and
                  row["validation_q_log_RMSE"]<stage4_baseline["validation"]["q_log_RMSE"] and guard and
                  row["validation_minimum_dissipation"]>=-1e-12)
        best_rows.append({**row,"checkpoint":str(path),"Rneq_guard_pass":guard,"feasible":feasible})
    comparison=pd.DataFrame(best_rows).sort_values(["feasible","validation_objective","parameter_count"],
                                                   ascending=[False,True,True])
    comparison.to_csv(output/"stage5b_candidate_comparison.csv",index=False)
    feasible=comparison[comparison.feasible]
    selected=None if feasible.empty else feasible.sort_values(["validation_objective","parameter_count"]).iloc[0]
    if selected is not None:
        source=Path(selected.checkpoint);destination=checkpoints/"stage5b_selected.pt";destination.write_bytes(source.read_bytes())
    summary={"schema_version":"2026-08-31-chapter3-stage5b-training-v1",
        "status":"reduced_training_completed","branch_scale_synchronized":synchronize,
        "normalization_diagnostic":normalization.to_dict("records"),"stage4_baseline":stage4_baseline,
        "selected_candidate":None if selected is None else selected.candidate,
        "selected_seed":None if selected is None else int(selected.seed),
        "selected_checkpoint":None if selected is None else str(checkpoints/"stage5b_selected.pt"),
        "candidate_comparison":comparison.to_dict("records"),
        "amplitude_labels_used_for_training":False,"source_workbook_sha256":raw["source_workbook_sha256"],
        "boundary":"Reduced differentiable training; production claims require full-tensor local-implicit validation."}
    save_json(output/"stage5b_training_summary.json",summary)
    print(json.dumps({"selected_candidate":summary["selected_candidate"],"selected_seed":summary["selected_seed"],
                      "branch_scale_synchronized":synchronize,"comparison":comparison[["candidate","validation_q_log_RMSE",
                      "validation_Rneq_RMSE","validation_storage_RMSE_scaled","validation_loss_log_RMSE","feasible"]].to_dict("records")},
                     ensure_ascii=False,indent=2))


if __name__=="__main__":main()
