"""Pointwise teacher-flow datasets, losses, metrics and stage-2 training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from dual_potential.dissipation import Stage2DissipationPICNN


NORMALIZATION = {
    "log_tau_over_1s": (0.0, 2.0),
    "total_I1_minus_3": (0.3030788629509967, 0.41706362389751866),
    "elastic_I1_minus_3": (0.06501872142323824, 0.1781529178183958),
    "viscous_I1_minus_3": (0.1951905443602957, 0.3487359989378981),
    "hardening_variable": (0.16718546022336916, 0.16596854899036956),
}


@dataclass
class FlowDataset:
    X: torch.Tensor
    D: torch.Tensor
    branch: torch.Tensor
    tau: torch.Tensor
    weight: torch.Tensor
    trajectory: torch.Tensor
    confidence: torch.Tensor
    features: dict[str, torch.Tensor]

    def select(self, mask):
        return FlowDataset(self.X[mask], self.D[mask], self.branch[mask],
                           self.tau[mask], self.weight[mask], self.trajectory[mask],
                           self.confidence[mask],
                           {name: value[mask] for name, value in self.features.items()})

    def to(self, device):
        return FlowDataset(self.X.to(device), self.D.to(device), self.branch.to(device),
                           self.tau.to(device), self.weight.to(device), self.trajectory.to(device),
                           self.confidence.to(device),
                           {name: value.to(device) for name, value in self.features.items()})

    @property
    def count(self): return len(self.X)


def load_datasets(cache: str | Path, device="cpu"):
    raw = np.load(cache)
    tensor = lambda name, dtype=torch.float64: torch.as_tensor(raw[name], dtype=dtype)
    all_data = FlowDataset(
        tensor("flow_X_MPa"), tensor("flow_D_per_s"), tensor("flow_branch_index", torch.long),
        torch.as_tensor(raw["tau_s"][raw["flow_branch_index"]], dtype=torch.float64),
        tensor("flow_sample_weight"), tensor("flow_trajectory_index", torch.long),
        tensor("flow_confidence_code", torch.long),
        {name: tensor(f"flow_{name}") for name in (
            "total_I1_minus_3", "elastic_I1_minus_3", "viscous_I1_minus_3", "hardening_variable")})
    split = tensor("flow_split_code", torch.long)
    use = torch.as_tensor(raw["flow_use_flow"], dtype=torch.bool)
    result = {
        "train": all_data.select((split == 0) & (all_data.confidence < 2) & use),
        "validation_high_medium": all_data.select((split == 1) & (all_data.confidence < 2) & use),
        "validation_low": all_data.select((split == 1) & (all_data.confidence == 2) & use),
        "test_high_medium": all_data.select((split == 2) & (all_data.confidence < 2) & use),
        "test_low": all_data.select((split == 2) & (all_data.confidence == 2) & use),
    }
    spectrum = {
        "tau_s": torch.as_tensor(raw["tau_s"], dtype=torch.float64, device=device),
        "branch_moduli_MPa": torch.as_tensor(0.12 * raw["branch_shares"], dtype=torch.float64, device=device),
    }
    return {name: value.to(device) for name, value in result.items()}, spectrum


def weighted_mean(value, weight):
    return torch.sum(value * weight) / weight.sum().clamp_min(1e-30)


def predict(model, data: FlowDataset, index=slice(None)):
    X = data.X[index]
    j = 0.5 * torch.einsum("...ij,...ij->...", X, X) / model.sigma_ref.square()
    features = {name: value[index] for name, value in data.features.items()}
    _, slope = model.potential_and_slope(j, data.branch[index], features)
    D = slope[..., None, None] * X / model.sigma_ref
    return slope, D


def loss_terms(model, data, index, asinh_scale=0.02):
    slope, D = predict(model, data, index)
    X, target, tau, weight = data.X[index], data.D[index], data.tau[index], data.weight[index]
    norm2 = torch.einsum("...ij,...ij->...", X, X)
    nonzero = norm2 > 1e-28
    target_slope = model.sigma_ref * torch.einsum("...ij,...ij->...", X, target) / norm2.clamp_min(1e-30)
    log_error = torch.log(slope.clamp_min(1e-16)) - torch.log(target_slope.clamp_min(1e-16))
    log_loss = weighted_mean(F.huber_loss(log_error, torch.zeros_like(log_error), reduction="none", delta=0.5),
                             weight * nonzero)
    predicted_scaled, target_scaled = tau[:, None, None] * D, tau[:, None, None] * target
    transformed = torch.asinh(predicted_scaled/asinh_scale) - torch.asinh(target_scaled/asinh_scale)
    tensor_loss = weighted_mean(torch.sum(transformed.square(), (-2, -1))/3.0, weight)
    teacher_energy = weighted_mean(torch.sum(target_scaled.square(), (-2, -1)), weight).clamp_min(1e-16)
    linear = weighted_mean(torch.sum((predicted_scaled-target_scaled).square(), (-2, -1)), weight)/teacher_energy
    total = log_loss + 0.5*tensor_loss + 0.25*linear
    return {"total": total, "log_slope": log_loss, "tensor": tensor_loss, "linear": linear}


@torch.no_grad()
def evaluate(model, data: FlowDataset, name: str):
    slope, D = predict(model, data)
    target_scaled = data.tau[:, None, None] * data.D
    predicted_scaled = data.tau[:, None, None] * D
    error = torch.linalg.matrix_norm(predicted_scaled-target_scaled)
    target_norm = torch.linalg.matrix_norm(target_scaled)
    nrmse = torch.sqrt(weighted_mean(error.square(), data.weight))/torch.sqrt(
        weighted_mean(target_norm.square(), data.weight)).clamp_min(1e-16)
    Xnorm = torch.linalg.matrix_norm(data.X)
    active = (Xnorm > 1e-10) & (target_norm > 1e-12)
    cosine = torch.sum(D[active]*data.D[active], (-2,-1))/(
        torch.linalg.matrix_norm(D[active])*torch.linalg.matrix_norm(data.D[active])).clamp_min(1e-30)
    angles = torch.rad2deg(torch.acos(cosine.clamp(-1,1))) if active.any() else torch.zeros(1, device=D.device)
    diss_pred = torch.sum(data.X*D, (-2,-1)); diss_true = torch.sum(data.X*data.D, (-2,-1))
    diss_rmse = torch.sqrt(weighted_mean((diss_pred-diss_true).square(), data.weight))/torch.sqrt(
        weighted_mean(diss_true.square(), data.weight)).clamp_min(1e-16)
    low = Xnorm <= torch.quantile(Xnorm, 0.1)
    fast = data.branch <= 3
    result = {"dataset": name, "branch_point_count": data.count,
              "trajectory_count": int(torch.unique(data.trajectory).numel()),
              "flow_nrmse": float(nrmse), "dissipation_nrmse": float(diss_rmse),
              "mean_direction_error_degree": float(angles.mean()),
              "maximum_direction_error_degree": float(angles.max()),
              "minimum_predicted_slope": float(slope.min()),
              "minimum_predicted_dissipation_MPa_per_s": float(diss_pred.min())}
    for label, mask in (("low_force", low), ("fast", fast), ("slow", ~fast)):
        subset_error_rms = torch.sqrt(weighted_mean(error[mask].square(), data.weight[mask]))
        subset_target_rms = torch.sqrt(weighted_mean(target_norm[mask].square(), data.weight[mask]))
        result[f"{label}_flow_nrmse"] = float(subset_error_rms/subset_target_rms.clamp_min(1e-16))
        result[f"{label}_dimensionless_flow_rmse"] = float(subset_error_rms)
        result[f"{label}_dimensionless_target_rms"] = float(subset_target_rms)
    result["branches"] = []
    for branch in range(7):
        mask = data.branch == branch
        relative = torch.sqrt(weighted_mean(error[mask].square(), data.weight[mask]))/torch.sqrt(
            weighted_mean(target_norm[mask].square(), data.weight[mask])).clamp_min(1e-16)
        result["branches"].append({"branch": branch, "tau_s": float(data.tau[mask][0]),
                                   "flow_nrmse": float(relative), "count": int(mask.sum())})
    return result


def train_model(architecture, seed, datasets, spectrum, *, maximum_epochs=180,
                minimum_epochs=80, patience=40, batch_size=8192, learning_rate=0.003):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = datasets["train"].X.device
    model = Stage2DissipationPICNN(architecture, spectrum["tau_s"], spectrum["branch_moduli_MPa"], 0.11565279376482657,
                                   NORMALIZATION, dtype=torch.float64, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maximum_epochs, eta_min=1e-4)
    train = datasets["train"]
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    start = perf_counter()
    for epoch in range(1, maximum_epochs+1):
        model.train(); order = torch.randperm(train.count, device=device)
        for offset in range(0, train.count, batch_size):
            index = order[offset:offset+batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = loss_terms(model, train, index)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0:
            model.eval()
            train_metric = evaluate(model, train, "train")
            validation = evaluate(model, datasets["validation_high_medium"], "validation_high_medium")
            row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                   "train_flow_nrmse": train_metric["flow_nrmse"],
                   "validation_flow_nrmse": validation["flow_nrmse"],
                   "elapsed_s": perf_counter()-start}
            history.append(row)
            score = validation["flow_nrmse"]
            if score < best-1e-7:
                best, best_epoch, stale = score, epoch, 0
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else: stale += 10
            if epoch >= minimum_epochs and stale >= patience: break
    model.load_state_dict({k:v.to(device) for k,v in best_state.items()}); model.eval()
    metrics = {name:evaluate(model,data,name) for name,data in datasets.items()}
    return model, {"architecture": architecture, "seed": seed, "best_epoch": best_epoch,
                   "completed_epochs": epoch, "training_time_s": perf_counter()-start,
                   "parameter_count": model.parameter_count, "history": history, "metrics": metrics}
