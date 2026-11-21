"""Stage 5A: positive small-strain spectrum and energy architecture prescreen."""

from __future__ import annotations

from project_paths import load_config

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
import torch
from torch import nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from dual_potential.constitutive import MaterialParameters
from stage3_real import equilibrium_shear_modulus, equilibrium_stress, load_real_data

CONFIG = ROOT / "configs" / "stage5a_linear_spectrum_energy_prescreen.yaml"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(value):
    value = np.asarray(value, float)
    return float(np.sqrt(np.mean(value * value)))


def spectrum_matrices(frequency_hz, tau_s):
    omega_tau = 2.0 * np.pi * np.asarray(frequency_hz)[:, None] * np.asarray(tau_s)[None, :]
    storage = omega_tau.square() / (1.0 + omega_tau.square()) if hasattr(omega_tau, "square") else omega_tau**2 / (1.0 + omega_tau**2)
    loss = omega_tau / (1.0 + omega_tau**2)
    return storage, loss


def spectrum_metrics(weights, rows, tau_s, geq):
    frequency = np.array([row["frequency_hz"] for row in rows], float)
    experimental_storage = np.array([row["storage_MPa"] for row in rows], float)
    experimental_loss = np.array([row["loss_MPa"] for row in rows], float)
    As, Al = spectrum_matrices(frequency, tau_s)
    predicted_storage = geq + As @ weights
    predicted_loss = Al @ weights
    storage_rmse = rms(predicted_storage - experimental_storage)
    loss_rmse = rms(predicted_loss - experimental_loss)
    storage_relative = rms((predicted_storage - experimental_storage) / experimental_storage)
    loss_relative = rms((predicted_loss - experimental_loss) / experimental_loss)
    neq_storage_scale = max(rms(experimental_storage - geq), 1e-10)
    loss_scale = max(rms(experimental_loss), 1e-10)
    balanced = math.sqrt(.5 * (storage_rmse / neq_storage_scale)**2 +
                         .5 * (loss_rmse / loss_scale)**2)
    return {"storage_RMSE_MPa": storage_rmse, "loss_RMSE_MPa": loss_rmse,
            "storage_relative_RMSE": storage_relative, "loss_relative_RMSE": loss_relative,
            "balanced_NRMSE": balanced, "predicted_storage": predicted_storage,
            "predicted_loss": predicted_loss}


def fit_positive_spectrum(raw, tau_s, geq, config, output):
    rows = raw["frequency_dma"]
    train = [row for row in rows if row["split"] == "training"]
    validation = [row for row in rows if row["split"] == "validation"]
    frequency = np.array([row["frequency_hz"] for row in train], float)
    ys = np.array([row["storage_MPa"] - geq for row in train], float)
    yl = np.array([row["loss_MPa"] for row in train], float)
    As, Al = spectrum_matrices(frequency, tau_s)
    storage_scale = max(rms(ys), 1e-10); loss_scale = max(rms(yl), 1e-10)
    identity = np.eye(len(tau_s))
    second = np.diff(np.eye(len(tau_s)), n=2, axis=0)
    candidates = []
    for loss_weight in config["linear_spectrum"]["loss_weight_grid"]:
        for ridge in config["linear_spectrum"]["ridge_grid"]:
            for smoothness in config["linear_spectrum"]["smoothness_grid"]:
                matrices = [As / storage_scale, math.sqrt(loss_weight) * Al / loss_scale]
                targets = [ys / storage_scale, math.sqrt(loss_weight) * yl / loss_scale]
                if ridge > 0:
                    matrices.append(math.sqrt(ridge) * identity); targets.append(np.zeros(len(tau_s)))
                if smoothness > 0:
                    matrices.append(math.sqrt(smoothness) * second); targets.append(np.zeros(len(tau_s)-2))
                result = lsq_linear(np.vstack(matrices), np.concatenate(targets), bounds=(0.0, np.inf),
                                    method="trf", lsmr_tol="auto", max_iter=2000)
                weights = result.x
                tm = spectrum_metrics(weights, train, tau_s, geq)
                vm = spectrum_metrics(weights, validation, tau_s, geq)
                am = spectrum_metrics(weights, rows, tau_s, geq)
                candidates.append({"loss_weight": loss_weight, "ridge": ridge, "smoothness": smoothness,
                    "optimizer_success": bool(result.success), "optimizer_cost": float(result.cost),
                    "training_storage_RMSE_MPa": tm["storage_RMSE_MPa"],
                    "training_loss_RMSE_MPa": tm["loss_RMSE_MPa"],
                    "training_balanced_NRMSE": tm["balanced_NRMSE"],
                    "validation_storage_RMSE_MPa": vm["storage_RMSE_MPa"],
                    "validation_loss_RMSE_MPa": vm["loss_RMSE_MPa"],
                    "validation_storage_relative_RMSE": vm["storage_relative_RMSE"],
                    "validation_loss_relative_RMSE": vm["loss_relative_RMSE"],
                    "validation_balanced_NRMSE": vm["balanced_NRMSE"],
                    "all_balanced_NRMSE": am["balanced_NRMSE"],
                    "total_modulus_MPa": float(weights.sum()),
                    "active_branch_count": int(np.sum(weights > config["linear_spectrum"]["active_weight_threshold_MPa"])),
                    "weights_json": json.dumps(weights.tolist())})
    frame = pd.DataFrame(candidates).sort_values(["validation_balanced_NRMSE", "all_balanced_NRMSE"])
    frame.to_csv(output / "linear_spectrum_candidates.csv", index=False)
    selected = frame.iloc[0]
    weights = np.array(json.loads(selected.weights_json), float)

    predictions = []
    all_metrics = spectrum_metrics(weights, rows, tau_s, geq)
    for index, row in enumerate(rows):
        predictions.append({"frequency_hz": row["frequency_hz"], "split": row["split"],
            "amplitude_fraction": row["amplitude_fraction"],
            "experimental_storage_MPa": row["storage_MPa"],
            "predicted_storage_MPa": all_metrics["predicted_storage"][index],
            "experimental_loss_MPa": row["loss_MPa"],
            "predicted_loss_MPa": all_metrics["predicted_loss"][index]})
    pd.DataFrame(predictions).to_csv(output / "linear_spectrum_predictions.csv", index=False)

    # Identifiability of the active, scaled design matrix.
    weighted_design = np.vstack((As / storage_scale,
        math.sqrt(selected.loss_weight) * Al / loss_scale))
    singular = np.linalg.svd(weighted_design, compute_uv=False)
    practical_rank = int(np.sum(singular / singular[0] >=
        config["linear_spectrum"]["practical_singular_value_ratio"]))
    condition = float(singular[0] / singular[-1])

    # Nonparametric frequency bootstrap with frozen hyperparameters.
    rng = np.random.default_rng(config["linear_spectrum"]["bootstrap_seed"])
    bootstrap = []
    for repeat in range(config["linear_spectrum"]["bootstrap_repeats"]):
        indices = rng.integers(0, len(train), len(train))
        matrices = [As[indices] / storage_scale,
                    math.sqrt(selected.loss_weight) * Al[indices] / loss_scale]
        targets = [ys[indices] / storage_scale,
                   math.sqrt(selected.loss_weight) * yl[indices] / loss_scale]
        if selected.ridge > 0:
            matrices.append(math.sqrt(selected.ridge) * identity); targets.append(np.zeros(len(tau_s)))
        if selected.smoothness > 0:
            matrices.append(math.sqrt(selected.smoothness) * second); targets.append(np.zeros(len(tau_s)-2))
        fit = lsq_linear(np.vstack(matrices), np.concatenate(targets), bounds=(0.0, np.inf),
                         method="trf", lsmr_tol="auto", max_iter=2000)
        for branch, (tau, value) in enumerate(zip(tau_s, fit.x)):
            bootstrap.append({"repeat": repeat, "branch": branch, "tau_s": tau, "weight_MPa": value})
    bootstrap_frame = pd.DataFrame(bootstrap)
    bootstrap_frame.to_csv(output / "linear_spectrum_bootstrap.csv", index=False)
    bootstrap_summary = bootstrap_frame.groupby(["branch", "tau_s"]).weight_MPa.agg(
        ["mean", "std", "min", "max"]).reset_index()
    bootstrap_summary.to_csv(output / "linear_spectrum_bootstrap_summary.csv", index=False)

    summary = {"tau_s": list(map(float, tau_s)), "weights_MPa": weights.tolist(),
        "total_modulus_MPa": float(weights.sum()), "selected_hyperparameters": {
            "loss_weight": float(selected.loss_weight), "ridge": float(selected.ridge),
            "smoothness": float(selected.smoothness)},
        "training_metrics": spectrum_metrics(weights, train, tau_s, geq) | {},
        "validation_metrics": spectrum_metrics(weights, validation, tau_s, geq) | {},
        "all_metrics": all_metrics | {}, "singular_values": singular.tolist(),
        "condition_number": condition, "practical_rank": practical_rank,
        "active_branch_count": int(np.sum(weights > config["linear_spectrum"]["active_weight_threshold_MPa"]))}
    for key in ("training_metrics", "validation_metrics", "all_metrics"):
        summary[key].pop("predicted_storage", None); summary[key].pop("predicted_loss", None)
    (output / "selected_linear_spectrum.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, weights


class SaturatingEnergy(nn.Module):
    """Positive mixture of saturating invariant ridge energies.

    Every ridge has unit derivative at the reference state. Consequently the
    small-strain shear modulus is exactly the prescribed group modulus, while
    the derivative may decrease smoothly at finite strain.
    """

    def __init__(self, name, group_moduli, ridge_counts, b_bounds, p_bounds, seed):
        super().__init__(); torch.manual_seed(seed)
        self.name = name
        self.register_buffer("group_moduli", torch.tensor(group_moduli, dtype=torch.float64))
        self.ridge_counts = list(ridge_counts)
        self.log_b_bounds = tuple(math.log(float(x)) for x in b_bounds)
        self.p_bounds = tuple(float(x) for x in p_bounds)
        self.raw_weight = nn.ParameterList(); self.raw_rho = nn.ParameterList()
        self.raw_b = nn.ParameterList(); self.raw_p = nn.ParameterList()
        self.raw_hardening = nn.ParameterList(); self.raw_hardening_rho = nn.ParameterList()
        for count in ridge_counts:
            self.raw_weight.append(nn.Parameter(.10*torch.randn(count, dtype=torch.float64)))
            self.raw_rho.append(nn.Parameter(torch.linspace(-.5, .5, count, dtype=torch.float64) +
                                              .20*torch.randn(count, dtype=torch.float64)))
            self.raw_b.append(nn.Parameter(torch.linspace(-1.0, 1.0, count, dtype=torch.float64) +
                                            .20*torch.randn(count, dtype=torch.float64)))
            self.raw_p.append(nn.Parameter(.20*torch.randn(count, dtype=torch.float64)))
            self.raw_hardening.append(nn.Parameter(torch.tensor(-2.0, dtype=torch.float64) +
                                                    .20*torch.randn((), dtype=torch.float64)))
            self.raw_hardening_rho.append(nn.Parameter(.20*torch.randn((), dtype=torch.float64)))

    def physical(self):
        groups = []
        for group in range(len(self.ridge_counts)):
            lo, hi = self.log_b_bounds; plo, phi = self.p_bounds
            groups.append({"modulus_MPa": float(self.group_moduli[group].detach()),
                "weights": torch.softmax(self.raw_weight[group], 0),
                "rho_I1": torch.sigmoid(self.raw_rho[group]),
                "b": torch.exp(lo + (hi-lo)*torch.sigmoid(self.raw_b[group])),
                "p": plo + (phi-plo)*torch.sigmoid(self.raw_p[group]),
                "hardening_MPa": F.softplus(self.raw_hardening[group]),
                "hardening_rho_I1": torch.sigmoid(self.raw_hardening_rho[group])})
        return groups

    @staticmethod
    def saturation(z, b, p):
        return ((1.0 + b*z).pow(p) - 1.0) / (b*p)

    def energy_invariants(self, x1, x2):
        total = x1.new_zeros(torch.broadcast_shapes(x1.shape, x2.shape))
        for group in self.physical():
            z = group["rho_I1"]*x1[..., None] + (1.0-group["rho_I1"])*x2[..., None]
            terms = self.saturation(z, group["b"], group["p"])
            total = total + .5*group["modulus_MPa"]*torch.sum(group["weights"]*terms, -1)
            zh = group["hardening_rho_I1"]*x1 + (1.0-group["hardening_rho_I1"])*x2
            total = total + .25*group["hardening_MPa"]*zh.square()
        return total

    def stress_path(self, strain, mode):
        if mode == "shear":
            x1 = strain.square(); x2 = x1
            dx1 = 2.0*strain; dx2 = dx1
        elif mode == "uniaxial":
            lam = 1.0 + strain
            x1 = lam.square() + 2.0/lam - 3.0
            x2 = 2.0*lam + lam.pow(-2) - 3.0
            dx1 = 2.0*lam - 2.0*lam.pow(-2)
            dx2 = 2.0 - 2.0*lam.pow(-3)
        else:
            raise ValueError(mode)
        total = strain.new_zeros(strain.shape)
        for group in self.physical():
            z = group["rho_I1"]*x1[..., None] + (1.0-group["rho_I1"])*x2[..., None]
            dz = group["rho_I1"]*dx1[..., None] + (1.0-group["rho_I1"])*dx2[..., None]
            slope = (1.0 + group["b"]*z).pow(group["p"]-1.0)
            total = total + .5*group["modulus_MPa"]*torch.sum(group["weights"]*slope*dz, -1)
            zh = group["hardening_rho_I1"]*x1 + (1.0-group["hardening_rho_I1"])*x2
            dzh = group["hardening_rho_I1"]*dx1 + (1.0-group["hardening_rho_I1"])*dx2
            total = total + .5*group["hardening_MPa"]*zh*dzh
        return total

    def serializable(self):
        output = {"name": self.name, "groups": []}
        for group in self.physical():
            output["groups"].append({"modulus_MPa": group["modulus_MPa"],
                "weights": group["weights"].detach().cpu().tolist(),
                "rho_I1": group["rho_I1"].detach().cpu().tolist(),
                "b": group["b"].detach().cpu().tolist(),
                "p": group["p"].detach().cpu().tolist(),
                "hardening_MPa": float(group["hardening_MPa"].detach()),
                "hardening_rho_I1": float(group["hardening_rho_I1"].detach())})
        return output


def architecture_spec(name, weights, tau_s, config, seed):
    b_bounds = config["energy_prescreen"]["saturation_b_bounds"]
    p_bounds = config["energy_prescreen"]["saturation_power_bounds"]
    fast = float(weights[np.asarray(tau_s) <= 1.0].sum())
    slow = float(weights[np.asarray(tau_s) > 1.0].sum())
    if name == "shared_low_parameter_saturation":
        return SaturatingEnergy(name, [weights.sum()], [1], b_bounds, p_bounds, seed)
    if name == "grouped_fast_slow_saturation":
        return SaturatingEnergy(name, [fast, slow], [1, 1], b_bounds, p_bounds, seed)
    if name == "shared_saturating_energy_network":
        return SaturatingEnergy(name, [weights.sum()],
            [config["energy_prescreen"]["ridge_count_shared"]], b_bounds, p_bounds, seed)
    if name == "hierarchical_fast_slow_energy_network":
        count = config["energy_prescreen"]["ridge_count_grouped"]
        return SaturatingEnergy(name, [fast, slow], [count, count], b_bounds, p_bounds, seed)
    raise ValueError(name)


def q_targets(relaxation):
    rows = []
    for condition in relaxation:
        peak = condition.loading_end_index
        exp_peak = float(condition.stress_MPa[peak])
        eq = float(condition.equilibrium_MPa)
        level = float(condition.strain[peak])
        rows.append({"condition": condition.key, "mode": condition.mode, "split": condition.split,
            "level": level, "equilibrium_MPa": eq,
            "target_nonequilibrium_stress_MPa": exp_peak-eq,
            "experimental_q": (exp_peak-eq)/max(abs(eq), 1e-12)})
    return rows


def energy_objective(model, targets, zhang_li, config):
    losses = []; predictions = []; by_mode = {"uniaxial": [], "shear": []}
    for row in targets:
        strain = torch.tensor(row["level"], dtype=torch.float64)
        predicted_stress = model.stress_path(strain, row["mode"])
        predicted_q = predicted_stress / max(abs(row["equilibrium_MPa"]), 1e-12)
        observed_q = predicted_q.new_tensor(row["experimental_q"])
        losses.append(torch.log(predicted_q.clamp_min(1e-12)/observed_q).square())
        by_mode[row["mode"]].append((strain, predicted_q, observed_q))
        predictions.append((row, predicted_stress, predicted_q))
    slope_losses = []; ordering_losses = []; ordering_correct = []
    for values in by_mode.values():
        values = sorted(values, key=lambda item: float(item[0]))
        for left, right in zip(values[:-1], values[1:]):
            pred = torch.log(right[1].clamp_min(1e-12)/left[1].clamp_min(1e-12))
            obs = torch.log(right[2]/left[2])
            slope_losses.append((pred-obs).square())
            target_gap = (right[2]-left[2]).clamp_min(1e-8)
            predicted_gap = right[1]-left[1]
            ordering_losses.append((torch.relu(.2*target_gap-predicted_gap)/target_gap).square())
            ordering_correct.append(float((predicted_gap > 0.0).detach()))
    # Penalize negative total path tangents without enforcing global convexity.
    tangent_penalties = []
    for mode, upper in (("uniaxial", 1.0), ("shear", .8)):
        grid = torch.linspace(.001, upper, 25, dtype=torch.float64, requires_grad=True)
        total_stress = equilibrium_stress(grid, mode, zhang_li) + model.stress_path(grid, mode)
        tangent = torch.autograd.grad(total_stress.sum(), grid, create_graph=True)[0]
        tangent_penalties.append(torch.relu(-tangent).square().mean())
    regularization = sum(parameter.square().mean() for parameter in model.parameters())
    loss = (config["energy_prescreen"]["q_log_weight"]*torch.stack(losses).mean() +
            config["energy_prescreen"]["mode_slope_weight"]*torch.stack(slope_losses).mean() +
            config["energy_prescreen"]["q_ordering_weight"]*torch.stack(ordering_losses).mean() +
            config["energy_prescreen"]["path_tangent_penalty"]*torch.stack(tangent_penalties).mean() +
            config["energy_prescreen"]["parameter_regularization"]*regularization)
    return loss, predictions, float(np.mean(ordering_correct))


def fit_energy_candidates(weights, tau_s, relaxation, parameters, config, output, checkpoint):
    targets = q_targets(relaxation)
    rows = []; prediction_rows = []; best_models = {}
    for name in config["energy_prescreen"]["candidates"]:
        best = None
        for seed in config["energy_prescreen"]["seeds"]:
            model = architecture_spec(name, weights, tau_s, config, seed)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["energy_prescreen"]["learning_rate"])
            best_state = None; best_loss = float("inf")
            for epoch in range(config["energy_prescreen"]["epochs"]):
                optimizer.zero_grad()
                loss, _, _ = energy_objective(model, targets, parameters.zhang_li, config)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config["energy_prescreen"]["gradient_clip"])
                optimizer.step()
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value
                    best_state = {key: item.detach().clone() for key, item in model.state_dict().items()}
            model.load_state_dict(best_state)
            _, predictions, ordering_accuracy = energy_objective(model, targets, parameters.zhang_li, config)
            observed = np.array([item[0]["experimental_q"] for item in predictions])
            predicted = np.array([float(item[2].detach()) for item in predictions])
            metric = rms(np.log(np.maximum(predicted, 1e-14)/observed))
            seed_row = {"candidate": name, "seed": seed, "objective": best_loss,
                        "q_ordering_accuracy": ordering_accuracy,
                        "q_log_RMSE": metric, "parameter_count": sum(p.numel() for p in model.parameters())}
            rows.append(seed_row)
            selection = (-ordering_accuracy, metric, best_loss)
            if best is None or selection < best[0]:
                best = (selection, model, predictions, seed, ordering_accuracy)
        best_models[name] = best[1]
        for row, stress, q in best[2]:
            prediction_rows.append({**row, "candidate": name, "seed": best[3],
                "q_ordering_accuracy": best[4],
                "predicted_nonequilibrium_stress_MPa": float(stress.detach()),
                "predicted_q": float(q.detach())})
        torch.save({"schema_version": "2026-08-31-stage5a-energy-prescreen-v2",
                    "candidate": name, "seed": best[3], "state_dict": best[1].state_dict(),
                    "physical_parameters": best[1].serializable(),
                    "linear_spectrum_weights_MPa": weights.tolist(), "tau_s": list(map(float, tau_s))},
                   checkpoint / f"{name}.pt")
    pd.DataFrame(rows).to_csv(output / "energy_architecture_seed_results.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "energy_architecture_q_predictions.csv", index=False)
    return best_models, pd.DataFrame(rows), pd.DataFrame(prediction_rows), targets


def invariants_from_F(F):
    J = torch.linalg.det(F)
    Fbar = F / J.pow(1.0/3.0)
    C = Fbar.transpose(-1, -2) @ Fbar
    I1 = torch.diagonal(C).sum(-1)
    I2 = .5*(I1.square() - torch.diagonal(C@C).sum(-1))
    return I1, I2


def total_energy(F, model, zhang_li):
    I1, I2 = invariants_from_F(F)
    first = (zhang_li["D"]*(I1-3.0) if abs(zhang_li["mu"]) < 1e-12 else
             zhang_li["D"]/zhang_li["mu"]*(1.0-torch.exp(-zhang_li["mu"]*(I1-3.0))))
    base = I2-3.0+zhang_li["N"]
    equilibrium = first + zhang_li["B"]*(base.pow(zhang_li["beta"])-zhang_li["N"]**zhang_li["beta"])
    return equilibrium + model.energy_invariants(I1-3.0, I2-3.0)


def deformation_path(kind, value):
    value = torch.as_tensor(value, dtype=torch.float64)
    one = value.new_tensor(1.0); zero = value.new_tensor(0.0)
    if kind == "uniaxial":
        lam = value; transverse = lam.pow(-.5)
        return torch.diag(torch.stack((lam, transverse, transverse)))
    if kind == "shear":
        return torch.stack((torch.stack((one, value, zero)),
                            torch.stack((zero, one, zero)),
                            torch.stack((zero, zero, one))))
    if kind == "planar":
        lam = value
        return torch.diag(torch.stack((lam, one, 1.0/lam)))
    if kind == "equibiaxial":
        lam = value
        return torch.diag(torch.stack((lam, lam, lam.pow(-2))))
    raise ValueError(kind)


def stability_screen(models, zhang_li, config, output):
    spec = config["stability_screen"]
    path_specs = {"uniaxial": spec["uniaxial_stretch"], "shear": spec["shear"],
                  "planar": spec["planar_stretch"], "equibiaxial": spec["equibiaxial_stretch"]}
    rng = np.random.default_rng(spec["acoustic_seed"])
    rows = []
    for name, model in models.items():
        identity = torch.eye(3, dtype=torch.float64, requires_grad=True)
        reference_energy = total_energy(identity, model, zhang_li)
        reference_stress = torch.autograd.grad(reference_energy, identity, create_graph=False)[0]
        minimum_energy = float("inf"); minimum_tangent = float("inf"); minimum_acoustic = float("inf")
        maximum_stress = 0.0; finite = True; state_count = 0
        for kind, (start, stop, count) in path_specs.items():
            for value in np.linspace(start, stop, int(count)):
                scalar = torch.tensor(float(value), dtype=torch.float64, requires_grad=True)
                F = deformation_path(kind, scalar)
                energy = total_energy(F, model, zhang_li)
                stress = torch.autograd.grad(energy, scalar, create_graph=True)[0]
                tangent = torch.autograd.grad(stress, scalar, create_graph=False)[0]
                values = (float(energy.detach()), float(stress.detach()), float(tangent.detach()))
                finite &= all(math.isfinite(x) for x in values)
                minimum_energy = min(minimum_energy, values[0]); maximum_stress = max(maximum_stress, abs(values[1]))
                minimum_tangent = min(minimum_tangent, values[2]); state_count += 1
                flat = F.detach().reshape(-1).requires_grad_(True)
                Hessian = torch.autograd.functional.hessian(
                    lambda x: total_energy(x.reshape(3,3), model, zhang_li), flat).detach().numpy()
                Fnp = F.detach().numpy()
                for _ in range(spec["acoustic_directions_per_state"]):
                    n = rng.normal(size=3); n /= np.linalg.norm(n)
                    normal = np.linalg.solve(Fnp.T, n); normal /= np.linalg.norm(normal)
                    m = rng.normal(size=3); m -= np.dot(m, normal)*normal
                    if np.linalg.norm(m) < 1e-10: continue
                    m /= np.linalg.norm(m)
                    rank_one = np.outer(m, n).reshape(-1)
                    acoustic = float(rank_one @ Hessian @ rank_one)
                    minimum_acoustic = min(minimum_acoustic, acoustic)
        rows.append({"candidate": name, "reference_energy_abs_MPa": abs(float(reference_energy.detach())),
            "reference_stress_max_abs_MPa": float(reference_stress.detach().abs().max()),
            "minimum_total_energy_MPa": minimum_energy, "minimum_path_tangent_MPa": minimum_tangent,
            "minimum_projected_acoustic_modulus_MPa": minimum_acoustic,
            "maximum_path_stress_abs_MPa": maximum_stress, "all_values_finite": finite,
            "state_count": state_count,
            "reference_pass": abs(float(reference_energy.detach())) <= 1e-10 and float(reference_stress.detach().abs().max()) <= 1e-9,
            "energy_pass": minimum_energy >= -1e-9,
            "path_tangent_pass": minimum_tangent >= spec["minimum_allowed_path_tangent_MPa"],
            "acoustic_pass": minimum_acoustic >= spec["minimum_allowed_projected_modulus_MPa"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "energy_architecture_stability.csv", index=False)
    return frame


def main():
    torch.set_default_dtype(torch.float64)
    config = load_config(CONFIG.read_text(encoding="utf-8"))
    output = Path(config["outputs"]["results_dir"]); output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(config["outputs"]["checkpoint_dir"]); checkpoint.mkdir(parents=True, exist_ok=True)
    parameters = MaterialParameters.from_json(config["teacher_parameter_file"])
    raw, relaxation, _ = load_real_data(config["real_training_cache"], parameters,
        points_per_decade=5, maximum_loading_points=14, device="cpu")
    tau_s = np.asarray(config["frozen"]["tau_s"], float)
    geq = equilibrium_shear_modulus(parameters.zhang_li)
    spectrum, weights = fit_positive_spectrum(raw, tau_s, geq, config, output)
    models, seed_results, q_predictions, targets = fit_energy_candidates(
        weights, tau_s, relaxation, parameters, config, output, checkpoint)
    stability = stability_screen(models, parameters.zhang_li, config, output)

    stage4 = pd.read_csv(ROOT / "results" / "stage4" / "final_relaxation_metrics.csv")
    stage4 = stage4[(stage4.model == "Stage4-GD-PICNN") &
                    (stage4.solver == "full_tensor_local_implicit")]
    stage4_q = rms(np.log(np.maximum(stage4.predicted_q.astype(float), 1e-14) /
                          np.maximum(stage4.experimental_q.astype(float), 1e-14)))
    best_rows = seed_results.sort_values(
        ["candidate", "q_ordering_accuracy", "q_log_RMSE", "objective"],
        ascending=[True, False, True, True]).groupby("candidate").first().reset_index()
    comparison = best_rows.merge(stability, on="candidate")
    comparison["q_improves_stage4"] = comparison.q_log_RMSE < stage4_q
    comparison["q_ordering_pass"] = comparison.q_ordering_accuracy >= 1.0-1e-12
    comparison["all_stability_pass"] = (comparison.reference_pass & comparison.energy_pass &
        comparison.path_tangent_pass & comparison.acoustic_pass & comparison.all_values_finite)
    comparison.to_csv(output / "energy_architecture_comparison.csv", index=False)
    feasible = comparison[comparison.q_improves_stage4 & comparison.q_ordering_pass & comparison.all_stability_pass]
    selected_name = None if feasible.empty else feasible.sort_values(["q_log_RMSE", "parameter_count"]).iloc[0].candidate
    acceptance = config["acceptance"]
    linear_pass = (spectrum["validation_metrics"]["balanced_NRMSE"] <= acceptance["linear_spectrum_validation_balanced_nrmse_maximum"] and
                   spectrum["validation_metrics"]["storage_relative_RMSE"] <= acceptance["linear_spectrum_storage_relative_rmse_maximum"] and
                   spectrum["validation_metrics"]["loss_relative_RMSE"] <= acceptance["linear_spectrum_loss_relative_rmse_maximum"])
    architecture_pass = selected_name is not None
    status = "passed" if linear_pass and architecture_pass else ("conditionally_passed" if architecture_pass else "failed")
    summary = {"schema_version": "2026-08-31-chapter3-stage5a-summary-v2",
        "status": status, "linear_spectrum_pass": bool(linear_pass),
        "energy_architecture_pass": bool(architecture_pass), "selected_energy_architecture": selected_name,
        "equilibrium_shear_modulus_MPa": geq, "stage4_q_log_RMSE": stage4_q,
        "linear_spectrum": spectrum,
        "energy_architecture_comparison": comparison.to_dict("records"),
        "data_provenance": {"real_training_cache": str(config["real_training_cache"]),
            "real_training_cache_sha256": sha256(config["real_training_cache"]),
            "frequency_amplitude_fraction": raw["frequency_dma_amplitude_fraction"],
            "amplitude_labels_loaded": raw["amplitude_labels_loaded"]},
        "boundary": "Static no-flow q is an architecture-capacity diagnostic, not the Stage-5B dynamic prediction."}
    (output / "stage5a_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")
    if selected_name:
        selected_model = models[selected_name]
        torch.save({"schema_version": "2026-08-31-stage5a-selected-energy-architecture-v2",
                    "candidate": selected_name, "state_dict": selected_model.state_dict(),
                    "physical_parameters": selected_model.serializable(),
                    "linear_spectrum_weights_MPa": weights.tolist(), "tau_s": tau_s.tolist(),
                    "status": status}, checkpoint / "stage5a_selected_energy_architecture.pt")
    print(json.dumps({"status": status, "linear_spectrum_pass": linear_pass,
        "selected_energy_architecture": selected_name,
        "linear_validation": spectrum["validation_metrics"],
        "stage4_q_log_RMSE": stage4_q,
        "candidate_q": comparison[["candidate","q_log_RMSE","q_ordering_accuracy",
            "q_ordering_pass","all_stability_pass"]].to_dict("records")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
