"""Stage-2 calibration of shared-shape one-term Ogden nonequilibrium energy.

Only Gneq and a_ogden are optimized.  The frozen Zhang--Li equilibrium network,
seven time scales, normalized old branch shares and existing T1/T2 flow
parameters are retained.  Complete-condition holdouts are never used in the
objective.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.stats import qmc, spearmanr


HERE = Path(__file__).resolve().parent
TEACHER = HERE.parent
ROOT = TEACHER.parent
T1_OLD = TEACHER / "3_PRF"
T2_OLD = TEACHER / "4_GroupedPRF"
T1_NEW = TEACHER / "5_T1_OgdenPRF"
T2_NEW = TEACHER / "6_T2_GroupedOgdenPRF"
for path in (HERE, T1_OLD / "code", T2_OLD / "code", T1_NEW / "code", T2_NEW / "code"):
    sys.path.insert(0, str(path))

from comparison_core import ZhangLi, equilibrium_stress, file_record, load_config, load_relaxation
from ogden_prf_core import normalized_shares
from t1_ogden.model import build_model as build_t1
from t2_ogden.model import build_model as build_t2
from teacher_model.model import ModelState
from teacher_model.solvers import simulate_history


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 8,
                     "axes.spines.right": False, "axes.spines.top": False,
                     "axes.linewidth": 0.8, "legend.frameon": False})

COLORS = {"experiment": "#272727", "old": "#767676", "new": "#B64342",
          "ideal": "#0F4D92", "real": "#B64342", "linear": "#9A4D8E"}


def write_json(path: Path, payload) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def save_figure(fig: plt.Figure, base: Path, rect=None) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.1, rect=rect)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def old_parameter_path(model_name: str) -> Path:
    base = T1_OLD if model_name == "T1" else T2_OLD
    return base / "results_paper_7branch" / "parameters" / f"{model_name}_full_data_teacher_parameters.json"


def build_model(model_name: str, payload: dict, Gneq: float, a_ogden: float,
                linear_only: bool = False):
    source = copy.deepcopy(payload)
    if linear_only:
        for flow in source["flows"].values():
            flow["nonlinear_multiplier"] = 0.0
    if model_name == "T1":
        return build_t1(source["zhang_li"], source["flows"]["shared"], source["tau_s"],
                        source["weights"], Gneq, a_ogden)
    return build_t2(source["zhang_li"], source["flows"]["fast"], source["flows"]["slow"],
                    source["tau_s"], source["weights"], Gneq, a_ogden, 1.0)


def curve_records(curves, zhang_li: ZhangLi, held_out: set[str]) -> list[dict]:
    rows = []
    for curve in curves:
        peak = int(np.argmax(curve.strain))
        equilibrium = float(equilibrium_stress(curve.mode, curve.strain[peak], zhang_li))
        peak_stress = float(curve.nominal_stress_MPa[peak])
        rows.append({"curve": curve, "mode": curve.mode, "condition": curve.label,
                     "key": f"{curve.mode}:{curve.label}",
                     "split": "held_out" if f"{curve.mode}:{curve.label}" in held_out else "training",
                     "strain": float(curve.strain[peak]), "peak_index": peak,
                     "equilibrium_MPa": equilibrium, "experimental_peak_MPa": peak_stress,
                     "experimental_q": (peak_stress - equilibrium) / equilibrium})
    return rows


def simulate_peak_q(model, record: dict, *, strict: bool = False) -> float:
    curve = record["curve"]
    peak = record["peak_index"]
    if strict:
        # Final verification replays every measured loading point.
        time_s = curve.time_s[:peak + 1]
        strain = curve.strain[:peak + 1]
        max_increment = 0.025
    else:
        # The measured loading ramps are piecewise linear.  During the scalar
        # shape search the same endpoints/rate are integrated with controlled
        # strain substeps, avoiding repeated Newton solves at redundant points.
        time_s = np.asarray([curve.time_s[0], curve.time_s[peak]], float)
        strain = np.asarray([curve.strain[0], curve.strain[peak]], float)
        max_increment = 0.05
    output = simulate_history(model, time_s, strain, curve.mode,
                              max_scalar_increment=max_increment)
    return float((output["nominal_stress_MPa"][-1] - record["equilibrium_MPa"]) /
                 record["equilibrium_MPa"])


def simulate_peak_q_fast(model, record: dict) -> float:
    """Second-order explicit exponential proxy used only for scalar shape search.

    The same constitutive rate and the measured loading rate are retained.  The
    exponential update preserves det(Fv)=1.  Selected parameters are always
    replayed with the production implicit solver before acceptance.
    """
    curve = record["curve"]
    peak = record["peak_index"]
    state = ModelState.identity(len(model.parameters.branches))
    for k in range(1, peak + 1):
        dt_total = float(curve.time_s[k] - curve.time_s[k - 1])
        x0 = float(curve.strain[k - 1]); x1 = float(curve.strain[k])
        subdivisions = max(1, int(np.ceil(dt_total / 0.025)),
                           int(np.ceil(abs(x1 - x0) / 0.01)))
        dt = dt_total / subdivisions
        for j in range(subdivisions):
            xa = x0 + (x1 - x0) * j / subdivisions
            xb = x0 + (x1 - x0) * (j + 1) / subdivisions
            xm = 0.5 * (xa + xb)
            Fm = model.deformation(xm, curve.mode)
            updated = []
            for Fv, branch in zip(state.Fv, model.parameters.branches):
                D0 = model._branch(Fm, Fv, branch)[0]
                Fvh = expm(0.5 * dt * D0) @ Fv
                Dm = model._branch(Fm, Fvh, branch)[0]
                updated.append(expm(dt * Dm) @ Fv)
            state = ModelState(updated)
    F = model.deformation(record["strain"], record["mode"])
    return float((model.measured_stress(F, state, record["mode"])
                  - record["equilibrium_MPa"]) / record["equilibrium_MPa"])


def frozen_q(model, record: dict) -> float:
    state = ModelState.identity(7)
    F = model.deformation(record["strain"], record["mode"])
    return float((model.measured_stress(F, state, record["mode"]) - record["equilibrium_MPa"]) /
                 record["equilibrium_MPa"])


def adjacent_trend_loss(rows: list[dict], q_values: np.ndarray, split="training") -> tuple[float, float]:
    trend, ordering = [], []
    for mode in ("uniaxial", "shear"):
        indices = [i for i, row in enumerate(rows) if row["mode"] == mode and row["split"] == split]
        indices = sorted(indices, key=lambda i: rows[i]["strain"])
        for left, right in zip(indices[:-1], indices[1:]):
            model_delta = np.log(q_values[right]) - np.log(q_values[left])
            target_delta = np.log(rows[right]["experimental_q"]) - np.log(rows[left]["experimental_q"])
            trend.append((model_delta - target_delta) ** 2)
            ordering.append(max(0.0, -model_delta) ** 2)
    return float(np.mean(trend)), float(np.mean(ordering))


class ShapeEvaluator:
    def __init__(self, model_name: str, payload: dict, rows: list[dict],
                 loading_retention: np.ndarray, G_bounds=(1e-4, 1.0)):
        self.model_name = model_name
        self.payload = payload
        self.rows = rows
        self.loading_retention = np.asarray(loading_retention, float)
        self.G_bounds = G_bounds
        self.cache: dict[float, dict] = {}

    def evaluate(self, a_ogden: float) -> dict:
        key = round(float(a_ogden), 10)
        if key in self.cache:
            return self.cache[key]
        started = time.time()
        result = {"model": self.model_name, "a_ogden": float(a_ogden), "success": False}
        try:
            unit_model = build_model(self.model_name, self.payload, 1.0, a_ogden)
            step_basis = np.asarray([frozen_q(unit_model, row) for row in self.rows])
            # Stage-1 strict replays provide the condition-wise fraction that
            # survives the measured loading ramp.  It is held fixed during the
            # cheap global shape screen; the selected candidate is then replayed
            # exactly below, so this proxy can never pass an acceptance gate.
            real_basis = step_basis * self.loading_retention
            if np.any(real_basis <= 0) or np.any(step_basis <= 0) or not np.all(np.isfinite(real_basis)):
                raise FloatingPointError("nonpositive/nonfinite unit q")
            train = np.asarray([row["split"] == "training" for row in self.rows])
            q_exp = np.asarray([row["experimental_q"] for row in self.rows])
            logG = float(np.mean(np.log(q_exp[train]) - np.log(real_basis[train])))
            Gneq = float(np.clip(np.exp(logG), *self.G_bounds))
            q_real = Gneq * real_basis
            q_step = Gneq * step_basis
            log_q_loss = float(np.mean((np.log(q_real[train]) - np.log(q_exp[train])) ** 2))
            trend_loss, ordering_loss = adjacent_trend_loss(self.rows, q_real)
            _, step_ordering = adjacent_trend_loss(self.rows, q_step)
            stress_scale = np.asarray([max(row["experimental_peak_MPa"], 0.05) for row in self.rows])
            predicted_peak = np.asarray([row["equilibrium_MPa"] for row in self.rows]) * (1.0 + q_real)
            observed_peak = np.asarray([row["experimental_peak_MPa"] for row in self.rows])
            stress_aux = float(np.mean(((predicted_peak[train] - observed_peak[train]) /
                                        stress_scale[train]) ** 2))
            objective = (log_q_loss + 0.75 * trend_loss + 1.5 * ordering_loss
                         + 0.25 * step_ordering + 0.05 * stress_aux)
            result.update({"success": True, "Gneq_MPa": Gneq, "objective": objective,
                           "log_q_loss": log_q_loss, "trend_loss": trend_loss,
                           "ordering_loss": ordering_loss, "step_ordering_loss": step_ordering,
                           "stress_auxiliary_loss": stress_aux, "real_q": q_real,
                           "step_q": q_step, "real_unit_q": real_basis,
                           "step_unit_q": step_basis})
        except Exception as error:
            result.update({"objective": 1e12, "error": repr(error)})
        result["elapsed_s"] = float(time.time() - started)
        self.cache[key] = result
        return result


def run_multistart(model_name: str, payload: dict, rows: list[dict], loading_retention,
                   seeds,
                   global_per_seed: int, a_bounds: tuple[float, float]) -> tuple[dict, list[dict], list[dict]]:
    evaluator = ShapeEvaluator(model_name, payload, rows, loading_retention)
    trials, run_best = [], []
    anchors = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.5, 7.5, 10.0, 12.0])
    for run_index, seed in enumerate(seeds):
        lhs = qmc.LatinHypercube(d=1, seed=seed).random(global_per_seed).ravel()
        candidates = a_bounds[0] + lhs * (a_bounds[1] - a_bounds[0])
        if run_index == 0:
            candidates = np.r_[candidates, anchors[(anchors >= a_bounds[0]) & (anchors <= a_bounds[1])]]
        seed_rows = []
        for a_value in candidates:
            row = dict(evaluator.evaluate(float(a_value)))
            row.update({"seed": int(seed), "stage": "global"})
            trials.append(row); seed_rows.append(row)
        best_global = min(seed_rows, key=lambda item: item["objective"])
        half_width = max(0.8, 0.18 * best_global["a_ogden"])
        lower = max(a_bounds[0], best_global["a_ogden"] - half_width)
        upper = min(a_bounds[1], best_global["a_ogden"] + half_width)
        local = minimize_scalar(lambda x: evaluator.evaluate(float(x))["objective"],
                                bounds=(lower, upper), method="bounded",
                                options={"xatol": 0.03, "maxiter": 12})
        local_row = dict(evaluator.evaluate(float(local.x)))
        local_row.update({"seed": int(seed), "stage": "local", "local_success": bool(local.success)})
        trials.append(local_row)
        best_seed = min([*seed_rows, local_row], key=lambda item: item["objective"])
        run_best.append(best_seed)
    selected = min(run_best, key=lambda item: item["objective"])
    return selected, trials, run_best


def trend_metrics(rows: list[dict], q_values: np.ndarray, label: str) -> list[dict]:
    result = []
    for mode in ("uniaxial", "shear"):
        indices = sorted([i for i, row in enumerate(rows) if row["mode"] == mode],
                         key=lambda i: rows[i]["strain"])
        x = np.asarray([rows[i]["strain"] for i in indices])
        q = q_values[indices]
        rho, p = spearmanr(x, q)
        result.append({"series": label, "mode": mode,
                       "slope": float(np.polyfit(x, q, 1)[0]),
                       "spearman_rho": float(rho), "spearman_p": float(p),
                       "strictly_increasing": bool(np.all(np.diff(q) > 0)),
                       "q_min": float(np.min(q)), "q_max": float(np.max(q))})
    return result


def constant_hold(model, hold_time: np.ndarray, strain: float, mode: str):
    state = ModelState.identity(7)
    F = model.deformation(strain, mode)
    stress = [model.measured_stress(F, state, mode)]
    for k in range(1, len(hold_time)):
        state, _ = model.advance(state, F, F, float(hold_time[k] - hold_time[k - 1]))
        stress.append(model.measured_stress(F, state, mode))
    return np.asarray(stress)


def final_diagnostics(model_name: str, payload: dict, selected: dict, rows: list[dict]):
    # Gneq cancels from the branch evolution because X, eta and drive_scale all
    # scale linearly with Gneq.  Therefore one strict unit-G replay supplies an
    # exact closed-form stiffness refinement and the final histories.
    unit_model = build_model(model_name, payload, 1.0, selected["a_ogden"])
    raw = []
    for row in rows:
        curve = row["curve"]; peak = row["peak_index"]
        hold_time = curve.time_s[peak:] - curve.time_s[peak]
        real_output = simulate_history(unit_model, curve.time_s, curve.strain, curve.mode,
                                       max_scalar_increment=0.025)
        raw.append((row, hold_time, real_output["nominal_stress_MPa"][peak:],
                    constant_hold(unit_model, hold_time, row["strain"], row["mode"])))
    train_unit_q = np.asarray([(real[0] - row["equilibrium_MPa"]) / row["equilibrium_MPa"]
                               for row, _time, real, _frozen in raw
                               if row["split"] == "training"])
    train_exp_q = np.asarray([row["experimental_q"] for row, _time, _real, _frozen in raw
                              if row["split"] == "training"])
    Gneq = float(np.clip(np.exp(np.mean(np.log(train_exp_q)-np.log(train_unit_q))),
                          1e-4, 1.0))
    selected["Gneq_MPa"] = Gneq
    metrics, histories = [], []
    for row, hold_time, real_unit, frozen_unit in raw:
        for scenario, unit_stress in (("real_ramp", real_unit),
                                      ("frozen_loading", frozen_unit)):
            stress = row["equilibrium_MPa"] + Gneq * (unit_stress-row["equilibrium_MPa"])
            peak_stress = float(stress[0]); noneq = peak_stress - row["equilibrium_MPa"]
            R = stress / peak_stress
            Rneq = (stress - row["equilibrium_MPa"]) / max(abs(noneq), 1e-14)
            metrics.append({"model": f"{model_name}-Ogden-1", "mode": row["mode"],
                            "condition": row["condition"], "split": row["split"],
                            "strain": row["strain"], "scenario": scenario,
                            "q": noneq / row["equilibrium_MPa"], "R_end": float(R[-1]),
                            "Rneq_end": float(Rneq[-1]), "peak_stress_MPa": peak_stress,
                            "equilibrium_MPa": row["equilibrium_MPa"],
                            "strict_unit_G_replay": True})
            for t, value, r_value, rn_value in zip(hold_time, stress, R, Rneq):
                histories.append({"model": f"{model_name}-Ogden-1", "mode": row["mode"],
                                  "condition": row["condition"], "split": row["split"],
                                  "scenario": scenario, "hold_time_s": float(t),
                                  "stress_MPa": float(value), "R": float(r_value),
                                  "Rneq": float(rn_value),
                                  "equilibrium_MPa": row["equilibrium_MPa"]})
    return pd.DataFrame(metrics), pd.DataFrame(histories), Gneq


def experimental_history(rows: list[dict]) -> pd.DataFrame:
    output = []
    for row in rows:
        curve = row["curve"]; peak = row["peak_index"]
        time_s = curve.time_s[peak:] - curve.time_s[peak]
        stress = curve.nominal_stress_MPa[peak:]
        R = stress / stress[0]
        Rneq = (stress - row["equilibrium_MPa"]) / (stress[0] - row["equilibrium_MPa"])
        for t, value, r_value, rn_value in zip(time_s, stress, R, Rneq):
            output.append({"model": "experiment", "mode": row["mode"],
                           "condition": row["condition"], "split": row["split"],
                           "scenario": "experimental", "hold_time_s": float(t),
                           "stress_MPa": float(value), "R": float(r_value),
                           "Rneq": float(rn_value), "equilibrium_MPa": row["equilibrium_MPa"]})
    return pd.DataFrame(output)


def condition_errors(rows: list[dict], metrics: pd.DataFrame, histories: pd.DataFrame,
                     experimental: pd.DataFrame) -> pd.DataFrame:
    output = []
    for row in rows:
        for model in ("T1-Ogden-1", "T2-Ogden-1"):
            pred_metric = metrics[(metrics["model"] == model) &
                                  (metrics["mode"] == row["mode"]) &
                                  (metrics["condition"].astype(str) == str(row["condition"])) &
                                  (metrics["scenario"] == "real_ramp")].iloc[0]
            pred = histories[(histories["model"] == model) &
                             (histories["mode"] == row["mode"]) &
                             (histories["condition"].astype(str) == str(row["condition"])) &
                             (histories["scenario"] == "real_ramp")].sort_values("hold_time_s")
            exp = experimental[(experimental["mode"] == row["mode"]) &
                               (experimental["condition"].astype(str) == str(row["condition"]))].sort_values("hold_time_s")
            output.append({"model": model, "mode": row["mode"], "condition": row["condition"],
                           "split": row["split"], "q_experimental": row["experimental_q"],
                           "q_model": pred_metric["q"],
                           "q_log_error": float(np.log(pred_metric["q"] / row["experimental_q"])),
                           "stress_RMSE_MPa": float(np.sqrt(np.mean(
                               (pred["stress_MPa"].to_numpy() - exp["stress_MPa"].to_numpy())**2))),
                           "R_RMSE": float(np.sqrt(np.mean((pred["R"].to_numpy()-exp["R"].to_numpy())**2))),
                           "Rneq_RMSE": float(np.sqrt(np.mean(
                               (pred["Rneq"].to_numpy()-exp["Rneq"].to_numpy())**2)))})
    return pd.DataFrame(output)


def plot_shape_search(search: pd.DataFrame, figures: Path,
                      strict_profile: pd.DataFrame | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, model in zip(axes, ("T1", "T2")):
        data = search[(search["model"] == model) & search["success"]].sort_values("a_ogden")
        ax.scatter(data["a_ogden"], data["objective"], s=14, alpha=0.55, color="#0F4D92")
        best = data.loc[data["objective"].idxmin()]
        ax.plot(best["a_ogden"], best["objective"], "*", ms=9, color="#0F4D92",
                label="Ideal-step screening")
        if strict_profile is not None:
            strict = strict_profile[strict_profile["model"] == model].sort_values("a_ogden")
            twin = ax.twinx()
            strict_line, = twin.plot(strict["a_ogden"], strict["objective"], "o-", ms=3.2,
                                     lw=1.2, color="#B64342",
                                     label="Measured-ramp profile")
            admissible = strict[(strict["a_ogden"] >= 1.0) &
                                strict["step_increasing_uniaxial"].astype(bool) &
                                strict["step_increasing_shear"].astype(bool)]
            best_strict = admissible.loc[admissible["objective"].idxmin()]
            twin.plot(best_strict["a_ogden"], best_strict["objective"], "*", ms=10,
                      color="#B64342")
            twin.set_ylabel("Ramp-profile objective", color="#B64342")
            twin.tick_params(axis="y", colors="#B64342")
        ax.set_xlabel(r"Ogden shape parameter $a_{\mathrm{O}}$")
        ax.set_ylabel("Stage 2 objective")
        ax.set_title(f"{model}-Ogden-1")
        ax.grid(alpha=0.18)
        handles, labels = ax.get_legend_handles_labels()
        if strict_profile is not None:
            handles.append(strict_line); labels.append(strict_line.get_label())
        ax.legend(handles, labels, fontsize=6.3, loc="upper center")
    save_figure(fig, figures / "ogden_shape_parameter_search")


def plot_q_comparison(rows, final_metrics: pd.DataFrame, old_metrics: pd.DataFrame,
                      figures: Path, scenario: str, stem: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex="col")
    for r, model in enumerate(("T1", "T2")):
        for c, mode in enumerate(("uniaxial", "shear")):
            ax = axes[r, c]
            selected_rows = sorted([row for row in rows if row["mode"] == mode], key=lambda x: x["strain"])
            x = np.asarray([row["strain"] for row in selected_rows])
            exp = np.asarray([row["experimental_q"] for row in selected_rows])
            new = final_metrics[(final_metrics["model"] == f"{model}-Ogden-1") &
                                (final_metrics["mode"] == mode) &
                                (final_metrics["scenario"] == scenario)].sort_values("strain")
            old = old_metrics[(old_metrics["model"] == model) &
                              (old_metrics["parameter_set"] == "full_data_teacher") &
                              (old_metrics["mode"] == mode) &
                              (old_metrics["scenario"] == scenario)].sort_values("strain_level")
            ax.plot(x, exp, "o-", color=COLORS["experiment"], lw=1.7, ms=4, label="Experiment")
            ax.plot(old["strain_level"], old["q"], "^--", color=COLORS["old"], lw=1.1,
                    ms=3.5, label=f"Baseline {model}")
            ax.plot(new["strain"], new["q"], "s-", color=COLORS["new"], lw=1.6,
                    ms=4, label=f"{model}-Ogden-1")
            ax.set_title(f"{model} · {'Uniaxial' if mode == 'uniaxial' else 'Simple shear'}")
            ax.set_xlabel("Engineering strain" if mode == "uniaxial" else "Shear strain")
            ax.set_ylabel(r"Normalized nonequilibrium peak $q$")
            ax.grid(alpha=0.18)
            ax.text(-0.15, 1.04, chr(ord("a")+2*r+c), transform=ax.transAxes,
                    fontweight="bold", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    save_figure(fig, figures / stem, rect=(0, 0, 1, 0.92))


def plot_ideal_real_gap(rows, metrics: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, model in zip(axes, ("T1", "T2")):
        real = metrics[(metrics["model"] == f"{model}-Ogden-1") &
                       (metrics["scenario"] == "real_ramp")]
        frozen = metrics[(metrics["model"] == f"{model}-Ogden-1") &
                         (metrics["scenario"] == "frozen_loading")]
        merged = real.merge(frozen, on=["model", "mode", "condition", "split", "strain"],
                            suffixes=("_real", "_frozen"))
        for mode, marker in (("uniaxial", "o"), ("shear", "s")):
            data = merged[merged["mode"] == mode].sort_values("strain")
            ax.plot(data["strain"], data["q_frozen"]-data["q_real"], marker+"-",
                    lw=1.4, ms=4, label="Uniaxial" if mode == "uniaxial" else "Simple shear")
        ax.set_xlabel("Strain level"); ax.set_ylabel(r"Peak reduction during loading $q_{step}-q_{real}$")
        ax.set_title(f"{model}-Ogden-1"); ax.grid(alpha=0.18); ax.legend()
    save_figure(fig, figures / "ogden_step_and_ramp_peaks")


def plot_rneq(histories: pd.DataFrame, experimental: pd.DataFrame, figures: Path) -> None:
    for model in ("T1-Ogden-1", "T2-Ogden-1"):
        for mode in ("uniaxial", "shear"):
            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            conditions = sorted(experimental[experimental["mode"] == mode]["condition"].unique(),
                                key=lambda x: float(str(x).replace("%", "")))
            palette = ["#0F4D92", "#42949E", "#E28E2C", "#B64342"]
            for color, condition in zip(palette, conditions):
                exp = experimental[(experimental["mode"] == mode) &
                                   (experimental["condition"].astype(str) == str(condition))]
                pred = histories[(histories["model"] == model) & (histories["mode"] == mode) &
                                 (histories["condition"].astype(str) == str(condition)) &
                                 (histories["scenario"] == "real_ramp")]
                ax.plot(exp["hold_time_s"], exp["Rneq"], "o", mfc="none", mew=0.7,
                        ms=2.6, markevery=max(1, len(exp)//24), color=color,
                        label=f"Experiment {condition}")
                ax.plot(pred["hold_time_s"], pred["Rneq"], "-", lw=1.4, color=color,
                        label=f"Model {condition}")
            ax.set_xlim(left=0); ax.set_xlabel("Hold time (s)")
            ax.set_ylabel(r"Normalized nonequilibrium relaxation $R_{neq}(t)$")
            ax.set_title(f"{model} · {'Uniaxial' if mode == 'uniaxial' else 'Simple shear'}")
            ax.grid(alpha=0.18); ax.legend(ncol=2, fontsize=6.5)
            save_figure(fig, figures / f"{model}_{mode}_Rneq")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-per-seed", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260827, 20260828, 20260829])
    parser.add_argument("--a-bounds", nargs=2, type=float, default=[0.75, 8.0])
    parser.add_argument("--results-root", type=Path,
                        default=TEACHER / "results_ogden_teacher_models" / "stage2_ogden_energy")
    parser.add_argument("--figures-root", type=Path,
                        default=TEACHER / "figures_ogden_teacher_models" / "stage2_ogden_energy")
    parser.add_argument("--strict-profile", type=Path,
                        default=TEACHER / "results_ogden_teacher_models" /
                        "stage2_ogden_energy" / "tables" / "stage2_strict_profile.csv")
    parser.add_argument("--profile-min-a", type=float, default=1.0)
    args = parser.parse_args()
    result_root = args.results_root.resolve(); figure_root = args.figures_root.resolve()
    tables = result_root / "tables"; parameters_dir = result_root / "parameters"
    tables.mkdir(parents=True, exist_ok=True); parameters_dir.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    config, config_path = load_config(HERE / "model_comparison_config.json")
    split_path = HERE / "model_comparison_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    curves = load_relaxation(config["data"]["viscoelastic"])
    rows = curve_records(curves, ZhangLi(**config["zhang_li"]),
                         set(split["relaxation"]["held_out"]))
    old_metrics = pd.read_csv(TEACHER / "results_ogden_teacher_models" /
                              "stage1_existing_model_diagnosis" / "tables" /
                              "T1_T2_q_R_Rneq_Rend_metrics.csv")

    selections, all_trials, run_bests = {}, [], []
    payloads = {}
    started = time.time()
    for model_name in ("T1", "T2"):
        payload = json.loads(old_parameter_path(model_name).read_text(encoding="utf-8"))
        payloads[model_name] = payload
        retention = []
        for row in rows:
            subset = old_metrics[(old_metrics["model"] == model_name) &
                                 (old_metrics["parameter_set"] == "full_data_teacher") &
                                 (old_metrics["mode"] == row["mode"]) &
                                 (old_metrics["condition"].astype(str) == str(row["condition"]))]
            q_real = float(subset[subset["scenario"] == "real_ramp"]["q"].iloc[0])
            q_step = float(subset[subset["scenario"] == "frozen_loading"]["q"].iloc[0])
            retention.append(q_real / q_step)
        selected, trials, best_runs = run_multistart(
            model_name, payload, rows, retention, args.seeds, args.global_per_seed,
            (float(args.a_bounds[0]), float(args.a_bounds[1])))
        selections[model_name] = selected
        all_trials.extend(trials); run_bests.extend(best_runs)
        print(model_name, "selected", selected["Gneq_MPa"], selected["a_ogden"],
              "objective", selected["objective"], flush=True)

    # The ideal-step-informed global screen only proposes a feasible shape
    # range.  When a strict profile is available, the final stage-2 candidate
    # is selected exclusively from production-integrator real-ramp replays.
    strict_profile = None
    if args.strict_profile.exists():
        strict_profile = pd.read_csv(args.strict_profile)
        for model_name in ("T1", "T2"):
            admissible = strict_profile[(strict_profile["model"] == model_name) &
                                        (strict_profile["a_ogden"] >= args.profile_min_a) &
                                        (strict_profile["step_increasing_uniaxial"].astype(bool)) &
                                        (strict_profile["step_increasing_shear"].astype(bool))]
            best = admissible.loc[admissible["objective"].idxmin()]
            selections[model_name]["ideal_screen_a_ogden"] = selections[model_name]["a_ogden"]
            selections[model_name]["ideal_screen_objective"] = selections[model_name]["objective"]
            selections[model_name]["a_ogden"] = float(best["a_ogden"])
            selections[model_name]["Gneq_MPa"] = float(best["Gneq_MPa"])
            selections[model_name]["strict_profile_objective"] = float(best["objective"])
            selections[model_name]["strict_profile_source"] = str(args.strict_profile.resolve())
            selections[model_name]["selection_constraint"] = (
                "ideal-step q strictly increasing in both uniaxial and shear")

    # Compact search tables; array-valued predictions are stored separately.
    scalar_keys = ["model", "seed", "stage", "success", "a_ogden", "Gneq_MPa",
                   "objective", "log_q_loss", "trend_loss", "ordering_loss",
                   "step_ordering_loss", "stress_auxiliary_loss", "elapsed_s", "error"]
    search_table = pd.DataFrame([{key: row.get(key) for key in scalar_keys} for row in all_trials])
    search_table.to_csv(tables / "stage2_search_trials.csv", index=False, encoding="utf-8-sig")
    best_table = pd.DataFrame([{key: row.get(key) for key in scalar_keys} for row in run_bests])
    best_table.to_csv(tables / "stage2_independent_run_best.csv", index=False, encoding="utf-8-sig")

    # Full histories are diagnostic only; no flow parameters are changed here.
    metric_frames, history_frames = [], []
    for model_name in ("T1", "T2"):
        search_G = selections[model_name]["Gneq_MPa"]
        metrics, histories, exact_G = final_diagnostics(
            model_name, payloads[model_name], selections[model_name], rows)
        selections[model_name]["search_Gneq_MPa"] = search_G
        selections[model_name]["Gneq_MPa"] = exact_G
        metric_frames.append(metrics); history_frames.append(histories)
    final_metrics = pd.concat(metric_frames, ignore_index=True)
    final_histories = pd.concat(history_frames, ignore_index=True)
    experimental = experimental_history(rows)
    errors = condition_errors(rows, final_metrics, final_histories, experimental)

    # Acceptance gates always use the strict replay, never the accelerated
    # search approximation.
    peak_rows, trend_rows = [], []
    exp_q = np.asarray([row["experimental_q"] for row in rows])
    trend_rows.extend(trend_metrics(rows, exp_q, "experiment"))
    for model_name, selected in selections.items():
        exact_real, exact_step = [], []
        for i, row in enumerate(rows):
            subset = final_metrics[(final_metrics["model"] == f"{model_name}-Ogden-1") &
                                   (final_metrics["mode"] == row["mode"]) &
                                   (final_metrics["condition"].astype(str) == str(row["condition"]))]
            real_q = float(subset[subset["scenario"] == "real_ramp"]["q"].iloc[0])
            step_q = float(subset[subset["scenario"] == "frozen_loading"]["q"].iloc[0])
            exact_real.append(real_q); exact_step.append(step_q)
            peak_rows.append({"model": f"{model_name}-Ogden-1", "mode": row["mode"],
                              "condition": row["condition"], "split": row["split"],
                              "strain": row["strain"], "experimental_q": row["experimental_q"],
                              "search_real_ramp_q": selected["real_q"][i],
                              "real_ramp_q": real_q, "ideal_step_q": step_q})
        exact_real = np.asarray(exact_real); exact_step = np.asarray(exact_step)
        train = np.asarray([row["split"] == "training" for row in rows])
        q_exp = np.asarray([row["experimental_q"] for row in rows])
        log_q_loss = float(np.mean((np.log(exact_real[train])-np.log(q_exp[train]))**2))
        trend_loss, ordering_loss = adjacent_trend_loss(rows, exact_real)
        _, step_ordering = adjacent_trend_loss(rows, exact_step)
        eq = np.asarray([row["equilibrium_MPa"] for row in rows])
        obs = np.asarray([row["experimental_peak_MPa"] for row in rows])
        scale = np.maximum(obs, 0.05)
        stress_aux = float(np.mean(((eq[train]*(1+exact_real[train])-obs[train]) /
                                    scale[train])**2))
        selected["strict_objective"] = (log_q_loss + 0.75*trend_loss
                                         + 1.5*ordering_loss + 0.25*step_ordering
                                         + 0.05*stress_aux)
        selected["strict_log_q_loss"] = log_q_loss
        selected["strict_trend_loss"] = trend_loss
        selected["strict_ordering_loss"] = ordering_loss
        selected["strict_step_ordering_loss"] = step_ordering
        selected["search_to_strict_max_abs_log_q_difference"] = float(
            np.max(np.abs(np.log(exact_real)-np.log(np.asarray(selected["real_q"])))))
        trend_rows.extend(trend_metrics(rows, exact_real,
                                        f"{model_name}-Ogden-1 real"))
        trend_rows.extend(trend_metrics(rows, exact_step,
                                        f"{model_name}-Ogden-1 step"))
    pd.DataFrame(peak_rows).to_csv(tables / "stage2_selected_peak_predictions.csv", index=False,
                                   encoding="utf-8-sig")
    trend_table = pd.DataFrame(trend_rows)
    trend_table.to_csv(tables / "stage2_q_trend_gates.csv", index=False, encoding="utf-8-sig")
    final_metrics.to_csv(tables / "stage2_final_scenario_metrics.csv", index=False,
                         encoding="utf-8-sig")
    final_histories.to_csv(tables / "stage2_final_relaxation_histories.csv", index=False,
                           encoding="utf-8-sig")
    experimental.to_csv(tables / "stage2_experimental_histories.csv", index=False,
                        encoding="utf-8-sig")
    errors.to_csv(tables / "stage2_condition_errors.csv", index=False, encoding="utf-8-sig")

    # Write independent parameter files into both new model directories and the shared result bundle.
    for model_name in ("T1", "T2"):
        payload = payloads[model_name]; selected = selections[model_name]
        new_payload = {
            "schema_version": "2026-08-27-stage2-ogden-energy-v1",
            "model": f"{model_name}-Ogden-1",
            "stage": "energy_only_old_flow_and_fixed_branch_shares",
            "zhang_li": payload["zhang_li"], "tau_s": payload["tau_s"],
            "old_weights": payload["weights"],
            "p_alpha": normalized_shares(payload["weights"]),
            "Gneq_MPa": selected["Gneq_MPa"], "a_ogden": selected["a_ogden"],
            "flows": payload["flows"], "transition_tau_s": 1.0,
            "objective": selected.get("strict_objective", selected["objective"]),
            "ideal_screen_objective": selected.get("ideal_screen_objective"),
            "strict_profile_objective": selected.get("strict_profile_objective"),
            "seeds": args.seeds,
            "held_out_conditions": sorted(split["relaxation"]["held_out"]),
            "flow_refitted": False, "branch_shares_refitted": False,
        }
        filename = f"{model_name}_Ogden_1_stage2_parameters.json"
        write_json(parameters_dir / filename, new_payload)
        destination = T1_NEW if model_name == "T1" else T2_NEW
        write_json(destination / "results" / "parameters" / filename, new_payload)

    plot_shape_search(search_table, figure_root, strict_profile)
    plot_q_comparison(rows, final_metrics, old_metrics, figure_root,
                      "frozen_loading", "step_peak_baseline_and_ogden")
    plot_q_comparison(rows, final_metrics, old_metrics, figure_root,
                      "real_ramp", "ramp_peak_baseline_and_ogden")
    plot_ideal_real_gap(rows, final_metrics, figure_root)
    plot_rneq(final_histories, experimental, figure_root)

    gate = {}
    for model_name in ("T1", "T2"):
        step = trend_table[trend_table["series"] == f"{model_name}-Ogden-1 step"]
        real = trend_table[trend_table["series"] == f"{model_name}-Ogden-1 real"]
        held = errors[(errors["model"] == f"{model_name}-Ogden-1") &
                      (errors["split"] == "held_out")]
        gate[model_name] = {
            "ideal_step_both_modes_increasing": bool(step["strictly_increasing"].all()),
            "real_ramp_both_modes_increasing": bool(real["strictly_increasing"].all()),
            "held_out_max_abs_log_q_error": float(held["q_log_error"].abs().max()),
            "parameter_not_at_boundary": bool(
                args.profile_min_a + 0.05 < selections[model_name]["a_ogden"] < args.a_bounds[1] - 0.05),
            "advance_to_flow_refit": bool(step["strictly_increasing"].all()),
        }

    source_files = [Path(config["data"]["viscoelastic"]), config_path, split_path,
                    old_parameter_path("T1"), old_parameter_path("T2"),
                    HERE / "ogden_prf_core.py", Path(__file__),
                    T1_NEW / "code" / "t1_ogden" / "model.py",
                    T2_NEW / "code" / "t2_ogden" / "model.py"]
    summary = {"stage": "stage2_ogden_energy_only", "elapsed_s": time.time()-started,
               "selected": {key: {field: value for field, value in selection.items()
                                    if field not in ("real_q", "step_q", "real_unit_q", "step_unit_q")}
                            for key, selection in selections.items()},
               "gate": gate,
               "aggregate_errors": errors.groupby(["model", "split"])[
                   ["q_log_error", "stress_RMSE_MPa", "R_RMSE", "Rneq_RMSE"]
               ].apply(lambda frame: pd.Series({
                   "q_log_RMSE": float(np.sqrt(np.mean(frame["q_log_error"]**2))),
                   "stress_RMSE_MPa": float(np.sqrt(np.mean(frame["stress_RMSE_MPa"]**2))),
                   "R_RMSE": float(np.sqrt(np.mean(frame["R_RMSE"]**2))),
                   "Rneq_RMSE": float(np.sqrt(np.mean(frame["Rneq_RMSE"]**2)))
               })).reset_index().to_dict(orient="records")}
    write_json(result_root / "stage2_summary.json", summary)
    manifest = {"stage": summary["stage"],
                "figure_contract": {"core_conclusion": "Test whether one shared Ogden shape corrects q before flow refitting.",
                                    "archetype": "quantitative grid", "backend": "Python/matplotlib only",
                                    "hero": "experimental versus old/new q trends",
                                    "controls": "ideal-real gap, multistart stability and Rneq histories"},
                "inputs": [file_record(path) for path in source_files],
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "numpy": np.__version__, "pandas": pd.__version__,
                                "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
                "outputs": {"tables": sorted(str(path) for path in tables.glob("*.csv")),
                            "parameters": sorted(str(path) for path in parameters_dir.glob("*.json")),
                            "figures": sorted(str(path) for path in figure_root.glob("*.*"))}}
    write_json(result_root / "stage2_reproducibility_manifest.json", manifest)
    print(json.dumps({"selected": summary["selected"], "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
