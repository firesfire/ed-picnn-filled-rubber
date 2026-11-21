"""Frozen data, metric and Pareto definitions shared by T0, T1 and T2.

The module deliberately contains no model-specific optimization.  Its purpose is
to make changes in constitutive structure, rather than changes in bookkeeping,
the only source of differences between the three paper baselines.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DECADE_EDGES = np.array([0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, np.inf])


@dataclass(frozen=True)
class ZhangLi:
    D: float
    mu: float
    B: float
    N: float
    beta: float


@dataclass
class RelaxationCurve:
    mode: str
    label: str
    time_s: np.ndarray
    strain: np.ndarray
    nominal_stress_MPa: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.mode}:{self.label}"


@dataclass
class DMAData:
    amplitude_percent: np.ndarray
    amplitude_frequency_hz: np.ndarray
    amplitude_storage_MPa: np.ndarray
    amplitude_loss_MPa: np.ndarray
    frequency_hz: np.ndarray
    frequency_storage_MPa: np.ndarray
    frequency_loss_MPa: np.ndarray


@dataclass
class CurveTarget:
    mode: str
    label: str
    split: str
    equilibrium_MPa: float
    hold_time_s: np.ndarray
    experimental_stress_MPa: np.ndarray
    experimental_peak_ratio: float
    experimental_shape: np.ndarray
    history_time_s: np.ndarray
    history_strain: np.ndarray
    peak_history_index: int

    @property
    def key(self) -> str:
        return f"{self.mode}:{self.label}"


def load_config(path: str | Path) -> tuple[dict, Path]:
    """Resolve data files, optionally overriding them with VISCOELASTIC_WORKBOOK or HYPERELASTIC_WORKBOOK."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    for key in ("viscoelastic", "hyperelastic"):
        override = os.environ.get(key.upper() + "_WORKBOOK")
        data_path = Path(override) if override else base / config["data"][key]
        config["data"][key] = str(data_path.resolve())
    return config, path


def file_record(path: str | Path) -> dict:
    path = Path(path).resolve()
    raw = path.read_bytes()
    stat = path.stat()
    return {
        "absolute_path": str(path),
        "sha256": sha256(raw).hexdigest(),
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
    }


def object_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def _numeric_triplet(frame: pd.DataFrame, col: int) -> np.ndarray:
    return np.asarray(
        frame.iloc[2:, [col, col + 1, col + 2]]
        .apply(pd.to_numeric, errors="coerce").dropna(), dtype=float)


def load_relaxation(path: str | Path) -> list[RelaxationCurve]:
    result: list[RelaxationCurve] = []
    definitions = [
        ("Uniaxial_SR", "uniaxial", ["25%", "50%", "75%", "100%"]),
        ("Simpleshear_SR", "shear", ["0.20", "0.40", "0.60", "0.80"]),
    ]
    for sheet, mode, labels in definitions:
        frame = pd.read_excel(path, sheet_name=sheet, header=None)
        for index, label in enumerate(labels):
            values = _numeric_triplet(frame, 4 * index)
            if len(values) < 3 or np.any(np.diff(values[:, 0]) <= 0):
                raise ValueError(f"invalid time history: {sheet}/{label}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"nonfinite value: {sheet}/{label}")
            result.append(RelaxationCurve(mode, label, values[:, 0],
                                          values[:, 1], values[:, 2]))
    return result


def load_dma(path: str | Path) -> DMAData:
    amp = np.asarray(pd.read_excel(path, sheet_name="Amplitude_DMA", header=None)
                     .iloc[3:, :].apply(pd.to_numeric, errors="coerce").dropna(), float)
    freq = np.asarray(pd.read_excel(path, sheet_name="Frequency_DMA", header=None)
                      .iloc[3:, :].apply(pd.to_numeric, errors="coerce").dropna(), float)
    return DMAData(amp[:, 6], amp[:, 3], amp[:, 8], amp[:, 7],
                   freq[:, 3], freq[:, 4], freq[:, 5])


def load_hyperelastic(path: str | Path) -> dict[str, np.ndarray]:
    frame = pd.read_excel(path, sheet_name="Sheet1", header=None)
    result = {}
    for name, col in (("uniaxial", 0), ("equibiaxial", 2), ("planar", 4)):
        result[name] = np.asarray(
            frame.iloc[2:, [col, col + 1]].apply(pd.to_numeric, errors="coerce").dropna(),
            float)
    return result


def invariants(C: np.ndarray) -> tuple[float, float]:
    i1 = float(np.trace(C))
    return i1, 0.5 * (i1 * i1 - float(np.trace(C @ C)))


def derivatives_from_invariants(i1, i2, p: ZhangLi):
    w1 = p.D * np.exp(-p.mu * (np.asarray(i1) - 3.0))
    w2 = p.B * p.beta * np.power(np.asarray(i2) - 3.0 + p.N, p.beta - 1.0)
    return w1, w2


def equilibrium_stress(mode: str, strain, p: ZhangLi):
    x = np.asarray(strain, float)
    if mode == "uniaxial":
        lam = 1.0 + x
        i1, i2 = lam * lam + 2.0 / lam, 2.0 * lam + lam ** -2
        w1, w2 = derivatives_from_invariants(i1, i2, p)
        return 2.0 * w1 * (lam - lam ** -2) + 2.0 * w2 * (1.0 - lam ** -3)
    if mode == "shear":
        inv = 3.0 + x * x
        w1, w2 = derivatives_from_invariants(inv, inv, p)
        return 2.0 * x * (w1 + w2)
    raise ValueError(mode)


def nominal_hyperelastic(stretch, mode: str, p: ZhangLi):
    lam = np.asarray(stretch, float)
    if mode == "uniaxial":
        return equilibrium_stress(mode, lam - 1.0, p)
    if mode == "planar":
        inv = lam * lam + 1.0 + lam ** -2
        w1, w2 = derivatives_from_invariants(inv, inv, p)
        return 2.0 * (lam - lam ** -3) * (w1 + w2)
    if mode == "equibiaxial":
        i1, i2 = 2.0 * lam * lam + lam ** -4, lam ** 4 + 2.0 * lam ** -2
        w1, w2 = derivatives_from_invariants(i1, i2, p)
        return 2.0 * (lam - lam ** -5) * (w1 + lam * lam * w2)
    raise ValueError(mode)


def equilibrium_shear_modulus(p: ZhangLi) -> float:
    return float(2.0 * (p.D + p.B * p.beta * p.N ** (p.beta - 1.0)))


def qlv_piecewise_linear_basis(time_s, equilibrium_history, tau_s):
    """QLV convolution basis for a piecewise-linear equilibrium stress history.

    Loading is retained.  This is not a step-strain approximation.
    """
    time_s = np.asarray(time_s, float)
    equilibrium_history = np.asarray(equilibrium_history, float)
    tau_s = np.asarray(tau_s, float)
    basis = np.zeros((len(time_s), len(tau_s)))
    for row in range(1, len(time_s)):
        dt = time_s[row] - time_s[row - 1]
        decay = np.exp(-dt / tau_s)
        ramp = tau_s / dt * (1.0 - decay)
        basis[row] = decay * basis[row - 1] + ramp * (
            equilibrium_history[row] - equilibrium_history[row - 1])
    return basis


def dma_design(frequency_hz, tau_s):
    z = 2.0 * np.pi * np.asarray(frequency_hz, float)[:, None] * np.asarray(tau_s, float)[None, :]
    return z * z / (1.0 + z * z), z / (1.0 + z * z)


def _hold_landmarks(duration_s: float, points_per_decade: int) -> np.ndarray:
    count = max(2, int(np.ceil((np.log10(max(duration_s, 1e-2)) + 2.0)
                               * points_per_decade)) + 1)
    positive = np.logspace(-2, np.log10(max(duration_s, 1e-2)), count)
    return np.unique(np.r_[0.0, positive[positive <= duration_s], duration_s])


def build_targets(curves: Iterable[RelaxationCurve], p: ZhangLi, split: dict,
                  points_per_decade: int) -> list[CurveTarget]:
    training = set(split["relaxation"]["training"])
    held = set(split["relaxation"]["held_out"])
    targets = []
    for curve in curves:
        if curve.key not in training | held:
            raise ValueError(f"condition absent from frozen split: {curve.key}")
        peak = int(np.argmax(curve.strain))
        peak_time = float(curve.time_s[peak])
        equilibrium = float(equilibrium_stress(curve.mode, curve.strain[peak], p))
        raw_time = curve.time_s[peak:] - peak_time
        raw_stress = curve.nominal_stress_MPa[peak:]
        hold_time = _hold_landmarks(float(raw_time[-1]), points_per_decade)
        hold_stress = np.interp(hold_time, raw_time, raw_stress)
        denominator = float(hold_stress[0] - equilibrium)
        if denominator <= 0:
            raise ValueError(f"nonpositive nonequilibrium peak: {curve.key}")
        history_time = np.r_[curve.time_s[:peak + 1], peak_time + hold_time[1:]]
        history_strain = np.r_[curve.strain[:peak + 1],
                               np.full(len(hold_time) - 1, curve.strain[peak])]
        targets.append(CurveTarget(
            curve.mode, curve.label, "training" if curve.key in training else "held_out",
            equilibrium, hold_time, hold_stress, denominator / equilibrium,
            (hold_stress - equilibrium) / denominator,
            history_time, history_strain, peak,
        ))
    return targets


def prediction_metrics(targets, bases, weights, splits=("training", "held_out")):
    rows = []
    weights = np.asarray(weights, float)
    for target, basis in zip(targets, bases):
        if target.split not in splits:
            continue
        pred = target.equilibrium_MPa + np.asarray(basis) @ weights
        pred_peak = pred[0] - target.equilibrium_MPa
        peak_ratio = pred_peak / target.equilibrium_MPa
        shape = (pred - target.equilibrium_MPa) / max(abs(pred_peak), 1e-14)
        error = shape - target.experimental_shape
        decade_values = []
        for left, right in zip(DECADE_EDGES[:-1], DECADE_EDGES[1:]):
            mask = (target.hold_time_s >= left) & (target.hold_time_s < right)
            if np.any(mask):
                decade_values.append(float(np.sqrt(np.mean(error[mask] ** 2))))
        rows.append({
            "mode": target.mode,
            "condition": target.label,
            "split": target.split,
            "experimental_peak_ratio": target.experimental_peak_ratio,
            "model_peak_ratio": float(peak_ratio),
            "normalized_peak_error": float(peak_ratio - target.experimental_peak_ratio),
            "normalized_curve_rmse": float(np.sqrt(np.mean(error ** 2))),
            "decade_equal_rmse": float(np.mean(decade_values)),
            "stress_rmse_MPa": float(np.sqrt(np.mean((pred - target.experimental_stress_MPa) ** 2))),
        })
    return rows


def aggregate_time_metrics(rows, split: str):
    selected = [row for row in rows if row["split"] == split]
    def rms(key):
        return float(np.sqrt(np.mean([row[key] ** 2 for row in selected])))
    return {
        "conditions": len(selected),
        "peak_rmse": rms("normalized_peak_error"),
        "curve_rmse": rms("normalized_curve_rmse"),
        "decade_equal_rmse": rms("decade_equal_rmse"),
        "stress_rmse_MPa": rms("stress_rmse_MPa"),
    }


def time_score(metrics: dict, shares=(0.4, 0.3, 0.3)) -> float:
    return float(np.sqrt(shares[0] * metrics["peak_rmse"] ** 2
                         + shares[1] * metrics["curve_rmse"] ** 2
                         + shares[2] * metrics["decade_equal_rmse"] ** 2))


def dma_metrics(storage_exp, loss_exp, storage_pred, loss_pred):
    storage_exp, loss_exp = np.asarray(storage_exp), np.asarray(loss_exp)
    storage_pred, loss_pred = np.asarray(storage_pred), np.asarray(loss_pred)
    gp_scale = max(float(np.ptp(storage_exp)), 0.05)
    gpp_scale = max(float(np.ptp(loss_exp)), 0.02)
    gp_rmse = float(np.sqrt(np.mean((storage_pred - storage_exp) ** 2)))
    gpp_rmse = float(np.sqrt(np.mean((loss_pred - loss_exp) ** 2)))
    return {
        "storage_rmse_MPa": gp_rmse,
        "loss_rmse_MPa": gpp_rmse,
        "storage_nrmse": gp_rmse / gp_scale,
        "loss_nrmse": gpp_rmse / gpp_scale,
        "balanced_nrmse": float(np.sqrt(0.5 * ((gp_rmse / gp_scale) ** 2
                                                 + (gpp_rmse / gpp_scale) ** 2))),
    }


def frequency_masks(frequency_hz, band=(5.0, 20.0)):
    frequency_hz = np.asarray(frequency_hz, float)
    held = (frequency_hz >= band[0]) & (frequency_hz <= band[1])
    return ~held, held


def nondominated(rows: list[dict], time_key="time_score", frequency_key="frequency_score"):
    result = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            if (other[time_key] <= row[time_key] and other[frequency_key] <= row[frequency_key]
                    and (other[time_key] < row[time_key] or other[frequency_key] < row[frequency_key])):
                dominated = True
                break
        if not dominated:
            result.append(row)
    return sorted(result, key=lambda item: item[time_key])


def representative_from_pareto(rows: list[dict], time_key="time_score",
                               frequency_key="frequency_score") -> dict:
    front = nondominated(rows, time_key, frequency_key)
    t = np.array([row[time_key] for row in front])
    f = np.array([row[frequency_key] for row in front])
    tn = (t - t.min()) / max(float(t.max() - t.min()), 1e-14)
    fn = (f - f.min()) / max(float(f.max() - f.min()), 1e-14)
    return front[int(np.argmin(np.hypot(tn, fn)))]


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return clean_json(asdict(value))
    return value


def write_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2), encoding="utf-8")

