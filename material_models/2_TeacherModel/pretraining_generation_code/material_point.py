"""Full-state three-dimensional material-point simulation for audited teachers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from path_library import PathSpec, classify_point


def _activate_model_packages(project: Path) -> None:
    for directory in (
        project / "common",
        project / "3_PRF" / "code",
        project / "4_GroupedPRF" / "code",
    ):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)


def load_teacher(project: Path, selection: dict):
    """Instantiate the frozen active T2--Ogden teacher."""
    _activate_model_packages(Path(project))
    from ogden_prf_core import OgdenPRFModel, build_t2_ogden_parameters

    payload = json.loads(Path(selection["selected_parameter_path"]).read_text(encoding="utf-8"))
    if selection["selected_model"] == "T2":
        parameters = build_t2_ogden_parameters(
            payload["zhang_li"], payload["flows"]["fast"], payload["flows"]["slow"],
            payload["tau_s"], payload["p_alpha"], payload["Gneq_MPa"],
            payload["a_ogden"], payload.get("transition_tau_s", 1.0),
        )
        return OgdenPRFModel(parameters), payload
    raise ValueError(f"selected teacher {selection['selected_model']} has no full-state tensor solver")


def deformation(mode: str, primary: float, secondary: float = 0.0) -> np.ndarray:
    """Isochoric deformation families; primary strains are engineering strains."""
    value = float(primary)
    if mode == "uniaxial":
        stretch = 1.0 + value
        if stretch <= 0.0:
            raise ValueError("nonpositive uniaxial stretch")
        return np.diag([stretch, stretch ** -0.5, stretch ** -0.5])
    if mode == "shear":
        return np.array([[1.0, value, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    if mode == "planar":
        stretch = 1.0 + value
        if stretch <= 0.0:
            raise ValueError("nonpositive planar stretch")
        return np.diag([stretch, 1.0, stretch ** -1.0])
    if mode == "equibiaxial":
        stretch = 1.0 + value
        if stretch <= 0.0:
            raise ValueError("nonpositive biaxial stretch")
        return np.diag([stretch, stretch, stretch ** -2.0])
    if mode == "mixed":
        return deformation("shear", secondary) @ deformation("uniaxial", value)
    raise ValueError(f"unsupported deformation mode: {mode}")


def _project_stress(stress: np.ndarray, mode: str) -> np.ndarray:
    pressure = (
        0.5 * float(stress[1, 1] + stress[2, 2])
        if mode == "uniaxial" else float(stress[2, 2])
    )
    return stress - pressure * np.eye(3)


def measured_stress(model, state, mode: str, primary: float, secondary: float = 0.0) -> float:
    F = deformation(mode, primary, secondary)
    cauchy = _project_stress(model.response(F, state)["cauchy_unpressurized"], mode)
    nominal = cauchy @ np.linalg.inv(F).T
    return float(cauchy[0, 1] if mode == "shear" else nominal[0, 0])


def advance_parts(
    model, state, mode: str, primary0: float, secondary0: float,
    primary1: float, secondary1: float, duration: float, parts: int = 1,
):
    if duration <= 0.0:
        return state.copy(), []
    count = max(int(parts), int(np.ceil(max(
        abs(float(primary1) - float(primary0)),
        abs(float(secondary1) - float(secondary0)),
    ) / 0.025)))
    count = max(count, 1)
    current = state.copy()
    diagnostics = []

    def advance_safely(current_state, left_primary, left_secondary, right_primary, right_secondary, step_dt, depth=0):
        left_F = deformation(mode, left_primary, left_secondary)
        right_F = deformation(mode, right_primary, right_secondary)
        try:
            with np.errstate(over="raise", invalid="raise"):
                return model.advance(current_state, left_F, right_F, step_dt)
        except (FloatingPointError, OverflowError, RuntimeError, np.linalg.LinAlgError, ValueError):
            if depth >= 8:
                raise
            middle_primary = 0.5 * (left_primary + right_primary)
            middle_secondary = 0.5 * (left_secondary + right_secondary)
            halfway, first_diagnostics = advance_safely(
                current_state, left_primary, left_secondary,
                middle_primary, middle_secondary, 0.5 * step_dt, depth + 1,
            )
            final, second_diagnostics = advance_safely(
                halfway, middle_primary, middle_secondary,
                right_primary, right_secondary, 0.5 * step_dt, depth + 1,
            )
            return final, first_diagnostics + second_diagnostics

    for index in range(count):
        left, right = index / count, (index + 1) / count
        current, step = advance_safely(
            current,
            primary0 + left * (primary1 - primary0),
            secondary0 + left * (secondary1 - secondary0),
            primary0 + right * (primary1 - primary0),
            secondary0 + right * (secondary1 - secondary0),
            float(duration) / count,
        )
        diagnostics.extend(step)
    return current, diagnostics


def _diagnostic_summary(diagnostics: list[dict]) -> dict:
    if not diagnostics:
        return {"newton_iterations": 0, "newton_residual": 0.0, "subdivisions": 0}
    return {
        "newton_iterations": int(sum(item.get("newton_iterations", 0) for item in diagnostics)),
        "newton_residual": float(max(item.get("residual", 0.0) for item in diagnostics)),
        "subdivisions": int(max(item.get("subdivisions", 0) for item in diagnostics)),
    }


def state_snapshot(model, state, spec: PathSpec, primary: float, secondary: float, diagnostics: list[dict]) -> dict:
    from ogden_prf_core import ogden_mandel_deviator, ogden_unit_energy
    from teacher_model.zhang_li import energy, unpressurized_cauchy

    F = deformation(spec.mode, primary, secondary)
    C = F.T @ F
    result = model.response(F, state)
    equilibrium = _project_stress(unpressurized_cauchy(F, model.parameters.zhang_li), spec.mode)
    branches = np.stack([_project_stress(value, spec.mode) for value in result["branch_cauchy"]])
    total = equilibrium + np.sum(branches, axis=0)
    inverse_transpose = np.linalg.inv(F).T
    nominal = total @ inverse_transpose
    secondary_stress = float(total[0, 1])
    Fv = np.stack(state.Fv)
    Fe = np.stack([F @ np.linalg.inv(value) for value in state.Fv])
    Ce = np.einsum("nji,njk->nik", Fe, Fe)
    moduli = np.array([
        model.parameters.Gneq_MPa * branch.p
        for branch in model.parameters.branches
    ], dtype=float)
    mandel_stress = np.stack([
        ogden_mandel_deviator(elastic, modulus, model.parameters.a_ogden)
        for elastic, modulus in zip(Fe, moduli)
    ])
    branch_energy = np.array([
        modulus * ogden_unit_energy(elastic.T @ elastic, model.parameters.a_ogden)
        for elastic, modulus in zip(Fe, moduli)
    ], dtype=float)
    details = result["branch"]
    admissible, distance = classify_point(spec, primary, secondary)
    equilibrium_energy = float(energy(C, model.parameters.zhang_li))
    snapshot = {
        "primary_strain": float(primary),
        "secondary_strain": float(secondary),
        "deformation_gradient": F,
        "right_cauchy_green": C,
        "jacobian": float(np.linalg.det(F)),
        "invariant_I1": float(np.trace(C)),
        "invariant_I2": float(0.5 * (np.trace(C) ** 2 - np.trace(C @ C))),
        "viscous_deformation_gradient": Fv,
        "elastic_deformation_gradient": Fe,
        "elastic_right_cauchy_green": Ce,
        "viscous_jacobian": np.linalg.det(Fv),
        "cauchy_stress_MPa": total,
        "first_piola_stress_MPa": nominal,
        "equilibrium_cauchy_stress_MPa": equilibrium,
        "branch_cauchy_stress_MPa": branches,
        "measured_stress_MPa": float(total[0, 1] if spec.mode == "shear" else nominal[0, 0]),
        "secondary_measured_stress_MPa": secondary_stress,
        "equilibrium_free_energy_MPa": equilibrium_energy,
        "branch_free_energy_MPa": branch_energy,
        "total_free_energy_MPa": float(equilibrium_energy + np.sum(branch_energy)),
        "mandel_stress_MPa": mandel_stress,
        "thermodynamic_driving_force_MPa": np.stack([item["X"] for item in details]),
        "viscous_flow_rate_per_s": np.stack([item["Dv"] for item in details]),
        "equivalent_driving_force_MPa": np.array([item["q_MPa"] for item in details]),
        "equivalent_flow_rate_per_s": np.array([item["gamma_dot_per_s"] for item in details]),
        "hardening_variable": np.array([item["h"] for item in details]),
        "hardening_multiplier": np.array([item["hardening"] for item in details]),
        "nonlinear_flow_multiplier": np.array([item["flow_multiplier"] for item in details]),
        "branch_dissipation_MPa_per_s": np.array([item["dissipation_MPa_per_s"] for item in details]),
        "total_dissipation_MPa_per_s": float(sum(item["dissipation_MPa_per_s"] for item in details)),
        "within_experimental_envelope": bool(admissible),
        "state_space_distance": float(distance),
        **_diagnostic_summary(diagnostics),
    }
    if not np.isclose(snapshot["total_free_energy_MPa"], result["energy_MPa"], atol=1e-9, rtol=1e-9):
        raise RuntimeError("equilibrium/branch free-energy decomposition is inconsistent")
    return snapshot


def _precondition(model, mode: str):
    from teacher_model.model import ModelState

    state = ModelState.identity(len(model.parameters.branches))
    peak = 0.25 if mode == "uniaxial" else 0.20
    state, _ = advance_parts(model, state, mode, 0.0, 0.0, peak, 0.0, 1.5, 12)
    state, _ = advance_parts(model, state, mode, peak, 0.0, peak, 0.0, 2.0, 4)
    state, _ = advance_parts(model, state, mode, peak, 0.0, 0.0, 0.0, 1.5, 12)
    return state


def _stress_controlled_step(model, spec: PathSpec, state, old_primary: float, target: float, duration: float):
    evaluations: dict[float, tuple[float, object, list[dict]]] = {}

    def residual(primary: float) -> float:
        primary = float(primary)
        if primary not in evaluations:
            trial, diagnostics = advance_parts(
                model, state, spec.mode, old_primary, 0.0, primary, 0.0, duration,
            )
            evaluations[primary] = (
                measured_stress(model, trial, spec.mode, primary) - target,
                trial,
                diagnostics,
            )
        return evaluations[primary][0]

    center = residual(old_primary)
    if abs(center) < 1e-10:
        primary = float(old_primary)
    else:
        step = 0.025
        if center < 0.0:
            lower, upper = float(old_primary), min(float(old_primary) + step, 1.30)
            while residual(upper) < 0.0 and upper < 1.30:
                step *= 1.8
                upper = min(float(old_primary) + step, 1.30)
        else:
            lower, upper = max(float(old_primary) - step, -0.10), float(old_primary)
            while residual(lower) > 0.0 and lower > -0.10:
                step *= 1.8
                lower = max(float(old_primary) - step, -0.10)
        if residual(lower) * residual(upper) > 0.0:
            raise RuntimeError(f"creep stress root is not bracketed at target={target:.6g} MPa")
        primary = float(brentq(residual, lower, upper, xtol=2e-9, rtol=2e-9, maxiter=32))
    _, next_state, diagnostics = evaluations[primary]
    return primary, next_state, diagnostics


def simulate_path(model, spec: PathSpec) -> dict:
    from teacher_model.model import ModelState

    state = _precondition(model, spec.mode) if spec.preconditioned else ModelState.identity(len(model.parameters.branches))
    primary = float(spec.primary[0])
    secondary = float(spec.secondary[0])
    snapshots = [state_snapshot(model, state, spec, primary, secondary, [])]
    for index in range(1, len(spec.time_s)):
        duration = float(spec.time_s[index] - spec.time_s[index - 1])
        if duration <= 0.0:
            raise ValueError(f"time is not strictly increasing at index {index}")
        if spec.control == "stress":
            if spec.target_stress_MPa is None:
                raise ValueError("stress-controlled path is missing target stresses")
            next_primary, state, diagnostics = _stress_controlled_step(
                model, spec, state, primary, float(spec.target_stress_MPa[index]), duration,
            )
            next_secondary = 0.0
        else:
            next_primary = float(spec.primary[index])
            next_secondary = float(spec.secondary[index])
            state, diagnostics = advance_parts(
                model, state, spec.mode, primary, secondary, next_primary, next_secondary, duration,
            )
        primary, secondary = next_primary, next_secondary
        snapshots.append(state_snapshot(model, state, spec, primary, secondary, diagnostics))
    arrays = {
        key: np.stack([np.asarray(snapshot[key]) for snapshot in snapshots])
        for key in snapshots[0]
    }
    arrays["time_s"] = spec.time_s.copy()
    if spec.target_stress_MPa is not None:
        arrays["target_stress_MPa"] = spec.target_stress_MPa.copy()
    return arrays
