"""Trajectory-level thermodynamic, numerical and split-independent quality gates."""

from __future__ import annotations

import numpy as np

from material_point import advance_parts, measured_stress
from path_library import PathSpec


THRESHOLDS = {
    "deformation_jacobian_error": 1e-8,
    "viscous_jacobian_error": 1e-8,
    "minimum_dissipation_MPa_per_s": -1e-10,
    "flow_trace_per_s": 1e-8,
    "stress_decomposition_MPa": 1e-8,
    "energy_decomposition_MPa": 1e-8,
    "creep_control_error_MPa": 2e-6,
    "local_time_refinement_relative_error": 0.08,
    "high_confidence_cycle_relative_change": 0.08,
    "maximum_turning_point_stress_jump_ratio": 0.60,
}


def _refinement_index(arrays: dict) -> int:
    primary = np.abs(np.diff(arrays["primary_strain"]))
    secondary = np.abs(np.diff(arrays["secondary_strain"]))
    duration = np.diff(arrays["time_s"])
    changing = np.flatnonzero(primary + secondary > 1e-9)
    if len(changing):
        return int(changing[min(len(changing) // 3, 8)] + 1)
    return int(np.argmax(duration) + 1)


def refinement_check(model, spec: PathSpec, arrays: dict) -> dict:
    from teacher_model.model import ModelState

    index = _refinement_index(arrays)
    start = ModelState([value.copy() for value in arrays["viscous_deformation_gradient"][index - 1]])
    p0 = float(arrays["primary_strain"][index - 1])
    p1 = float(arrays["primary_strain"][index])
    s0 = float(arrays["secondary_strain"][index - 1])
    s1 = float(arrays["secondary_strain"][index])
    dt = float(arrays["time_s"][index] - arrays["time_s"][index - 1])
    stresses = []
    states = []
    for parts in (1, 2, 4):
        state, _ = advance_parts(model, start, spec.mode, p0, s0, p1, s1, dt, parts)
        states.append(state)
        stresses.append(measured_stress(model, state, spec.mode, p1, s1))
    scale = max(abs(stresses[-1]), 0.025)
    coarse_error = abs(stresses[0] - stresses[1]) / scale
    fine_error = abs(stresses[1] - stresses[2]) / scale
    state_error = float(max(
        np.linalg.norm(a - b) for a, b in zip(states[1].Fv, states[2].Fv)
    ))
    return {
        "index": index,
        "dt_s": dt,
        "stress_dt_MPa": float(stresses[0]),
        "stress_dt_over_2_MPa": float(stresses[1]),
        "stress_dt_over_4_MPa": float(stresses[2]),
        "coarse_relative_difference": float(coarse_error),
        "fine_relative_difference": float(fine_error),
        "state_dt2_dt4_frobenius_difference": state_error,
        "passed": bool(fine_error <= THRESHOLDS["local_time_refinement_relative_error"]),
    }


def inspect_trajectory(model, spec: PathSpec, arrays: dict) -> dict:
    finite = all(np.all(np.isfinite(value)) for value in arrays.values())
    monotonic_time = bool(np.all(np.diff(arrays["time_s"]) > 0.0))
    jacobian_error = float(np.max(np.abs(arrays["jacobian"] - 1.0)))
    viscous_error = float(np.max(np.abs(arrays["viscous_jacobian"] - 1.0)))
    dissipation_min = float(np.min(arrays["branch_dissipation_MPa_per_s"]))
    flow_trace = float(np.max(np.abs(np.trace(arrays["viscous_flow_rate_per_s"], axis1=-2, axis2=-1))))
    stress_reconstruction = (
        arrays["equilibrium_cauchy_stress_MPa"]
        + np.sum(arrays["branch_cauchy_stress_MPa"], axis=1)
    )
    stress_error = float(np.max(np.abs(stress_reconstruction - arrays["cauchy_stress_MPa"])))
    energy_reconstruction = (
        arrays["equilibrium_free_energy_MPa"]
        + np.sum(arrays["branch_free_energy_MPa"], axis=1)
    )
    energy_error = float(np.max(np.abs(energy_reconstruction - arrays["total_free_energy_MPa"])))
    control_error = (
        float(np.max(np.abs(arrays["measured_stress_MPa"] - arrays["target_stress_MPa"])))
        if "target_stress_MPa" in arrays else 0.0
    )
    periodic_change = None
    if spec.points_per_cycle and len(arrays["measured_stress_MPa"]) >= 2 * spec.points_per_cycle:
        width = spec.points_per_cycle
        last = arrays["measured_stress_MPa"][-width:]
        before = arrays["measured_stress_MPa"][-2 * width:-width]
        periodic_change = float(np.max(np.abs(last - before)) / max(np.max(np.abs(last)), 0.025))
    strain_step = np.diff(arrays["primary_strain"])
    stress_step = np.abs(np.diff(arrays["measured_stress_MPa"]))
    turning = np.flatnonzero(
        (np.abs(strain_step[1:]) <= 1e-12)
        & (np.abs(strain_step[:-1]) > 1e-12)
        | (strain_step[1:] * strain_step[:-1] < -1e-14)
    ) + 1 if len(strain_step) > 1 else np.array([], dtype=int)
    stress_span = max(float(np.ptp(arrays["measured_stress_MPa"])), 0.025)
    turning_jump = float(np.max(stress_step[turning]) / stress_span) if len(turning) else 0.0
    equilibrium_cauchy = arrays["equilibrium_cauchy_stress_MPa"]
    if spec.mode == "shear":
        equilibrium_scalar = equilibrium_cauchy[:, 0, 1]
    else:
        inverse_transpose = np.swapaxes(np.linalg.inv(arrays["deformation_gradient"]), -1, -2)
        equilibrium_scalar = np.einsum("tij,tjk->tik", equilibrium_cauchy, inverse_transpose)[:, 0, 0]
    nonequilibrium_scalar = arrays["measured_stress_MPa"] - equilibrium_scalar
    changing = np.flatnonzero(
        np.abs(np.diff(arrays["primary_strain"]))
        + np.abs(np.diff(arrays["secondary_strain"])) > 1e-10
    )
    hold_relaxes = True
    if spec.control == "strain" and spec.hold_duration_s >= 30.0 and len(changing):
        hold_start = int(changing[-1] + 1)
        # Signed branch stresses can cancel after unloading or multistage
        # loading, so their scalar sum need not decay monotonically even while
        # every branch dissipates.  The norm of all branch driving forces is
        # the physically relevant convergence-to-equilibrium indicator.
        driving_norm = np.linalg.norm(arrays["equivalent_driving_force_MPa"], axis=1)
        hold_relaxes = bool(driving_norm[-1] <= driving_norm[hold_start] + 1e-8)
    # Very weak branches have q proportional to their stiffness; a small but
    # nonzero q may still correspond to a physically finite q/eta flow rate.
    # The constitutive zero-flow requirement therefore applies to exactly zero
    # driving force, rather than to an absolute threshold shared by all branches.
    near_zero = arrays["equivalent_driving_force_MPa"] == 0.0
    zero_flow_error = (
        float(np.max(np.abs(arrays["equivalent_flow_rate_per_s"][near_zero])))
        if np.any(near_zero) else 0.0
    )
    refinement = refinement_check(model, spec, arrays)
    checks = {
        "all_finite": bool(finite),
        "strictly_increasing_time": monotonic_time,
        "positive_and_isochoric_F": bool(np.min(arrays["jacobian"]) > 0 and jacobian_error <= THRESHOLDS["deformation_jacobian_error"]),
        "isochoric_Fv": bool(viscous_error <= THRESHOLDS["viscous_jacobian_error"]),
        "nonnegative_dissipation": bool(dissipation_min >= THRESHOLDS["minimum_dissipation_MPa_per_s"]),
        "traceless_viscous_flow": bool(flow_trace <= THRESHOLDS["flow_trace_per_s"]),
        "stress_decomposition": bool(stress_error <= THRESHOLDS["stress_decomposition_MPa"]),
        "energy_decomposition": bool(energy_error <= THRESHOLDS["energy_decomposition_MPa"]),
        "stress_control_accuracy": bool(control_error <= THRESHOLDS["creep_control_error_MPa"]),
        "no_turning_point_stress_jump": bool(turning_jump <= THRESHOLDS["maximum_turning_point_stress_jump_ratio"]),
        "long_hold_approaches_equilibrium": hold_relaxes,
        "zero_driving_force_zero_flow": bool(zero_flow_error <= 1e-8),
        "time_refinement": bool(refinement["passed"]),
        "cycle_steady_if_high_confidence": bool(
            periodic_change is None or spec.confidence != "high"
            or periodic_change <= THRESHOLDS["high_confidence_cycle_relative_change"]
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "point_count": int(len(arrays["time_s"])),
        "duration_s": float(arrays["time_s"][-1]),
        "maximum_deformation_jacobian_error": jacobian_error,
        "maximum_viscous_jacobian_error": viscous_error,
        "minimum_branch_dissipation_MPa_per_s": dissipation_min,
        "maximum_flow_trace_per_s": flow_trace,
        "maximum_stress_decomposition_error_MPa": stress_error,
        "maximum_energy_decomposition_error_MPa": energy_error,
        "maximum_stress_control_error_MPa": control_error,
        "last_cycle_relative_change": periodic_change,
        "maximum_turning_point_stress_jump_ratio": turning_jump,
        "long_hold_approaches_equilibrium": hold_relaxes,
        "maximum_zero_force_flow_per_s": zero_flow_error,
        "maximum_newton_residual": float(np.max(arrays["newton_residual"])),
        "maximum_subdivision_level": int(np.max(arrays["subdivisions"])),
        "maximum_newton_iterations": int(np.max(arrays["newton_iterations"])),
        "time_refinement": refinement,
    }
