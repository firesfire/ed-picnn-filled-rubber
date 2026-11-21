"""Material-point solvers for stress-controlled creep and preloaded cyclic shear."""

from project_paths import load_config
from pathlib import Path
import hashlib
import numpy as np
import torch
from dual_potential.solvers import extract_dynamic_moduli
from dual_potential.tensor import deformation_gradient
from stage3_real import _coordinate
from stage5b_energy import advance_reduced, reduced_response
from stage6_joint import synchronize_dissipation_spectrum
from stage7_joint import load_stage7_material

def read_yaml(path: Path) -> dict:
    return load_config(path.read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_final_material(config: dict):
    checkpoint = Path(config["final_checkpoint"])
    actual_hash = sha256(checkpoint)
    expected_hash = config["expected_checkpoint_sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError(f"final checkpoint SHA-256 mismatch: {actual_hash}")
    stage7_config = read_yaml(Path(config["stage7_config"]))
    material, _, _ = load_stage7_material(stage7_config, "cpu", load_checkpoint=False)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    material.nonequilibrium_energy.load_state_dict(saved["energy_state_dict"], strict=False)
    material.dissipation_potential.load_state_dict(
        saved["dissipation_state_dict"], strict=False)
    synchronize_dissipation_spectrum(material)
    material.eval()
    for parameter in material.parameters():
        parameter.requires_grad_(False)
    return material, actual_hash

def reduced_energy(material, strain, internal, mode: str) -> torch.Tensor:
    F = deformation_gradient(strain, mode, dtype=material.dtype, device=material.device)
    value = material.equilibrium_energy(F)
    coordinate = _coordinate(strain, mode)
    for branch in range(7):
        gap = coordinate - internal[branch]
        if mode == "uniaxial":
            lam = torch.exp(gap)
            Fe = torch.diag(torch.stack((lam, lam.rsqrt(), lam.rsqrt())))
        else:
            Fe = deformation_gradient(gap, "shear", dtype=material.dtype,
                                      device=material.device)
        value = value + material.nonequilibrium_energy(Fe, branch)
    return value

def reduced_creep(material, stress: torch.Tensor, time: torch.Tensor, mode: str,
                  tolerance: float, maximum_iterations: int):
    internal = torch.zeros(7, dtype=material.dtype, device=material.device)
    strain = torch.tensor(0.0, dtype=material.dtype, device=material.device)
    strains, stresses, energies, dissipations = [strain], [], [], []
    force_norms, flow_norms, iterations_used = [], [], [0]

    def observe(value, state):
        response = reduced_response(material, value, state, mode)
        force_norm = torch.sqrt(2.0 * response["j"].clamp_min(0.0)) * \
            material.dissipation_potential.sigma_ref
        flow_norm = response["slope"] * force_norm / \
            material.dissipation_potential.sigma_ref
        return response, force_norm.max(), flow_norm.max()

    initial, force, flow = observe(strain, internal)
    stresses.append(initial["stress"]); energies.append(reduced_energy(material, strain, internal, mode))
    dissipations.append(initial["dissipation"].sum()); force_norms.append(force); flow_norms.append(flow)
    for index in range(1, len(time)):
        dt = (time[index] - time[index-1]).clamp_min(1.0e-12)
        target = stress[index]

        def trial(value):
            candidate = advance_reduced(material, internal, strain, value, dt, mode)
            response = reduced_response(material, value, candidate, mode)
            return response["stress"], candidate

        guess = strain.clone(); accepted = None; used = 0
        lower, upper = ((-0.75, 3.0) if mode == "uniaxial" else (-2.0, 2.0))
        for iteration in range(maximum_iterations):
            used = iteration + 1
            value, candidate = trial(guess)
            residual = value - target
            if float(residual.abs()) <= tolerance:
                accepted = candidate; break
            step = 1.0e-5 * max(1.0, abs(float(guess)))
            tangent = (trial(guess + step)[0] - trial(guess - step)[0]) / (2.0 * step)
            if not bool(torch.isfinite(tangent)) or float(tangent.abs()) < 1.0e-9:
                break
            guess = torch.clamp(guess - residual / tangent, min=lower, max=upper)
        if accepted is None:
            lo = torch.tensor(lower, dtype=material.dtype); hi = torch.tensor(upper, dtype=material.dtype)
            flo = trial(lo)[0] - target; fhi = trial(hi)[0] - target
            if float(flo * fhi) > 0.0:
                raise RuntimeError(f"reduced creep failed to bracket target at index {index}")
            for offset in range(80):
                guess = 0.5 * (lo + hi); value, candidate = trial(guess)
                residual = value - target
                used = maximum_iterations + offset + 1
                if float(residual.abs()) <= tolerance:
                    accepted = candidate; break
                if float(flo * residual) <= 0.0:
                    hi = guess
                else:
                    lo, flo = guess, residual
            if accepted is None:
                raise RuntimeError(f"reduced creep stress control failed at index {index}")
        strain, internal = guess.detach(), accepted.detach()
        response, force, flow = observe(strain, internal)
        strains.append(strain); stresses.append(response["stress"])
        energies.append(reduced_energy(material, strain, internal, mode))
        dissipations.append(response["dissipation"].sum())
        force_norms.append(force); flow_norms.append(flow); iterations_used.append(used)
    return {"strain": torch.stack(strains), "stress": torch.stack(stresses),
            "energy": torch.stack(energies), "dissipation": torch.stack(dissipations),
            "force_norm": torch.stack(force_norms), "flow_norm": torch.stack(flow_norms),
            "iterations": iterations_used}

def creep_time_and_stress(spec: dict, target: float, dtype: torch.dtype):
    load_time = torch.linspace(
        0.0, float(spec["loading_time_s"]), int(spec["loading_points"]), dtype=dtype)
    hold_elapsed = torch.tensor(spec["hold_elapsed_s"], dtype=dtype)
    time = torch.cat((load_time, float(spec["loading_time_s"]) + hold_elapsed))
    stress = torch.cat((target * load_time / float(spec["loading_time_s"]),
                        torch.full_like(hold_elapsed, target)))
    return time, stress

def periodic_preloaded_shear(material, frequency_hz: float, amplitude: float,
                             preload: float, spec: dict):
    internal = torch.full((7,), preload, dtype=material.dtype, device=material.device)
    prior_oscillatory = None
    change = float("inf")
    result = None
    dynamic = None
    cycles_used = 0
    period = 1.0 / frequency_hz
    points = int(spec["points_per_cycle"])
    tolerance = float(spec["steady_tolerance"])
    for cycle in range(int(spec["maximum_cycles"])):
        local_time = torch.linspace(0.0, period, points + 1, dtype=material.dtype)
        strain = preload + amplitude * torch.sin(2.0 * torch.pi * frequency_hz * local_time)
        stress_values=[]; energy_values=[]; dissipation_values=[]
        for step in range(1, len(local_time)):
            dt=local_time[step]-local_time[step-1]
            internal=advance_reduced(material,internal,strain[step-1],strain[step],dt,"shear").detach()
            response=reduced_response(material,strain[step],internal,"shear")
            stress_values.append(response["stress"].detach())
            energy_values.append(reduced_energy(material,strain[step],internal,"shear").detach())
            dissipation_values.append(response["dissipation"].sum().detach())
        stress=torch.stack(stress_values)
        oscillatory = stress - stress.mean()
        cycles_used = cycle + 1
        if prior_oscillatory is not None:
            scale = max(float(oscillatory.abs().max()), 1.0e-10)
            change = float((oscillatory - prior_oscillatory).abs().max()) / scale
        dynamic = extract_dynamic_moduli(
            strain[1:], stress, frequency_hz, amplitude, change, tolerance)
        result={"time":local_time[1:],"strain":strain[1:],"stress":stress,
                "energy":torch.stack(energy_values),
                "dissipation":torch.stack(dissipation_values)}
        if cycles_used >= int(spec["minimum_cycles"]) and change <= tolerance:
            break
        prior_oscillatory = oscillatory.detach()
    return result, dynamic, cycles_used

def dynamic_row(config, sweep: str, preload: float, frequency: float,
                amplitude: float, history, dynamic, cycles_used: int,
                wall_time_s: float) -> dict:
    return {
        "model": config["model"], "sweep": sweep,
        "preload_type": "mean_simple_shear_strain",
        "shear_preload": preload, "frequency_Hz": frequency,
        "strain_amplitude_fraction": amplitude,
        "strain_amplitude_percent": 100.0 * amplitude,
        "storage_modulus_MPa": dynamic.storage_MPa,
        "loss_modulus_MPa": dynamic.loss_MPa,
        "tan_delta": dynamic.loss_factor,
        "second_harmonic_ratio": dynamic.harmonic_ratios[2],
        "third_harmonic_ratio": dynamic.harmonic_ratios[3],
        "fifth_harmonic_ratio": dynamic.harmonic_ratios[5],
        "cycles_used": cycles_used, "cycle_change": dynamic.cycle_change,
        "steady": dynamic.steady,
        "minimum_dissipation_MPa_per_s": float(history["dissipation"].min()),
        "maximum_detFv_error": 0.0,
        "wall_time_s": wall_time_s,
    }
