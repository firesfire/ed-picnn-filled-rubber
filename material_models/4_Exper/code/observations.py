"""Creep and harmonic-response evaluation on measured condition coordinates."""
from time import perf_counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from generate_final_model_creep_preload_dma import creep_time_and_stress, reduced_creep
from stage5b_energy import advance_reduced, reduced_response

def predict_creep(material, observations: pd.DataFrame, config: dict,
                  results_dir: Path) -> pd.DataFrame:
    output = []
    spec_base = dict(config["creep"])
    for (mode, stress), group in observations.groupby(["mode", "target_stress_MPa"]):
        valid_times = np.sort(group.loc[group.hold_time_s >= 0.0, "hold_time_s"].unique())
        spec = dict(spec_base)
        spec["hold_elapsed_s"] = valid_times.tolist()
        time, prescribed = creep_time_and_stress(spec, float(stress), material.dtype)
        print(f"creep {mode}, {stress:g} MPa, {len(valid_times)} comparison points",
              flush=True)
        started = perf_counter()
        history = reduced_creep(
            material, prescribed, time, mode,
            float(spec["stress_tolerance_MPa"]),
            int(spec["maximum_outer_iterations"]),
        )
        offset = len(time) - len(valid_times)
        prediction = pd.DataFrame({
            "hold_time_s": valid_times,
            "predicted_strain": history["strain"][offset:].detach().cpu().numpy(),
            "computed_stress_MPa": history["stress"][offset:].detach().cpu().numpy(),
            "total_energy_MPa": history["energy"][offset:].detach().cpu().numpy(),
            "total_dissipation_MPa_per_s": (
                history["dissipation"][offset:].detach().cpu().numpy()),
            "maximum_driving_force_norm_MPa": (
                history["force_norm"][offset:].detach().cpu().numpy()),
            "maximum_flow_rate_norm_per_s": (
                history["flow_norm"][offset:].detach().cpu().numpy()),
        })
        prediction["predicted_compliance_per_MPa"] = (
            prediction.predicted_strain / float(stress))
        prediction["mode"] = mode
        prediction["target_stress_MPa"] = float(stress)
        prediction["condition_wall_time_s"] = perf_counter() - started
        output.append(prediction)
    predictions = pd.concat(output, ignore_index=True)
    merged = observations.merge(
        predictions,
        on=["mode", "target_stress_MPa", "hold_time_s"],
        how="left",
    )
    merged["compliance_error_per_MPa"] = (
        merged.predicted_compliance_per_MPa
        - merged.experimental_compliance_per_MPa)
    merged["compliance_relative_error"] = (
        merged.compliance_error_per_MPa
        / merged.experimental_compliance_per_MPa)
    merged["strain_error"] = merged.predicted_strain - merged.experimental_strain
    merged["strain_relative_error"] = (
        merged.strain_error / merged.experimental_strain)
    merged.to_csv(results_dir / "creep_experiment_prediction.csv", index=False)
    return merged

def batch_periodic_dma(material, cases: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Integrate all experimental DMA cases in one tensor batch.

    This is algebraically the same reduced exponential update used by the
    scalar solver. All cases use the same phase grid, while their physical
    time increments are set by their individual frequencies.
    """
    preload = torch.tensor(cases.shear_preload.to_numpy(), dtype=material.dtype)
    frequency = torch.tensor(cases.frequency_Hz.to_numpy(), dtype=material.dtype)
    amplitude = torch.tensor(
        cases.strain_amplitude_fraction.to_numpy(), dtype=material.dtype)
    count = len(cases)
    points = int(spec["points_per_cycle"])
    maximum_cycles = int(spec["maximum_cycles"])
    phase = 2.0 * torch.pi * torch.arange(
        0, points + 1, dtype=material.dtype) / points
    dt = 1.0 / (frequency * points)
    internal = preload[:, None].repeat(1, 7)
    minimum_dissipation = torch.full(
        (count,), float("inf"), dtype=material.dtype)
    prior_oscillatory = None
    change = torch.full((count,), float("inf"), dtype=material.dtype)
    final_stress = None
    started = perf_counter()
    for cycle in range(maximum_cycles):
        stress_values = []
        for step in range(1, points + 1):
            strain0 = preload + amplitude * torch.sin(phase[step - 1])
            strain1 = preload + amplitude * torch.sin(phase[step])
            internal = advance_reduced(
                material, internal, strain0, strain1, dt, "shear").detach()
            response = reduced_response(material, strain1, internal, "shear")
            stress_values.append(response["stress"].detach())
            dissipation = response["dissipation"].sum(-1).detach()
            minimum_dissipation = torch.minimum(minimum_dissipation, dissipation)
        final_stress = torch.stack(stress_values, dim=1)
        oscillatory = final_stress - final_stress.mean(dim=1, keepdim=True)
        if prior_oscillatory is not None:
            scale = oscillatory.abs().amax(dim=1).clamp_min(1.0e-10)
            change = (oscillatory - prior_oscillatory).abs().amax(dim=1) / scale
        prior_oscillatory = oscillatory
        print(f"batch DMA cycle {cycle + 1}/{maximum_cycles}, "
              f"max change={float(change.max()):.4g}", flush=True)

    phase_samples = phase[1:]
    storage = 2.0 * torch.mean(
        final_stress * torch.sin(phase_samples)[None, :], dim=1) / amplitude
    loss = 2.0 * torch.mean(
        final_stress * torch.cos(phase_samples)[None, :], dim=1) / amplitude
    fundamental = torch.sqrt(storage.square() + loss.square()).clamp_min(1.0e-14)
    harmonics = {}
    for order in (2, 3, 5):
        a = 2.0 * torch.mean(
            final_stress * torch.sin(order * phase_samples)[None, :], dim=1) / amplitude
        b = 2.0 * torch.mean(
            final_stress * torch.cos(order * phase_samples)[None, :], dim=1) / amplitude
        harmonics[order] = torch.sqrt(a.square() + b.square()) / fundamental
    elapsed = perf_counter() - started
    out = cases.copy().reset_index(drop=True)
    out["model"] = "ED-PICNN-stage8-final"
    out["sweep"] = "exact_experiment_point"
    out["preload_type"] = "mean_simple_shear_strain"
    out["strain_amplitude_percent"] = 100.0 * out.strain_amplitude_fraction
    out["storage_modulus_MPa"] = storage.detach().cpu().numpy()
    out["loss_modulus_MPa"] = loss.detach().cpu().numpy()
    out["tan_delta"] = (loss / storage.clamp_min(1.0e-14)).detach().cpu().numpy()
    out["second_harmonic_ratio"] = harmonics[2].detach().cpu().numpy()
    out["third_harmonic_ratio"] = harmonics[3].detach().cpu().numpy()
    out["fifth_harmonic_ratio"] = harmonics[5].detach().cpu().numpy()
    out["cycles_used"] = maximum_cycles
    out["cycle_change"] = change.detach().cpu().numpy()
    out["steady"] = out.cycle_change <= float(spec["steady_tolerance"])
    out["minimum_dissipation_MPa_per_s"] = (
        minimum_dissipation.detach().cpu().numpy())
    out["maximum_detFv_error"] = 0.0
    out["wall_time_s"] = elapsed / max(count, 1)
    return out
