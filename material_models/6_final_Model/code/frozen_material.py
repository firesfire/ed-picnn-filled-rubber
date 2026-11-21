"""Load the two final potentials and the shared equilibrium material for prediction."""
from pathlib import Path

import torch
from dual_potential.constitutive import DualPotentialMaterial, MaterialParameters
from dual_potential.dissipation import Stage2DissipationPICNN, Stage6GatedDissipationPotential
from dual_potential.energies import TangentSecantEnergy, ZhangLiEquilibriumEnergy
from project_paths import load_config
from stage2_training import NORMALIZATION
from stage6_joint import synchronize_dissipation_spectrum


def load_frozen(checkpoint, teacher_parameters):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    energy_state = saved["energy_state_dict"]
    flow_state = saved["dissipation_state_dict"]
    p = MaterialParameters.from_json(teacher_parameters)
    cfg_dir = Path(__file__).resolve().parents[1] / "configs"
    energy_cfg = load_config((cfg_dir / "stage5b_epicnn_energy_finetuning.yaml").read_text(encoding="utf-8"))["energy"]
    gate_cfg = load_config((cfg_dir / "stage7_joint_consistency.yaml").read_text(encoding="utf-8"))["dissipation_gate"]
    times = energy_state["tau_s"]
    group_shapes = []
    for group in range(2):
        slots = torch.ones_like(energy_state[f"raw_weight.{group}"])
        group_shapes.append(dict(weights=slots, rho_I1=slots/2, b=slots, p=slots*0.75,
                                 hardening_MPa=1.0, hardening_rho_I1=0.5))
    energy = TangentSecantEnergy(energy_state["branch_moduli_MPa"], times, group_shapes, grouped=True,
                                 b_bounds=energy_cfg["saturation_b_bounds"],
                                 p_bounds=energy_cfg["saturation_power_bounds"],
                                 transition_tau_s=energy_cfg["group_transition_tau_s"])
    energy.load_state_dict(energy_state, strict=True)
    channels = flow_state["base.encoder.0.weight"].shape[1]
    architecture = {1: "PICNN-A", 4: "PICNN-B", 5: "PICNN-C"}[channels]
    base = Stage2DissipationPICNN(architecture, times, flow_state["branch_moduli_MPa"],
                                  float(flow_state["base.sigma_ref"]), NORMALIZATION)
    flow = Stage6GatedDissipationPotential(base, flow_state["branch_moduli_MPa"], times,
                                          transition_tau_s=gate_cfg["transition_tau_s"],
                                          state_scales=gate_cfg["state_scales"])
    flow.load_state_dict(flow_state, strict=True)
    material = DualPotentialMaterial(ZhangLiEquilibriumEnergy(p.zhang_li), energy, flow, times)
    material.equilibrium_energy_parameters = p.zhang_li
    synchronize_dissipation_spectrum(material)
    return material.eval().requires_grad_(False)
