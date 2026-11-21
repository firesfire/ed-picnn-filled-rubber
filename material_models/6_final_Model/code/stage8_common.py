"""Frozen Stage-8 model registry and reproducibility helpers."""

from __future__ import annotations

from project_paths import load_config

import hashlib
import json
from pathlib import Path

import torch
import yaml

from dual_potential.constitutive import DualPotentialMaterial, MaterialParameters
from dual_potential.dissipation import T2GroupedPowerPotential
from dual_potential.energies import FixedOgdenEnergy, ZhangLiEquilibriumEnergy
from stage2_rollout import load_stage2_material
from stage4_energy import load_stage4_checkpoint
from stage5b_energy import load_stage5b_material
from stage6_joint import synchronize_dissipation_spectrum
from stage7_joint import load_stage7_material


ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path):
    return load_config(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")


def build_t2(parameter_file, device="cpu"):
    p = MaterialParameters.from_json(parameter_file)
    energy = FixedOgdenEnergy(p.branch_moduli_MPa, p.a_ogden,
                              dtype=torch.float64, device=device)
    dissipation = T2GroupedPowerPotential(
        p.branch_moduli_MPa, p.tau_s, p.equilibrium_shear_MPa,
        p.fast_flow, p.slow_flow, p.transition_tau_s,
        dtype=torch.float64, device=device)
    material = DualPotentialMaterial(
        ZhangLiEquilibriumEnergy(p.zhang_li, dtype=torch.float64, device=device),
        energy, dissipation, p.tau_s, dtype=torch.float64, device=device)
    material.equilibrium_energy_parameters = p.zhang_li
    material.eval()
    return material, p


def load_registry(config, device="cpu"):
    """Load every eligible frozen candidate without changing any parameter."""
    parameter_file = config["teacher_parameter_file"]
    stage4_config = read_yaml(ROOT / "configs" / "stage4_gdpicnn_energy_release.yaml")
    stage5_config = read_yaml(ROOT / "configs" / "stage5b_epicnn_energy_finetuning.yaml")
    stage7_config = read_yaml(config["stage7_config"])

    t2, parameters = build_t2(parameter_file, device)

    stage3_path = ROOT / "checkpoints" / "stage3" / "dpicnn_stage3_selected.pt"
    d_picnn, _ = load_stage2_material(stage3_path, parameter_file, device)

    stage4_path = ROOT / "checkpoints" / "stage4" / "gdpicnn_stage4_selected.pt"
    gd_picnn, _, _ = load_stage4_checkpoint(stage4_path, stage4_config, device)

    stage5_path = ROOT / "checkpoints" / "stage5b" / "hierarchical_fast_slow_energy_network_pareto.pt"
    e_picnn, _, _ = load_stage5b_material(
        stage5_config, "hierarchical_fast_slow_energy_network", device,
        synchronize_branch_scale=True)
    stage5_saved = torch.load(stage5_path, map_location=device, weights_only=False)
    e_picnn.nonequilibrium_energy.load_state_dict(stage5_saved["energy_state_dict"], strict=False)
    e_picnn.eval()

    ed_picnn, _, _ = load_stage7_material(stage7_config, device, load_checkpoint=False)
    stage7_saved = torch.load(config["stage7_checkpoint"], map_location=device,
                              weights_only=False)
    ed_picnn.nonequilibrium_energy.load_state_dict(stage7_saved["energy_state_dict"], strict=False)
    ed_picnn.dissipation_potential.load_state_dict(
        stage7_saved["dissipation_state_dict"], strict=False)
    synchronize_dissipation_spectrum(ed_picnn)
    ed_picnn.eval()

    registry = {
        "T2-Ogden": t2,
        "D-PICNN": d_picnn,
        "GD-PICNN": gd_picnn,
        "E-PICNN": e_picnn,
        "ED-PICNN": ed_picnn,
    }
    sources = {
        "T2-Ogden": Path(parameter_file),
        "D-PICNN": stage3_path,
        "GD-PICNN": stage4_path,
        "E-PICNN": stage5_path,
        "ED-PICNN": Path(config["stage7_checkpoint"]),
    }
    return registry, sources, parameters


def registry_manifest(registry, sources):
    rows = []
    for name, material in registry.items():
        source = sources[name]
        rows.append({
            "model": name,
            "source": str(source),
            "sha256": sha256(source),
            "trainable_parameter_count": sum(p.numel() for p in material.parameters()),
            "frozen_for_stage8": True,
        })
    return rows
