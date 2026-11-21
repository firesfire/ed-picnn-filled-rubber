"""Convert the frozen HDF5 teacher database to a Torch-readable NPZ cache."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "2_TeacherModel" / "pretraining_data" / "full" / "teacher_pretraining_full.h5"
OUT = ROOT / "audit" / "stage2"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "stage2_teacher_cache.npz"

SPLIT = {"train": 0, "validation": 1, "test": 2}
CONFIDENCE = {"high": 0, "medium": 1, "low": 2}
MODE = {"uniaxial": 0, "shear": 1, "planar": 2, "equibiaxial": 3, "mixed": 4}


def dev(A):
    return A - np.trace(A, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3.0


def main():
    flow_records = []
    trajectory_records = []
    metadata = []
    with h5py.File(SOURCE, "r") as handle:
        tau = handle["branch_metadata/time_scale_s"][:]
        p = handle["branch_metadata/stiffness_weight"][:]
        for trajectory_index, identifier in enumerate(sorted(handle["trajectories"].keys())):
            group = handle[f"trajectories/{identifier}"]
            attrs = dict(group.attrs)
            split = str(attrs["split"])
            confidence = str(attrs["confidence"])
            mode = str(attrs["deformation_mode"])
            use_flow = bool(attrs["use_for_flow_supervision"])
            use_rollout = bool(attrs["use_for_rollout_supervision"])
            X = dev(0.5 * (group["thermodynamic_driving_force_MPa"][:] +
                           np.swapaxes(group["thermodynamic_driving_force_MPa"][:], -1, -2)))
            D = dev(0.5 * (group["viscous_flow_rate_per_s"][:] +
                           np.swapaxes(group["viscous_flow_rate_per_s"][:], -1, -2)))
            Fv = group["viscous_deformation_gradient"][:]
            Ce = group["elastic_right_cauchy_green"][:]
            Cv = np.swapaxes(Fv, -1, -2) @ Fv
            elastic_I1 = np.trace(Ce, axis1=-2, axis2=-1) - 3.0
            viscous_I1 = np.trace(Cv, axis1=-2, axis2=-1) - 3.0
            n = len(group["time_s"])
            sample_weight = group["sample_weight"][:]
            flow_records.append({
                "trajectory_index": np.full(n * 7, trajectory_index, np.int64),
                "split_code": np.full(n * 7, SPLIT[split], np.int8),
                "confidence_code": np.full(n * 7, CONFIDENCE[confidence], np.int8),
                "branch_index": np.tile(np.arange(7, dtype=np.int64), n),
                "X_MPa": X.reshape(-1, 3, 3),
                "D_per_s": D.reshape(-1, 3, 3),
                "Fv": Fv.reshape(-1, 3, 3),
                "total_I1_minus_3": np.repeat(group["invariant_I1"][:] - 3.0, 7),
                "elastic_I1_minus_3": elastic_I1.reshape(-1),
                "viscous_I1_minus_3": viscous_I1.reshape(-1),
                "hardening_variable": group["hardening_variable"][:].reshape(-1),
                "state_space_distance": np.repeat(group["state_space_distance"][:], 7),
                "sample_weight": np.repeat(sample_weight / 7.0, 7),
                "use_flow": np.full(n * 7, use_flow, bool),
            })
            trajectory_records.append({
                "trajectory_index": np.full(n, trajectory_index, np.int64),
                "time_s": group["time_s"][:],
                "F": group["deformation_gradient"][:],
                "Fv": Fv,
                "teacher_stress_MPa": group["measured_stress_MPa"][:],
            })
            metadata.append({
                "trajectory_id": identifier,
                "trajectory_index": trajectory_index,
                "split": split,
                "confidence": confidence,
                "mode": mode,
                "mode_code": MODE[mode],
                "path_family": str(attrs["path_family"]),
                "use_flow": use_flow,
                "use_rollout": use_rollout,
                "point_count": n,
            })

    arrays = {}
    for name in flow_records[0]:
        arrays[f"flow_{name}"] = np.concatenate([record[name] for record in flow_records], axis=0)
    for name in trajectory_records[0]:
        arrays[f"trajectory_{name}"] = np.concatenate([record[name] for record in trajectory_records], axis=0)
    arrays["tau_s"] = tau
    arrays["branch_shares"] = p
    np.savez_compressed(CACHE, **arrays)
    (OUT / "stage2_trajectory_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "source": str(SOURCE), "cache": str(CACHE),
        "trajectory_count": len(metadata),
        "flow_branch_point_count": int(len(arrays["flow_X_MPa"])),
        "trajectory_point_count": int(len(arrays["trajectory_F"])),
        "supervised_train_trajectory_count": sum(item["split"] == "train" and item["confidence"] in ("high", "medium") and item["use_flow"] for item in metadata),
        "low_confidence_train_trajectory_count_excluded": sum(item["split"] == "train" and item["confidence"] == "low" for item in metadata),
        "energy_supervision_used": False,
        "experimental_labels_used": False,
    }
    (OUT / "stage2_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
