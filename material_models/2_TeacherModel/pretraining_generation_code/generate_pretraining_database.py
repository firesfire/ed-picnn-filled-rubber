"""Generate evidence-selected, confidence-weighted material-point pretraining data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

for thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import h5py
import numpy as np
import scipy

from material_point import load_teacher, simulate_path
from path_library import (
    CONFIDENCE_WEIGHTS, PathSpec, build_diagnostic_specs, build_formal_specs,
    select_pilot_specs,
)
from quality_control import THRESHOLDS, inspect_trajectory
from teacher_selection import evaluate_teachers, write_selection_csv


SCHEMA_VERSION = "2026-08-30-T2-Ogden-dissipation-pretraining-v2"
GLOBAL_MODEL = None

FIELD_SCHEMA = {
    "time_s": ("s", "physical elapsed time"),
    "time_step_s": ("s", "elapsed time since preceding recorded state"),
    "primary_strain": ("1", "engineering axial strain or engineering shear"),
    "secondary_strain": ("1", "engineering shear in the mixed path"),
    "deformation_gradient": ("1", "F[i,j], current spatial row and reference material column"),
    "right_cauchy_green": ("1", "C=F.T@F in material coordinates"),
    "jacobian": ("1", "det(F)"),
    "invariant_I1": ("1", "trace(C)"),
    "invariant_I2": ("1", "(trace(C)^2-trace(C@C))/2"),
    "viscous_deformation_gradient": ("1", "Fv[branch,i,j] in the intermediate configuration"),
    "elastic_deformation_gradient": ("1", "Fe[branch,i,j]=F@inverse(Fv)"),
    "elastic_right_cauchy_green": ("1", "Ce[branch]=Fe.T@Fe"),
    "viscous_jacobian": ("1", "det(Fv) for every branch"),
    "cauchy_stress_MPa": ("MPa", "pressure-projected total Cauchy stress tensor"),
    "first_piola_stress_MPa": ("MPa", "P=sigma@inverse(F).T for J=1"),
    "equilibrium_cauchy_stress_MPa": ("MPa", "fixed Zhang-Li equilibrium stress contribution"),
    "branch_cauchy_stress_MPa": ("MPa", "pressure-projected nonequilibrium branch stresses"),
    "measured_stress_MPa": ("MPa", "P11 for axial families; sigma12 for simple shear"),
    "secondary_measured_stress_MPa": ("MPa", "sigma12 accompanying the selected primary mode"),
    "equilibrium_free_energy_MPa": ("MPa", "equilibrium free-energy density"),
    "branch_free_energy_MPa": ("MPa", "nonequilibrium free-energy density per branch"),
    "total_free_energy_MPa": ("MPa", "equilibrium plus all nonequilibrium free-energy densities"),
    "mandel_stress_MPa": ("MPa", "branch Mandel stress in intermediate coordinates"),
    "thermodynamic_driving_force_MPa": ("MPa", "deviatoric Mandel driving-force tensor"),
    "viscous_flow_rate_per_s": ("1/s", "symmetric deviatoric viscous velocity-gradient tensor"),
    "equivalent_driving_force_MPa": ("MPa", "sqrt(3/2 X:X) per branch"),
    "equivalent_flow_rate_per_s": ("1/s", "branch equivalent viscous flow rate"),
    "hardening_variable": ("1", "branch equivalent viscous Hencky strain"),
    "hardening_multiplier": ("1", "branch hardening multiplier"),
    "nonlinear_flow_multiplier": ("1", "linear plus nonlinear flow multiplier"),
    "branch_dissipation_MPa_per_s": ("MPa/s", "X:Dv per branch"),
    "total_dissipation_MPa_per_s": ("MPa/s", "sum of branch dissipation rates"),
    "within_experimental_envelope": ("bool", "whether the state has supported experimental geometry and magnitude"),
    "state_space_distance": ("1", "heuristic distance outside the supported experimental state envelope"),
    "sample_weight": ("1", "within-trajectory balanced supervision weight"),
    "newton_iterations": ("count", "sum of branch Newton iterations in the recorded increment"),
    "newton_residual": ("1", "maximum Newton residual in the recorded increment"),
    "subdivisions": ("count", "maximum implicit-integrator subdivision depth"),
    "target_stress_MPa": ("MPa", "prescribed scalar nominal stress for creep paths only"),
}


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _dump_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _worker_initialize(project: str, selection: dict) -> None:
    global GLOBAL_MODEL
    GLOBAL_MODEL, _ = load_teacher(Path(project), selection)


def _point_weights(spec: PathSpec, arrays: dict) -> np.ndarray:
    primary_change = np.abs(np.diff(arrays["primary_strain"], prepend=arrays["primary_strain"][0]))
    secondary_change = np.abs(np.diff(arrays["secondary_strain"], prepend=arrays["secondary_strain"][0]))
    stress_change = np.abs(np.diff(arrays["measured_stress_MPa"], prepend=arrays["measured_stress_MPa"][0]))
    movement = primary_change + secondary_change
    movement /= max(float(np.max(movement)), 1e-12)
    stress_change /= max(float(np.max(stress_change)), 1e-12)
    weights = 1.0 + 1.5 * movement + 1.0 * stress_change
    return spec.confidence_weight * weights / float(np.sum(weights))


def _worker_simulate(spec: PathSpec) -> dict:
    started = time.perf_counter()
    try:
        arrays = simulate_path(GLOBAL_MODEL, spec)
        arrays["time_step_s"] = np.diff(arrays["time_s"], prepend=arrays["time_s"][0])
        arrays["sample_weight"] = _point_weights(spec, arrays)
        quality = inspect_trajectory(GLOBAL_MODEL, spec, arrays)
        return {
            "trajectory_id": spec.trajectory_id,
            "arrays": arrays,
            "quality": quality,
            "elapsed_s": float(time.perf_counter() - started),
            "error": None if quality["passed"] else "quality gates failed: " + ", ".join(
                name for name, passed in quality["checks"].items() if not passed
            ),
        }
    except Exception as error:
        return {
            "trajectory_id": spec.trajectory_id,
            "arrays": None,
            "quality": None,
            "elapsed_s": float(time.perf_counter() - started),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }


def _branch_groups(payload: dict, model_name: str) -> np.ndarray:
    if model_name == "T2":
        transition = float(payload.get("transition_tau_s", 1.0))
        return np.array(["fast" if tau <= transition else "slow" for tau in payload["tau_s"]], dtype="S8")
    return np.array(["shared"] * len(payload["tau_s"]), dtype="S8")


def _branch_weights(payload: dict) -> np.ndarray:
    values = payload.get("p_alpha", payload.get("weights"))
    if values is None:
        raise KeyError("teacher payload contains neither p_alpha nor weights")
    return np.asarray(values, dtype=float)


def _initialize_hdf(handle: h5py.File, selection: dict, payload: dict, stage: str, seed: int) -> None:
    handle.attrs.update({
        "schema_version": SCHEMA_VERSION,
        "teacher_model": selection["selected_model"],
        "teacher_fidelity": selection["teacher_fidelity"],
        "teacher_parameter_sha256": selection["selected_parameter_sha256"],
        "experimental_data_sha256": selection["data_sha256"],
        "configuration_sha256": selection["configuration_sha256"],
        "experimental_split_sha256": selection["split_sha256"],
        "synthetic_source": "synthetic_teacher",
        "tensor_component_order": "full row-major 3x3; no Voigt shear-factor scaling",
        "energy_reference": "Zhang-Li equilibrium and Ogden branch energies are normalized to zero at reference",
        "stage": stage,
        "random_seed": int(seed),
    })
    branch = handle.create_group("branch_metadata")
    branch.create_dataset("branch_index", data=np.arange(len(payload["tau_s"]), dtype=int))
    branch.create_dataset("time_scale_s", data=np.asarray(payload["tau_s"], dtype=float))
    branch.create_dataset("stiffness_weight", data=_branch_weights(payload))
    branch.create_dataset("group", data=_branch_groups(payload, selection["selected_model"]))
    handle.create_group("trajectories")


def _store_trajectory(handle: h5py.File, spec: PathSpec, result: dict, selection: dict, seed: int) -> None:
    group = handle["trajectories"].create_group(spec.trajectory_id)
    masks = spec.supervision_masks
    group.attrs.update({
        "trajectory_id": spec.trajectory_id,
        "condition_id": f"{spec.family}:{spec.mode}:{spec.trajectory_id}",
        "path_family": spec.family,
        "deformation_mode": spec.mode,
        "control": spec.control,
        "confidence": spec.confidence,
        "trajectory_weight": spec.confidence_weight,
        "frequency_hz": spec.frequency_hz,
        "amplitude": spec.amplitude,
        "preload": spec.preload,
        "rate_per_s": spec.rate_per_s,
        "hold_duration_s": spec.hold_duration_s,
        "points_per_cycle": spec.points_per_cycle,
        "preconditioned": spec.preconditioned,
        "teacher_model": selection["selected_model"],
        "teacher_parameter_sha256": selection["selected_parameter_sha256"],
        "source": "synthetic_teacher",
        "random_seed": int(seed),
        "quality_passed": result["quality"]["passed"],
        "quality_json": json.dumps(result["quality"], ensure_ascii=False, default=_json_default),
        "diagnostic_only": spec.diagnostic_only,
        "use_for_flow_supervision": masks["flow"],
        "use_for_rollout_supervision": masks["rollout"],
        "use_for_stress_supervision": masks["stress"],
        "use_for_energy_supervision": masks["energy"],
    })
    for name, values in result["arrays"].items():
        dataset = group.create_dataset(name, data=values, compression="gzip", compression_opts=4, shuffle=True)
        unit, description = FIELD_SCHEMA.get(name, ("1", "documented in field_schema.json"))
        dataset.attrs["unit"] = unit
        dataset.attrs["description"] = description


def _manifest_record(spec: PathSpec, quality: dict, elapsed: float, split: str = "unassigned") -> dict:
    masks = spec.supervision_masks
    return {
        "trajectory_id": spec.trajectory_id,
        "condition_id": f"{spec.family}:{spec.mode}:{spec.trajectory_id}",
        "path_family": spec.family,
        "deformation_mode": spec.mode,
        "control": spec.control,
        "confidence": spec.confidence,
        "trajectory_weight": spec.confidence_weight,
        "split": split,
        "point_count": quality["point_count"],
        "duration_s": quality["duration_s"],
        "maximum_primary_strain": float(np.max(spec.primary)) if spec.control == "strain" else "stress_controlled",
        "minimum_primary_strain": float(np.min(spec.primary)) if spec.control == "strain" else "stress_controlled",
        "frequency_hz": spec.frequency_hz,
        "amplitude": spec.amplitude,
        "preload": spec.preload,
        "rate_per_s": spec.rate_per_s,
        "hold_duration_s": spec.hold_duration_s,
        "preconditioned": spec.preconditioned,
        "diagnostic_only": spec.diagnostic_only,
        "use_for_flow_supervision": masks["flow"],
        "use_for_rollout_supervision": masks["rollout"],
        "use_for_stress_supervision": masks["stress"],
        "use_for_energy_supervision": masks["energy"],
        "quality_passed": quality["passed"],
        "minimum_branch_dissipation_MPa_per_s": quality["minimum_branch_dissipation_MPa_per_s"],
        "maximum_viscous_jacobian_error": quality["maximum_viscous_jacobian_error"],
        "local_time_refinement_relative_error": quality["time_refinement"]["fine_relative_difference"],
        "last_cycle_relative_change": quality["last_cycle_relative_change"],
        "generation_elapsed_s": elapsed,
        "source": "synthetic_teacher",
        "hdf5_group": f"/trajectories/{spec.trajectory_id}",
    }


def _write_csv(path: Path, rows: list[dict], fallback_fields: list[str] | None = None) -> None:
    fields = list(rows[0]) if rows else (fallback_fields or ["trajectory_id", "error"])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _simulate_stage(
    project: Path, selection: dict, payload: dict, specs: list[PathSpec],
    output: Path, stage: str, seed: int, workers: int, pilot_source: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    spec_by_id = {spec.trajectory_id: spec for spec in specs}
    records: list[dict] = []
    failures: list[dict] = []
    completed: set[str] = set()
    with h5py.File(output, "w") as destination:
        _initialize_hdf(destination, selection, payload, stage, seed)
        if pilot_source and pilot_source.exists():
            with h5py.File(pilot_source, "r") as source:
                for identifier in sorted(source["trajectories"]):
                    if identifier not in spec_by_id:
                        continue
                    source.copy(source["trajectories"][identifier], destination["trajectories"], name=identifier)
                    quality = json.loads(source["trajectories"][identifier].attrs["quality_json"])
                    records.append(_manifest_record(spec_by_id[identifier], quality, 0.0))
                    completed.add(identifier)
            print(f"[{stage}] reused {len(completed)} quality-approved pilot histories", flush=True)

        remaining = [spec for spec in specs if spec.trajectory_id not in completed]
        if remaining:
            with ProcessPoolExecutor(
                max_workers=max(1, min(workers, len(remaining))),
                initializer=_worker_initialize,
                initargs=(str(project), selection),
            ) as pool:
                futures = {pool.submit(_worker_simulate, spec): spec for spec in remaining}
                for future in as_completed(futures):
                    spec = futures[future]
                    result = future.result()
                    index = len(records) + len(failures) + 1
                    if result["error"]:
                        failures.append({
                            "trajectory_id": spec.trajectory_id,
                            "path_family": spec.family,
                            "deformation_mode": spec.mode,
                            "confidence": spec.confidence,
                            "error": result["error"],
                            "traceback": result.get("traceback", ""),
                        })
                        print(f"[{stage}] {index}/{len(specs)} FAILED {spec.trajectory_id} {result['error']}", flush=True)
                        continue
                    _store_trajectory(destination, spec, result, selection, seed)
                    records.append(_manifest_record(spec, result["quality"], result["elapsed_s"]))
                    print(
                        f"[{stage}] {index}/{len(specs)} {spec.trajectory_id} {spec.family} "
                        f"confidence={spec.confidence} points={result['quality']['point_count']} "
                        f"elapsed={result['elapsed_s']:.1f}s", flush=True,
                    )
        destination.attrs["accepted_trajectory_count"] = len(records)
        destination.attrs["failed_trajectory_count"] = len(failures)
    records.sort(key=lambda row: row["trajectory_id"])
    failures.sort(key=lambda row: row["trajectory_id"])
    return records, failures


def _assign_splits(records: list[dict], seed: int) -> None:
    rng = np.random.default_rng(seed + 417)
    groups = defaultdict(list)
    for row in records:
        groups[(row["confidence"], row["path_family"], row["deformation_mode"])].append(row)
    for _, members in sorted(groups.items()):
        order = rng.permutation(len(members))
        if len(members) >= 5:
            validation = max(1, round(.15 * len(members)))
            test = max(1, round(.15 * len(members)))
        elif len(members) >= 3:
            validation, test = 1, 1
        elif len(members) == 2:
            validation, test = 0, 1
        else:
            validation, test = 0, 0
        for rank, position in enumerate(order):
            members[position]["split"] = (
                "validation" if rank < validation
                else "test" if rank < validation + test
                else "train"
            )


def _write_split_files(full_path: Path, destination_root: Path, records: list[dict], selection: dict, payload: dict, seed: int) -> None:
    by_split = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)
    with h5py.File(full_path, "r+") as source:
        for record in records:
            source["trajectories"][record["trajectory_id"]].attrs["split"] = record["split"]
        for split in ("train", "validation", "test"):
            directory = destination_root / split
            directory.mkdir(parents=True, exist_ok=True)
            target_path = directory / f"teacher_pretraining_{split}.h5"
            with h5py.File(target_path, "w") as target:
                _initialize_hdf(target, selection, payload, split, seed)
                for record in by_split[split]:
                    identifier = record["trajectory_id"]
                    source.copy(source["trajectories"][identifier], target["trajectories"], name=identifier)
                target.attrs["accepted_trajectory_count"] = len(by_split[split])
            _write_csv(directory / f"{split}_manifest.csv", by_split[split])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(project: Path, metadata: Path, selection: dict, payload: dict, records: list[dict], seed: int, workers: int) -> dict:
    schema = {
        "schema_version": SCHEMA_VERSION,
        "tensor_convention": {
            "storage_order": "(time, branch where applicable, spatial/material row, material column)",
            "tensor_shape": [3, 3],
            "flattening_if_needed": "C-order: 11,12,13,21,22,23,31,32,33",
            "shear_convention": "full tensor, no engineering-shear or Voigt factor applied to tensor components",
            "stress_unit": "MPa",
            "energy_density_unit": "MPa = MJ/m^3",
            "dissipation_unit": "MPa/s = MW/m^3",
        },
        "fields": {
            name: {"unit": unit, "description": description}
            for name, (unit, description) in FIELD_SCHEMA.items()
        },
        "branch_metadata": {
            "branch_index": list(range(len(payload["tau_s"]))),
            "time_scale_s": payload["tau_s"],
            "stiffness_weight": _branch_weights(payload).tolist(),
            "group": [value.decode() for value in _branch_groups(payload, selection["selected_model"])],
        },
    }
    confidence = {
        "high": {
            "trajectory_weight": CONFIDENCE_WEIGHTS["high"],
            "domain": "directly tested finite-deformation uniaxial/shear relaxation at 0.2/s",
        },
        "medium": {
            "trajectory_weight": CONFIDENCE_WEIGHTS["medium"],
            "domain": "interpolated uniaxial/shear relaxation, unloading, multistage and random piecewise paths inside strain envelope",
        },
        "low": {
            "trajectory_weight": CONFIDENCE_WEIGHTS["low"],
            "domain": "compression, planar/biaxial/mixed paths, creep, continued internal states, preload, or cyclic amplitude above 0.1%",
            "restriction": "never claim amplitude- or preload-dependent experimental nonlinearities from teacher pseudolabels",
        },
        "diagnostic": {
            "trajectory_weight": 0.0,
            "domain": "frequency DMA, amplitude DMA, preload cycles and creep inversion",
            "restriction": "stored for diagnostics only; all supervision masks are false",
        },
        "within_history_normalization": "sample weights sum to trajectory_weight; changes in strain and stress receive extra emphasis",
    }
    reproducibility = {
        "random_seed": seed,
        "worker_count": workers,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
        "teacher_model": selection["selected_model"],
        "teacher_parameter_file": selection["selected_parameter_path"],
        "teacher_parameter_sha256": selection["selected_parameter_sha256"],
        "experimental_data_sha256": selection["data_sha256"],
        "configuration_sha256": selection["configuration_sha256"],
        "experimental_split_sha256": selection["split_sha256"],
        "generation_command": f"python pretraining_generation_code/generate_pretraining_database.py --stage all --seed {seed} --workers {workers}",
        "code_sha256": {
            path.name: _sha256(path)
            for path in sorted((project / "pretraining_generation_code").glob("*.py"))
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    split_policy = {
        "unit": "complete material-point loading history; never split a trajectory into multiple subsets",
        "stratification": ["confidence", "path_family", "deformation_mode"],
        "target_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "small_group_rule": "groups of 3-4 allocate one validation and one test history; groups of two allocate one train and one test",
        "source_label": "synthetic_teacher",
        "experimental_isolation": "train/validation/test contain synthetic time-domain trajectories only; DMA/preload/creep are isolated in diagnostic_only",
        "counts": dict(Counter(row["split"] for row in records)),
    }
    for filename, content in (
        ("field_schema.json", schema),
        ("confidence_domains.json", confidence),
        ("split_policy.json", split_policy),
        ("quality_thresholds.json", THRESHOLDS),
        ("reproducibility.json", reproducibility),
        ("teacher_selection.json", selection),
    ):
        _dump_json(metadata / filename, content)
    write_selection_csv(selection, metadata / "teacher_selection_scorecard.csv")
    return reproducibility


def _selection_report(project: Path, selection: dict, records: list[dict]) -> None:
    counts = Counter(row["confidence"] for row in records)
    lines = [
        "# Teacher selection and trajectory coverage", "",
        f"Selected candidate: `{selection['selected_candidate_id']}`.", "",
        "The frozen T2-Ogden model supplies synthetic state and flow histories for pretraining.",
        "The split is by complete history. Teacher energy is archived without energy supervision.", "",
        "| Confidence | Trajectory weight | Histories |", "|---|---:|---|",
        "| High | 1.00 | Experimentally constrained uniaxial and shear relaxation |",
        "| Medium | 0.35 | Multirate, unload-reload, multistep and piecewise paths within the envelope |",
        "| Low | 0.08 | Compression, planar, biaxial, mixed, reverse shear and state continuation |",
        "| Diagnostic | 0.00 | Frequency and amplitude sweeps, preloaded cycles and creep |", "",
        f"Formal histories: {counts['high']} high, {counts['medium']} medium and {counts['low']} low confidence.",
    ]
    (project / "teacher_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _quality_report(project: Path, selection: dict, pilot: list[dict], records: list[dict], failures: list[dict], reproducibility: dict) -> None:
    families = Counter(row["path_family"] for row in records)
    splits = Counter(row["split"] for row in records)
    confidence = Counter(row["confidence"] for row in records)
    points = sum(row["point_count"] for row in records)
    minimum_dissipation = min((row["minimum_branch_dissipation_MPa_per_s"] for row in records), default=float("nan"))
    maximum_volume_error = max((row["maximum_viscous_jacobian_error"] for row in records), default=float("nan"))
    maximum_refinement_error = max((row["local_time_refinement_relative_error"] for row in records), default=float("nan"))
    lines = [
        "# Teacher data generation and quality checks", "",
        f"Teacher: `{selection['selected_model_variant']}`.", "",
        f"Pilot histories: {len(pilot)}. Formal histories: {len(records)}. Failures: {len(failures)}.",
        f"Material-point observations: {points}. Branch observations: {7 * points}.",
        f"Histories by split: {splits['train']} training, {splits['validation']} validation, {splits['test']} test.",
        f"Confidence: {confidence['high']} high, {confidence['medium']} medium, {confidence['low']} low.", "",
        "| Path family | Histories |", "|---|---:|",
    ]
    lines.extend(f"|`{name}`|{count}|" for name, count in sorted(families.items()))
    lines.extend([
        "", "## Numerical checks", "",
        f"- Minimum branch dissipation: {minimum_dissipation:.3e} MPa/s.",
        f"- Maximum absolute viscous volume-ratio error: {maximum_volume_error:.3e}.",
        f"- Maximum local relative stress change between half and quarter steps: {maximum_refinement_error:.3%}.", "",
        "The HDF5 files store full tensors and seven branch histories. Per-trajectory flags specify flow, rollout and stress supervision. Diagnostic histories have zero supervision weight.",
        "Tensor conventions, units, thresholds and the split policy are in `pretraining_data/metadata/`.", "",
        f"Generation command: `{reproducibility['generation_command']}`.",
    ])
    (project / "teacher_data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "all"), default="all")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--pilot-max", type=int, default=30)
    parser.add_argument("--teacher-parameters", type=Path, help="Frozen T2 material parameter file")
    parser.add_argument("--selection", type=Path, help="Chapter 2 teacher-selection decision file")
    parser.add_argument("--experiment", type=Path, help="Experimental workbook from calibration")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    output = project / "pretraining_data"
    metadata = output / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    selection = evaluate_teachers(project, args.selection, args.teacher_parameters, args.experiment)
    _, payload = load_teacher(project, selection)
    specs = build_formal_specs(args.seed)
    diagnostic_specs = build_diagnostic_specs(args.seed)
    pilot_specs = select_pilot_specs(specs, args.pilot_max)
    print(
        f"Selected {selection['selected_model']} from audited comparison; "
        f"pilot={len(pilot_specs)}, formal={len(specs)}, diagnostic={len(diagnostic_specs)}, "
        f"workers={args.workers}", flush=True,
    )
    pilot_path = output / "pilot" / "teacher_pretraining_pilot.h5"
    pilot_records, pilot_failures = _simulate_stage(
        project, selection, payload, pilot_specs, pilot_path, "pilot", args.seed,
        min(args.workers, 6),
    )
    _write_csv(output / "pilot" / "pilot_manifest.csv", pilot_records)
    _write_csv(metadata / "pilot_failed_trajectories.csv", pilot_failures)
    if pilot_failures:
        raise RuntimeError(f"pilot produced {len(pilot_failures)} failed histories; formal generation is blocked")
    if args.stage == "pilot":
        _metadata(project, metadata, selection, payload, pilot_records, args.seed, args.workers)
        _selection_report(project, selection, pilot_records)
        print("Pilot database completed and all quality checks passed.", flush=True)
        return

    full_path = output / "full" / "teacher_pretraining_full.h5"
    records, failures = _simulate_stage(
        project, selection, payload, specs, full_path, "full", args.seed, args.workers, pilot_path,
    )
    diagnostic_path = output / "diagnostic_only" / "teacher_diagnostic_only.h5"
    diagnostic_records, diagnostic_failures = _simulate_stage(
        project, selection, payload, diagnostic_specs, diagnostic_path,
        "diagnostic_only", args.seed, args.workers,
    )
    for row in diagnostic_records:
        row["split"] = "diagnostic_only"
    _write_csv(output / "diagnostic_only" / "diagnostic_manifest.csv", diagnostic_records)
    _write_csv(metadata / "diagnostic_failed_trajectories.csv", diagnostic_failures)
    _assign_splits(records, args.seed)
    _write_split_files(full_path, output, records, selection, payload, args.seed)
    _write_csv(project / "dataset_manifest.csv", records)
    _write_csv(metadata / "dataset_manifest.csv", records)
    _write_csv(metadata / "failed_trajectories.csv", failures)
    reproducibility = _metadata(project, metadata, selection, payload, records, args.seed, args.workers)
    _selection_report(project, selection, records)
    _quality_report(project, selection, pilot_records, records, failures, reproducibility)
    summary = {
        "selected_teacher": selection["selected_model"],
        "pilot_trajectories": len(pilot_records),
        "formal_trajectories": len(records),
        "failed_trajectories": len(failures),
        "diagnostic_trajectories": len(diagnostic_records),
        "diagnostic_failed_trajectories": len(diagnostic_failures),
        "formal_points": sum(row["point_count"] for row in records),
        "split_counts": dict(Counter(row["split"] for row in records)),
        "confidence_counts": dict(Counter(row["confidence"] for row in records)),
        "formal_hdf5": str(full_path),
        "formal_hdf5_size_bytes": full_path.stat().st_size,
        "diagnostic_hdf5": str(diagnostic_path),
        "diagnostic_hdf5_size_bytes": diagnostic_path.stat().st_size,
    }
    _dump_json(metadata / "generation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"formal generation retained {len(failures)} failed histories; inspect failed_trajectories.csv")


if __name__ == "__main__":
    main()
