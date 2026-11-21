"""Load the frozen chapter-2 T2--Ogden low-fidelity teacher decision."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_teachers(project: Path, decision_path: Path | None = None,
                      parameter_path: Path | None = None,
                      data_path: Path | None = None) -> dict:
    """Return the immutable final teacher snapshot rather than rerunning selection."""
    project = Path(project)
    decision_path = decision_path or project / "results_chapter2_final" / "teacher_selection_final.json"
    decision = _read_json(decision_path)
    if decision.get("selected_teacher") != "T2":
        raise RuntimeError("the frozen chapter-2 decision does not select T2")

    parameter_path = parameter_path or (
        project / "results_chapter2_fair_comparison" / "parameters"
        / "T2_chapter2_joint_evaluation_parameters.json"
    )
    parameters = _read_json(parameter_path)
    if parameters.get("model") != "T2-Ogden-1":
        raise RuntimeError("active teacher parameter file is not T2-Ogden-1")

    data_path = data_path or Path(parameters["data"]["absolute_path"])
    if _sha256(data_path) != parameters["data"]["sha256"]:
        raise RuntimeError("experimental data hash changed after teacher freeze")

    candidates = []
    for row in decision["score_table"]:
        candidates.append({
            "model": row["model"],
            "eligible": bool(row["eligible"]),
            "predictive_utility_P": row.get("predictive_utility_P"),
            "S_teacher": row.get("S_teacher"),
            "S_stability": row.get("S_stability"),
            "S_numerical": row.get("S_numerical"),
            "S_cost": row.get("S_cost"),
            "reason": row.get("ineligibility_reason", ""),
        })

    return {
        "schema_version": "2026-08-30-frozen-T2-Ogden-pretraining-v2",
        "selection_type": "frozen dual-layer chapter-2 teacher rule",
        "decision_path": str(decision_path),
        "decision_sha256": _sha256(decision_path),
        "amplitude_DMA_used_in_score": False,
        "candidates": candidates,
        "selected_model": "T2",
        "selected_model_variant": "T2-Ogden-1",
        "selected_parameter_path": str(parameter_path),
        "selected_parameter_sha256": _sha256(parameter_path),
        "selected_candidate_id": parameters["selected_candidate_id"],
        "data_sha256": parameters["data"]["sha256"],
        "configuration_sha256": parameters["configuration_sha256"],
        "split_sha256": parameters["split_sha256"],
        "protocol_sha256": parameters["protocol_sha256"],
        "teacher_fidelity": (
            "physically acceptable low-fidelity T2-Ogden prior; "
            "high confidence only for finite-deformation relaxation"
        ),
        "known_exclusions": [
            "0.1% frequency DMA experimental pseudolabels",
            "1 Hz amplitude DMA experimental pseudolabels",
            "preload nonlinearity",
            "far multiaxial extrapolation",
        ],
    }


def write_selection_csv(result: dict, path: Path) -> None:
    fields = [
        "model", "eligible", "predictive_utility_P", "S_teacher",
        "S_stability", "S_numerical", "S_cost", "reason",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["candidates"])
