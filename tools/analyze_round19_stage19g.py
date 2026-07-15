#!/usr/bin/env python3
"""Strict completeness gate and evidence-only analyzer for Round 19G."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from tools.round19_stage19f_ensemble import REQUIRED_MEMBER_IDS

VERDICTS = {"SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"}
OUTPUT_CSVS = (
    "round19g_atom_occlusion.csv",
    "round19g_connected_substructure_masking.csv",
    "round19g_scaffold_sidechain_ablation.csv",
    "round19g_bond_occlusion.csv",
    "round19g_pooled_drug_occlusion.csv",
    "round19g_maccs_ablation.csv",
    "round19g_omics_group_ablation.csv",
    "round19g_context_sensitivity.csv",
    "round19g_routing_audit.csv",
    "round19g_routing_counterfactual.csv",
    "round19g_case_summary.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locked_hash(experiment_lock: dict, key: str) -> str:
    for container in (experiment_lock, experiment_lock.get("hashes", {}), experiment_lock.get("locks", {})):
        value = container.get(key) if isinstance(container, dict) else None
        if value:
            return str(value)
    if key == "final_lock_sha256":
        value = experiment_lock.get("final_role_lock", {}).get("file_sha256")
        if value:
            return str(value)
    raise KeyError(f"experiment lock does not pin {key}")


def validate_complete(
    output_dir: Path,
    *,
    expected_case_ids: set[str],
    final_lock: Path,
    experiment_lock: Path,
) -> dict:
    frames = {}
    for filename in OUTPUT_CSVS:
        path = Path(output_dir) / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frames[filename] = pd.read_csv(path)

    atom = frames["round19g_atom_occlusion.csv"]
    required = {"case_id", "member_id", "control_type", "repeat"}
    if required - set(atom.columns):
        raise KeyError(f"atom output missing {sorted(required - set(atom.columns))}")
    if set(atom["case_id"].astype(str)) != expected_case_ids:
        raise AssertionError("atom output does not contain all and only locked cases")
    applicable = atom.copy()
    if "applicable" in applicable:
        applicable = applicable[
            applicable["applicable"].astype(str).str.lower().isin({"true", "1"})
        ]
    if applicable.empty:
        raise AssertionError("atom output contains no applicable graph perturbations")
    if "source_candidate_id" not in applicable:
        applicable["source_candidate_id"] = "legacy_unspecified"
    for (case_id, source), group in applicable.groupby(["case_id", "source_candidate_id"]):
        if set(group["member_id"].astype(str)) != set(REQUIRED_MEMBER_IDS):
            raise AssertionError(f"case/source {(case_id, source)} does not contain all 15 members")
        for member, member_rows in group.groupby("member_id"):
            random = member_rows[member_rows["control_type"] == "matched_random"]
            repeats = set(pd.to_numeric(random["repeat"], errors="coerce").dropna().astype(int))
            if repeats != set(range(20)):
                raise AssertionError(
                    f"case/source {(case_id, source)} member {member} lacks 20 matched-random repeats"
                )

    for filename, frame in frames.items():
        if "case_id" in frame:
            missing = expected_case_ids - set(frame["case_id"].astype(str))
            if missing:
                raise AssertionError(f"{filename} missing cases: {sorted(missing)}")

    routing = frames["round19g_routing_audit.csv"]
    if not {"seen_drug", "seen_scaffold", "seen_cancer_type", "routing_match"} <= set(routing):
        raise KeyError("routing audit lacks seen booleans or routing_match")
    if routing.empty or not routing["routing_match"].astype(bool).all():
        raise AssertionError("routing_match is not 100%")

    lock = json.loads(Path(experiment_lock).read_text(encoding="utf-8"))
    if sha256_file(final_lock) != _locked_hash(lock, "final_lock_sha256"):
        raise AssertionError("final lock hash changed")
    experiment_hash = sha256_file(experiment_lock)
    sidecar = Path(output_dir) / "experiment_lock.sha256"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != experiment_hash:
        raise AssertionError("experiment lock hash changed or attestation is missing")
    return {"complete": True, "routing_match_rate": 1.0, "csv_files": list(OUTPUT_CSVS)}


def write_report(output_dir: Path, completeness: dict, verdict: str) -> None:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {sorted(VERDICTS)}")
    payload = {
        **completeness,
        "verdict": verdict,
        "classification": "post_lock_descriptive",
        "roles_changed": False,
    }
    Path(output_dir, "round19g_analysis_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--final-lock", required=True)
    parser.add_argument("--experiment-lock", required=True)
    parser.add_argument("--verdict", choices=sorted(VERDICTS), required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if not args.require_complete:
        raise SystemExit("Analyzer requires --require-complete for formal output")
    cases = pd.read_csv(args.case_manifest)
    case_column = "case_id" if "case_id" in cases else "eval_row_id"
    if case_column not in cases:
        raise KeyError("case manifest requires case_id or eval_row_id")
    completeness = validate_complete(
        Path(args.output_dir),
        expected_case_ids=set(cases[case_column].astype(str)),
        final_lock=Path(args.final_lock),
        experiment_lock=Path(args.experiment_lock),
    )
    write_report(Path(args.output_dir), completeness, args.verdict)


if __name__ == "__main__":
    main()
