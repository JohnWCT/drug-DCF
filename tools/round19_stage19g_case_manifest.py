#!/usr/bin/env python3
"""Validate the Round 19G case manifest and build pinned task manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from tools.round19_stage19f_final_lock import canonical_sha256, sha256_file
from tools.round19_stage19g_case_selector import (
    CASE_COLUMNS,
    PREDICTION_COLUMNS,
    SELECTION_REASONS,
)

METHODS = ("attention", "occlusion", "omics", "routing")
SHARD_SIZE = 32


def _bool_series(values: pd.Series, name: str) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    out = values.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        raise ValueError(f"{name} must contain explicit booleans")
    return out.astype(bool)


def validate_case_manifest(
    frame: pd.DataFrame,
    *,
    minimum_patient_samples: int = 20,
    minimum_patient_cancer_types: int = 3,
    patient_per_drug_cap: int = 30,
) -> pd.DataFrame:
    missing = set(CASE_COLUMNS) - set(frame)
    if missing:
        raise KeyError(f"Case manifest missing fields: {sorted(missing)}")
    forbidden = PREDICTION_COLUMNS & set(frame)
    if forbidden:
        raise AssertionError(f"Predictions must not be persisted in case manifest: {sorted(forbidden)}")
    out = frame.loc[:, CASE_COLUMNS].copy()
    if not out["case_id"].astype(str).equals(out["eval_row_id"].astype(str)):
        raise AssertionError("case_id compatibility alias must equal eval_row_id")
    if out["eval_row_id"].isna().any() or out["eval_row_id"].duplicated().any():
        raise AssertionError("eval_row_id must be non-null and unique")
    valid_namespace = out["eval_row_id"].astype(str).str.match(r"^19(?:e|f):")
    if not valid_namespace.all():
        raise AssertionError("Every eval_row_id must carry a 19e: or 19f: namespace")
    if not set(out["selection_reason"].astype(str)).issubset(SELECTION_REASONS):
        raise AssertionError("selection_reason contains values outside the fixed enum")
    for column in (
        "is_posthoc_contrastive",
        "is_tcga_exploratory",
        "posthoc_case",
        "selection_eligible",
        "prediction_used_for_selection",
    ):
        out[column] = _bool_series(out[column], column)
    contrastive = out["selection_reason"].astype(str).str.startswith("contrastive_")
    if not out["is_posthoc_contrastive"].equals(contrastive):
        raise AssertionError("Contrastive cases must be explicitly marked post-hoc")
    if not out.loc[contrastive, "prediction_used_for_selection"].all():
        raise AssertionError("Post-hoc contrastive cases must disclose prediction use")
    non_contrastive = ~contrastive
    if out.loc[non_contrastive, "prediction_used_for_selection"].any():
        raise AssertionError("Representative/patient selection may not use predictions")
    tcga = (out["source_stage"] == "19F") & (out["source_target"] != "internal_test")
    if not out.loc[tcga, "is_tcga_exploratory"].all():
        raise AssertionError("All TCGA cases must be exploratory-only")
    if out.loc[~tcga, "is_tcga_exploratory"].any():
        raise AssertionError("Non-TCGA rows cannot carry the TCGA exploratory flag")
    if not (out.loc[tcga, "selection_reason"] == "tcga_exploratory").all():
        raise AssertionError("TCGA cases require selection_reason=tcga_exploratory")
    if not out.loc[tcga, "posthoc_case"].all() or out.loc[tcga, "selection_eligible"].any():
        raise AssertionError("TCGA cases must be posthoc and selection-ineligible")
    if out.loc[~tcga, "selection_reason"].eq("tcga_exploratory").any():
        raise AssertionError("TCGA exploratory reason cannot appear in primary cohorts")
    expected_posthoc = contrastive | tcga
    if not out["posthoc_case"].equals(expected_posthoc):
        raise AssertionError("posthoc_case must identify contrastive and TCGA rows exactly")
    if (
        out["graph_smiles_metadata_status"]
        != "legacy_graph_resolution_required"
    ).any():
        raise AssertionError("graph_smiles metadata status must require exporter resolution")
    if not out["graph_smiles"].astype(str).equals(out["canonical_smiles"].astype(str)):
        raise AssertionError("Pre-export graph_smiles must alias canonical_smiles")
    if set(pd.to_numeric(out["selection_seed"], errors="raise").astype(int)) != {19091}:
        raise AssertionError("Round 19G case selection seed must be 19091")

    patient = out[out["selection_reason"] == "patient_conditioned"]
    if patient.empty:
        raise AssertionError("At least one qualified patient-conditioned drug is required")
    patient_failures = []
    for drug, group in patient.groupby("drug_id", sort=True):
        evidence = {
            "drug_id": str(drug),
            "selected_rows": int(len(group)),
            "selected_unique_modelids": int(group["ModelID"].nunique()),
            "selected_cancer_types": int(group["cancer_type"].nunique()),
            "selected_labels": sorted(group["Label"].astype(int).unique().tolist()),
            "available_unique_modelids": int(
                pd.to_numeric(group["available_drug_unique_modelids"], errors="raise").min()
            ),
            "available_cancer_types": int(
                pd.to_numeric(group["available_drug_cancer_types"], errors="raise").min()
            ),
            "available_labels": sorted(
                {
                    int(value)
                    for text in group["available_drug_labels"].astype(str)
                    for value in text.split(",")
                    if value != ""
                }
            ),
        }
        if (
            evidence["selected_rows"] > patient_per_drug_cap
            or evidence["selected_unique_modelids"] < minimum_patient_samples
            or evidence["selected_cancer_types"] < minimum_patient_cancer_types
            or set(evidence["selected_labels"]) != {0, 1}
            or evidence["available_unique_modelids"] < minimum_patient_samples
            or evidence["available_cancer_types"] < minimum_patient_cancer_types
            or set(evidence["available_labels"]) != {0, 1}
            or set(group["source_stage"]) != {"19E"}
            or group["posthoc_case"].any()
            or not group["selection_eligible"].all()
        ):
            patient_failures.append(evidence)
    if patient_failures:
        raise AssertionError(
            "patient-conditioned per-drug semantics violated: "
            + json.dumps(patient_failures, sort_keys=True)
        )
    if not 150 <= len(out) <= 250:
        raise AssertionError(f"Case manifest total must remain within 150-250: {len(out)}")
    return out


def _load_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("lock_type") != "round19_final_role_lock"
        or lock.get("schema_version") != 1
        or lock.get("immutable") is not True
    ):
        raise AssertionError("Task manifests require the immutable schema-v1 final role lock")
    payload = json.loads(json.dumps(lock, allow_nan=False))
    expected = payload["hashes"].pop("lock_payload_sha256")
    if expected != canonical_sha256(payload):
        raise AssertionError("Final role lock payload hash mismatch")
    return lock


def _resolve_role_sources(lock: Mapping[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    inventory_sources = sorted(
        {str(row["source_candidate_id"]) for row in lock["hashes"]["checkpoint_inventory"]}
    )
    aliases: dict[str, list[str]] = defaultdict(list)
    for role, record in lock["roles"].items():
        requested = str(record["source_candidate_id"])
        matches = [
            source
            for source in inventory_sources
            if source == requested
            or source.startswith(requested + "_")
            or requested.startswith(source + "_")
        ]
        if len(matches) != 1:
            raise AssertionError(f"Locked role {role} does not resolve uniquely: {matches}")
        aliases[matches[0]].append(str(role))
    if set(aliases) != set(inventory_sources):
        raise AssertionError("Every actual locked source must be represented by a role")
    return {key: sorted(value) for key, value in aliases.items()}, inventory_sources


def _candidate_methods(
    lock: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, list[str]]:
    aliases, all_sources = _resolve_role_sources(lock)
    components = config.get("candidate_components")
    if not isinstance(components, Mapping):
        raise KeyError("Interpretability config requires candidate_components")
    if set(components) != set(all_sources):
        raise AssertionError("candidate_components must cover actual locked sources exactly")
    attention = sorted(
        source
        for source in all_sources
        if str(components[source].get("predictor_id")) == "P2"
    )
    if not attention:
        raise AssertionError("No locked P2 source is available for attention tasks")
    methods = config.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(METHODS):
        raise AssertionError(f"Config methods must be exactly {METHODS}")
    selected = {
        "attention": attention,
        "occlusion": all_sources,
        "omics": all_sources,
        "routing": all_sources,
    }
    for method in METHODS:
        if methods[method].get("enabled") is not True:
            selected[method] = []
    # Touch aliases here so a future role-schema change cannot bypass dedup semantics.
    assert all(aliases[source] for source in all_sources)
    return selected


def _verify_inventory(
    lock: Mapping[str, Any], project_root: Path
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for item in lock["hashes"]["checkpoint_inventory"]:
        source = str(item["source_candidate_id"])
        member = str(item["member_id"])
        key = (source, member)
        if key in seen:
            raise AssertionError(f"Duplicate checkpoint member: {key}")
        seen.add(key)
        path = Path(str(item["checkpoint_path"]))
        path = path if path.is_absolute() else project_root / path
        if not path.is_file() or sha256_file(path) != item["checkpoint_sha256"]:
            raise AssertionError(f"Checkpoint hash mismatch: {path}")
        grouped[source].append(dict(item))
    for source, items in grouped.items():
        if len(items) != 15 or len({row["member_id"] for row in items}) != 15:
            raise AssertionError(f"{source} must have exactly 15 unique checkpoint members")
    return grouped


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise AssertionError(f"Refusing to write an empty task manifest: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_task_manifests(
    *,
    case_manifest_path: Path,
    config_path: Path,
    final_lock_path: Path,
    project_root: Path,
    output_dir: Path,
    case_selection_config_path: Optional[Path] = None,
    shard_size: int = SHARD_SIZE,
) -> dict[str, Any]:
    if shard_size != 32:
        raise AssertionError("Round 19G protocol fixes case shard size at 32")
    cases = validate_case_manifest(pd.read_csv(case_manifest_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lock = _load_lock(final_lock_path)
    candidates = _candidate_methods(lock, config)
    inventory = _verify_inventory(lock, project_root)
    aliases, _ = _resolve_role_sources(lock)
    pins = {
        "case_manifest_sha256": sha256_file(case_manifest_path),
        "config_sha256": sha256_file(config_path),
        "case_selection_config_sha256": (
            sha256_file(case_selection_config_path) if case_selection_config_path else ""
        ),
        "final_lock_file_sha256": sha256_file(final_lock_path),
        "final_lock_payload_sha256": lock["hashes"]["lock_payload_sha256"],
    }
    shards = []
    cohorts = (
        ("primary_faithfulness", cases[cases["selection_eligible"]]),
        ("tcga_exploratory", cases[cases["selection_reason"] == "tcga_exploratory"]),
    )
    for cohort_scope, cohort_cases in cohorts:
        cohort_cases = cohort_cases.reset_index(drop=True)
        for index in range(math.ceil(len(cohort_cases) / shard_size)):
            start = index * shard_size
            stop = min(start + shard_size, len(cohort_cases))
            ids = cohort_cases.iloc[start:stop]["eval_row_id"].astype(str).tolist()
            shards.append(
                {
                    "case_shard_id": f"{cohort_scope}_case_shard_{index:04d}",
                    "cohort_scope": cohort_scope,
                    "case_start": start,
                    "case_stop_exclusive": stop,
                    "case_count": len(ids),
                    "case_ids_sha256": hashlib.sha256(
                        "\n".join(ids).encode("utf-8")
                    ).hexdigest(),
                }
            )
    outputs: dict[str, Any] = {}
    for method in METHODS:
        rows = []
        for source in candidates[method]:
            for checkpoint in sorted(inventory[source], key=lambda row: row["member_id"]):
                for shard in shards:
                    rows.append(
                        {
                            "task_id": (
                                f"{method}__{source}__{checkpoint['member_id']}__"
                                f"{shard['case_shard_id']}"
                            ),
                            "method": method,
                            "source_candidate_id": source,
                            "role_aliases": ",".join(aliases[source]),
                            "member_id": checkpoint["member_id"],
                            "checkpoint_path": checkpoint["checkpoint_path"],
                            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                            **shard,
                            **pins,
                            "classification": "confirmatory_internal_or_exploratory_tcga_by_case_flag",
                        }
                    )
        if rows:
            output = output_dir / f"round19g_{method}_task_manifest.csv"
            _write_csv(output, rows)
            outputs[method] = {
                "path": str(output),
                "sha256": sha256_file(output),
                "tasks": len(rows),
                "candidates": len(candidates[method]),
                "members_per_candidate": 15,
                "case_shards": len(shards),
            }
    summary = {
        "artifact_type": "round19_stage19g_task_manifest_summary",
        "schema_version": 1,
        "shard_size": shard_size,
        "case_count": len(cases),
        "pins": pins,
        "manifests": outputs,
    }
    summary["summary_payload_sha256"] = canonical_sha256(summary)
    summary_path = output_dir / "round19g_task_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def finalize_case_manifest(selected_path: Path, output_path: Path) -> dict[str, Any]:
    cases = validate_case_manifest(pd.read_csv(selected_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases.sort_values(["selection_reason", "eval_row_id"], kind="mergesort").to_csv(
        output_path, index=False
    )
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "rows": len(cases),
        "reason_counts": cases["selection_reason"].value_counts().sort_index().to_dict(),
        "tcga_exploratory_rows": int(cases["is_tcga_exploratory"].sum()),
    }


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Finalize 19G cases and build task manifests")
    parser.add_argument("--project-root", default=str(root_default))
    parser.add_argument("--selected-cases", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument(
        "--config", default="config/round19_stage19g_interpretability.json"
    )
    parser.add_argument(
        "--case-selection-config",
        default="config/round19_stage19g_case_selection.json",
    )
    parser.add_argument(
        "--final-lock",
        default="result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json",
    )
    parser.add_argument(
        "--task-output-dir",
        default="result/optimization_runs/round19_factorial/manifests/round19g",
    )
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()

    def rooted(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    case_summary = finalize_case_manifest(
        rooted(args.selected_cases), rooted(args.case_manifest)
    )
    task_summary = build_task_manifests(
        case_manifest_path=rooted(args.case_manifest),
        config_path=rooted(args.config),
        final_lock_path=rooted(args.final_lock),
        project_root=project_root,
        output_dir=rooted(args.task_output_dir),
        case_selection_config_path=rooted(args.case_selection_config),
    )
    print(json.dumps({"case_manifest": case_summary, "tasks": task_summary}, indent=2))


if __name__ == "__main__":
    main()
