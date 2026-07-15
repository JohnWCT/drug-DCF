#!/usr/bin/env python3
"""Build the Round 19F proposal-only checkpoint inventory.

This stage deliberately does not create internal-test or TCGA inference
manifests: proposal roles have not yet been locked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Union

import pandas as pd

from tools.round19_stage19f_ensemble import (
    N_REQUIRED_MEMBERS,
    REQUIRED_FOLDS,
    REQUIRED_SEEDS,
    member_id,
)


PROPOSAL_ROLE_NAMES = (
    "historical_anchor",
    "source_performance_champion",
    "parsimonious_context_model",
    "cancer_shift_specialist",
    "chemical_shift_specialist",
    "source_only_domain_candidate",
    "efficient_model",
    "general_recommended_model",
)


def _load_frame(value: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.read_csv(value)


def _load_roles(value: object) -> object:
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text(encoding="utf-8"))
    return value


def _normalise_roles(proposal_roles: object) -> dict[str, dict]:
    payload = _load_roles(proposal_roles)
    if isinstance(payload, Mapping):
        for key in ("proposal_roles", "roles"):
            if key in payload:
                payload = payload[key]
                break

    normalised: dict[str, dict] = {}
    if isinstance(payload, Mapping):
        for role, value in payload.items():
            record = dict(value) if isinstance(value, Mapping) else {"candidate_id": value}
            record.setdefault("role_name", str(role))
            normalised[str(role)] = record
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, Mapping):
                raise TypeError("proposal role list entries must be mappings")
            record = dict(item)
            role = record.get("role_name") or record.get("role") or record.get("name")
            if not role:
                raise KeyError("proposal role entry missing role_name")
            if str(role) in normalised:
                raise AssertionError(f"duplicate proposal role: {role}")
            normalised[str(role)] = record
    else:
        raise TypeError("proposal_roles must be a mapping, list, or JSON path")

    missing = set(PROPOSAL_ROLE_NAMES) - set(normalised)
    if missing:
        raise AssertionError(f"missing proposal roles: {sorted(missing)}")
    # Null roles are valid (e.g. no universal general model), but do not create
    # checkpoint inventory rows.
    return {
        role: normalised[role]
        for role in PROPOSAL_ROLE_NAMES
        if normalised[role].get("candidate_id") is not None
    }


def _resolve_source_candidate(
    record: Mapping[str, object], available: set[str]
) -> str:
    explicit = record.get("source_candidate_id")
    if explicit:
        source = str(explicit)
    else:
        candidate = (
            record.get("candidate_id")
            or record.get("proposal_candidate_id")
            or record.get("selected_candidate_id")
            or record.get("stage19e_candidate_id")
        )
        if not candidate:
            raise KeyError("proposal role missing candidate_id/source_candidate_id")
        candidate = str(candidate)
        if candidate in available:
            source = candidate
        elif candidate.startswith("E") and candidate[1:].split("_", 1)[0].isdigit():
            source = "F" + candidate[1:].split("_", 1)[0]
        else:
            source = candidate

    if source in available:
        return source
    matches = sorted(
        candidate
        for candidate in available
        if candidate.startswith(source + "_")
        or source.startswith(candidate + "_")
    )
    if len(matches) != 1:
        raise KeyError(
            f"cannot uniquely map proposal candidate {source!r} to stage19d source; "
            f"matches={matches}"
        )
    return matches[0]


def build_checkpoint_inventory(
    stage19d_manifest: Union[str, Path, pd.DataFrame],
    proposal_roles: object,
    *,
    output_path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """Return one row per unique source candidate × 15 checkpoint members."""
    manifest = _load_frame(stage19d_manifest)
    required = {"candidate_id", "split_seed", "fold_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"stage19d manifest missing columns: {sorted(missing)}")
    if "checkpoint_path" not in manifest and "result_dir" not in manifest:
        raise KeyError("stage19d manifest requires checkpoint_path or result_dir")

    manifest = manifest.copy()
    manifest["candidate_id"] = manifest["candidate_id"].astype(str)
    available = set(manifest["candidate_id"])
    roles = _normalise_roles(proposal_roles)
    source_to_roles: dict[str, list[str]] = {}
    for role, record in roles.items():
        source = _resolve_source_candidate(record, available)
        source_to_roles.setdefault(source, []).append(role)

    rows = []
    expected_pairs = {
        (seed, fold) for seed in REQUIRED_SEEDS for fold in REQUIRED_FOLDS
    }
    for source_candidate_id, role_names in source_to_roles.items():
        source_rows = manifest[manifest["candidate_id"] == source_candidate_id].copy()
        source_rows["split_seed"] = source_rows["split_seed"].astype(int)
        source_rows["fold_id"] = source_rows["fold_id"].astype(int)
        pairs = list(zip(source_rows["split_seed"], source_rows["fold_id"]))
        if len(source_rows) != N_REQUIRED_MEMBERS or set(pairs) != expected_pairs:
            raise AssertionError(
                f"{source_candidate_id} requires exactly 15 stage19d checkpoints "
                f"for seeds {REQUIRED_SEEDS} × folds {REQUIRED_FOLDS}; got {len(source_rows)}"
            )
        if len(pairs) != len(set(pairs)):
            raise AssertionError(f"{source_candidate_id} has duplicate seed/fold checkpoints")

        identity_cols = [
            column
            for column in ("drug_id", "predictor_id", "omics_id")
            if column in source_rows
        ]
        for column in identity_cols:
            if source_rows[column].nunique(dropna=False) != 1:
                raise AssertionError(
                    f"{source_candidate_id} has inconsistent {column} metadata"
                )

        for _, row in source_rows.sort_values(["split_seed", "fold_id"]).iterrows():
            checkpoint_path = (
                str(row["checkpoint_path"])
                if "checkpoint_path" in source_rows and pd.notna(row["checkpoint_path"])
                else str(Path(str(row["result_dir"])) / "checkpoint.pt")
            )
            output = {
                "candidate_id": source_candidate_id,
                "source_candidate_id": source_candidate_id,
                "role_names": ",".join(sorted(role_names)),
                "split_seed": int(row["split_seed"]),
                "fold_id": int(row["fold_id"]),
                "member_id": member_id(row["split_seed"], row["fold_id"]),
                "checkpoint_path": checkpoint_path,
            }
            for column in identity_cols:
                output[column] = row[column]
            rows.append(output)

    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise AssertionError("proposal roles produced an empty checkpoint inventory")
    if inventory.duplicated(["source_candidate_id", "member_id"]).any():
        raise AssertionError("source candidate/member pairs must be unique")
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        inventory.to_csv(path, index=False)
    return inventory


build_stage19f_manifest = build_checkpoint_inventory
build_stage19f_checkpoint_inventory = build_checkpoint_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19F checkpoint inventory")
    parser.add_argument("--stage19d-manifest", required=True)
    parser.add_argument("--proposal-roles", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inventory = build_checkpoint_inventory(
        args.stage19d_manifest, args.proposal_roles, output_path=args.output
    )
    print(
        json.dumps(
            {
                "written": args.output,
                "n_unique_source_candidates": int(
                    inventory["source_candidate_id"].nunique()
                ),
                "n_checkpoints": int(len(inventory)),
                "proposal_only": True,
                "internal_manifest_created": False,
                "tcga_manifest_created": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
