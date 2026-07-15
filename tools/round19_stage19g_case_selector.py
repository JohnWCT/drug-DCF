#!/usr/bin/env python3
"""Build the Round 19G metadata spine and deterministic case selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

SELECTION_SEED = 19091
SHIFTS = ("drug_heldout", "scaffold_heldout", "cancer_type_heldout")
SELECTION_REASONS = {
    "representative_stratified_random",
    "contrastive_cancer_gain",
    "contrastive_cancer_loss",
    "contrastive_chemical_gain",
    "contrastive_chemical_loss",
    "contrastive_role_disagreement",
    "patient_conditioned",
    "tcga_exploratory",
}
PREDICTION_COLUMNS = {"logit", "probability", "probability_std"}
TCGA_SOURCES = {
    "gdsc_intersect13": "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv",
    "tcga_only3": "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv",
    "dapl": "data/TCGA/TCGA_drug_response_from_DAPL.csv",
    "aacdr_tcga_only": "data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv",
    "aacdr_gdsc_intersect": "data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv",
}
CASE_COLUMNS = (
    "case_id",
    "eval_row_id",
    "candidate_role",
    "candidate_id",
    "source_stage",
    "source_target",
    "source_row_id",
    "shift_strategy",
    "ModelID",
    "Patient_id",
    "DRUG_NAME",
    "Label",
    "cancer_type",
    "normalized_drug_id",
    "drug_id",
    "scaffold_id",
    "canonical_smiles",
    "graph_smiles",
    "graph_smiles_metadata_status",
    "selection_reason",
    "selection_detail",
    "posthoc_case",
    "selection_eligible",
    "is_posthoc_contrastive",
    "is_tcga_exploratory",
    "prediction_used_for_selection",
    "selection_seed",
    "available_drug_unique_modelids",
    "available_drug_cancer_types",
    "available_drug_labels",
)


class PatientCohortUnavailable(AssertionError):
    """Raised when the declared patient cohort semantics cannot be met."""


def _normal(value: object) -> str:
    return str(value).strip().casefold()


def _hash_order(value: object, *, seed: int, namespace: str) -> str:
    return hashlib.sha256(f"{seed}|{namespace}|{value}".encode("utf-8")).hexdigest()


def deterministic_take(
    frame: pd.DataFrame,
    n: int,
    *,
    seed: int,
    namespace: str,
    exclude: Iterable[str] = (),
) -> pd.DataFrame:
    """Select by identity hash only; prediction values never affect ordering."""
    blocked = set(map(str, exclude))
    pool = frame[~frame["eval_row_id"].astype(str).isin(blocked)].copy()
    pool["_selection_order"] = pool["eval_row_id"].map(
        lambda value: _hash_order(value, seed=seed, namespace=namespace)
    )
    return (
        pool.sort_values(["_selection_order", "eval_row_id"], kind="mergesort")
        .head(max(0, n))
        .drop(columns="_selection_order")
    )


def _read_maps(
    round_root: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    splits = round_root / "splits"
    cancers = pd.read_csv(splits / "round19e_modelid_cancer_type_map.csv")
    drugs = pd.read_csv(splits / "round19e_drug_group_table.csv")
    scaffolds = pd.read_csv(splits / "round19e_scaffold_group_table.csv")
    return (
        cancers.set_index("ModelID")["cancer_type"].astype(str).to_dict(),
        drugs.set_index("DRUG_NAME")["normalized_drug_id"].astype(str).to_dict(),
        scaffolds.set_index("DRUG_NAME")["scaffold_id"].astype(str).to_dict(),
        drugs.set_index("DRUG_NAME")["canonical_smiles"].astype(str).to_dict(),
    )


def _enrich(
    frame: pd.DataFrame,
    *,
    cancer_map: Mapping[str, str],
    drug_map: Mapping[str, str],
    scaffold_map: Mapping[str, str],
    smiles_map: Mapping[str, str],
) -> pd.DataFrame:
    out = frame.copy()
    out["ModelID"] = out["ModelID"].astype(str)
    out["DRUG_NAME"] = out["DRUG_NAME"].astype(str)
    out["Label"] = pd.to_numeric(out["Label"], errors="raise").astype(int)
    if "Patient_id" not in out:
        out["Patient_id"] = ""
    out["cancer_type"] = out.get("cancer_type", pd.Series(index=out.index, dtype=object))
    out["cancer_type"] = out["cancer_type"].fillna(out["ModelID"].map(cancer_map))
    out["normalized_drug_id"] = out["DRUG_NAME"].map(drug_map)
    out["normalized_drug_id"] = out["normalized_drug_id"].fillna(
        out["DRUG_NAME"].map(_normal)
    )
    out["scaffold_id"] = out["DRUG_NAME"].map(scaffold_map).fillna("")
    mapped_smiles = out["DRUG_NAME"].map(smiles_map)
    existing_smiles = out.get(
        "canonical_smiles", pd.Series(index=out.index, dtype=object)
    ).replace("", pd.NA)
    out["canonical_smiles"] = existing_smiles.fillna(mapped_smiles).fillna("")
    out["graph_smiles"] = out["canonical_smiles"]
    out["graph_smiles_metadata_status"] = "legacy_graph_resolution_required"
    out["drug_id"] = out["normalized_drug_id"]
    return out


def load_stage19e(
    round_root: Path,
    candidate_id: str,
    *,
    cancer_map: Mapping[str, str],
    drug_map: Mapping[str, str],
    scaffold_map: Mapping[str, str],
    smiles_map: Mapping[str, str],
) -> pd.DataFrame:
    rows = []
    for shift in SHIFTS:
        paths = sorted((round_root / "stage19e" / shift).glob(f"{candidate_id}__*/*val_predictions.csv"))
        if not paths:
            raise FileNotFoundError(f"No {candidate_id} predictions for {shift}")
        for path in paths:
            part = pd.read_csv(path)
            required = {"_row_id", "ModelID", "DRUG_NAME", "Label", "probability"}
            if required - set(part):
                raise KeyError(f"{path} missing {sorted(required - set(part))}")
            part["shift_strategy"] = shift
            rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    if out.duplicated(["shift_strategy", "_row_id"]).any():
        raise AssertionError(f"{candidate_id} has duplicate shift × _row_id predictions")
    out["eval_row_id"] = (
        "19e:" + out["shift_strategy"].astype(str) + ":" + out["_row_id"].astype(str)
    )
    out["source_stage"] = "19E"
    out["source_target"] = out["shift_strategy"]
    out["source_row_id"] = out["_row_id"].astype(str)
    return _enrich(
        out,
        cancer_map=cancer_map,
        drug_map=drug_map,
        scaffold_map=scaffold_map,
        smiles_map=smiles_map,
    )


def _tcga_metadata(project_root: Path) -> pd.DataFrame:
    rows = []
    for target, relative in TCGA_SOURCES.items():
        source = pd.read_csv(project_root / relative).reset_index(drop=True)
        patient_col = "Patient_id" if "Patient_id" in source else "patient"
        drug_col = "drug_name" if "drug_name" in source else "DRUG_NAME"
        cancer_col = next(
            (name for name in ("cancer_type", "cancers", "primary_disease") if name in source),
            None,
        )
        required = {patient_col, drug_col, "Label"}
        if required - set(source):
            raise KeyError(f"{relative} missing patient/drug/Label fields")
        part = pd.DataFrame(
            {
                "source_target": target,
                "source_row_id": source.index.astype(str),
                "ModelID": source[patient_col].astype(str),
                "Patient_id": source[patient_col].astype(str),
                "DRUG_NAME": source[drug_col].astype(str),
                "Label": source["Label"],
                "cancer_type": source[cancer_col].astype(str) if cancer_col else "",
                "canonical_smiles": (
                    source["smiles"].astype(str) if "smiles" in source else ""
                ),
            }
        )
        part["eval_row_id"] = (
            "19f:"
            + target
            + "|"
            + part["Patient_id"]
            + "|"
            + part["DRUG_NAME"]
            + "|"
            + part["Label"].astype(str)
            + "|"
            + part["source_row_id"]
        )
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def load_stage19f_spine(
    ensemble_path: Path,
    *,
    project_root: Path,
    cancer_map: Mapping[str, str],
    drug_map: Mapping[str, str],
    scaffold_map: Mapping[str, str],
    smiles_map: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(ensemble_path)
    required = {"candidate_id", "eval_row_id", "probability", "Label", "target_key"}
    if required - set(predictions):
        raise KeyError(f"19F ensemble missing {sorted(required - set(predictions))}")
    predictions = predictions.copy()
    predictions["eval_row_id"] = "19f:" + predictions["eval_row_id"].astype(str)
    metadata = (
        predictions.sort_values(["candidate_id", "eval_row_id"])
        .drop_duplicates("eval_row_id")
        .copy()
    )
    metadata["source_stage"] = "19F"
    metadata["source_target"] = metadata["target_key"].astype(str)
    metadata["source_row_id"] = metadata["eval_row_id"].astype(str)
    metadata["shift_strategy"] = ""
    metadata["Patient_id"] = metadata.get("Patient_id", metadata["ModelID"]).fillna("")
    if "cancer_type" not in metadata:
        metadata["cancer_type"] = ""
    if "canonical_smiles" not in metadata:
        metadata["canonical_smiles"] = ""
    tcga_meta = _tcga_metadata(project_root)
    tcga_lookup = tcga_meta.set_index("eval_row_id")
    tcga_mask = metadata["target_key"].astype(str) != "internal_test"
    for column in ("Patient_id", "cancer_type", "source_row_id", "canonical_smiles"):
        mapped = metadata.loc[tcga_mask, "eval_row_id"].map(tcga_lookup[column])
        metadata.loc[tcga_mask, column] = mapped.fillna(metadata.loc[tcga_mask, column])
    metadata = _enrich(
        metadata,
        cancer_map=cancer_map,
        drug_map=drug_map,
        scaffold_map=scaffold_map,
        smiles_map=smiles_map,
    )
    return metadata, predictions


def build_metadata_spine(
    *,
    round_root: Path,
    project_root: Path,
    ensemble_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cancer_map, drug_map, scaffold_map, smiles_map = _read_maps(round_root)
    e1 = load_stage19e(
        round_root,
        "E1",
        cancer_map=cancer_map,
        drug_map=drug_map,
        scaffold_map=scaffold_map,
        smiles_map=smiles_map,
    )
    e3 = load_stage19e(
        round_root,
        "E3",
        cancer_map=cancer_map,
        drug_map=drug_map,
        scaffold_map=scaffold_map,
        smiles_map=smiles_map,
    )
    stage19f, predictions19f = load_stage19f_spine(
        ensemble_path,
        project_root=project_root,
        cancer_map=cancer_map,
        drug_map=drug_map,
        scaffold_map=scaffold_map,
        smiles_map=smiles_map,
    )
    spine = pd.concat(
        [
            e1.drop(columns=list(PREDICTION_COLUMNS & set(e1)), errors="ignore"),
            stage19f.drop(columns=list(PREDICTION_COLUMNS & set(stage19f)), errors="ignore"),
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates("eval_row_id")
    return spine, e1, e3, predictions19f


def _mark(
    frame: pd.DataFrame,
    reason: str,
    *,
    detail: str,
    prediction_used: bool,
) -> pd.DataFrame:
    if reason not in SELECTION_REASONS:
        raise ValueError(f"Unknown selection_reason: {reason}")
    out = frame.copy()
    out["case_id"] = out["eval_row_id"].astype(str)
    out["selection_reason"] = reason
    out["selection_detail"] = detail
    out["is_posthoc_contrastive"] = reason.startswith("contrastive_")
    out["is_tcga_exploratory"] = reason == "tcga_exploratory"
    out["posthoc_case"] = out["is_posthoc_contrastive"] | out["is_tcga_exploratory"]
    out["selection_eligible"] = ~out["is_tcga_exploratory"]
    out["candidate_role"] = "all_locked_roles"
    out["candidate_id"] = "all_locked_sources"
    if reason.startswith("contrastive_"):
        out["candidate_role"] = "posthoc_contrastive_pair"
        out["candidate_id"] = "E1_vs_E3"
    out["prediction_used_for_selection"] = bool(prediction_used)
    out["selection_seed"] = SELECTION_SEED
    for column in (
        "available_drug_unique_modelids",
        "available_drug_cancer_types",
        "available_drug_labels",
    ):
        if column not in out:
            out[column] = ""
    return out


def _representative(e1: pd.DataFrame, per_stratum: int) -> pd.DataFrame:
    selected = []
    used: set[str] = set()
    for shift in SHIFTS:
        for label in (0, 1):
            group = e1[(e1["shift_strategy"] == shift) & (e1["Label"] == label)]
            take = deterministic_take(
                group,
                per_stratum,
                seed=SELECTION_SEED,
                namespace=f"representative:{shift}:{label}",
                exclude=used,
            )
            used.update(take["eval_row_id"].astype(str))
            selected.append(
                _mark(
                    take,
                    "representative_stratified_random",
                    detail=f"primary:{shift}:label={label}",
                    prediction_used=False,
                )
            )
    target = min(per_stratum * len(SHIFTS) * 2, len(e1))
    current = pd.concat(selected, ignore_index=True) if selected else e1.iloc[0:0]
    if len(current) < target:
        fallback = deterministic_take(
            e1,
            target - len(current),
            seed=SELECTION_SEED,
            namespace="representative:deterministic_fallback",
            exclude=used,
        )
        current = pd.concat(
            [
                current,
                _mark(
                    fallback,
                    "representative_stratified_random",
                    detail="deterministic_fallback_for_unavailable_strata",
                    prediction_used=False,
                ),
            ],
            ignore_index=True,
        )
    return current


def _correct(frame: pd.DataFrame, suffix: str) -> pd.Series:
    return (frame[f"probability_{suffix}"] >= 0.5).astype(int) == frame["Label"].astype(int)


def _contrastive_e1_e3(
    e1: pd.DataFrame, e3: pd.DataFrame, per_direction: int, exclude: set[str]
) -> list[pd.DataFrame]:
    keys = ["eval_row_id"]
    merged = e1.merge(
        e3[["eval_row_id", "probability"]],
        on=keys,
        suffixes=("_e1", "_e3"),
        validate="one_to_one",
    )
    merged["e1_correct"] = _correct(merged, "e1")
    merged["e3_correct"] = _correct(merged, "e3")
    outputs = []
    domains: Sequence[tuple[str, Sequence[str]]] = (
        ("cancer", ("cancer_type_heldout",)),
        ("chemical", ("drug_heldout", "scaffold_heldout")),
    )
    for domain, shifts in domains:
        domain_rows = merged[merged["shift_strategy"].isin(shifts)]
        for direction, mask in (
            ("gain", domain_rows["e1_correct"] & ~domain_rows["e3_correct"]),
            ("loss", ~domain_rows["e1_correct"] & domain_rows["e3_correct"]),
        ):
            reason = f"contrastive_{domain}_{direction}"
            take = deterministic_take(
                domain_rows[mask],
                per_direction,
                seed=SELECTION_SEED,
                namespace=reason,
                exclude=exclude,
            )
            exclude.update(take["eval_row_id"].astype(str))
            outputs.append(
                _mark(
                    take,
                    reason,
                    detail="post_hoc:E1_vs_E3:correctness_discordance",
                    prediction_used=True,
                )
            )
    return outputs


def _role_disagreement(
    spine: pd.DataFrame,
    predictions: pd.DataFrame,
    final_lock: Mapping[str, Any],
    n: int,
    exclude: set[str],
) -> pd.DataFrame:
    roles = final_lock["roles"]
    sources = {
        str(roles["cancer_shift_specialist"]["source_candidate_id"]),
        str(roles["chemical_shift_specialist"]["source_candidate_id"]),
    }
    available = set(predictions["candidate_id"].astype(str))
    resolved = []
    for source in sorted(sources):
        matches = sorted(
            value
            for value in available
            if value == source or value.startswith(source + "_") or source.startswith(value + "_")
        )
        if len(matches) != 1:
            raise AssertionError(f"Cannot resolve locked role source {source}: {matches}")
        resolved.append(matches[0])
    internal = predictions[
        (predictions["target_key"].astype(str) == "internal_test")
        & predictions["candidate_id"].astype(str).isin(resolved)
    ]
    pivot = internal.pivot(index="eval_row_id", columns="candidate_id", values="probability")
    eligible_ids = pivot.index[(pivot[resolved[0]] >= 0.5) != (pivot[resolved[1]] >= 0.5)]
    eligible = spine[spine["eval_row_id"].astype(str).isin(eligible_ids.astype(str))]
    take = deterministic_take(
        eligible,
        n,
        seed=SELECTION_SEED,
        namespace="contrastive_role_disagreement",
        exclude=exclude,
    )
    exclude.update(take["eval_row_id"].astype(str))
    return _mark(
        take,
        "contrastive_role_disagreement",
        detail=f"post_hoc:locked_roles:{resolved[0]}_vs_{resolved[1]}",
        prediction_used=True,
    )


def _patient_conditioned(
    development: pd.DataFrame,
    *,
    minimum_samples: int,
    minimum_cancer_types: int,
    per_drug_cap: int,
    maximum_drugs: int,
    exclude: set[str],
) -> pd.DataFrame:
    pool = development[development["shift_strategy"] == "drug_heldout"].copy()
    if pool.duplicated("eval_row_id").any():
        raise AssertionError("Development patient-conditioned pool has duplicate rows")
    availability = []
    eligible: dict[str, pd.DataFrame] = {}
    for drug, group in pool.groupby("normalized_drug_id", sort=True):
        evidence = {
            "drug_id": str(drug),
            "available_unique_modelids": int(group["ModelID"].nunique()),
            "available_cancer_types": int(group["cancer_type"].nunique()),
            "available_labels": sorted(group["Label"].astype(int).unique().tolist()),
        }
        availability.append(evidence)
        if (
            evidence["available_unique_modelids"] >= minimum_samples
            and evidence["available_cancer_types"] >= minimum_cancer_types
            and set(evidence["available_labels"]) == {0, 1}
        ):
            eligible[str(drug)] = group
    drug_order = sorted(
        eligible,
        key=lambda drug: (_hash_order(drug, seed=SELECTION_SEED, namespace="patient:drug"), drug),
    )
    selected_groups = []
    for drug in drug_order:
        available = eligible[drug]
        remaining = available[
            ~available["eval_row_id"].astype(str).isin(exclude)
        ].copy()
        ordered = deterministic_take(
            remaining,
            len(remaining),
            seed=SELECTION_SEED,
            namespace=f"patient:rows:{drug}",
        )
        chosen_ids: set[str] = set()
        chosen_rows = []

        def add_first(candidates: pd.DataFrame) -> None:
            for _, row in candidates.iterrows():
                row_id = str(row["eval_row_id"])
                if row_id not in chosen_ids and len(chosen_rows) < per_drug_cap:
                    chosen_ids.add(row_id)
                    chosen_rows.append(row)
                    return

        cancer_order = sorted(
            ordered["cancer_type"].dropna().astype(str).unique(),
            key=lambda value: (
                _hash_order(value, seed=SELECTION_SEED, namespace=f"patient:cancer:{drug}"),
                value,
            ),
        )
        for cancer in cancer_order[:minimum_cancer_types]:
            add_first(ordered[ordered["cancer_type"].astype(str) == cancer])
        for label in (0, 1):
            add_first(ordered[ordered["Label"].astype(int) == label])
        for _, row in ordered.iterrows():
            if len(chosen_rows) >= per_drug_cap:
                break
            row_id = str(row["eval_row_id"])
            if row_id not in chosen_ids:
                chosen_ids.add(row_id)
                chosen_rows.append(row)
        chosen = pd.DataFrame(chosen_rows)
        if (
            len(chosen) <= per_drug_cap
            and chosen["ModelID"].nunique() >= minimum_samples
            and chosen["cancer_type"].nunique() >= minimum_cancer_types
            and set(chosen["Label"].astype(int).unique()) == {0, 1}
        ):
            chosen["available_drug_unique_modelids"] = int(available["ModelID"].nunique())
            chosen["available_drug_cancer_types"] = int(available["cancer_type"].nunique())
            chosen["available_drug_labels"] = ",".join(
                map(str, sorted(available["Label"].astype(int).unique()))
            )
            selected_groups.append(chosen)
            exclude.update(chosen["eval_row_id"].astype(str))
            if len(selected_groups) >= maximum_drugs:
                break
    if not selected_groups:
        raise PatientCohortUnavailable(
            "No development drug satisfies per-drug patient-conditioned constraints: "
            + json.dumps(
                {
                    "minimum_unique_modelids_per_drug": minimum_samples,
                    "minimum_cancer_types_per_drug": minimum_cancer_types,
                    "required_labels": [0, 1],
                    "per_drug_cap": per_drug_cap,
                    "available_drugs": availability,
                },
                sort_keys=True,
            )
        )
    return _mark(
        pd.concat(selected_groups, ignore_index=True),
        "patient_conditioned",
        detail="development_19E_same_drug_multiple_ModelID_omics_samples",
        prediction_used=False,
    )


def _tcga_exploratory(
    spine: pd.DataFrame, *, target_cases: int, exclude: set[str]
) -> pd.DataFrame:
    tcga = spine[
        (spine["source_stage"] == "19F")
        & (spine["source_target"] != "internal_test")
    ].copy()
    selected = []
    targets = sorted(tcga["source_target"].astype(str).unique())
    base, remainder = divmod(target_cases, len(targets))
    for index, target in enumerate(targets):
        take = deterministic_take(
            tcga[tcga["source_target"].astype(str) == target],
            base + (1 if index < remainder else 0),
            seed=SELECTION_SEED,
            namespace=f"tcga_exploratory:{target}",
            exclude=exclude,
        )
        exclude.update(take["eval_row_id"].astype(str))
        selected.append(take)
    cohort = pd.concat(selected, ignore_index=True)
    if len(cohort) != target_cases:
        raise AssertionError(
            f"TCGA exploratory target unavailable: requested={target_cases} selected={len(cohort)}"
        )
    return _mark(
        cohort,
        "tcga_exploratory",
        detail="post_hoc_exploratory_external_benchmark;excluded_from_primary_faithfulness",
        prediction_used=False,
    )


def select_cases(
    *,
    spine: pd.DataFrame,
    e1: pd.DataFrame,
    e3: pd.DataFrame,
    predictions19f: pd.DataFrame,
    final_lock: Mapping[str, Any],
    representative_per_stratum: int = 20,
    contrastive_per_direction: int = 10,
    disagreement_cases: int = 20,
    patient_minimum_samples: int = 20,
    patient_minimum_cancer_types: int = 3,
    patient_per_drug_cap: int = 30,
    patient_maximum_drugs: int = 1,
    tcga_exploratory_cases: int = 20,
) -> pd.DataFrame:
    representative = _representative(e1, representative_per_stratum)
    used = set(representative["eval_row_id"].astype(str))
    parts = [representative]
    parts.extend(
        _contrastive_e1_e3(e1, e3, contrastive_per_direction, used)
    )
    parts.append(
        _role_disagreement(spine, predictions19f, final_lock, disagreement_cases, used)
    )
    parts.append(
        _patient_conditioned(
            e1,
            minimum_samples=patient_minimum_samples,
            minimum_cancer_types=patient_minimum_cancer_types,
            per_drug_cap=patient_per_drug_cap,
            maximum_drugs=patient_maximum_drugs,
            exclude=used,
        )
    )
    parts.append(
        _tcga_exploratory(
            spine,
            target_cases=tcga_exploratory_cases,
            exclude=used,
        )
    )
    selected = pd.concat(parts, ignore_index=True, sort=False)
    if selected["eval_row_id"].duplicated().any():
        raise AssertionError("Selected eval_row_id values must be unique")
    if not set(selected["selection_reason"]).issubset(SELECTION_REASONS):
        raise AssertionError("Selection reason enum drift")
    if not 150 <= len(selected) <= 250:
        raise AssertionError(f"Round 19G total case count outside 150-250: {len(selected)}")
    return selected.reindex(columns=CASE_COLUMNS)


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Select deterministic Round 19G cases")
    parser.add_argument("--project-root", default=str(root_default))
    parser.add_argument(
        "--round-root", default="result/optimization_runs/round19_factorial"
    )
    parser.add_argument(
        "--ensemble-predictions",
        default=(
            "result/optimization_runs/round19_factorial/reports/"
            "round19_stage19f_posthoc/round19f_15member_ensemble_predictions.csv"
        ),
    )
    parser.add_argument(
        "--final-lock",
        default="result/optimization_runs/round19_factorial/reports/round19_final_role_lock.json",
    )
    parser.add_argument("--metadata-spine-output", required=True)
    parser.add_argument("--selected-output", required=True)
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()

    def rooted(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    spine, e1, e3, predictions = build_metadata_spine(
        round_root=rooted(args.round_root),
        project_root=project_root,
        ensemble_path=rooted(args.ensemble_predictions),
    )
    lock = json.loads(rooted(args.final_lock).read_text(encoding="utf-8"))
    selected = select_cases(
        spine=spine,
        e1=e1,
        e3=e3,
        predictions19f=predictions,
        final_lock=lock,
    )
    spine_output = rooted(args.metadata_spine_output)
    selected_output = rooted(args.selected_output)
    spine_output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    spine.to_csv(spine_output, index=False)
    selected.to_csv(selected_output, index=False)
    print(
        json.dumps(
            {
                "metadata_spine_rows": len(spine),
                "selected_rows": len(selected),
                "selection_reason_counts": selected["selection_reason"].value_counts().to_dict(),
                "tcga_exploratory_rows": int(selected["is_tcga_exploratory"].sum()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
