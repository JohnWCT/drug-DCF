#!/usr/bin/env python3
"""Exploratory-only Round 19F post-hoc aggregation, analysis, and reporting."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from tools.analyze_round18_external_eval import paired_bootstrap_delta
from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics
from tools.round19_stage19f_ensemble import (
    N_REQUIRED_MEMBERS,
    REQUIRED_MEMBER_IDS,
    ensemble_predictions,
)
from tools.round19_stage19f_final_lock import REQUIRED_ROLES, verify_final_lock


CLASSIFICATION = "exploratory_post_hoc"
INTERNAL_TARGET = "internal_test"
TCGA_TARGETS = (
    "gdsc_intersect13",
    "tcga_only3",
    "dapl",
    "aacdr_tcga_only",
    "aacdr_gdsc_intersect",
)
METRICS = ("DrugMacro_AUC", "DrugMacro_AUPRC", "Global_AUC", "Global_AUPRC")
PREDICTION_FILENAMES = {"internal_test_predictions.csv", "tcga_predictions.csv"}
SCHEMA_VERSION = 1


def _read_json_object(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _locked_inventory(lock: Mapping[str, Any]) -> pd.DataFrame:
    try:
        items = lock["hashes"]["checkpoint_inventory"]
    except (KeyError, TypeError) as exc:
        raise AssertionError("Final lock has no checkpoint inventory") from exc
    if not isinstance(items, list) or not items:
        raise AssertionError("Final lock checkpoint inventory is empty")
    frame = pd.DataFrame(items)
    required = {
        "source_candidate_id",
        "member_id",
        "checkpoint_path",
        "checkpoint_sha256",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Lock checkpoint inventory missing columns: {sorted(missing)}")
    if frame.duplicated(["source_candidate_id", "member_id"]).any():
        raise AssertionError("Lock candidate/member identities are duplicated")
    if len(frame) != 90 or frame["source_candidate_id"].nunique() != 6:
        raise AssertionError("Final lock must contain six candidates and 90 checkpoints")
    expected_members = set(REQUIRED_MEMBER_IDS)
    for candidate, group in frame.groupby("source_candidate_id", sort=False):
        members = set(group["member_id"].astype(str))
        if len(group) != N_REQUIRED_MEMBERS or members != expected_members:
            raise AssertionError(
                f"Locked candidate {candidate} does not have the required 15-member grid"
            )
    return frame


def _resolve_locked_candidate(value: Any, candidates: Iterable[str]) -> Optional[str]:
    if value is None:
        return None
    candidate = str(value)
    available = set(str(item) for item in candidates)
    if candidate in available:
        return candidate
    matches = sorted(
        item
        for item in available
        if item.startswith(candidate + "_") or candidate.startswith(item + "_")
    )
    if len(matches) != 1:
        raise AssertionError(
            f"Role candidate {candidate!r} cannot be mapped uniquely to lock inventory; "
            f"matches={matches}"
        )
    return matches[0]


def _role_aliases(lock: Mapping[str, Any], candidates: Iterable[str]) -> pd.DataFrame:
    roles = lock.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != REQUIRED_ROLES:
        raise AssertionError("Final lock role schema is incomplete or unexpected")
    rows = []
    for role_name, record in roles.items():
        if not isinstance(record, Mapping):
            raise TypeError(f"Role {role_name} must be an object")
        source = record.get("source_candidate_id")
        if source is None:
            source = record.get("candidate_id")
        resolved = _resolve_locked_candidate(source, candidates) if source is not None else None
        rows.append(
            {
                "role_name": str(role_name),
                "locked_candidate_id": resolved,
                "role_status": "locked_alias" if resolved is not None else "locked_null",
                "classification": CLASSIFICATION,
            }
        )
    return pd.DataFrame(rows)


def _discover_predictions(prediction_root: Path, output_dir: Path) -> pd.DataFrame:
    if not prediction_root.is_dir():
        raise NotADirectoryError(prediction_root)
    paths = []
    output_resolved = output_dir.resolve()
    for path in sorted(prediction_root.rglob("*.csv")):
        if path.name not in PREDICTION_FILENAMES:
            continue
        try:
            path.resolve().relative_to(output_resolved)
            continue
        except ValueError:
            pass
        paths.append(path)
    if not paths:
        raise FileNotFoundError(
            f"No {sorted(PREDICTION_FILENAMES)} found below prediction root {prediction_root}"
        )
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            raise AssertionError(f"Prediction file is empty: {path}")
        frame["_prediction_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _validate_prediction_provenance(
    predictions: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    frame = predictions.copy()
    if "source_candidate_id" not in frame.columns:
        raise KeyError("Predictions require source_candidate_id")
    if "candidate_id" in frame.columns:
        mismatch = frame["candidate_id"].astype(str) != frame["source_candidate_id"].astype(str)
        if mismatch.any():
            raise AssertionError("candidate_id/source_candidate_id identity drift")
    frame["candidate_id"] = frame["source_candidate_id"].astype(str)
    required = {"candidate_id", "member_id", "checkpoint_path", "target_key"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Predictions missing provenance columns: {sorted(missing)}")

    lock_keys = inventory[
        ["source_candidate_id", "member_id", "checkpoint_path", "checkpoint_sha256"]
    ].rename(
        columns={
            "checkpoint_path": "_locked_checkpoint_path",
            "checkpoint_sha256": "_locked_checkpoint_sha256",
        }
    )
    checked = frame.merge(
        lock_keys,
        how="left",
        left_on=["candidate_id", "member_id"],
        right_on=["source_candidate_id", "member_id"],
        validate="many_to_one",
    )
    if checked["_locked_checkpoint_sha256"].isna().any():
        bad = checked.loc[
            checked["_locked_checkpoint_sha256"].isna(), ["candidate_id", "member_id"]
        ].iloc[0]
        raise AssertionError(f"Prediction identity is not in final lock: {bad.to_dict()}")
    path_drift = (
        checked["checkpoint_path"].astype(str)
        != checked["_locked_checkpoint_path"].astype(str)
    )
    if path_drift.any():
        bad = checked.loc[path_drift, ["candidate_id", "member_id"]].iloc[0]
        raise AssertionError(f"Prediction checkpoint path identity drift: {bad.to_dict()}")
    if "checkpoint_sha256" in checked:
        hash_drift = (
            checked["checkpoint_sha256"].astype(str)
            != checked["_locked_checkpoint_sha256"].astype(str)
        )
        if hash_drift.any():
            bad = checked.loc[hash_drift, ["candidate_id", "member_id"]].iloc[0]
            raise AssertionError(f"Prediction checkpoint identity drift: {bad.to_dict()}")
    checked = checked.drop(
        columns=[
            "source_candidate_id_y",
            "_locked_checkpoint_path",
            "_locked_checkpoint_sha256",
        ]
    ).rename(columns={"source_candidate_id_x": "source_candidate_id"})

    expected_candidates = set(inventory["source_candidate_id"].astype(str))
    observed_candidates = set(checked["candidate_id"].astype(str))
    if observed_candidates != expected_candidates:
        raise AssertionError(
            "Prediction candidate coverage differs from final lock: "
            f"missing={sorted(expected_candidates - observed_candidates)}, "
            f"extra={sorted(observed_candidates - expected_candidates)}"
        )
    expected_targets = {INTERNAL_TARGET, *TCGA_TARGETS}
    observed_targets = set(checked["target_key"].astype(str))
    if observed_targets != expected_targets:
        raise AssertionError(
            "Prediction target coverage must be internal_test plus the fixed five TCGA targets: "
            f"missing={sorted(expected_targets - observed_targets)}, "
            f"extra={sorted(observed_targets - expected_targets)}"
        )
    candidate_targets = checked.groupby("candidate_id")["target_key"].agg(
        lambda values: set(values.astype(str))
    )
    if not all(targets == expected_targets for targets in candidate_targets):
        raise AssertionError("Every locked candidate must cover internal and all five TCGA targets")
    candidates = sorted(expected_candidates)
    for target in sorted(expected_targets):
        row_sets = {
            candidate: set(
                checked.loc[
                    (checked["candidate_id"] == candidate)
                    & (checked["target_key"].astype(str) == target),
                    "eval_row_id",
                ].astype(str)
            )
            for candidate in candidates
        }
        if len({frozenset(values) for values in row_sets.values()}) != 1:
            raise AssertionError(
                f"Candidate eval-row coverage differs for target={target}"
            )
    return checked


def _metrics(predictions: pd.DataFrame) -> dict:
    values = calculate_robust_drug_macro_metrics(predictions)
    return {
        "DrugMacro_AUC": values["DrugMacro_AUC"],
        "DrugMacro_AUPRC": values["DrugMacro_AUPRC"],
        "Global_AUC": values["Global_AUC"],
        "Global_AUPRC": values["Global_AUPRC"],
        "n_valid_auc_drugs": int(values["n_valid_auc_drugs"]),
        "n_valid_auprc_drugs": int(values["n_valid_auprc_drugs"]),
        "n_total_drugs": int(values["n_total_drugs"]),
        "n_rows": int(len(predictions)),
        "classification": CLASSIFICATION,
    }


def _metric_tables(ensembled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for (candidate, target), group in ensembled.groupby(
        ["candidate_id", "target_key"], sort=True
    ):
        rows.append(
            {
                "candidate_id": candidate,
                "target_key": target,
                **_metrics(group),
            }
        )
    all_metrics = pd.DataFrame(rows)
    internal = all_metrics[all_metrics["target_key"] == INTERNAL_TARGET].reset_index(drop=True)
    tcga = all_metrics[all_metrics["target_key"].isin(TCGA_TARGETS)].reset_index(drop=True)
    if len(internal) != ensembled["candidate_id"].nunique():
        raise AssertionError("Internal per-candidate metrics are incomplete")
    expected_tcga_rows = ensembled["candidate_id"].nunique() * len(TCGA_TARGETS)
    if len(tcga) != expected_tcga_rows:
        raise AssertionError("TCGA per-candidate/per-target metrics are incomplete")

    integrated_rows = []
    for candidate, group in tcga.groupby("candidate_id", sort=True):
        if set(group["target_key"]) != set(TCGA_TARGETS):
            raise AssertionError(f"Integrated5 target drift for {candidate}")
        auc_complete = bool(group["DrugMacro_AUC"].notna().all())
        auprc_complete = bool(group["DrugMacro_AUPRC"].notna().all())
        integrated_rows.append(
            {
                "candidate_id": candidate,
                "Integrated5_n_targets": len(TCGA_TARGETS),
                "Integrated5_DrugMacro_TCGA_AUC": (
                    float(group["DrugMacro_AUC"].mean()) if auc_complete else None
                ),
                "Integrated5_DrugMacro_TCGA_AUPRC": (
                    float(group["DrugMacro_AUPRC"].mean()) if auprc_complete else None
                ),
                "Integrated5_target_weighting": "equal_target_mean",
                "Integrated5_status": (
                    "computed"
                    if auc_complete and auprc_complete
                    else "not_estimable_incomplete_target_metrics"
                ),
                "classification": CLASSIFICATION,
            }
        )
    return internal, tcga, pd.DataFrame(integrated_rows)


def _bootstrap_table(
    ensembled: pd.DataFrame,
    aliases: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    n_jobs: int = 0,
) -> pd.DataFrame:
    anchor_rows = aliases[aliases["role_name"] == "historical_anchor"]
    if len(anchor_rows) != 1 or pd.isna(anchor_rows.iloc[0]["locked_candidate_id"]):
        raise AssertionError("Final lock must contain a non-null historical_anchor")
    anchor = str(anchor_rows.iloc[0]["locked_candidate_id"])
    candidates = sorted(set(ensembled["candidate_id"]) - {anchor})
    tasks = []
    for target in (INTERNAL_TARGET, *TCGA_TARGETS):
        baseline = ensembled[
            (ensembled["candidate_id"] == anchor) & (ensembled["target_key"] == target)
        ]
        for candidate in candidates:
            current = ensembled[
                (ensembled["candidate_id"] == candidate)
                & (ensembled["target_key"] == target)
            ]
            for metric in METRICS:
                tasks.append(
                    {
                        "target_key": target,
                        "candidate_id": candidate,
                        "reference_candidate_id": anchor,
                        "metric": metric,
                        "current": current,
                        "baseline": baseline,
                        "n_bootstrap": n_bootstrap,
                        "seed": bootstrap_seed,
                    }
                )

    workers = int(n_jobs)
    if workers <= 0:
        workers = min(16, max(1, (os.cpu_count() or 4) - 1))
    print(
        f"[19F post-hoc] bootstrap jobs={len(tasks)} "
        f"n_bootstrap={n_bootstrap} n_jobs={workers}",
        file=sys.stderr,
        flush=True,
    )
    if workers == 1 or len(tasks) <= 1:
        rows = [_bootstrap_worker(task) for task in tasks]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_bootstrap_worker, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                if completed % 10 == 0 or completed == len(tasks):
                    print(
                        f"[19F post-hoc] bootstrap progress {completed}/{len(tasks)}",
                        file=sys.stderr,
                        flush=True,
                    )
    return pd.DataFrame(rows)


def _bootstrap_worker(task: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "target_key": task["target_key"],
        "candidate_id": task["candidate_id"],
        "reference_candidate_id": task["reference_candidate_id"],
        "metric": task["metric"],
        "comparison_direction": "candidate_minus_reference",
        "classification": CLASSIFICATION,
    }
    try:
        result = paired_bootstrap_delta(
            task["current"],
            task["baseline"],
            metric=task["metric"],
            group_col="ModelID",
            n_bootstrap=task["n_bootstrap"],
            seed=task["seed"],
        )
        row.update(result)
        row.update({"status": "computed", "reason": ""})
    except (ValueError, RuntimeError) as exc:
        row.update(
            {
                "mean_delta": None,
                "ci_lower": None,
                "ci_upper": None,
                "probability_delta_gt_zero": None,
                "n_bootstrap": 0,
                "status": "not_estimable",
                "reason": str(exc),
            }
        )
    return row


def _role_view(
    aliases: pd.DataFrame,
    internal: pd.DataFrame,
    tcga: pd.DataFrame,
    integrated: pd.DataFrame,
) -> pd.DataFrame:
    metric_rows = pd.concat(
        [
            internal,
            tcga,
            integrated.assign(target_key="Integrated5"),
        ],
        ignore_index=True,
        sort=False,
    )
    view = aliases.merge(
        metric_rows,
        how="left",
        left_on="locked_candidate_id",
        right_on="candidate_id",
        validate="many_to_many",
        suffixes=("_role", ""),
    )
    view["classification"] = CLASSIFICATION
    return view


def _write_report(
    path: Path,
    *,
    lock_path: Path,
    prediction_root: Path,
    candidates: int,
    bootstrap: pd.DataFrame,
) -> None:
    computed = int((bootstrap["status"] == "computed").sum()) if not bootstrap.empty else 0
    text = f"""# Round 19F exploratory post-hoc report

> Classification: **exploratory post-hoc**. These results are descriptive only.
> Locked roles remain unchanged; this report makes no selection or lock recommendation.

- Final lock: `{lock_path}`
- Prediction root: `{prediction_root}`
- Locked source candidates: {candidates}
- Ensemble: arithmetic mean of exactly 15 member probabilities
- Internal output: per-candidate DrugMacro and global metrics
- TCGA output: per-candidate metrics for each fixed target
- Integrated5: equal mean across the five target-level DrugMacro metrics
- Paired bootstrap rows computed: {computed}/{len(bootstrap)}
- Role view: aliases the immutable lock roles to candidate metrics

All generated tables include `classification={CLASSIFICATION}`.
"""
    path.write_text(text, encoding="utf-8")


def analyze_posthoc(
    final_lock_path: Path,
    prediction_root: Path,
    *,
    output_dir: Optional[Path] = None,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 42,
    n_jobs: int = 1,
) -> dict:
    """Validate locked provenance, aggregate predictions, and write exploratory reports."""
    final_lock_path = Path(final_lock_path)
    prediction_root = Path(prediction_root)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else prediction_root / "reports" / "round19_stage19f_posthoc"
    )
    project_root = Path(__file__).resolve().parents[1]
    lock = _read_json_object(final_lock_path)
    verify_final_lock(lock, project_root)
    if (
        lock.get("posthoc_classification") != CLASSIFICATION
        or lock.get("single_champion") is not None
        or lock.get("role_immutability", {}).get("internal_test_may_change_roles") is not False
        or lock.get("role_immutability", {}).get("tcga_may_change_roles") is not False
    ):
        raise AssertionError("Final lock does not authorize immutable exploratory post-hoc use")
    inventory = _locked_inventory(lock)
    aliases = _role_aliases(lock, inventory["source_candidate_id"].astype(str))
    raw = _discover_predictions(prediction_root, output_dir)
    validated = _validate_prediction_provenance(raw, inventory)
    ensembled = ensemble_predictions(validated)
    internal, tcga, integrated = _metric_tables(ensembled)
    bootstrap = _bootstrap_table(
        ensembled,
        aliases,
        n_bootstrap=int(n_bootstrap),
        bootstrap_seed=int(bootstrap_seed),
        n_jobs=int(n_jobs),
    )
    role_view = _role_view(aliases, internal, tcga, integrated)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "ensemble_predictions": output_dir / "round19f_15member_ensemble_predictions.csv",
        "internal_metrics": output_dir / "round19f_internal_candidate_metrics.csv",
        "tcga_metrics": output_dir / "round19f_tcga_per_target_metrics.csv",
        "integrated5": output_dir / "round19f_integrated5_equal_target_mean.csv",
        "paired_bootstrap": output_dir / "round19f_paired_bootstrap_deltas.csv",
        "role_alias_view": output_dir / "round19f_role_alias_view.csv",
        "report": output_dir / "round19f_exploratory_posthoc_report.md",
        "summary": output_dir / "round19f_exploratory_posthoc_summary.json",
    }
    ensembled.assign(classification=CLASSIFICATION).to_csv(
        artifacts["ensemble_predictions"], index=False
    )
    internal.to_csv(artifacts["internal_metrics"], index=False)
    tcga.to_csv(artifacts["tcga_metrics"], index=False)
    integrated.to_csv(artifacts["integrated5"], index=False)
    bootstrap.to_csv(artifacts["paired_bootstrap"], index=False)
    role_view.to_csv(artifacts["role_alias_view"], index=False)
    _write_report(
        artifacts["report"],
        lock_path=final_lock_path,
        prediction_root=prediction_root,
        candidates=int(inventory["source_candidate_id"].nunique()),
        bootstrap=bootstrap,
    )
    summary = {
        "artifact_type": "round19_stage19f_exploratory_posthoc_summary",
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "role_lock_immutable": True,
        "roles_changed": False,
        "aggregation": {
            "method": "mean_probability",
            "required_members": N_REQUIRED_MEMBERS,
            "best_fold_allowed": False,
        },
        "coverage": {
            "n_candidates": int(inventory["source_candidate_id"].nunique()),
            "n_internal_targets": 1,
            "n_tcga_targets": len(TCGA_TARGETS),
            "tcga_targets": list(TCGA_TARGETS),
        },
        "artifacts": {key: str(value) for key, value in artifacts.items() if key != "summary"},
    }
    artifacts["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**summary, "summary_path": str(artifacts["summary"])}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round 19F exploratory post-hoc analyzer (no inference)"
    )
    parser.add_argument("--final-lock", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Bootstrap worker processes; 0 selects up to 16 automatically.",
    )
    args = parser.parse_args()
    result = analyze_posthoc(
        Path(args.final_lock),
        Path(args.prediction_root),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
