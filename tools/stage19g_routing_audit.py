#!/usr/bin/env python3
"""Fold-aware 19E / full-development TCGA routing audit through the final lock."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from tools.round19_deployment_policy import select_role
from tools.round19_stage19g_lock_adapter import route_locked

SEEN_COLUMNS = ("seen_drug", "seen_scaffold", "seen_cancer_type")


def novelty_class(row: Mapping) -> str:
    seen = {column: bool(row[column]) for column in SEEN_COLUMNS}
    if not seen["seen_drug"]:
        return "unseen_drug"
    if not seen["seen_scaffold"]:
        return "unseen_scaffold"
    if not seen["seen_cancer_type"]:
        return "unseen_cancer_type"
    return "source_like"


def audit_routing(
    cases: pd.DataFrame,
    support: pd.DataFrame,
    *,
    final_lock: Path,
    project_root: Path,
    router: Callable = route_locked,
) -> pd.DataFrame:
    required = {"case_id", "evaluation_scope", "drug_id", "scaffold_id", "cancer_type"}
    missing = required - set(cases.columns)
    if missing:
        raise KeyError(f"cases missing columns: {sorted(missing)}")
    rows = []
    for record in cases.to_dict("records"):
        scope = str(record["evaluation_scope"])
        eligible = support
        if scope == "19E":
            if "fold_id" not in record or pd.isna(record["fold_id"]):
                raise ValueError("19E support must be fold-relative")
            fold_values = pd.to_numeric(eligible["fold_id"], errors="coerce")
            eligible = eligible[fold_values == float(record["fold_id"])]
            support_basis = "19E_fold_relative"
        elif scope == "TCGA":
            eligible = eligible[eligible["support_scope"] == "full_development"]
            support_basis = "TCGA_full_development"
        else:
            raise ValueError(f"unknown evaluation_scope={scope!r}")
        seen = {
            "seen_drug": record["drug_id"] in set(eligible["drug_id"]),
            "seen_scaffold": record["scaffold_id"] in set(eligible["scaffold_id"]),
            "seen_cancer_type": record["cancer_type"] in set(eligible["cancer_type"]),
        }
        novelty = novelty_class(seen)
        expected = select_role(novelty)
        routed = router(final_lock, novelty, project_root=project_root)
        rows.append({
            **record,
            **seen,
            "support_basis": support_basis,
            "novelty_class": novelty,
            "expected_role": expected,
            "selected_role": routed["selected_role"],
            "routing_match": routed["selected_role"] == expected,
            "lock_file_sha256": routed["lock_file_sha256"],
        })
    result = pd.DataFrame(rows)
    if result.empty or not result["routing_match"].all():
        raise AssertionError("routing_match must be 100%")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--support", required=True)
    parser.add_argument("--final-lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_routing(
        pd.read_csv(args.cases), pd.read_csv(args.support),
        final_lock=Path(args.final_lock), project_root=Path(args.project_root),
    ).to_csv(output, index=False)


if __name__ == "__main__":
    main()
