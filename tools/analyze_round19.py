#!/usr/bin/env python3
"""Round 19 analyzer: Stage 19B templates and Stage 19C composition merge."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round19_manifest_validator import assert_selection_frame_has_no_tcga
from tools.round19_selection_lock import scan_mapping_for_forbidden

EFFECT_REPORTS_19B = [
    "round19b_architecture_ranking.csv",
    "round19b_node_capacity_effect.csv",
    "round19b_graph_bottleneck_effect.csv",
    "round19b_bond_content_effect.csv",
    "round19b_predictor_integration_effect.csv",
    "round19b_context_dependency.csv",
    "round19b_resource_summary.csv",
]

EFFECT_REPORTS_19C = [
    "round19c_full_composition_ranking.csv",
    "round19c_omics_effects.csv",
    "round19c_context_redundancy.csv",
    "round19c_source_only_control.csv",
    "round19c_context_shuffle_control.csv",
    "round19c_omics_predictor_interaction.csv",
    "round19c_omics_drug_interaction.csv",
    "round19c_role_candidate_summary.csv",
    "round19c_resource_summary.csv",
]

OMICS_EFFECT_PAIRS = [
    ("O1", "O0", "summary_effect"),
    ("O2", "O0", "context_effect"),
    ("O3", "O1", "context_added_to_summary"),
    ("O3", "O2", "summary_added_to_context"),
    ("O4", "O0", "source_only_effect"),
    ("O2", "O4", "target_informed_vs_source_only"),
]


def write_empty_effect_templates(outdir: str, names: List[str]) -> dict:
    reports = Path(outdir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names:
        path = reports / name
        if not path.is_file():
            pd.DataFrame(columns=["architecture_id", "mean_DrugMacro_AUC"]).to_csv(path, index=False)
        df = pd.read_csv(path)
        assert_selection_frame_has_no_tcga(df)
        written.append(str(path))
    return {"templates": written}


def _load_job_metrics_from_dir(root: Path, manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.is_file():
        return pd.DataFrame()
    manifest = pd.read_csv(manifest_path)
    rows = []
    for _, row in manifest.iterrows():
        result_dir = Path(str(row["result_dir"]))
        mpath = result_dir / "val_metrics.json"
        status_path = result_dir / "job_status.json"
        status = "missing"
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status", "unknown")
        metrics = {}
        if mpath.is_file():
            metrics = json.loads(mpath.read_text(encoding="utf-8"))
        rows.append(
            {
                "job_id": row.get("job_id"),
                "drug_id": row.get("drug_id") or row.get("drug_representation_id"),
                "predictor_id": row.get("predictor_id"),
                "omics_id": row.get("omics_id"),
                "fold_id": int(row.get("fold_id", 0)),
                "control_type": row.get("control_type", "none"),
                "context_control": row.get("context_control", "none"),
                "role": row.get("role", ""),
                "status": status,
                "DrugMacro_AUC": metrics.get("DrugMacro_AUC"),
                "DrugMacro_AUPRC": metrics.get("DrugMacro_AUPRC"),
                "Global_AUC": metrics.get("Global_AUC"),
                "result_dir": str(result_dir),
            }
        )
    df = pd.DataFrame(rows)
    assert_selection_frame_has_no_tcga(df)
    return df


def _cell_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    done = metrics[metrics["status"] == "done"].copy()
    if done.empty:
        return pd.DataFrame()
    rows = []
    for (d, p, o), g in done.groupby(["drug_id", "predictor_id", "omics_id"]):
        rows.append(
            {
                "drug_id": d,
                "predictor_id": p,
                "omics_id": o,
                "n_folds": int(len(g)),
                "mean_drugmacro_auc": float(g["DrugMacro_AUC"].mean()),
                "std_drugmacro_auc": float(g["DrugMacro_AUC"].std(ddof=0)) if len(g) > 1 else 0.0,
                "mean_drugmacro_auprc": float(g["DrugMacro_AUPRC"].mean())
                if g["DrugMacro_AUPRC"].notna().all()
                else None,
                "mean_global_auc": float(g["Global_AUC"].mean()) if g["Global_AUC"].notna().all() else None,
            }
        )
    return pd.DataFrame(rows)


def analyze_stage19c(outdir: str, *, require_complete: bool = False) -> dict:
    root = Path(outdir)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    lock_path = reports / "round19_stage19c_candidate_lock.json"
    unique_cells = []
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        scan_mapping_for_forbidden(lock)
        unique_cells = lock.get("unique_cells") or []

    m19b = root / "manifests" / "stage19b_drug_predictor_manifest.csv"
    m19c = root / "manifests" / "stage19c_manifest.csv"

    metrics19b = _load_job_metrics_from_dir(root, m19b)
    metrics19c = _load_job_metrics_from_dir(root, m19c)

    if unique_cells:
        keys = {(str(c["drug_id"]), str(c["predictor_id"])) for c in unique_cells}
        metrics19b = metrics19b[
            metrics19b.apply(lambda r: (str(r["drug_id"]), str(r["predictor_id"])) in keys, axis=1)
        ]
        metrics19b = metrics19b[metrics19b["omics_id"].isin(["O1", "O2", "O3"])]

    core19c = metrics19c[metrics19c.get("control_type", "none") == "none"] if not metrics19c.empty else metrics19c
    ctrl19c = (
        metrics19c[metrics19c.get("control_type", "none") == "context_shuffle"]
        if not metrics19c.empty
        else metrics19c
    )

    n_core_expected = len(unique_cells) * 2 * 3 if unique_cells else None
    n_core_done = int((core19c["status"] == "done").sum()) if not core19c.empty else 0
    n_ctrl_done = int((ctrl19c["status"] == "done").sum()) if not ctrl19c.empty else 0

    if require_complete and n_core_expected is not None:
        if n_core_done != n_core_expected or n_ctrl_done != 12:
            raise RuntimeError(
                f"Stage19C incomplete: core {n_core_done}/{n_core_expected}, controls {n_ctrl_done}/12"
            )
    elif n_core_expected is not None and (n_core_done < n_core_expected or n_ctrl_done < 12):
        warnings.warn(
            f"Stage19C partial: core {n_core_done}/{n_core_expected}, controls {n_ctrl_done}/12",
            stacklevel=2,
        )

    merged = pd.concat([metrics19b, core19c], ignore_index=True)
    composition = _cell_summary(merged)
    if not composition.empty:
        composition = composition.sort_values(
            ["mean_drugmacro_auc", "mean_drugmacro_auprc"],
            ascending=[False, False],
            na_position="last",
        )
        composition.to_csv(reports / "round19c_full_composition_ranking.csv", index=False)

    # Omics effects (paired per fold)
    effect_rows = []
    if not merged.empty:
        done = merged[merged["status"] == "done"]
        for (d, p), cell_df in done.groupby(["drug_id", "predictor_id"]):
            for a, b, label in OMICS_EFFECT_PAIRS:
                da = cell_df[cell_df["omics_id"] == a]
                db = cell_df[cell_df["omics_id"] == b]
                for fold in sorted(set(da["fold_id"]).intersection(set(db["fold_id"]))):
                    ra = da[da["fold_id"] == fold].iloc[0]
                    rb = db[db["fold_id"] == fold].iloc[0]
                    effect_rows.append(
                        {
                            "drug_id": d,
                            "predictor_id": p,
                            "comparison": label,
                            "omics_a": a,
                            "omics_b": b,
                            "fold_id": int(fold),
                            "auc_a": ra["DrugMacro_AUC"],
                            "auc_b": rb["DrugMacro_AUC"],
                            "delta_auc": float(ra["DrugMacro_AUC"]) - float(rb["DrugMacro_AUC"]),
                            "delta_auprc": float(ra["DrugMacro_AUPRC"]) - float(rb["DrugMacro_AUPRC"])
                            if pd.notna(ra["DrugMacro_AUPRC"]) and pd.notna(rb["DrugMacro_AUPRC"])
                            else None,
                        }
                    )
    effects = pd.DataFrame(effect_rows)
    if not effects.empty:
        effects.to_csv(reports / "round19c_omics_effects.csv", index=False)
        summary = (
            effects.groupby(["drug_id", "predictor_id", "comparison"])
            .agg(
                mean_delta=("delta_auc", "mean"),
                median_delta=("delta_auc", "median"),
                positive_fold_count=("delta_auc", lambda s: int((s > 0).sum())),
                total_fold_count=("delta_auc", "count"),
            )
            .reset_index()
        )
        summary.to_csv(reports / "round19c_context_redundancy.csv", index=False)

    # Source-only O4 control
    if not composition.empty:
        o4 = composition[composition["omics_id"] == "O4"].copy()
        o4.to_csv(reports / "round19c_source_only_control.csv", index=False)

    # Context shuffle control
    shuffle_rows = []
    if not ctrl19c.empty:
        true19b = metrics19b[metrics19b["omics_id"].isin(["O2", "O3"])]
        for _, crow in ctrl19c.iterrows():
            if crow["status"] != "done":
                continue
            match = true19b[
                (true19b["drug_id"] == crow["drug_id"])
                & (true19b["predictor_id"] == crow["predictor_id"])
                & (true19b["omics_id"] == crow["omics_id"])
                & (true19b["fold_id"] == crow["fold_id"])
            ]
            true_auc = float(match.iloc[0]["DrugMacro_AUC"]) if len(match) else np.nan
            shuf_auc = float(crow["DrugMacro_AUC"]) if pd.notna(crow["DrugMacro_AUC"]) else np.nan
            shuffle_rows.append(
                {
                    "drug_id": crow["drug_id"],
                    "predictor_id": crow["predictor_id"],
                    "omics_id": crow["omics_id"],
                    "fold_id": int(crow["fold_id"]),
                    "true_drugmacro_auc": true_auc,
                    "shuffled_drugmacro_auc": shuf_auc,
                    "delta_true_minus_shuffled": true_auc - shuf_auc if pd.notna(true_auc) else np.nan,
                }
            )
    shuffle_df = pd.DataFrame(shuffle_rows)
    if not shuffle_df.empty:
        shuffle_df.to_csv(reports / "round19c_context_shuffle_control.csv", index=False)

    # Role summary stub
    if lock_path.is_file() and not composition.empty:
        role_rows = []
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        for cell in lock.get("unique_cells", []):
            d, p = str(cell["drug_id"]), str(cell["predictor_id"])
            sub = composition[(composition.drug_id == d) & (composition.predictor_id == p)]
            role_rows.append(
                {
                    "role": cell.get("primary_role", ""),
                    "drug_id": d,
                    "predictor_id": p,
                    "n_omics_modes": int(len(sub)),
                    "best_omics_by_auc": sub.sort_values("mean_drugmacro_auc", ascending=False)
                    .iloc[0]["omics_id"]
                    if len(sub)
                    else None,
                    "mean_auc_best_omics": float(sub["mean_drugmacro_auc"].max()) if len(sub) else None,
                }
            )
        pd.DataFrame(role_rows).to_csv(reports / "round19c_role_candidate_summary.csv", index=False)

    # Stubs for interaction/resource reports if missing
    for name in [
        "round19c_omics_predictor_interaction.csv",
        "round19c_omics_drug_interaction.csv",
        "round19c_resource_summary.csv",
    ]:
        path = reports / name
        if not path.is_file():
            pd.DataFrame(columns=["note"]).to_csv(path, index=False)

    for path in reports.glob("round19c_*.csv"):
        assert_selection_frame_has_no_tcga(pd.read_csv(path))

    return {
        "stage": "19c",
        "n_composition_rows": int(len(composition)),
        "n_core_done": n_core_done,
        "n_core_expected": n_core_expected,
        "n_control_done": n_ctrl_done,
        "reports_dir": str(reports),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="19b", choices=["19b", "19c", "selection"])
    parser.add_argument("--outdir", default="result/optimization_runs/round19_factorial")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args()
    if args.stage == "19b":
        out = write_empty_effect_templates(args.outdir, EFFECT_REPORTS_19B)
        print(json.dumps(out, indent=2))
        return
    if args.stage == "19c":
        out = analyze_stage19c(args.outdir, require_complete=args.require_complete)
        print(json.dumps(out, indent=2))
        return
    if args.stage == "selection":
        if not args.write_lock:
            raise SystemExit("selection stage requires --write-lock after 19B/19C complete")
        raise SystemExit("Refuse lock: Round 19B/19C results not complete (smoke guard)")
    raise SystemExit(f"Unsupported stage {args.stage}")


if __name__ == "__main__":
    main()
