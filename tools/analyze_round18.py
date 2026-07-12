#!/usr/bin/env python3
"""Round 18 analyzer: screening ranking, resource summary, selection lock."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def collect_job_rows(manifest_path: Path) -> pd.DataFrame:
    man = pd.read_csv(manifest_path)
    rows = []
    for _, job in man.iterrows():
        result_dir = Path(str(job["result_dir"]))
        status_path = result_dir / "job_status.json"
        metrics_path = result_dir / "val_metrics.json"
        summary_path = result_dir / "train_summary.json"
        resource_path = result_dir / "runtime_resource_summary.json"
        status = _load_json(status_path) if status_path.is_file() else {"status": "missing"}
        metrics = _load_json(metrics_path) if metrics_path.is_file() else {}
        summary = _load_json(summary_path) if summary_path.is_file() else {}
        resource = _load_json(resource_path) if resource_path.is_file() else {}
        rows.append(
            {
                "job_id": job.get("job_id"),
                "stage": job.get("stage"),
                "architecture_id": job.get("architecture_id"),
                "architecture_family": job.get("architecture_family"),
                "omics_mode": job.get("omics_mode"),
                "transformer_config_id": job.get("transformer_config_id", ""),
                "residual_mode": job.get("residual_mode", ""),
                "fold_id": int(job.get("fold_id", -1)),
                "result_dir": str(result_dir),
                "status": status.get("status", "missing"),
                "DrugMacro_AUC": _safe_float(metrics.get("DrugMacro_AUC")),
                "DrugMacro_AUPRC": _safe_float(metrics.get("DrugMacro_AUPRC")),
                "Global_AUC": _safe_float(metrics.get("Global_AUC")),
                "Global_AUPRC": _safe_float(metrics.get("Global_AUPRC")),
                "n_valid_auc_drugs": metrics.get("n_valid_auc_drugs"),
                "best_epoch": summary.get("best_epoch"),
                "best_score": _safe_float(summary.get("best_score")),
                "n_epochs": summary.get("n_epochs"),
                "micro_batch_size": resource.get("micro_batch_size")
                or status.get("successful_micro_batch"),
                "accumulation_steps": resource.get("accumulation_steps")
                or status.get("gradient_accumulation_steps"),
                "peak_gpu_mem_mb": resource.get("peak_gpu_mem_mb"),
                "oom_retry_count": status.get("oom_retry_count"),
                "oom_batch_history": json.dumps(status.get("oom_batch_history") or []),
            }
        )
    return pd.DataFrame(rows)


def rank_screening_architectures(job_df: pd.DataFrame) -> pd.DataFrame:
    """3-fold mean DrugMacro AUC ranking; no internal test / TCGA."""
    if job_df is None or job_df.empty or "status" not in job_df.columns:
        return pd.DataFrame()
    done = job_df[job_df["status"] == "done"].copy()
    if done.empty:
        return pd.DataFrame()
    gcols = [
        "architecture_id",
        "architecture_family",
        "omics_mode",
        "transformer_config_id",
        "residual_mode",
    ]
    rows = []
    for keys, g in done.groupby(gcols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(gcols, keys))
        aucs = [x for x in g["DrugMacro_AUC"].tolist() if x is not None]
        auprcs = [x for x in g["DrugMacro_AUPRC"].tolist() if x is not None]
        rec.update(
            {
                "n_folds_done": int(len(g)),
                "n_folds_with_auc": int(len(aucs)),
                "mean_DrugMacro_AUC": float(np.mean(aucs)) if aucs else None,
                "std_DrugMacro_AUC": float(np.std(aucs, ddof=0)) if len(aucs) > 1 else (0.0 if aucs else None),
                "mean_DrugMacro_AUPRC": float(np.mean(auprcs)) if auprcs else None,
                "mean_Global_AUC": float(np.nanmean(g["Global_AUC"].astype(float)))
                if g["Global_AUC"].notna().any()
                else None,
                "fold_DrugMacro_AUC": json.dumps(
                    {int(r.fold_id): r.DrugMacro_AUC for r in g.itertuples()}
                ),
            }
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        by=["mean_DrugMacro_AUC", "mean_DrugMacro_AUPRC"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def select_formal_candidates(
    ranking: pd.DataFrame,
    *,
    top_k_pooled: int = 2,
    top_k_cross: int = 2,
) -> List[dict]:
    """Pick top candidates per family for 18D; screening only."""
    if ranking.empty:
        return []
    selected = []
    seen = set()
    # always include best overall
    for _, row in ranking.iterrows():
        if row["architecture_id"] in seen:
            continue
        fam = str(row["architecture_family"])
        if fam in {"pooled_mlp", "pooled_transformer"}:
            n_fam = sum(1 for s in selected if s["architecture_family"] == fam)
            if n_fam >= top_k_pooled and fam != ranking.iloc[0]["architecture_family"]:
                # still allow best overall below
                pass
        selected.append(
            {
                "architecture_id": row["architecture_id"],
                "architecture_family": row["architecture_family"],
                "omics_mode": row["omics_mode"],
                "transformer_config_id": row.get("transformer_config_id") or "",
                "residual_mode": row.get("residual_mode") or "",
                "selection_metric": "CV_DrugMacro_AUC",
                "mean_DrugMacro_AUC": row["mean_DrugMacro_AUC"],
                "mean_DrugMacro_AUPRC": row["mean_DrugMacro_AUPRC"],
            }
        )
        seen.add(row["architecture_id"])
        # stop rules: keep top pooled + top cross
        n_mlp = sum(1 for s in selected if s["architecture_family"] == "pooled_mlp")
        n_tf = sum(1 for s in selected if s["architecture_family"] == "pooled_transformer")
        n_x = sum(1 for s in selected if s["architecture_family"] == "cross_attention")
        if n_mlp >= 1 and n_tf >= top_k_pooled - 1 and n_x >= top_k_cross and len(selected) >= 4:
            break
        if len(selected) >= 6:
            break
    # Ensure at least one of each pooled family if present in ranking
    for fam, k in (("pooled_mlp", 1), ("pooled_transformer", 1), ("cross_attention", top_k_cross)):
        have = sum(1 for s in selected if s["architecture_family"] == fam)
        if have >= k:
            continue
        for _, row in ranking[ranking["architecture_family"] == fam].iterrows():
            if row["architecture_id"] in seen:
                continue
            selected.append(
                {
                    "architecture_id": row["architecture_id"],
                    "architecture_family": row["architecture_family"],
                    "omics_mode": row["omics_mode"],
                    "transformer_config_id": row.get("transformer_config_id") or "",
                    "residual_mode": row.get("residual_mode") or "",
                    "selection_metric": "CV_DrugMacro_AUC",
                    "mean_DrugMacro_AUC": row["mean_DrugMacro_AUC"],
                    "mean_DrugMacro_AUPRC": row["mean_DrugMacro_AUPRC"],
                }
            )
            seen.add(row["architecture_id"])
            have += 1
            if have >= k:
                break
    return selected


def write_locked_selection(
    outdir: Path,
    *,
    ranking: pd.DataFrame,
    formal_candidates: List[dict],
    settings_path: str,
) -> Path:
    reports = outdir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if ranking.empty:
        raise ValueError("Cannot write selection lock without screening ranking")
    best = ranking.iloc[0].to_dict()
    split_meta = outdir / "splits" / "split_metadata.json"
    feature_meta = (
        outdir / "data" / "round18_feature_coverage_preflight.json"
    )
    # Prefer own_plus_summary feature metadata hash if present
    settings = _load_json(Path(settings_path))
    feat_dir = (
        Path(settings.get("feature_root", "result/optimization_runs/round17r_18class/features"))
        / settings.get("feature_model_key", "r13_exp_008")
        / str(best.get("omics_mode") or "own_plus_summary")
    )
    feat_meta_path = feat_dir / "feature_metadata.json"

    lock = {
        "architecture_id": best["architecture_id"],
        "architecture_family": best["architecture_family"],
        "omics_mode": best["omics_mode"],
        "transformer_config_id": best.get("transformer_config_id") or "",
        "residual_mode": best.get("residual_mode") or "",
        "transformer_config": {},
        "gin_config": settings.get("gin", {}),
        "split_manifest_sha256": _sha256_file(split_meta),
        "feature_metadata_sha256": _sha256_file(feat_meta_path),
        "feature_coverage_sha256": _sha256_file(feature_meta),
        "selection_metric": "CV_DrugMacro_AUC",
        "selected_mean_DrugMacro_AUC": best.get("mean_DrugMacro_AUC"),
        "selected_without_internal_test": True,
        "selected_without_tcga": True,
        "formal_candidates": formal_candidates,
        "notes": [
            "Selection uses screening fold-mean DrugMacro AUC only.",
            "Omics encoder is frozen pretrained/transductive; grouped response CV holds out labels/ModelIDs.",
            "own_proto_context_projected_16 uses unlabeled TCGA prototype context (not TCGA response labels).",
        ],
    }
    path = reports / "round18_locked_selection.json"
    path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return path


def analyze_round18(
    outdir: str,
    *,
    settings_path: str = "config/round18_architecture_settings.json",
    write_lock: bool = False,
    screening_manifest: Optional[str] = None,
    cross_manifest: Optional[str] = None,
    formal_manifest: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(outdir)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    manifests = []
    for default, override in (
        (root / "manifests" / "stage18b_screening_manifest.csv", screening_manifest),
        (root / "manifests" / "stage18c_cross_attention_manifest.csv", cross_manifest),
        (root / "manifests" / "stage18d_formal_5cv_manifest.csv", formal_manifest),
    ):
        path = Path(override) if override else default
        if path.is_file():
            manifests.append(path)

    all_jobs = []
    for m in manifests:
        df = collect_job_rows(m)
        df["manifest"] = str(m)
        all_jobs.append(df)
    jobs = pd.concat(all_jobs, ignore_index=True) if all_jobs else pd.DataFrame()
    jobs_path = reports / "round18_job_completion_summary.csv"
    jobs.to_csv(jobs_path, index=False)

    # Screening = 18b + 18c done jobs
    screening = jobs[jobs["stage"].astype(str).isin(["18b", "18c"])].copy() if not jobs.empty else jobs
    ranking = rank_screening_architectures(screening)
    ranking_path = reports / "round18_screening_architecture_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    omics_summary = pd.DataFrame()
    if not ranking.empty:
        omics_summary = (
            ranking.groupby("omics_mode", dropna=False)
            .agg(
                n_architectures=("architecture_id", "count"),
                best_mean_DrugMacro_AUC=("mean_DrugMacro_AUC", "max"),
                mean_of_means=("mean_DrugMacro_AUC", "mean"),
            )
            .reset_index()
        )
    omics_path = reports / "round18_omics_mode_summary.csv"
    omics_summary.to_csv(omics_path, index=False)

    resource = screening[
        [
            "job_id",
            "architecture_id",
            "omics_mode",
            "fold_id",
            "status",
            "micro_batch_size",
            "accumulation_steps",
            "peak_gpu_mem_mb",
            "oom_retry_count",
            "oom_batch_history",
            "n_epochs",
        ]
    ].copy() if not screening.empty else pd.DataFrame()
    resource_path = reports / "round18_resource_usage_summary.csv"
    resource.to_csv(resource_path, index=False)
    oom_path = reports / "round18_oom_summary.csv"
    resource.to_csv(oom_path, index=False)

    # Formal 5CV if present
    formal = jobs[jobs["stage"].astype(str) == "18d"].copy() if not jobs.empty else pd.DataFrame()
    formal_summary = rank_screening_architectures(formal) if not formal.empty else pd.DataFrame()
    # rename for clarity
    formal_path = reports / "round18_formal_5cv_summary.csv"
    formal_summary.to_csv(formal_path, index=False)

    # Split balance copy
    split_balance_src = root / "splits" / "fold_balance_report.csv"
    split_balance_dst = reports / "round18_split_balance_summary.csv"
    if split_balance_src.is_file():
        pd.read_csv(split_balance_src).to_csv(split_balance_dst, index=False)

    lock_path = None
    formal_candidates = select_formal_candidates(ranking) if not ranking.empty else []
    if write_lock:
        # Require all screening jobs done for lock (18b at minimum)
        b_man = root / "manifests" / "stage18b_screening_manifest.csv"
        if b_man.is_file():
            b_jobs = collect_job_rows(b_man)
            if not (b_jobs["status"] == "done").all():
                raise RuntimeError(
                    "Cannot write selection lock until all stage18b jobs are done "
                    "({}/{} done)".format(int((b_jobs["status"] == "done").sum()), len(b_jobs))
                )
        lock_path = write_locked_selection(
            root,
            ranking=ranking,
            formal_candidates=formal_candidates,
            settings_path=settings_path,
        )

    # Minimal markdown report
    md_path = reports / "round18_final_report.md"
    lines = [
        "# Round 18 Analysis Report",
        "",
        "## Screening ranking (fold-mean DrugMacro AUC)",
        "",
        "Selection uses screening CV only; internal test and TCGA are excluded.",
        "",
    ]
    if not ranking.empty:
        lines.append(ranking.head(15).to_markdown(index=False))
    else:
        lines.append("_No completed screening jobs yet._")
    lines.extend(
        [
            "",
            "## Scientific notes",
            "",
            "- Omics encoder is frozen; CV is grouped response-prediction CV with",
            "  pretrained/transductive omics representations (not fully inductive).",
            "- `own_proto_context_projected_16` includes unlabeled TCGA prototype context.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "jobs": str(jobs_path),
        "ranking": str(ranking_path),
        "omics_summary": str(omics_path),
        "resource": str(resource_path),
        "formal_summary": str(formal_path),
        "report_md": str(md_path),
        "lock": str(lock_path) if lock_path else None,
        "n_jobs": int(len(jobs)),
        "n_ranking": int(len(ranking)),
        "formal_candidates": formal_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 18 results")
    parser.add_argument("--outdir", default="result/optimization_runs/round18_architecture")
    parser.add_argument("--settings", default="config/round18_architecture_settings.json")
    parser.add_argument("--write-lock", action="store_true")
    parser.add_argument("--screening-manifest", default=None)
    parser.add_argument("--cross-manifest", default=None)
    parser.add_argument("--formal-manifest", default=None)
    args = parser.parse_args()
    out = analyze_round18(
        args.outdir,
        settings_path=args.settings,
        write_lock=args.write_lock,
        screening_manifest=args.screening_manifest,
        cross_manifest=args.cross_manifest,
        formal_manifest=args.formal_manifest,
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
