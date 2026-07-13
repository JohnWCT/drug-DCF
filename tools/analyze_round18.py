#!/usr/bin/env python3
"""Round 18 analyzer: screening ranking, resource summary, selection lock."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CONTEXT16 = "own_proto_context_projected_16"
OWN_PLUS = "own_plus_summary"
MLP_OWN_PLUS_ID = "pooled_mlp__own_plus_summary"
P3_CONTEXT16_PREFIX = "pooled_transformer__P3_deeper128__"
P1_CONTEXT16_PREFIX = "pooled_transformer__P1_compact64__"
REQUIRED_SCREENING_FOLDS = 3


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


def filter_complete_screening(
    ranking: pd.DataFrame,
    *,
    n_folds: int = REQUIRED_SCREENING_FOLDS,
) -> pd.DataFrame:
    """Keep only architectures with all screening folds done and with AUC."""
    if ranking is None or ranking.empty:
        return pd.DataFrame()
    mask = (ranking["n_folds_done"] == n_folds) & (ranking["n_folds_with_auc"] == n_folds)
    out = ranking.loc[mask].copy()
    if out.empty:
        return out
    out = out.sort_values(
        by=["mean_DrugMacro_AUC", "mean_DrugMacro_AUPRC"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def assert_manifests_complete_for_lock(root: Path) -> Dict[str, Dict[str, int]]:
    """Hard gate: both 18B and 18C manifests must exist and be 100% done."""
    required_manifests = [
        root / "manifests" / "stage18b_screening_manifest.csv",
        root / "manifests" / "stage18c_cross_attention_manifest.csv",
    ]
    completion: Dict[str, Dict[str, int]] = {}
    for manifest in required_manifests:
        if not manifest.is_file():
            raise RuntimeError(f"Cannot write lock: missing {manifest}")
        jobs = collect_job_rows(manifest)
        n_total = int(len(jobs))
        if n_total == 0 or "status" not in jobs.columns:
            raise RuntimeError(
                f"Cannot write lock: {manifest.name} has 0 jobs"
            )
        n_done = int((jobs["status"] == "done").sum())
        completion[manifest.name] = {"n_done": n_done, "n_total": n_total}
        if n_done != n_total:
            raise RuntimeError(
                f"Cannot write lock: {manifest.name} has {n_done}/{n_total} completed jobs"
            )
    return completion


def _fold_lookup(jobs: pd.DataFrame, architecture_id: str) -> Dict[int, float]:
    sub = jobs[
        (jobs["architecture_id"] == architecture_id)
        & (jobs["status"] == "done")
        & jobs["DrugMacro_AUC"].notna()
    ]
    return {int(r.fold_id): float(r.DrugMacro_AUC) for r in sub.itertuples()}


def _pair_rows(
    *,
    comparison: str,
    candidate_a: str,
    candidate_b: str,
    fold_map_a: Dict[int, float],
    fold_map_b: Dict[int, float],
    extra: Optional[dict] = None,
) -> List[dict]:
    folds = sorted(set(fold_map_a) & set(fold_map_b))
    if not folds:
        return []
    deltas = [fold_map_a[f] - fold_map_b[f] for f in folds]
    mean_delta = float(np.mean(deltas))
    n_pos = int(sum(1 for d in deltas if d > 0))
    rows = []
    for f, d in zip(folds, deltas):
        rec = {
            "comparison": comparison,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "fold_id": f,
            "auc_a": fold_map_a[f],
            "auc_b": fold_map_b[f],
            "paired_delta": d,
            "mean_delta": mean_delta,
            "n_folds": len(folds),
            "n_folds_positive": n_pos,
            "std_delta": float(np.std(deltas, ddof=0)) if len(deltas) > 1 else 0.0,
        }
        if extra:
            rec.update(extra)
        rows.append(rec)
    return rows


def build_cross_attention_paired_deltas(jobs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Paired fold comparisons for 18C analysis."""
    if jobs is None or jobs.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    done = jobs[(jobs["status"] == "done") & jobs["DrugMacro_AUC"].notna()].copy()
    if done.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    paired_rows: List[dict] = []
    residual_rows: List[dict] = []
    omics_rows: List[dict] = []

    mlp_own = _fold_lookup(done, MLP_OWN_PLUS_ID)
    p3_ids = sorted(
        {
            str(x)
            for x in done.loc[
                (done["architecture_family"] == "pooled_transformer")
                & (done["omics_mode"] == CONTEXT16)
                & done["architecture_id"].astype(str).str.contains("P3_deeper128", na=False),
                "architecture_id",
            ].unique()
        }
    )
    p3_context = _fold_lookup(done, p3_ids[0]) if p3_ids else {}

    ca = done[done["architecture_family"] == "cross_attention"].copy()
    ca_archs = ca["architecture_id"].dropna().unique().tolist()

    for arch in ca_archs:
        meta = ca[ca["architecture_id"] == arch].iloc[0]
        if str(meta["omics_mode"]) != CONTEXT16:
            continue
        fold_a = _fold_lookup(done, arch)
        if mlp_own:
            paired_rows.extend(
                _pair_rows(
                    comparison="cross_attn_context16_vs_mlp_own_plus_summary",
                    candidate_a=arch,
                    candidate_b=MLP_OWN_PLUS_ID,
                    fold_map_a=fold_a,
                    fold_map_b=mlp_own,
                    extra={
                        "transformer_config_id": meta.get("transformer_config_id", ""),
                        "residual_mode": meta.get("residual_mode", ""),
                    },
                )
            )
        if p3_context:
            paired_rows.extend(
                _pair_rows(
                    comparison="cross_attn_context16_vs_p3_context16",
                    candidate_a=arch,
                    candidate_b=p3_ids[0],
                    fold_map_a=fold_a,
                    fold_map_b=p3_context,
                    extra={
                        "transformer_config_id": meta.get("transformer_config_id", ""),
                        "residual_mode": meta.get("residual_mode", ""),
                    },
                )
            )

    for (cfg, omics), g in ca.groupby(["transformer_config_id", "omics_mode"], dropna=False):
        pure_ids = g.loc[g["residual_mode"] == "pure", "architecture_id"].unique()
        res_ids = g.loc[g["residual_mode"] == "pooled_residual", "architecture_id"].unique()
        if len(pure_ids) != 1 or len(res_ids) != 1:
            continue
        fold_res = _fold_lookup(done, res_ids[0])
        fold_pure = _fold_lookup(done, pure_ids[0])
        rows = _pair_rows(
            comparison="pooled_residual_minus_pure",
            candidate_a=str(res_ids[0]),
            candidate_b=str(pure_ids[0]),
            fold_map_a=fold_res,
            fold_map_b=fold_pure,
            extra={"transformer_config_id": cfg, "omics_mode": omics, "residual_mode": "pooled_residual"},
        )
        paired_rows.extend(rows)
        if rows:
            residual_rows.append(
                {
                    "transformer_config_id": cfg,
                    "omics_mode": omics,
                    "candidate_a": rows[0]["candidate_a"],
                    "candidate_b": rows[0]["candidate_b"],
                    "mean_delta": rows[0]["mean_delta"],
                    "std_delta": rows[0]["std_delta"],
                    "n_folds": rows[0]["n_folds"],
                    "n_folds_positive": rows[0]["n_folds_positive"],
                }
            )

    for (cfg, residual), g in ca.groupby(["transformer_config_id", "residual_mode"], dropna=False):
        own_ids = g.loc[g["omics_mode"] == OWN_PLUS, "architecture_id"].unique()
        ctx_ids = g.loc[g["omics_mode"] == CONTEXT16, "architecture_id"].unique()
        if len(own_ids) != 1 or len(ctx_ids) != 1:
            continue
        fold_ctx = _fold_lookup(done, ctx_ids[0])
        fold_own = _fold_lookup(done, own_ids[0])
        rows = _pair_rows(
            comparison="context16_minus_own_plus_summary",
            candidate_a=str(ctx_ids[0]),
            candidate_b=str(own_ids[0]),
            fold_map_a=fold_ctx,
            fold_map_b=fold_own,
            extra={"transformer_config_id": cfg, "residual_mode": residual, "omics_mode": CONTEXT16},
        )
        paired_rows.extend(rows)
        if rows:
            omics_rows.append(
                {
                    "transformer_config_id": cfg,
                    "residual_mode": residual,
                    "candidate_a": rows[0]["candidate_a"],
                    "candidate_b": rows[0]["candidate_b"],
                    "mean_delta": rows[0]["mean_delta"],
                    "std_delta": rows[0]["std_delta"],
                    "n_folds": rows[0]["n_folds"],
                    "n_folds_positive": rows[0]["n_folds_positive"],
                }
            )

    paired = pd.DataFrame(paired_rows)
    residual_summary = pd.DataFrame(residual_rows)
    omics_summary = pd.DataFrame(omics_rows)
    if not residual_summary.empty:
        residual_summary = residual_summary.sort_values("mean_delta", ascending=False).reset_index(drop=True)
    if not omics_summary.empty:
        omics_summary = omics_summary.sort_values("mean_delta", ascending=False).reset_index(drop=True)
    return paired, residual_summary, omics_summary


def select_formal_candidates(
    ranking: pd.DataFrame,
    *,
    top_k_pooled: int = 2,
    top_k_cross: int = 2,
) -> List[dict]:
    """Legacy generic picker (kept for tests); prefer select_formal_candidates_policy for lock."""
    if ranking.empty:
        return []
    selected = []
    seen = set()
    for _, row in ranking.iterrows():
        if row["architecture_id"] in seen:
            continue
        fam = str(row["architecture_family"])
        if fam in {"pooled_mlp", "pooled_transformer"}:
            n_fam = sum(1 for s in selected if s["architecture_family"] == fam)
            if n_fam >= top_k_pooled and fam != ranking.iloc[0]["architecture_family"]:
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
        n_mlp = sum(1 for s in selected if s["architecture_family"] == "pooled_mlp")
        n_tf = sum(1 for s in selected if s["architecture_family"] == "pooled_transformer")
        n_x = sum(1 for s in selected if s["architecture_family"] == "cross_attention")
        if n_mlp >= 1 and n_tf >= top_k_pooled - 1 and n_x >= top_k_cross and len(selected) >= 4:
            break
        if len(selected) >= 6:
            break
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


def _row_to_candidate(row: pd.Series, *, role: str) -> dict:
    return {
        "role": role,
        "architecture_id": row["architecture_id"],
        "architecture_family": row["architecture_family"],
        "omics_mode": row["omics_mode"],
        "transformer_config_id": row.get("transformer_config_id") or "",
        "residual_mode": row.get("residual_mode") or "",
        "selection_metric": "CV_DrugMacro_AUC",
        "mean_DrugMacro_AUC": row["mean_DrugMacro_AUC"],
        "mean_DrugMacro_AUPRC": row["mean_DrugMacro_AUPRC"],
    }


def _find_exact_or_prefix(ranking: pd.DataFrame, architecture_id: str = "", prefix: str = "") -> Optional[pd.Series]:
    if ranking.empty:
        return None
    if architecture_id:
        hit = ranking[ranking["architecture_id"] == architecture_id]
        if not hit.empty:
            return hit.iloc[0]
    if prefix:
        hit = ranking[ranking["architecture_id"].astype(str).str.startswith(prefix)]
        if not hit.empty:
            ctx = hit[hit["omics_mode"] == CONTEXT16]
            return (ctx if not ctx.empty else hit).iloc[0]
    return None


def select_formal_candidates_policy(ranking: pd.DataFrame) -> List[dict]:
    """
    Explicit 18D lock policy (5 candidates):
      1. MLP x own_plus_summary
      2. P3 deeper128 x context16
      3. P1 compact64 x context16
      4. best pure cross-attention
      5. best pooled-residual cross-attention
    """
    if ranking is None or ranking.empty:
        return []
    complete = filter_complete_screening(ranking)
    if complete.empty:
        raise RuntimeError(
            "Cannot select formal candidates: no architecture has "
            f"n_folds_done=={REQUIRED_SCREENING_FOLDS} and n_folds_with_auc=={REQUIRED_SCREENING_FOLDS}"
        )

    selected: List[dict] = []
    seen = set()

    def _add(row: Optional[pd.Series], role: str) -> None:
        if row is None:
            raise RuntimeError(f"Cannot select formal candidates: missing required role '{role}'")
        aid = row["architecture_id"]
        if aid in seen:
            return
        selected.append(_row_to_candidate(row, role=role))
        seen.add(aid)

    mlp = _find_exact_or_prefix(complete, architecture_id=MLP_OWN_PLUS_ID)
    _add(mlp, "anchor_mlp_own_plus_summary")

    p3 = _find_exact_or_prefix(
        complete,
        architecture_id=f"pooled_transformer__P3_deeper128__{CONTEXT16}",
        prefix=P3_CONTEXT16_PREFIX,
    )
    _add(p3, "best_pooled_transformer_p3_context16")

    p1 = _find_exact_or_prefix(
        complete,
        architecture_id=f"pooled_transformer__P1_compact64__{CONTEXT16}",
        prefix=P1_CONTEXT16_PREFIX,
    )
    _add(p1, "efficient_transformer_p1_context16")

    ca = complete[complete["architecture_family"] == "cross_attention"]
    pure = ca[ca["residual_mode"] == "pure"]
    residual = ca[ca["residual_mode"] == "pooled_residual"]
    if pure.empty:
        raise RuntimeError("Cannot select formal candidates: no complete pure cross-attention")
    if residual.empty:
        raise RuntimeError("Cannot select formal candidates: no complete pooled_residual cross-attention")
    _add(pure.iloc[0], "best_cross_attention_pure")
    _add(residual.iloc[0], "best_cross_attention_pooled_residual")

    return selected


def write_locked_selection(
    outdir: Path,
    *,
    ranking: pd.DataFrame,
    formal_candidates: List[dict],
    settings_path: str,
    completion: Optional[Dict[str, Dict[str, int]]] = None,
) -> Path:
    reports = outdir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if ranking.empty:
        raise ValueError("Cannot write selection lock without screening ranking")
    if not formal_candidates:
        raise ValueError("Cannot write selection lock without formal candidates")
    best = ranking.iloc[0].to_dict()
    split_meta = outdir / "splits" / "split_metadata.json"
    feature_meta = outdir / "data" / "round18_feature_coverage_preflight.json"
    settings = _load_json(Path(settings_path))
    feat_dir = (
        Path(settings.get("feature_root", "result/optimization_runs/round17r_18class/features"))
        / settings.get("feature_model_key", "r13_exp_008")
        / str(best.get("omics_mode") or OWN_PLUS)
    )
    feat_meta_path = feat_dir / "feature_metadata.json"

    b18 = (completion or {}).get("stage18b_screening_manifest.csv", {})
    c18 = (completion or {}).get("stage18c_cross_attention_manifest.csv", {})

    lock = {
        "selection_policy": "round18_explicit_5candidate",
        "architecture_id": formal_candidates[0]["architecture_id"],
        "architecture_family": formal_candidates[0]["architecture_family"],
        "omics_mode": formal_candidates[0]["omics_mode"],
        "transformer_config_id": formal_candidates[0].get("transformer_config_id") or "",
        "residual_mode": formal_candidates[0].get("residual_mode") or "",
        "transformer_config": {},
        "gin_config": settings.get("gin", {}),
        "split_manifest_sha256": _sha256_file(split_meta),
        "feature_metadata_sha256": _sha256_file(feat_meta_path),
        "feature_coverage_sha256": _sha256_file(feature_meta),
        "selection_metric": "CV_DrugMacro_AUC",
        "selected_mean_DrugMacro_AUC": formal_candidates[0].get("mean_DrugMacro_AUC"),
        "selected_without_internal_test": True,
        "selected_without_tcga": True,
        "internal_test_used": False,
        "tcga_used": False,
        "fixed_split_seed": 42,
        "fixed_model_seed": 101,
        "stage18b_completion": f"{b18.get('n_done', '?')}/{b18.get('n_total', '?')}",
        "stage18c_completion": f"{c18.get('n_done', '?')}/{c18.get('n_total', '?')}",
        "screening_best_overall": {
            "architecture_id": best.get("architecture_id"),
            "mean_DrugMacro_AUC": best.get("mean_DrugMacro_AUC"),
        },
        "formal_candidates": formal_candidates,
        "notes": [
            "Selection uses screening fold-mean DrugMacro AUC only.",
            "Lock uses explicit 5-candidate policy (not generic top-k).",
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

    screening = jobs[jobs["stage"].astype(str).isin(["18b", "18c"])].copy() if not jobs.empty else jobs
    ranking_raw = rank_screening_architectures(screening)
    ranking = filter_complete_screening(ranking_raw)
    ranking_path = reports / "round18_screening_architecture_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    ranking_raw_path = reports / "round18_screening_architecture_ranking_raw.csv"
    ranking_raw.to_csv(ranking_raw_path, index=False)

    paired, residual_effect, omics_interaction = build_cross_attention_paired_deltas(screening)
    paired_path = reports / "round18_cross_attention_paired_deltas.csv"
    residual_path = reports / "round18_residual_effect_summary.csv"
    omics_arch_path = reports / "round18_omics_architecture_interaction.csv"
    paired.to_csv(paired_path, index=False)
    residual_effect.to_csv(residual_path, index=False)
    omics_interaction.to_csv(omics_arch_path, index=False)

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

    resource = (
        screening[
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
        ].copy()
        if not screening.empty
        else pd.DataFrame()
    )
    resource_path = reports / "round18_resource_usage_summary.csv"
    resource.to_csv(resource_path, index=False)
    oom_path = reports / "round18_oom_summary.csv"
    resource.to_csv(oom_path, index=False)

    formal = jobs[jobs["stage"].astype(str) == "18d"].copy() if not jobs.empty else pd.DataFrame()
    formal_summary = rank_screening_architectures(formal) if not formal.empty else pd.DataFrame()
    formal_path = reports / "round18_formal_5cv_summary.csv"
    formal_summary.to_csv(formal_path, index=False)

    split_balance_src = root / "splits" / "fold_balance_report.csv"
    split_balance_dst = reports / "round18_split_balance_summary.csv"
    if split_balance_src.is_file():
        pd.read_csv(split_balance_src).to_csv(split_balance_dst, index=False)

    lock_path = None
    formal_candidates: List[dict] = []
    completion = None
    if write_lock:
        completion = assert_manifests_complete_for_lock(root)
        formal_candidates = select_formal_candidates_policy(ranking_raw)
        lock_path = write_locked_selection(
            root,
            ranking=ranking if not ranking.empty else ranking_raw,
            formal_candidates=formal_candidates,
            settings_path=settings_path,
            completion=completion,
        )
    elif not ranking.empty:
        try:
            formal_candidates = select_formal_candidates_policy(ranking_raw)
        except RuntimeError:
            formal_candidates = select_formal_candidates(ranking)

    md_path = reports / "round18_final_report.md"
    lines = [
        "# Round 18 Analysis Report",
        "",
        "## Screening ranking (complete 3-fold DrugMacro AUC only)",
        "",
        "Selection uses screening CV only; internal test and TCGA are excluded.",
        "Architectures enter ranking only when `n_folds_done == 3` and `n_folds_with_auc == 3`.",
        "",
    ]
    if not ranking.empty:
        lines.append(ranking.head(15).to_markdown(index=False))
    else:
        lines.append("_No complete screening architectures yet._")
    lines.extend(
        [
            "",
            "## Scientific notes",
            "",
            "- Omics encoder is frozen; CV is grouped response-prediction CV with",
            "  pretrained/transductive omics representations (not fully inductive).",
            "- `own_proto_context_projected_16` includes unlabeled TCGA prototype context.",
            "- Paired deltas: see `round18_cross_attention_paired_deltas.csv`.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "jobs": str(jobs_path),
        "ranking": str(ranking_path),
        "ranking_raw": str(ranking_raw_path),
        "omics_summary": str(omics_path),
        "resource": str(resource_path),
        "formal_summary": str(formal_path),
        "paired_deltas": str(paired_path),
        "residual_effect": str(residual_path),
        "omics_architecture_interaction": str(omics_arch_path),
        "report_md": str(md_path),
        "lock": str(lock_path) if lock_path else None,
        "n_jobs": int(len(jobs)),
        "n_ranking": int(len(ranking)),
        "formal_candidates": formal_candidates,
        "completion": completion,
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
