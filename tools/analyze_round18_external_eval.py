#!/usr/bin/env python3
"""Round 18E external evaluation: ensemble + metrics + paired bootstrap."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics
from tools.round18_prediction_ensemble import (
    assert_fold_prediction_uniqueness,
    assert_no_best_fold_selection,
    ensemble_fold_predictions,
)


COMPARE_PAIRS = [
    ("cross_attn__X3__pure__own_proto_context_projected_16", "pooled_mlp__own_plus_summary"),
    ("cross_attn__X3__pure__own_proto_context_projected_16", "pooled_transformer__P3_deeper128__own_proto_context_projected_16"),
    ("cross_attn__X3__pure__own_proto_context_projected_16", "pooled_transformer__P1_compact64__own_proto_context_projected_16"),
    ("cross_attn__X3__pure__own_proto_context_projected_16", "cross_attn__X3__pooled_residual__own_proto_context_projected_16"),
]

METRIC_KEYS = [
    "DrugMacro_AUC",
    "DrugMacro_AUPRC",
    "Global_AUC",
    "Global_AUPRC",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_pred_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def collect_fold_predictions(
    root: Path,
    *,
    kind: str,
    architecture_id: str,
    target_key: str,
    n_folds: int = 5,
) -> pd.DataFrame:
    frames = []
    for fold_id in range(n_folds):
        if kind == "internal":
            rd = root / "stage18e_internal" / architecture_id / f"fold_{fold_id}"
            pred = _read_pred_csv(rd / "internal_test_predictions.csv")
        else:
            rd = root / "stage18e_tcga" / architecture_id / target_key / f"fold_{fold_id}"
            pred = _read_pred_csv(rd / "tcga_predictions.csv")
        if pred is None:
            raise FileNotFoundError(
                f"Missing predictions for {architecture_id} {target_key} fold {fold_id}: {rd}"
            )
        frames.append(pred)
    df = pd.concat(frames, ignore_index=True)
    assert_no_best_fold_selection(df)
    assert_fold_prediction_uniqueness(df, required_folds=n_folds)
    return df


def metrics_from_pred(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    work = df.copy()
    if "DRUG_NAME" not in work.columns and "drug_name" in work.columns:
        work["DRUG_NAME"] = work["drug_name"]
    m = calculate_robust_drug_macro_metrics(work)
    return {
        "DrugMacro_AUC": m.get("DrugMacro_AUC"),
        "DrugMacro_AUPRC": m.get("DrugMacro_AUPRC"),
        "Global_AUC": m.get("Global_AUC"),
        "Global_AUPRC": m.get("Global_AUPRC"),
        "n_valid_auc_drugs": m.get("n_valid_auc_drugs"),
        "n_rows": int(len(work)),
    }


def _normalize_pred(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = df.copy()
    if "DRUG_NAME" not in out.columns and "drug_name" in out.columns:
        out["DRUG_NAME"] = out["drug_name"]
    if group_col not in out.columns and group_col == "Patient_id" and "ModelID" in out.columns:
        out[group_col] = out["ModelID"]
    out[group_col] = out[group_col].astype(str)
    out["DRUG_NAME"] = out["DRUG_NAME"].astype(str)
    out["Label"] = out["Label"].astype(int)
    out["probability"] = out["probability"].astype(float)
    return out


def _safe_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return None


def _safe_auprc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(average_precision_score(y, p))
    except ValueError:
        return None


def _pack_patient_lists(
    df: pd.DataFrame, group_col: str, ids: List[str], *, need_drug: bool
) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[List[np.ndarray]]]:
    """Ordered per-id arrays matching ``ids`` for integer-index resampling."""
    y_list: List[np.ndarray] = []
    p_list: List[np.ndarray] = []
    d_list: Optional[List[np.ndarray]] = [] if need_drug else None
    grouped = {str(gid): g for gid, g in df.groupby(group_col, sort=False)}
    for gid in ids:
        g = grouped[gid]
        y_list.append(g["Label"].to_numpy(dtype=np.int32))
        p_list.append(g["probability"].to_numpy(dtype=np.float64))
        if d_list is not None:
            d_list.append(g["DRUG_NAME"].to_numpy(dtype=object))
    return y_list, p_list, d_list


def _metric_from_concat(
    y: np.ndarray,
    p: np.ndarray,
    drug: Optional[np.ndarray],
    *,
    metric: str,
    min_samples: int = 10,
    min_pos: int = 2,
    min_neg: int = 2,
) -> Optional[float]:
    if metric == "Global_AUC":
        return _safe_auc(y, p)
    if metric == "Global_AUPRC":
        return _safe_auprc(y, p)
    if drug is None:
        raise ValueError(f"metric={metric} requires drug labels")
    # DrugMacro: group once via argsort instead of unique+boolean mask each drug
    order = np.argsort(drug, kind="mergesort")
    y_s = y[order]
    p_s = p[order]
    d_s = drug[order]
    scores: List[float] = []
    n = len(d_s)
    i = 0
    while i < n:
        j = i + 1
        while j < n and d_s[j] == d_s[i]:
            j += 1
        yd = y_s[i:j]
        pd_ = p_s[i:j]
        n_pos = int((yd == 1).sum())
        n_neg = int((yd == 0).sum())
        if len(yd) >= min_samples and n_pos >= min_pos and n_neg >= min_neg:
            if metric == "DrugMacro_AUC":
                s = _safe_auc(yd, pd_)
            else:
                s = _safe_auprc(yd, pd_)
            if s is not None:
                scores.append(s)
        i = j
    if not scores:
        return None
    return float(np.mean(scores))


def paired_bootstrap_delta(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    *,
    metric: str,
    group_col: str = "Patient_id",
    n_bootstrap: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, float]:
    """
    Paired bootstrap over unique group IDs (ModelID / Patient_id).
    Both models use identical sampled IDs each repeat.
    """
    a = _normalize_pred(pred_a, group_col)
    b = _normalize_pred(pred_b, group_col)
    ids = sorted(set(a[group_col]) & set(b[group_col]))
    if len(ids) < 5:
        raise ValueError(f"Too few shared {group_col} for bootstrap: {len(ids)}")

    need_drug = metric.startswith("DrugMacro")
    ya_list, pa_list, da_list = _pack_patient_lists(a, group_col, ids, need_drug=need_drug)
    yb_list, pb_list, db_list = _pack_patient_lists(b, group_col, ids, need_drug=need_drug)
    n_ids = len(ids)
    rng = np.random.default_rng(seed)
    deltas: List[float] = []

    for _ in range(int(n_bootstrap)):
        sample_idx = rng.integers(0, n_ids, size=n_ids)
        ya = np.concatenate([ya_list[i] for i in sample_idx])
        pa = np.concatenate([pa_list[i] for i in sample_idx])
        yb = np.concatenate([yb_list[i] for i in sample_idx])
        pb = np.concatenate([pb_list[i] for i in sample_idx])
        if need_drug:
            assert da_list is not None and db_list is not None
            da = np.concatenate([da_list[i] for i in sample_idx])
            db = np.concatenate([db_list[i] for i in sample_idx])
        else:
            da = db = None
        ma = _metric_from_concat(ya, pa, da, metric=metric)
        mb = _metric_from_concat(yb, pb, db, metric=metric)
        if ma is None or mb is None:
            continue
        deltas.append(float(ma) - float(mb))

    if not deltas:
        raise RuntimeError(f"No valid bootstrap deltas for metric={metric}")
    arr = np.asarray(deltas, dtype=float)
    alpha = (1.0 - ci) / 2.0
    return {
        "mean_delta": float(np.mean(arr)),
        "ci_lower": float(np.quantile(arr, alpha)),
        "ci_upper": float(np.quantile(arr, 1.0 - alpha)),
        "probability_delta_gt_zero": float(np.mean(arr > 0)),
        "n_bootstrap": int(len(arr)),
    }


def _bootstrap_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    meta = {
        "target_key": task["target_key"],
        "candidate_a": task["candidate_a"],
        "candidate_b": task["candidate_b"],
        "metric": task["metric"],
    }
    try:
        stats = paired_bootstrap_delta(
            task["pred_a"],
            task["pred_b"],
            metric=task["metric"],
            group_col=task["group_col"],
            n_bootstrap=task["n_bootstrap"],
            seed=task["seed"],
        )
        return {**meta, **stats, "error": ""}
    except Exception as exc:  # noqa: BLE001
        return {
            **meta,
            "mean_delta": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "probability_delta_gt_zero": float("nan"),
            "n_bootstrap": 0,
            "error": str(exc),
        }


def analyze_round18_external_eval(
    outdir: str,
    *,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 42,
    reuse_ensemble: bool = True,
    n_jobs: int = 0,
) -> Dict:
    root = Path(outdir)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    lock = _load_json(reports / "round18_locked_selection.json")
    candidates = lock.get("formal_candidates") or []
    if len(candidates) < 1:
        raise RuntimeError("No formal_candidates in lock")

    settings = _load_json(Path("config/round18_architecture_settings.json"))
    tcga_targets = [t["key"] for t in settings.get("tcga", {}).get("eval_targets", [])]
    n_folds = int(settings["formal_cv"]["n_splits"])

    summary_rows = []
    bootstrap_rows = []
    ensemble_by_key: Dict[Tuple[str, str], pd.DataFrame] = {}

    for cand in candidates:
        arch = cand["architecture_id"]
        role = cand.get("role") or arch

        # Internal test
        ens_path = reports / f"round18e_internal_{arch}_ensemble_predictions.csv"
        fold_path = reports / f"round18e_internal_{arch}_fold_predictions.csv"
        if reuse_ensemble and ens_path.is_file():
            ens = pd.read_csv(ens_path)
        else:
            fold_df = collect_fold_predictions(
                root, kind="internal", architecture_id=arch, target_key="internal_test", n_folds=n_folds
            )
            fold_df.to_csv(fold_path, index=False)
            ens = ensemble_fold_predictions(fold_df, required_folds=n_folds)
            ens.to_csv(ens_path, index=False)
        met = metrics_from_pred(ens)
        summary_rows.append(
            {
                "target_key": "internal_test",
                "candidate_id": role,
                "architecture_id": arch,
                "architecture_family": cand.get("architecture_family"),
                "omics_mode": cand.get("omics_mode"),
                **met,
            }
        )
        ensemble_by_key[(arch, "internal_test")] = ens

        for tkey in tcga_targets:
            ens_path = reports / f"round18e_tcga_{arch}_{tkey}_ensemble_predictions.csv"
            fold_path = reports / f"round18e_tcga_{arch}_{tkey}_fold_predictions.csv"
            if reuse_ensemble and ens_path.is_file():
                ens = pd.read_csv(ens_path)
            else:
                fold_df = collect_fold_predictions(
                    root, kind="tcga", architecture_id=arch, target_key=tkey, n_folds=n_folds
                )
                fold_df.to_csv(fold_path, index=False)
                ens = ensemble_fold_predictions(fold_df, required_folds=n_folds)
                ens.to_csv(ens_path, index=False)
            met = metrics_from_pred(ens)
            summary_rows.append(
                {
                    "target_key": tkey,
                    "candidate_id": role,
                    "architecture_id": arch,
                    "architecture_family": cand.get("architecture_family"),
                    "omics_mode": cand.get("omics_mode"),
                    **met,
                }
            )
            ensemble_by_key[(arch, tkey)] = ens

    summary = pd.DataFrame(summary_rows)
    summary_path = reports / "round18_external_eval_summary.csv"
    summary.to_csv(summary_path, index=False)

    internal = summary[summary["target_key"] == "internal_test"].copy()
    internal_path = reports / "round18_internal_test_summary.csv"
    internal.to_csv(internal_path, index=False)

    tcga_sum = summary[summary["target_key"] != "internal_test"].copy()
    tcga_path = reports / "round18_five_target_tcga_summary.csv"
    tcga_sum.to_csv(tcga_path, index=False)

    integrated_rows = []
    for cand in candidates:
        arch = cand["architecture_id"]
        sub = tcga_sum[tcga_sum.architecture_id == arch]
        integrated_rows.append(
            {
                "architecture_id": arch,
                "candidate_id": cand.get("role") or arch,
                "Integrated5_n_tcga_eval_targets": int(sub["target_key"].nunique()),
                "Integrated5_DrugMacro_TCGA_AUC": float(sub["DrugMacro_AUC"].mean()) if len(sub) else np.nan,
                "Integrated5_DrugMacro_TCGA_AUPRC": float(sub["DrugMacro_AUPRC"].mean()) if len(sub) else np.nan,
                "Integrated5_Global_TCGA_AUC": float(sub["Global_AUC"].mean()) if len(sub) else np.nan,
                "Integrated5_Global_TCGA_AUPRC": float(sub["Global_AUPRC"].mean()) if len(sub) else np.nan,
            }
        )
    integ_df = pd.DataFrame(integrated_rows)
    integ_path = reports / "round18_integrated5_summary.csv"
    integ_df.to_csv(integ_path, index=False)

    targets_all = ["internal_test"] + list(tcga_targets)
    tasks: List[Dict[str, Any]] = []
    for cand_a, cand_b in COMPARE_PAIRS:
        for tkey in targets_all:
            if (cand_a, tkey) not in ensemble_by_key or (cand_b, tkey) not in ensemble_by_key:
                continue
            for metric in METRIC_KEYS:
                tasks.append(
                    {
                        "pred_a": ensemble_by_key[(cand_a, tkey)],
                        "pred_b": ensemble_by_key[(cand_b, tkey)],
                        "metric": metric,
                        "group_col": "Patient_id",
                        "n_bootstrap": int(n_bootstrap),
                        "seed": int(bootstrap_seed),
                        "target_key": tkey,
                        "candidate_a": cand_a,
                        "candidate_b": cand_b,
                    }
                )

    workers = int(n_jobs)
    if workers <= 0:
        workers = min(16, max(1, (os.cpu_count() or 4) - 1))
    print(
        f"[18E analyze] paired bootstrap jobs={len(tasks)} n_bootstrap={n_bootstrap} n_jobs={workers}",
        flush=True,
    )
    if workers == 1 or len(tasks) <= 1:
        for i, task in enumerate(tasks, 1):
            bootstrap_rows.append(_bootstrap_worker(task))
            if i % 4 == 0 or i == len(tasks):
                print(f"[18E analyze] bootstrap progress {i}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_bootstrap_worker, task) for task in tasks]
            done = 0
            for fut in as_completed(futures):
                bootstrap_rows.append(fut.result())
                done += 1
                if done % 4 == 0 or done == len(tasks):
                    print(f"[18E analyze] bootstrap progress {done}/{len(tasks)}", flush=True)

    boot = pd.DataFrame(bootstrap_rows)
    boot_path = reports / "round18e_paired_bootstrap_deltas.csv"
    boot.to_csv(boot_path, index=False)

    x3_pure = "cross_attn__X3__pure__own_proto_context_projected_16"
    x3_res = "cross_attn__X3__pooled_residual__own_proto_context_projected_16"
    mlp = "pooled_mlp__own_plus_summary"
    verdict = {"cross_attention_external_success": False, "notes": []}
    try:
        it_x3 = float(internal[internal.architecture_id == x3_pure].iloc[0]["DrugMacro_AUC"])
        it_mlp = float(internal[internal.architecture_id == mlp].iloc[0]["DrugMacro_AUC"])
        x3_ok_internal = it_x3 >= it_mlp
        tcga_x3 = tcga_sum[tcga_sum.architecture_id == x3_pure]
        tcga_mlp = tcga_sum[tcga_sum.architecture_id == mlp].set_index("target_key")
        n_non_worse = 0
        for _, row in tcga_x3.iterrows():
            tkey = row["target_key"]
            if tkey in tcga_mlp.index and float(row["DrugMacro_AUC"]) >= float(
                tcga_mlp.loc[tkey, "DrugMacro_AUC"]
            ):
                n_non_worse += 1
        verdict["cross_attention_external_success"] = bool(x3_ok_internal and n_non_worse >= 3)
        verdict["notes"].append(
            f"internal X3_pure={it_x3:.4f} vs MLP={it_mlp:.4f}; TCGA non-worse={n_non_worse}/5"
        )
        it_res = internal[internal.architecture_id == x3_res]
        if not it_res.empty:
            res_auc = float(it_res.iloc[0]["DrugMacro_AUC"])
            verdict["notes"].append(
                f"internal residual={res_auc:.4f} (delta vs pure={res_auc - it_x3:.4f})"
            )
        # Prefer pure if nearly identical
        verdict["prefer_pure_for_18F"] = True
        if not it_res.empty and abs(float(it_res.iloc[0]["DrugMacro_AUC"]) - it_x3) < 0.002:
            verdict["notes"].append("pure≈residual on internal test; prefer pure for 18F")
    except Exception as exc:  # noqa: BLE001
        verdict["notes"].append(f"verdict computation failed: {exc}")

    verdict_path = reports / "round18e_success_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    out = {
        "summary": str(summary_path),
        "internal_summary": str(internal_path),
        "tcga_summary": str(tcga_path),
        "integrated5": str(integ_path),
        "paired_bootstrap": str(boot_path),
        "verdict": str(verdict_path),
        "n_summary_rows": int(len(summary)),
        "success": verdict.get("cross_attention_external_success"),
    }
    print(json.dumps(out, indent=2, default=str))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Round 18E external evaluation")
    parser.add_argument("--outdir", default="result/optimization_runs/round18_architecture")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="Parallel workers for paired bootstrap (0 = auto, min(16, cpu-1))",
    )
    parser.add_argument("--no-reuse-ensemble", action="store_true")
    args = parser.parse_args()
    analyze_round18_external_eval(
        args.outdir,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        reuse_ensemble=not args.no_reuse_ensemble,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
