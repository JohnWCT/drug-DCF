#!/usr/bin/env python3
"""Round 20 Stage 20B: pooled E3 vs gated pooled fusion (winner dimension)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result/optimization_runs/round20_unseen_drug_closure"
DEV_ROWS = PROJECT_ROOT / "result/optimization_runs/round19_factorial/splits/development_rows.csv"
SPLIT_SEEDS = [52, 62, 72]
N_SPLITS = 5
MODEL_SEED = 101


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_dimension_lock(path: Path) -> dict:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED":
        raise ValueError(f"Stage 20A lock not LOCKED: {path}")
    if lock.get("selected_context") not in {"C16", "C32"}:
        raise ValueError(f"Invalid selected_context: {lock.get('selected_context')}")
    return lock


def build_stage20b_manifest(
    *,
    dimension_lock_path: Path,
    resolved_e3_path: Path = RESULT_ROOT / "stage20_0/resolved_e3.json",
    outdir: Path = RESULT_ROOT / "stage20b_predictor",
    reuse_e3_from_20a: bool = True,
) -> dict:
    """Build Stage 20B jobs.

    If ``reuse_e3_from_20a`` and the Stage 20A winner arm completed, only emit
    the 15 gated jobs and point B_E3 at the existing Stage 20A job directories.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lock = _load_dimension_lock(dimension_lock_path)
    ctx = lock["selected_context"]
    omics_dim = int(lock["selected_omics_dim"])
    feature_dir = str(lock["selected_feature_dir"])
    lock_hash = _sha256_file(Path(dimension_lock_path))
    e3 = json.loads(Path(resolved_e3_path).read_text(encoding="utf-8"))
    e3_body = e3.get("resolved_e3", e3)
    e3_hash = _sha256_text(json.dumps(e3_body, sort_keys=True))
    feat_sha = _sha256_file(Path(feature_dir) / "ccle_latent_proto.pkl")

    stage20a_jobs_root = RESULT_ROOT / "stage20a_dimension" / "jobs"
    jobs: List[dict] = []
    reused = 0
    for seed in SPLIT_SEEDS:
        assign = RESULT_ROOT / "splits" / f"round20a_drug_heldout_seed{seed}_assignments.csv"
        assign_sha = _sha256_file(assign)
        for fold in range(N_SPLITS):
            # Baseline E3 — reuse Stage 20A winner job when possible
            e3_src = stage20a_jobs_root / f"r20a__A_{ctx}_E3__ss{seed}__f{fold}__ms{MODEL_SEED}"
            e3_job_id = f"r20b__B_E3__{ctx}__ss{seed}__f{fold}__ms{MODEL_SEED}"
            if reuse_e3_from_20a and (e3_src / "metrics.json").is_file():
                jobs.append(
                    {
                        "job_id": e3_job_id,
                        "stage": "20B",
                        "candidate_id": "B_E3",
                        "predictor_id": "pooled_e3",
                        "predictor_kind": "pooled_e3",
                        "context_id": ctx,
                        "omics_dim": omics_dim,
                        "drug_encoder_id": "D0",
                        "split_seed": int(seed),
                        "fold": int(fold),
                        "model_seed": MODEL_SEED,
                        "split_assignment_path": str(assign),
                        "split_assignment_sha256": assign_sha,
                        "feature_store_path": feature_dir,
                        "feature_store_sha256": feat_sha,
                        "e3_contract_sha256": e3_hash,
                        "stage20a_lock_sha256": lock_hash,
                        "response_path": str(DEV_ROWS),
                        "output_dir": str(e3_src),  # reuse
                        "reused_from_stage20a": True,
                        "skip_train": True,
                    }
                )
                reused += 1
            else:
                jobs.append(
                    {
                        "job_id": e3_job_id,
                        "stage": "20B",
                        "candidate_id": "B_E3",
                        "predictor_id": "pooled_e3",
                        "predictor_kind": "pooled_e3",
                        "context_id": ctx,
                        "omics_dim": omics_dim,
                        "drug_encoder_id": "D0",
                        "split_seed": int(seed),
                        "fold": int(fold),
                        "model_seed": MODEL_SEED,
                        "split_assignment_path": str(assign),
                        "split_assignment_sha256": assign_sha,
                        "feature_store_path": feature_dir,
                        "feature_store_sha256": feat_sha,
                        "e3_contract_sha256": e3_hash,
                        "stage20a_lock_sha256": lock_hash,
                        "response_path": str(DEV_ROWS),
                        "output_dir": str(outdir / "jobs" / e3_job_id),
                        "reused_from_stage20a": False,
                        "skip_train": False,
                    }
                )

            gated_id = f"r20b__B_GATED__{ctx}__ss{seed}__f{fold}__ms{MODEL_SEED}"
            jobs.append(
                {
                    "job_id": gated_id,
                    "stage": "20B",
                    "candidate_id": "B_GATED",
                    "predictor_id": "gated_pooled_fusion",
                    "predictor_kind": "gated_pooled_fusion",
                    "context_id": ctx,
                    "omics_dim": omics_dim,
                    "drug_encoder_id": "D0",
                    "split_seed": int(seed),
                    "fold": int(fold),
                    "model_seed": MODEL_SEED,
                    "split_assignment_path": str(assign),
                    "split_assignment_sha256": assign_sha,
                    "feature_store_path": feature_dir,
                    "feature_store_sha256": feat_sha,
                    "e3_contract_sha256": e3_hash,
                    "stage20a_lock_sha256": lock_hash,
                    "response_path": str(DEV_ROWS),
                    "output_dir": str(outdir / "jobs" / gated_id),
                    "reused_from_stage20a": False,
                    "skip_train": False,
                }
            )

    manifest_path = outdir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, sort_keys=True) + "\n")

    train_jobs = [j for j in jobs if not j.get("skip_train")]
    validation = {
        "jobs_total": len(jobs),
        "train_jobs": len(train_jobs),
        "reused_e3_jobs": reused,
        "selected_context": ctx,
        "selected_omics_dim": omics_dim,
        "paired_comparisons": 15,
        "manifest_path": str(manifest_path),
        "schema_status": "PASS" if len(jobs) == 30 else "FAIL",
    }
    _write_json(outdir / "manifest_validation.json", validation)
    return validation


def analyze_stage20b(
    *,
    stage_dir: Path = RESULT_ROOT / "stage20b_predictor",
    auprc_max_drop: float = 0.01,
    major_fail_auc_delta: float = -0.02,
    min_nonworse_seeds: int = 2,
    strict: bool = True,
) -> dict:
    stage_dir = Path(stage_dir)
    jobs = [
        json.loads(line)
        for line in (stage_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    missing = []
    for job in jobs:
        mp = Path(job["output_dir"]) / "metrics.json"
        if not mp.is_file():
            missing.append(job["job_id"])
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m.get("status") != "COMPLETE":
            missing.append(job["job_id"])
            continue
        rows.append(
            {
                "job_id": job["job_id"],
                "candidate_id": job["candidate_id"],
                "split_seed": job["split_seed"],
                "fold": job["fold"],
                "DrugMacro_AUC": m["metrics"].get("DrugMacro_AUC"),
                "DrugMacro_AUPRC": m["metrics"].get("DrugMacro_AUPRC"),
            }
        )
    if strict and missing:
        raise RuntimeError(f"Stage 20B incomplete: {missing}")

    df = pd.DataFrame(rows)
    reports = stage_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    pivot = df.pivot_table(
        index=["split_seed", "fold"], columns="candidate_id",
        values=["DrugMacro_AUC", "DrugMacro_AUPRC"], aggfunc="first",
    )
    pair_rows = []
    for (seed, fold), r in pivot.iterrows():
        e3_auc = r[("DrugMacro_AUC", "B_E3")]
        g_auc = r[("DrugMacro_AUC", "B_GATED")]
        e3_ap = r[("DrugMacro_AUPRC", "B_E3")]
        g_ap = r[("DrugMacro_AUPRC", "B_GATED")]
        pair_rows.append(
            {
                "split_seed": int(seed),
                "fold": int(fold),
                "e3_auc": e3_auc,
                "gated_auc": g_auc,
                "delta_auc": g_auc - e3_auc,
                "e3_auprc": e3_ap,
                "gated_auprc": g_ap,
                "delta_auprc": g_ap - e3_ap,
            }
        )
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(reports / "stage20b_pairwise.csv", index=False)

    cand = (
        df.groupby("candidate_id")
        .agg(
            mean_auc=("DrugMacro_AUC", "mean"),
            std_auc=("DrugMacro_AUC", "std"),
            mean_auprc=("DrugMacro_AUPRC", "mean"),
            std_auprc=("DrugMacro_AUPRC", "std"),
            worst_fold_auc=("DrugMacro_AUC", "min"),
            completed_jobs=("job_id", "count"),
        )
        .reset_index()
    )
    cand.to_csv(reports / "stage20b_candidate_summary.csv", index=False)

    seed_rows = []
    for seed, g in pairwise.groupby("split_seed"):
        seed_rows.append(
            {
                "split_seed": int(seed),
                "e3_mean_auc": g["e3_auc"].mean(),
                "gated_mean_auc": g["gated_auc"].mean(),
                "delta_auc": g["gated_auc"].mean() - g["e3_auc"].mean(),
                "e3_mean_auprc": g["e3_auprc"].mean(),
                "gated_mean_auprc": g["gated_auprc"].mean(),
                "delta_auprc": g["gated_auprc"].mean() - g["e3_auprc"].mean(),
            }
        )
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(reports / "stage20b_seed_summary.csv", index=False)

    mean_auc_delta = float(df.loc[df.candidate_id == "B_GATED", "DrugMacro_AUC"].mean()
                           - df.loc[df.candidate_id == "B_E3", "DrugMacro_AUC"].mean())
    mean_ap_delta = float(df.loc[df.candidate_id == "B_GATED", "DrugMacro_AUPRC"].mean()
                          - df.loc[df.candidate_id == "B_E3", "DrugMacro_AUPRC"].mean())
    seed_deltas = {str(int(r.split_seed)): float(r.delta_auc) for r in seed_summary.itertuples()}
    nonworse = sum(1 for d in seed_deltas.values() if d >= 0)
    worst = min(seed_deltas.values()) if seed_deltas else 0.0

    guardrails = {
        "g1_mean_auc_nonworse": mean_auc_delta >= 0,
        "g2_seed_majority": nonworse >= min_nonworse_seeds,
        "g3_auprc": mean_ap_delta >= -auprc_max_drop,
        "g4_no_major_fail": worst >= major_fail_auc_delta,
        "g5_complete": len(missing) == 0 and len(df) == 30,
    }
    report = {
        "candidate": "B_GATED",
        "baseline": "B_E3",
        "mean_auc_delta": mean_auc_delta,
        "mean_auprc_delta": mean_ap_delta,
        "seed_auc_deltas": seed_deltas,
        "guardrails": guardrails,
        "all_pass": all(guardrails.values()),
        "missing_jobs": missing,
    }
    _write_json(stage_dir / "stage20b_guardrail_report.json", report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build-manifest")
    b.add_argument(
        "--dimension-lock",
        default=str(RESULT_ROOT / "stage20a_dimension/stage20a_dimension_decision.json"),
    )
    b.add_argument("--outdir", default=str(RESULT_ROOT / "stage20b_predictor"))
    a = sub.add_parser("analyze")
    a.add_argument("--stage-dir", default=str(RESULT_ROOT / "stage20b_predictor"))
    a.add_argument("--no-strict", action="store_true")
    args = p.parse_args()
    if args.command == "build-manifest":
        print(json.dumps(build_stage20b_manifest(
            dimension_lock_path=Path(args.dimension_lock), outdir=Path(args.outdir)
        ), indent=2))
    else:
        print(json.dumps(analyze_stage20b(
            stage_dir=Path(args.stage_dir), strict=not args.no_strict
        ), indent=2))


if __name__ == "__main__":
    main()
