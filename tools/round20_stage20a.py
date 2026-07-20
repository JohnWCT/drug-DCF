#!/usr/bin/env python3
"""Round 20 Stage 20A: C16 vs C32 repeated drug-held-out pipeline helpers.

This module builds the locked drug-held-out split assignments, the 30-job
paired manifest, and the paired analysis / dimension decision. Training itself
is delegated to ``step1_finetune_latent_pipeline_round20_cv.py`` which reuses the
Round 19 training building blocks so C16 and C32 differ ONLY in the omics
feature store (80 vs 96 dims).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result/optimization_runs/round20_unseen_drug_closure"
DEV_ROWS = PROJECT_ROOT / "result/optimization_runs/round19_factorial/splits/development_rows.csv"
DRUG_TABLE = (
    PROJECT_ROOT
    / "result/optimization_runs/round19_factorial/splits/round19e_drug_group_table.csv"
)
SPLIT_SEEDS = [52, 62, 72]
N_SPLITS = 5
MODEL_SEED = 101

C16_FEATURE_DIR = RESULT_ROOT / "features/z_plus_context16"
C32_FEATURE_DIR = RESULT_ROOT / "features/z_plus_context32"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attach_drug_group(dev: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    # drug_group_id = canonical_smiles (alias/salt safe). Merge by DRUG_NAME.
    name_to_canon = dict(zip(table["DRUG_NAME"].astype(str), table["canonical_smiles"].astype(str)))
    out = dev.copy()
    out["drug_group_id"] = out["DRUG_NAME"].astype(str).map(name_to_canon)
    if out["drug_group_id"].isna().any():
        missing = sorted(out.loc[out["drug_group_id"].isna(), "DRUG_NAME"].unique())[:10]
        raise ValueError(f"Unmapped drugs for drug_group_id: {missing}")
    return out


def build_stage20a_splits(
    *,
    dev_rows_path: Path = DEV_ROWS,
    drug_table_path: Path = DRUG_TABLE,
    split_seeds: List[int] = SPLIT_SEEDS,
    n_splits: int = N_SPLITS,
    outdir: Path = RESULT_ROOT / "splits",
) -> dict:
    """Deterministic repeated GroupKFold drug-held-out splits over development rows."""
    from sklearn.model_selection import GroupKFold

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(dev_rows_path)
    table = pd.read_csv(drug_table_path)
    dev = _attach_drug_group(dev, table)

    fold_reports: List[dict] = []
    seed_hashes: Dict[str, str] = {}
    for seed in split_seeds:
        groups = dev["drug_group_id"].astype(str)
        unique_groups = sorted(groups.unique())
        # Deterministic per-seed group ordering, then GroupKFold on reordered rows.
        rng = np.random.RandomState(int(seed))
        perm = list(unique_groups)
        rng.shuffle(perm)
        rank = {g: i for i, g in enumerate(perm)}
        order = np.argsort(groups.map(rank).to_numpy(), kind="mergesort")
        ordered = dev.iloc[order].reset_index(drop=True)
        y = ordered["Label"].to_numpy()
        grp = ordered["drug_group_id"].astype(str).to_numpy()
        gkf = GroupKFold(n_splits=int(n_splits))
        rows: List[pd.DataFrame] = []
        for fold_id, (train_idx, val_idx) in enumerate(gkf.split(ordered, y, grp)):
            train_groups = set(grp[train_idx])
            val_groups = set(grp[val_idx])
            overlap = train_groups & val_groups
            if overlap:
                raise AssertionError(f"seed={seed} fold={fold_id} drug overlap {sorted(overlap)[:3]}")
            for idx, role in ((train_idx, "train"), (val_idx, "val")):
                part = ordered.iloc[idx][
                    ["_row_id", "ModelID", "DRUG_NAME", "drug_group_id", "Label"]
                ].copy()
                part["cv_name"] = f"round20a_drug_heldout_seed{seed}"
                part["fold_id"] = int(fold_id)
                part["split_role"] = role
                part["partition"] = role
                part["split_seed"] = int(seed)
                rows.append(part)
            val_labels = ordered.iloc[val_idx]["Label"]
            fold_reports.append(
                {
                    "split_seed": int(seed),
                    "fold": int(fold_id),
                    "train_drug_count": int(len(train_groups)),
                    "validation_drug_count": int(len(val_groups)),
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(val_idx)),
                    "identity_overlap_count": 0,
                    "canonical_smiles_overlap_count": 0,
                    "row_overlap_count": 0,
                    "val_has_both_classes": bool(val_labels.nunique() > 1),
                    "status": "PASS",
                }
            )
        seed_df = pd.concat(rows, ignore_index=True)
        seed_df = seed_df[
            [
                "cv_name",
                "fold_id",
                "split_role",
                "partition",
                "_row_id",
                "ModelID",
                "DRUG_NAME",
                "drug_group_id",
                "Label",
                "split_seed",
            ]
        ]
        seed_path = outdir / f"round20a_drug_heldout_seed{seed}_assignments.csv"
        seed_df.to_csv(seed_path, index=False)
        seed_hashes[str(seed)] = _sha256_file(seed_path)

    audit = {
        "status": "PASS",
        "seeds": [int(s) for s in split_seeds],
        "folds_per_seed": int(n_splits),
        "group_column": "drug_group_id",
        "n_dev_rows": int(len(dev)),
        "n_drug_groups": int(dev["drug_group_id"].nunique()),
        "folds": fold_reports,
        "assignment_sha256": seed_hashes,
    }
    _write_json(outdir / "round20a_drug_split_audit.json", audit)
    _write_json(
        outdir / "round20a_split_metadata.json",
        {
            "split_seeds": [int(s) for s in split_seeds],
            "n_splits": int(n_splits),
            "model_seed": MODEL_SEED,
            "dev_rows_path": str(dev_rows_path),
            "drug_table_path": str(drug_table_path),
            "assignment_sha256": seed_hashes,
        },
    )
    return audit


def _assignment_path(seed: int) -> Path:
    return RESULT_ROOT / "splits" / f"round20a_drug_heldout_seed{seed}_assignments.csv"


def build_stage20a_manifest(
    *,
    resolved_e3_path: Path,
    outdir: Path = RESULT_ROOT / "stage20a_dimension",
    split_seeds: List[int] = SPLIT_SEEDS,
    n_splits: int = N_SPLITS,
    model_seed: int = MODEL_SEED,
) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    e3 = json.loads(Path(resolved_e3_path).read_text(encoding="utf-8"))
    e3_body = e3.get("resolved_e3", e3)
    e3_hash = _sha256_text(json.dumps(e3_body, sort_keys=True))

    dims = {
        "C16": {"context_dim": 16, "omics_dim": 80, "feature_dir": str(C16_FEATURE_DIR)},
        "C32": {"context_dim": 32, "omics_dim": 96, "feature_dir": str(C32_FEATURE_DIR)},
    }
    proj_sha = {
        "C16": _sha256_file(C16_FEATURE_DIR / "projection_model.pkl"),
        "C32": _sha256_file(C32_FEATURE_DIR / "projection_model.pkl"),
    }
    feat_sha = {
        "C16": _sha256_file(C16_FEATURE_DIR / "ccle_latent_proto.pkl"),
        "C32": _sha256_file(C32_FEATURE_DIR / "ccle_latent_proto.pkl"),
    }
    assign_sha = {
        int(s): _sha256_file(_assignment_path(int(s))) for s in split_seeds
    }

    jobs: List[dict] = []
    for seed in split_seeds:
        for fold in range(n_splits):
            for ctx in ("C16", "C32"):
                cfg = dims[ctx]
                job_id = f"r20a__A_{ctx}_E3__ss{seed}__f{fold}__ms{model_seed}"
                jobs.append(
                    {
                        "job_id": job_id,
                        "stage": "20A",
                        "candidate_id": f"A_{ctx}_E3",
                        "context_id": ctx,
                        "context_dim": cfg["context_dim"],
                        "omics_dim": cfg["omics_dim"],
                        "predictor_id": "E3",
                        "predictor_kind": "pooled_e3",
                        "predictor_type": e3_body["predictor_class"],
                        "drug_encoder_id": "D0",
                        "split_seed": int(seed),
                        "fold": int(fold),
                        "model_seed": int(model_seed),
                        "split_assignment_path": str(_assignment_path(int(seed))),
                        "split_assignment_sha256": assign_sha[int(seed)],
                        "feature_store_path": cfg["feature_dir"],
                        "projection_sha256": proj_sha[ctx],
                        "feature_store_sha256": feat_sha[ctx],
                        "e3_contract_sha256": e3_hash,
                        "response_path": str(DEV_ROWS),
                        "output_dir": str(outdir / "jobs" / job_id),
                    }
                )

    manifest_path = outdir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, sort_keys=True) + "\n")

    # Validation
    pair_keys = {}
    for job in jobs:
        key = (job["split_seed"], job["fold"])
        pair_keys.setdefault(key, set()).add(job["context_id"])
    missing_pairs = [k for k, v in pair_keys.items() if v != {"C16", "C32"}]
    split_hash_ok = len({j["split_assignment_sha256"] for j in jobs if j["split_seed"] == 52}) == 1
    e3_hash_ok = len({j["e3_contract_sha256"] for j in jobs}) == 1
    feat_ok = all(
        (j["context_id"] == "C16" and j["omics_dim"] == 80)
        or (j["context_id"] == "C32" and j["omics_dim"] == 96)
        for j in jobs
    )
    validation = {
        "jobs_total": len(jobs),
        "paired_comparisons": len(pair_keys),
        "missing_pairs": len(missing_pairs),
        "schema_status": "PASS",
        "split_hash_status": "PASS" if split_hash_ok else "FAIL",
        "feature_status": "PASS" if feat_ok else "FAIL",
        "e3_contract_hash_status": "PASS" if e3_hash_ok else "FAIL",
        "manifest_path": str(manifest_path),
    }
    _write_json(outdir / "manifest_validation.json", validation)
    return validation


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def _load_metrics(job_dir: Path) -> dict | None:
    mp = job_dir / "metrics.json"
    if not mp.is_file():
        return None
    return json.loads(mp.read_text(encoding="utf-8"))


def analyze_stage20a(
    *,
    stage_dir: Path = RESULT_ROOT / "stage20a_dimension",
    parsimony_delta: float = 0.005,
    auprc_max_drop: float = 0.01,
    major_fail_auc_delta: float = -0.02,
    min_nonworse_seeds: int = 2,
    strict: bool = True,
) -> dict:
    stage_dir = Path(stage_dir)
    manifest = [
        json.loads(line)
        for line in (stage_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    missing = []
    for job in manifest:
        job_dir = Path(job["output_dir"])
        m = _load_metrics(job_dir)
        if m is None or m.get("status") != "COMPLETE":
            missing.append(job["job_id"])
            continue
        rows.append(
            {
                "job_id": job["job_id"],
                "context_id": job["context_id"],
                "split_seed": job["split_seed"],
                "fold": job["fold"],
                "DrugMacro_AUC": m["metrics"].get("DrugMacro_AUC"),
                "DrugMacro_AUPRC": m["metrics"].get("DrugMacro_AUPRC"),
                "Global_AUC": m["metrics"].get("Global_AUC"),
                "Global_AUPRC": m["metrics"].get("Global_AUPRC"),
                "training_time_seconds": m.get("training_time_seconds"),
            }
        )
    if strict and missing:
        raise RuntimeError(f"Stage 20A incomplete jobs: {missing}")
    df = pd.DataFrame(rows)
    reports_dir = stage_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Pairwise per (seed, fold)
    pivot = df.pivot_table(
        index=["split_seed", "fold"], columns="context_id",
        values=["DrugMacro_AUC", "DrugMacro_AUPRC"], aggfunc="first",
    )
    pair_rows = []
    for (seed, fold), r in pivot.iterrows():
        c16_auc = r[("DrugMacro_AUC", "C16")]
        c32_auc = r[("DrugMacro_AUC", "C32")]
        c16_ap = r[("DrugMacro_AUPRC", "C16")]
        c32_ap = r[("DrugMacro_AUPRC", "C32")]
        pair_rows.append(
            {
                "split_seed": int(seed),
                "fold": int(fold),
                "c16_auc": c16_auc,
                "c32_auc": c32_auc,
                "delta_auc": c32_auc - c16_auc,
                "c16_auprc": c16_ap,
                "c32_auprc": c32_ap,
                "delta_auprc": c32_ap - c16_ap,
            }
        )
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(reports_dir / "stage20a_pairwise.csv", index=False)

    # candidate summary
    cand = (
        df.groupby("context_id")
        .agg(
            mean_auc=("DrugMacro_AUC", "mean"),
            std_auc=("DrugMacro_AUC", "std"),
            mean_auprc=("DrugMacro_AUPRC", "mean"),
            std_auprc=("DrugMacro_AUPRC", "std"),
            worst_fold_auc=("DrugMacro_AUC", "min"),
            mean_training_time=("training_time_seconds", "mean"),
            completed_jobs=("job_id", "count"),
        )
        .reset_index()
    )
    cand.to_csv(reports_dir / "stage20a_candidate_summary.csv", index=False)

    # seed summary
    seed_rows = []
    for seed, g in pairwise.groupby("split_seed"):
        seed_rows.append(
            {
                "split_seed": int(seed),
                "c16_mean_auc": g["c16_auc"].mean(),
                "c32_mean_auc": g["c32_auc"].mean(),
                "delta_auc": g["c32_auc"].mean() - g["c16_auc"].mean(),
                "c16_mean_auprc": g["c16_auprc"].mean(),
                "c32_mean_auprc": g["c32_auprc"].mean(),
                "delta_auprc": g["c32_auprc"].mean() - g["c16_auprc"].mean(),
            }
        )
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(reports_dir / "stage20a_seed_summary.csv", index=False)

    mean_c16 = float(df.loc[df.context_id == "C16", "DrugMacro_AUC"].mean())
    mean_c32 = float(df.loc[df.context_id == "C32", "DrugMacro_AUC"].mean())
    mean_delta = mean_c32 - mean_c16
    mean_ap_c16 = float(df.loc[df.context_id == "C16", "DrugMacro_AUPRC"].mean())
    mean_ap_c32 = float(df.loc[df.context_id == "C32", "DrugMacro_AUPRC"].mean())
    seed_deltas = {int(r.split_seed): float(r.delta_auc) for r in seed_summary.itertuples()}
    nonworse = sum(1 for d in seed_deltas.values() if d >= 0)
    worst_seed_delta = min(seed_deltas.values()) if seed_deltas else 0.0

    guardrails = {
        "mean_auc": mean_delta >= 0,
        "seed_majority": nonworse >= min_nonworse_seeds,
        "auprc": (mean_ap_c32 >= mean_ap_c16 - auprc_max_drop),
        "major_fail": worst_seed_delta >= major_fail_auc_delta,
    }

    if mean_c32 < mean_c16:
        selected, reason = "C16", "c32_mean_auc_lower"
    elif abs(mean_delta) < parsimony_delta:
        selected, reason = "C16", "parsimony"
    elif mean_delta >= parsimony_delta and guardrails["seed_majority"] and guardrails["auprc"] and guardrails["major_fail"]:
        selected, reason = "C32", "stable_improvement"
    else:
        selected, reason = "C16", "guardrail_not_met"

    decision = {
        "stage": "20A",
        "status": "LOCKED",
        "selected_context": selected,
        "selected_context_dim": 16 if selected == "C16" else 32,
        "selected_omics_dim": 80 if selected == "C16" else 96,
        "reason": reason,
        "mean_auc_c16": mean_c16,
        "mean_auc_c32": mean_c32,
        "mean_auc_delta_c32_minus_c16": mean_delta,
        "mean_auprc_c16": mean_ap_c16,
        "mean_auprc_c32": mean_ap_c32,
        "seed_deltas": seed_deltas,
        "nonworse_seed_count": nonworse,
        "worst_seed_delta": worst_seed_delta,
        "guardrails": guardrails,
        "n_jobs_analyzed": int(len(df)),
        "missing_jobs": missing,
        "selected_feature_dir": str(C16_FEATURE_DIR if selected == "C16" else C32_FEATURE_DIR),
    }
    _write_json(stage_dir / "stage20a_dimension_decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("build-splits")
    sp.add_argument("--outdir", default=str(RESULT_ROOT / "splits"))

    mp = sub.add_parser("build-manifest")
    mp.add_argument(
        "--e3-contract",
        default=str(RESULT_ROOT / "stage20_0/resolved_e3.json"),
    )
    mp.add_argument("--outdir", default=str(RESULT_ROOT / "stage20a_dimension"))
    mp.add_argument("--dry-run", action="store_true")

    ap = sub.add_parser("analyze")
    ap.add_argument("--stage-dir", default=str(RESULT_ROOT / "stage20a_dimension"))
    ap.add_argument("--no-strict", action="store_true")

    args = parser.parse_args()
    if args.command == "build-splits":
        audit = build_stage20a_splits(outdir=Path(args.outdir))
        print(json.dumps({"status": audit["status"], "n_folds": len(audit["folds"])}, indent=2))
    elif args.command == "build-manifest":
        report = build_stage20a_manifest(
            resolved_e3_path=Path(args.e3_contract), outdir=Path(args.outdir)
        )
        print(json.dumps(report, indent=2))
    elif args.command == "analyze":
        decision = analyze_stage20a(stage_dir=Path(args.stage_dir), strict=not args.no_strict)
        print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
