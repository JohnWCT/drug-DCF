#!/usr/bin/env python3
"""Train BioCDA B1/B2 on Round18 formal 5-folds; infer TCGA (eval3 Stage24B).

Design:
- Source folds = Round18 formal_5fold (train/val only).
- Checkpoint selection = source validation DrugMacro AUC only (never TCGA).
- Features = C32 (z64+context32).
- B1 = BioCDAPredictive (pooled E3), optional Round20 GIN warm-start.
- B2 = biocda_xa_fresh with Round23 freeze schedule.
- Parallelism = up to max_jobs_per_gpu subprocess workers on one GPU.
- After each fold: TCGA inference on all 5 targets → fold metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from biocda.data.xa_dataset import biocda_collate_fn, build_xa_dataset
from biocda.models.predictive.pooled_e3 import BioCDAPredictive
from biocda.models.xa.factory import build_xa_v2
from biocda.training.freeze_schedule import FreezePhase
from biocda.training.xa_loop import pos_weight_from_labels, train_xa_run
from biocda.training.xa_v2_trainer import train_xa_v2_run
from biocda.training.graph_cache_io import ensure_graph_cache, load_graph_cache
from biocda.utils.gpu import build_efficient_dataloader_kwargs, configure_gpu_efficiency
from biocda.validation.tcga_benchmark import (
    _build_tcga_dataset,
    _predict_loader,
    _resolve_round20_lock,
    prepare_tcga_frames,
)
from tools.biocda_telegram_notify import biocda_notify
from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics, metrics_to_jsonable


def _load_cfg(path: Path) -> Dict[str, Any]:
    p = path if path.is_absolute() else ROOT / path
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _xa_model_config() -> Dict[str, Any]:
    # Match configs/biocda/xa_v2_closure.yaml model block
    return {
        "model": {
            "type": "biocda_xa_fresh",
            "drug_encoder": {
                "input_dim": 78,
                "node_hidden_dim": 32,
                "num_layers": 5,
                "jk_mode": "last",
                "dropout": 0.1,
                "use_batch_norm": True,
            },
            "cross_attention": {
                "d_model": 128,
                "num_heads": 4,
                "num_layers": 2,
                "ffn_dim": 256,
                "attention_dropout": 0.1,
                "block_dropout": 0.2,
            },
            "response_head": {"hidden_dim": 128, "dropout": 0.1},
        },
        "optimizer": {
            "learning_rate": 0.0003,
            "last_gin_learning_rate": 0.00001,
            "weight_decay": 0.0001,
        },
        "experiment": {"architecture_version": "biocda-xa-v2"},
        "training": {},
    }


def _job_done(result_dir: Path) -> bool:
    return (
        (result_dir / "best.pt").is_file()
        and (result_dir / "tcga_fold_metrics.json").is_file()
        and (result_dir / "status.json").is_file()
        and json.loads((result_dir / "status.json").read_text()).get("status") == "complete"
    )


def _subset_loader(dataset, row_ids: List[int], *, batch_size: int, shuffle: bool) -> DataLoader:
    id_to_idx = {int(r): i for i, r in enumerate(dataset.df["_row_id"].astype(int))}
    indices = [id_to_idx[r] for r in row_ids if r in id_to_idx]
    kwargs = build_efficient_dataloader_kwargs(batch_size=batch_size)
    # Docker /dev/shm is too small for multi-worker graph batches → Bus error.
    # Keep num_workers=0; fill GPU with large micro-batch + accumulation instead.
    kwargs["num_workers"] = 0
    kwargs.pop("persistent_workers", None)
    kwargs.pop("prefetch_factor", None)
    kwargs.update({"shuffle": shuffle, "collate_fn": biocda_collate_fn})
    return DataLoader(Subset(dataset, indices), **kwargs)


def _ensure_tcga_cache(cache_path: Path) -> Dict[str, Any]:
    if cache_path.is_file():
        with cache_path.open("rb") as f:
            return pickle.load(f)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock = _resolve_round20_lock()
    # Force C32 feature_dir from Round24 config defaults
    lock = dict(lock)
    lock["selected_context"] = dict(lock["selected_context"])
    lock["selected_context"]["id"] = "C32"
    lock["selected_context"]["omics_dimension"] = 96
    lock["selected_context"]["feature_dir"] = str(
        ROOT / "result/optimization_runs/round20_unseen_drug_closure/features/z_plus_context32"
    )
    frames, patient_latent = prepare_tcga_frames(lock)
    # Convert frames to plain dicts of records for pickle stability
    payload = {
        "frames": {k: v.copy() for k, v in frames.items()},
        "patient_latent": patient_latent,
        "lock_context": lock["selected_context"],
    }
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=4)
    return payload


def _infer_tcga_fold(
    model: torch.nn.Module,
    *,
    result_dir: Path,
    tcga_cache: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    out: Dict[str, Any] = {}
    frames = tcga_cache["frames"]
    patient_latent = tcga_cache["patient_latent"]
    for target_key, frame in frames.items():
        loader = _build_tcga_dataset(frame, patient_latent)
        pred = _predict_loader(model, loader, device)
        pred["target_key"] = target_key
        pred.to_csv(result_dir / f"tcga_predictions__{target_key}.csv", index=False)
        metrics = metrics_to_jsonable(calculate_robust_drug_macro_metrics(pred))
        out[target_key] = metrics
        print(
            f"  TCGA {target_key}: DrugMacro_AUC={metrics.get('DrugMacro_AUC')} n={len(pred)}",
            flush=True,
        )
    return out


def train_one_job(job: Dict[str, Any], cfg: Dict[str, Any], *, smoke: bool = False) -> Dict[str, Any]:
    result_dir = Path(job["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    if _job_done(result_dir) and not smoke:
        return {"status": "skipped_complete", "result_dir": str(result_dir)}

    configure_gpu_efficiency(target_utilization=float(cfg.get("target_gpu_utilization", 0.9)))
    # When multiple jobs share one GPU, reserve a fraction of memory
    n_parallel = int(os.environ.get("ROUND24_PARALLEL_SLOTS", "1"))
    if torch.cuda.is_available() and n_parallel > 1:
        frac = min(0.9 / n_parallel, 0.45)
        try:
            torch.cuda.set_per_process_memory_fraction(frac, 0)
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dir = str(ROOT / cfg["paths"]["feature_c32"])
    smiles = str(ROOT / cfg["paths"]["drug_smiles"])
    r18 = ROOT / cfg["paths"]["round18_root"]
    dev = pd.read_csv(r18 / "splits" / "development_rows.csv")
    assigns = pd.read_csv(r18 / "splits" / "formal_5fold_assignments.csv")
    fold_id = int(job["fold_id"])
    fold_part = assigns[assigns["fold_id"] == fold_id]
    train_ids = fold_part[fold_part["split_role"] == "train"]["_row_id"].astype(int).tolist()
    val_ids = fold_part[fold_part["split_role"] == "val"]["_row_id"].astype(int).tolist()

    cache_root = ROOT / cfg["paths"]["run_root"] / "stage24b"
    # Prefer existing Round23 shared cache if present (same smiles/feature space)
    r23_cache = ROOT / "outputs/xa_v2_closure"
    if (r23_cache / "shared_graph_cache.pkl").is_file():
        graph_cache = load_graph_cache(r23_cache)
    else:
        ensure_graph_cache(
            dev_rows_path=r18 / "splits" / "development_rows.csv",
            feature_dir=feature_dir,
            drug_smiles_path=smiles,
            cache_root=cache_root,
        )
        graph_cache = load_graph_cache(cache_root)
    print(f"[fold {fold_id}] building dataset n={len(dev)} cache_drugs={len(graph_cache)}", flush=True)
    dataset = build_xa_dataset(dev, feature_dir=feature_dir, drug_smiles_path=smiles, graph_cache=graph_cache)
    print(f"[fold {fold_id}] dataset ready omics_dim={dataset.omics_dim}", flush=True)
    tr = cfg["training"]
    bs = 64 if smoke else int(tr.get("micro_batch_size", 512))
    # Keep large micro-batch for GPU fill; parallel slots share GPU via memory fraction.
    train_loader = _subset_loader(dataset, train_ids, batch_size=bs, shuffle=True)
    val_loader = _subset_loader(dataset, val_ids, batch_size=bs, shuffle=False)

    id_to_idx = {int(r): i for i, r in enumerate(dataset.df["_row_id"].astype(int))}
    train_labels = torch.tensor(
        [int(dataset.df.iloc[id_to_idx[r]]["Label"]) for r in train_ids if r in id_to_idx],
        dtype=torch.float32,
    )
    pw = pos_weight_from_labels(train_labels, str(device))

    architecture = job["architecture"]
    xa_cfg = _xa_model_config()
    pred_ckpt = ROOT / "result/optimization_runs/round20_unseen_drug_closure/stage20e_release/checkpoints/seed52_fold0.pt"

    t0 = time.perf_counter()
    print(f"[fold {fold_id}] start train arch={architecture} bs={bs} n_train={len(train_ids)} n_val={len(val_ids)} device={device}", flush=True)
    if architecture in {"biocda_predictive_e3", "biocda_predictive"}:
        model = BioCDAPredictive()
        if bool(tr.get("predictive_warmstart_gin", True)) and pred_ckpt.is_file():
            from biocda.training.gin_transfer import transfer_e3_gin_into_module

            transfer_e3_gin_into_module(pred_ckpt, model.encoder, strict=True)
        max_epochs = 2 if smoke else int(tr["max_epochs"])
        patience = 2 if smoke else int(tr["early_stopping_patience"])
        result = train_xa_run(
            model,
            train_loader,
            val_loader,
            val_loader,  # GDSC test unused for selection; diagnostic only
            run_dir=result_dir,
            max_epochs=max_epochs,
            patience=patience,
            lr=float(tr.get("learning_rate", 3e-4)),
            weight_decay=float(tr.get("weight_decay", 1e-4)),
            grad_clip=float(tr.get("gradient_clip_norm", 1.0)),
            use_amp=bool(tr.get("mixed_precision", True)),
            accumulation_steps=int(tr.get("accumulation_steps", 2)),
            model_type="biocda_predictive",
            architecture_version="biocda-predictive-e3",
            config={"round24": True, "fold_id": fold_id, **xa_cfg},
            pos_weight=pw,
        )
    elif architecture in {"biocda_xa_fresh"}:
        model = build_xa_v2(xa_cfg, model_type="biocda_xa_fresh")
        if smoke:
            phases = [FreezePhase("attention_warmup", epochs=2, freeze_gin_layers=[0, 1, 2, 3, 4], other_lr=3e-4)]
            patience = 2
        else:
            phases = [
                FreezePhase("attention_warmup", epochs=15, freeze_gin_layers=[0, 1, 2, 3, 4], other_lr=3e-4),
                FreezePhase(
                    "last_gin_adaptation",
                    epochs=40,
                    freeze_gin_layers=[0, 1, 2, 3],
                    last_gin_lr=1e-5,
                    other_lr=3e-4,
                ),
                FreezePhase(
                    "joint_stabilization",
                    epochs=145,
                    freeze_gin_layers=[0, 1, 2, 3],
                    last_gin_lr=1e-5,
                    other_lr=3e-4,
                ),
            ]
            patience = int(tr["early_stopping_patience"])
        result = train_xa_v2_run(
            model,
            train_loader,
            val_loader,
            val_loader,
            run_dir=result_dir,
            phases=phases,
            patience=patience,
            weight_decay=float(tr.get("weight_decay", 1e-4)),
            grad_clip=float(tr.get("gradient_clip_norm", 1.0)),
            use_amp=bool(tr.get("mixed_precision", True)),
            accumulation_steps=int(tr.get("accumulation_steps", 2)),
            model_type="biocda_xa_fresh",
            architecture_version="biocda-xa-v2",
            config={"round24": True, "fold_id": fold_id, **xa_cfg},
            pos_weight=pw,
            teacher=None,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Reload best checkpoint for TCGA infer
    blob = torch.load(result_dir / "best.pt", map_location=device)
    if architecture in {"biocda_predictive_e3", "biocda_predictive"}:
        model = BioCDAPredictive().to(device)
        model.load_state_dict(blob["model_state_dict"], strict=True)
    else:
        model = build_xa_v2(xa_cfg, model_type="biocda_xa_fresh").to(device)
        model.load_state_dict(blob["model_state_dict"], strict=True)

    cache_path = ROOT / cfg["paths"]["run_root"] / "stage24b" / "tcga_c32_cache.pkl"
    tcga_cache = _ensure_tcga_cache(cache_path)
    tcga_metrics = _infer_tcga_fold(model, result_dir=result_dir, tcga_cache=tcga_cache, device=device)
    (result_dir / "tcga_fold_metrics.json").write_text(
        json.dumps(tcga_metrics, indent=2, default=str) + "\n"
    )

    summary = {
        "status": "complete",
        "candidate_id": job["candidate_id"],
        "architecture": architecture,
        "fold_id": fold_id,
        "best_epoch": result.best_epoch,
        "best_val_DrugMacro_AUC": result.metrics_validation.get("DrugMacro_AUC"),
        "training_time_sec": time.perf_counter() - t0,
        "tcga_used_in_selection": False,
        "early_stopping_metric": "validation_drug_macro_auc",
        "batch_size": bs,
        "n_parallel_slots": n_parallel,
    }
    (result_dir / "fold_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (result_dir / "status.json").write_text(json.dumps({"status": "complete"}) + "\n")
    # Alias for evaluate scripts expecting best_checkpoint.pt
    if not (result_dir / "best_checkpoint.pt").is_file():
        (result_dir / "best_checkpoint.pt").symlink_to(result_dir / "best.pt")
    return summary


def aggregate_candidate(cfg: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
    run_root = ROOT / cfg["paths"]["run_root"] / "stage24b" / candidate_id
    report_root = ROOT / cfg["paths"]["reports_root"] / "stage24b" / candidate_id
    report_root.mkdir(parents=True, exist_ok=True)
    fold_aucs: Dict[str, List[float]] = {t["key"]: [] for t in cfg["targets"]}
    fold_auprcs: Dict[str, List[float]] = {t["key"]: [] for t in cfg["targets"]}
    fold_rows = []
    n_folds = int(cfg["n_folds"])
    missing = [fold_id for fold_id in range(n_folds) if not (run_root / f"fold_{fold_id}" / "tcga_fold_metrics.json").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing TCGA metrics for folds {missing} under {run_root}")
    for fold_id in range(n_folds):
        d = run_root / f"fold_{fold_id}"
        mpath = d / "tcga_fold_metrics.json"
        metrics = json.loads(mpath.read_text())
        for key in fold_aucs:
            auc = metrics[key].get("DrugMacro_AUC")
            auprc = metrics[key].get("DrugMacro_AUPRC")
            fold_aucs[key].append(float(auc) if auc is not None else float("nan"))
            fold_auprcs[key].append(float(auprc) if auprc is not None else float("nan"))
            fold_rows.append(
                {
                    "candidate_id": candidate_id,
                    "fold_id": fold_id,
                    "target_key": key,
                    "DrugMacro_AUC": auc,
                    "DrugMacro_AUPRC": auprc,
                    "Global_AUC": metrics[key].get("Global_AUC"),
                    "n_rows": metrics[key].get("n_rows"),
                }
            )
    # Ensemble: mean probability across folds per target
    ens_aucs = {}
    for t in cfg["targets"]:
        key = t["key"]
        frames = []
        for fold_id in range(n_folds):
            p = run_root / f"fold_{fold_id}" / f"tcga_predictions__{key}.csv"
            frames.append(pd.read_csv(p))
        base = frames[0][["_row_id", "DRUG_NAME", "Label"]].copy()
        for i, df in enumerate(frames):
            base = base.merge(
                df[["_row_id", "probability"]].rename(columns={"probability": f"p{i}"}),
                on="_row_id",
                how="left",
            )
        pcols = [c for c in base.columns if c.startswith("p")]
        base["probability"] = base[pcols].mean(axis=1)
        base.to_csv(report_root / f"ensemble_predictions__{key}.csv", index=False)
        em = metrics_to_jsonable(calculate_robust_drug_macro_metrics(base))
        ens_aucs[key] = em

    per_target_fold_mean_auc = {k: float(np.nanmean(v)) for k, v in fold_aucs.items()}
    per_target_fold_mean_auprc = {k: float(np.nanmean(v)) for k, v in fold_auprcs.items()}
    per_target_fold_std_auc = {k: float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in fold_aucs.items()}

    arch = next(c["architecture"] for c in cfg["candidates_24b"] if c["id"] == candidate_id)
    feat = next(c["feature"] for c in cfg["candidates_24b"] if c["id"] == candidate_id)
    payload = {
        "candidate_id": candidate_id,
        "architecture": arch,
        "feature": feat,
        "n_folds": n_folds,
        "per_target_fold_mean_auc": per_target_fold_mean_auc,
        "per_target_fold_mean_auprc": per_target_fold_mean_auprc,
        "per_target_fold_std_auc": per_target_fold_std_auc,
        "per_target_ensemble": {k: {"DrugMacro_AUC": v.get("DrugMacro_AUC"), "DrugMacro_AUPRC": v.get("DrugMacro_AUPRC")} for k, v in ens_aucs.items()},
        "status": "complete",
    }
    pd.DataFrame(fold_rows).to_csv(report_root / "fold_metrics.csv", index=False)
    (report_root / "candidate_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    # also under stage24b/<id> as evaluate expects
    return payload


def dispatch(manifest: Path, cfg: Dict[str, Any], *, max_jobs: int, smoke: bool) -> int:
    jobs = pd.read_csv(manifest).to_dict(orient="records")
    if smoke:
        jobs = jobs[:1]
    # Prebuild TCGA + graph caches once (serial) before parallel workers
    cache_path = ROOT / cfg["paths"]["run_root"] / "stage24b" / "tcga_c32_cache.pkl"
    print("Building/loading TCGA C32 cache...", flush=True)
    _ensure_tcga_cache(cache_path)
    r23_cache = ROOT / "outputs/xa_v2_closure"
    if not (r23_cache / "shared_graph_cache.pkl").is_file():
        print("Building shared graph cache...", flush=True)
        ensure_graph_cache(
            dev_rows_path=ROOT / cfg["paths"]["round18_root"] / "splits" / "development_rows.csv",
            feature_dir=str(ROOT / cfg["paths"]["feature_c32"]),
            drug_smiles_path=str(ROOT / cfg["paths"]["drug_smiles"]),
            cache_root=ROOT / cfg["paths"]["run_root"] / "stage24b",
        )
    else:
        print("Reusing Round23 shared_graph_cache.pkl", flush=True)
    biocda_notify(f"Round24 Stage24B fold-train START jobs={len(jobs)} parallel={max_jobs}")

    pending = [j for j in jobs if smoke or not _job_done(Path(j["result_dir"]))]
    skipped = len(jobs) - len(pending)
    results: List[Dict[str, Any]] = [{"status": "skipped_complete"} for _ in range(skipped)]

    def _launch(job: Dict[str, Any]) -> Dict[str, Any]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        env["ROUND24_PARALLEL_SLOTS"] = str(max_jobs)
        cmd = [
            sys.executable,
            str(ROOT / "scripts/round24/train_biocda_on_round18_folds.py"),
            "--config",
            str(ROOT / "configs/round24/eval3.yaml"),
            "--single-job",
            json.dumps(job),
        ]
        if smoke:
            cmd.append("--smoke")
        print("+", " ".join(cmd[:6]), f"... fold={job['fold_id']} {job['candidate_id']}", flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
        status_path = Path(job["result_dir"]) / "status.json"
        if status_path.is_file():
            return json.loads(status_path.read_text())
        return {"status": "failed", "rc": rc, "result_dir": job["result_dir"]}

    if max_jobs <= 1 or len(pending) <= 1:
        for job in pending:
            results.append(train_one_job(job, cfg, smoke=smoke))
    else:
        with ThreadPoolExecutor(max_workers=max_jobs) as ex:
            futs = {ex.submit(_launch, j): j for j in pending}
            for fut in as_completed(futs):
                results.append(fut.result())

    # Aggregate completed candidates
    for cid in sorted({j["candidate_id"] for j in jobs}):
        try:
            payload = aggregate_candidate(cfg, cid)
            print(f"Aggregated {cid}: {payload['per_target_fold_mean_auc']}", flush=True)
        except FileNotFoundError as exc:
            print(f"Skip aggregate {cid}: {exc}", flush=True)

    n_ok = sum(1 for r in results if r.get("status") in {"complete", "skipped_complete"})
    out = ROOT / cfg["paths"]["reports_root"] / "stage24b" / "train_dispatch_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_ok": n_ok, "n_jobs": len(jobs), "results": results}, indent=2, default=str) + "\n")
    biocda_notify(f"Round24 Stage24B fold-train DONE ok={n_ok}/{len(jobs)}")
    return 0 if n_ok == len(jobs) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--single-job", type=str, default=None, help="JSON job payload")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--max-jobs-per-gpu", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    if args.single_job:
        job = json.loads(args.single_job)
        summary = train_one_job(job, cfg, smoke=args.smoke)
        print(json.dumps(summary, indent=2, default=str))
        return 0 if summary.get("status") in {"complete", "skipped_complete"} else 1
    if args.aggregate_only:
        for cid in ("B1", "B2"):
            try:
                print(json.dumps(aggregate_candidate(cfg, cid), indent=2))
            except FileNotFoundError as e:
                print(e)
        return 0
    if args.manifest is None:
        raise SystemExit("--manifest required unless --single-job / --aggregate-only")
    return dispatch(args.manifest, cfg, max_jobs=int(args.max_jobs_per_gpu), smoke=args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
