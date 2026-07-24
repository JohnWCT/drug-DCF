#!/usr/bin/env python3
"""Round 24 Stage24E: NoHoldout confirmation of preferred architectures.

Candidates (preregistered):
  E-NH0  pooled_mlp × own_plus_summary × NoHoldout  (reuse ablation)
  E-NH1  biocda_predictive_e3 × C16 × NoHoldout     (main)
  E-NH2  biocda_predictive_e3 × C32 × NoHoldout     (feature contrast)
  E-REF2 / E-REF3  holdout F2/F3 references only

Hard gate (stest0): aacdr_gdsc_intersect > 0.5279 AND aacdr_tcga_only > 0.4804.
Telegram only on full round completion (finalize).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
from biocda.training.graph_cache_io import load_graph_cache
from biocda.training.xa_loop import pos_weight_from_labels, train_xa_run
from biocda.utils.gpu import build_efficient_dataloader_kwargs, configure_gpu_efficiency
from biocda.validation.round24_gate import (
    build_lock_manifest,
    evaluate_all_target_gate,
    rank_passing_candidates,
    write_lock_manifest,
)
from biocda.validation.round24_protocol import gate_table, load_eval3_config
from biocda.validation.tcga_benchmark import _resolve_round20_lock
from tools.biocda_telegram_notify import biocda_notify
from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics, metrics_to_jsonable
from tools.round18_tcga_dataset import load_tcga_omics_latent_dict, prepare_tcga_response_frame
from tools.round19_dataset import Round19ResponseDataset
from tools.round20_tcga import build_tcga_o2_features

NOHOLDOUT_SPLITS = ROOT / "result/optimization_runs/round24_train_source_ablation/NoHoldout/splits"
ABLATION_REPORT = ROOT / "reports/round24/train_source_ablation/NoHoldout"
ABLATION_RUNS = ROOT / "result/optimization_runs/round24_train_source_ablation/NoHoldout/runs"

CANDIDATES = [
    {
        "id": "E-NH0",
        "architecture": "pooled_mlp",
        "feature": "own_plus_summary",
        "train_source": "no_holdout",
        "role": "data_baseline",
        "reuse_from": "reports/round24/train_source_ablation/NoHoldout",
        "action": "reuse",
    },
    {
        "id": "E-NH1",
        "architecture": "biocda_predictive_e3",
        "feature": "z_plus_context16",
        "feature_path": "result/optimization_runs/round20_unseen_drug_closure/features/z_plus_context16",
        "tcga_mode": "c16_build",
        "train_source": "no_holdout",
        "role": "primary_confirm",
        "action": "train",
    },
    {
        "id": "E-NH2",
        "architecture": "biocda_predictive_e3",
        "feature": "z_plus_context32",
        "feature_path": "result/optimization_runs/round20_unseen_drug_closure/features/z_plus_context32",
        "tcga_mode": "c32_build",
        "train_source": "no_holdout",
        "role": "feature_contrast",
        "action": "train",
    },
    {
        "id": "E-REF2",
        "architecture": "biocda_predictive_e3",
        "feature": "z_plus_context16",
        "train_source": "holdout_ref",
        "role": "holdout_reference",
        "reuse_from": "reports/round24/stage24c/F2",
        "action": "reuse_ref",
    },
    {
        "id": "E-REF3",
        "architecture": "biocda_predictive_e3",
        "feature": "z_plus_context32",
        "train_source": "holdout_ref",
        "role": "holdout_reference",
        "reuse_from": "reports/round24/stage24c/F3",
        "action": "reuse_ref",
    },
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    kwargs["num_workers"] = 0
    kwargs.pop("persistent_workers", None)
    kwargs.pop("prefetch_factor", None)
    kwargs.update({"shuffle": shuffle, "collate_fn": biocda_collate_fn})
    return DataLoader(Subset(dataset, indices), **kwargs)


def _feature_dim(feature_dir: Path) -> int:
    meta = feature_dir / "feature_metadata.json"
    if meta.is_file():
        m = json.loads(meta.read_text())
        d = m.get("response_input_dim") or m.get("feature_dim") or m.get("omics_dimension")
        if d is not None:
            return int(d)
    import pickle

    with (feature_dir / "ccle_latent_proto.pkl").open("rb") as f:
        lat = pickle.load(f)
    return int(np.asarray(next(iter(lat.values()))).reshape(-1).shape[0])


def _patient_latent_from_map(latent: Dict[str, Any]) -> Dict[str, Any]:
    patient_latent: Dict[str, Any] = {}
    for key, vec in latent.items():
        parts = str(key).split("-")
        if len(parts) >= 3:
            patient_latent.setdefault("-".join(parts[:3]), vec)
        patient_latent[str(key)] = vec
    return patient_latent


def _tcga_cache_path(cand: Dict[str, Any]) -> Path:
    return ROOT / "result/optimization_runs/round24_tcga_recovery/stage24e/cache" / f"tcga_patient_latent__{cand['id']}.pkl"


def _load_tcga_latent(cand: Dict[str, Any], feature_dir: Path) -> Dict[str, Any]:
    """Build once per candidate and reuse across folds (avoids CPU thrash / GPU idle)."""
    import pickle

    cache_path = _tcga_cache_path(cand)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        with cache_path.open("rb") as f:
            print(f"[{cand['id']}] TCGA latent cache HIT {cache_path}", flush=True)
            return pickle.load(f)

    mode = cand["tcga_mode"]
    if mode not in {"c16_build", "c32_build"}:
        raise ValueError(mode)
    lock = _resolve_round20_lock()
    lock = dict(lock)
    lock["selected_context"] = dict(lock["selected_context"])
    if mode == "c16_build":
        lock["selected_context"]["id"] = "C16"
        lock["selected_context"]["omics_dimension"] = 80
    else:
        lock["selected_context"]["id"] = "C32"
        lock["selected_context"]["omics_dimension"] = 96
    lock["selected_context"]["feature_dir"] = str(feature_dir)
    print(f"[{cand['id']}] building TCGA latent cache → {cache_path}", flush=True)
    patient_latent = _patient_latent_from_map(build_tcga_o2_features(lock))
    tmp = cache_path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(patient_latent, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_path)
    return patient_latent


def prebuild_tcga_caches(cfg: Dict[str, Any], candidate_ids: List[str]) -> None:
    """Serialize TCGA feature builds before parallel train/infer (GPU then stays busy)."""
    for cand in CANDIDATES:
        if cand["id"] not in candidate_ids or cand.get("action") != "train":
            continue
        feature_dir = ROOT / cand["feature_path"]
        _load_tcga_latent(cand, feature_dir)


@torch.no_grad()
def _infer_tcga(model, *, cand, feature_dir, smiles, result_dir, device, cfg) -> Dict[str, Any]:
    patient_latent = _load_tcga_latent(cand, feature_dir)
    out: Dict[str, Any] = {}
    own = ROOT / cfg["paths"]["feature_own_plus_summary"]
    # Larger infer batch to keep GPU fed after train
    infer_bs = 1024 if device.type == "cuda" else 256
    for t in cfg["targets"]:
        key = t["key"]
        path = str(ROOT / t["path"])
        frame, _, _ = prepare_tcga_response_frame(
            path,
            feature_dir=str(own),
            drug_smiles_path=smiles,
            target_key=key,
        )
        keep = [str(r["ModelID"]) in patient_latent for _, r in frame.iterrows()]
        frame = frame.loc[keep].reset_index(drop=True)
        if frame.empty:
            out[key] = {"DrugMacro_AUC": None, "n_rows": 0, "error": "empty_after_latent_filter"}
            continue
        row_latent = {str(r["ModelID"]): patient_latent[str(r["ModelID"])] for _, r in frame.iterrows()}
        ds = Round19ResponseDataset(
            frame,
            feature_dir=str(feature_dir),
            drug_smiles_path=smiles,
            encoder_type="gin",
            graph_cache={},
            latent_by_id=row_latent,
        )
        loader = DataLoader(ds, batch_size=infer_bs, shuffle=False, num_workers=0, collate_fn=biocda_collate_fn)
        model.eval()
        rows = []
        for batch in loader:
            omics = batch["omics"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            drug = batch["drug_graph"].to(device)
            logits = model(omics, context, drug).logits
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels = batch["labels"].cpu().numpy().astype(int)
            for i in range(len(labels)):
                rows.append(
                    {
                        "_row_id": int(batch["_row_id"][i]),
                        "DRUG_NAME": str(batch["DRUG_NAME"][i]),
                        "Label": int(labels[i]),
                        "probability": float(probs[i]),
                    }
                )
        pred = pd.DataFrame(rows)
        pred.to_csv(result_dir / f"tcga_predictions__{key}.csv", index=False)
        metrics = metrics_to_jsonable(calculate_robust_drug_macro_metrics(pred))
        metrics["n_rows"] = int(len(pred))
        out[key] = metrics
        print(f"  [{cand['id']}] {key} AUC={metrics.get('DrugMacro_AUC')} n={len(pred)}", flush=True)
    return out


def write_manifest(cfg: Dict[str, Any]) -> Path:
    out = ROOT / cfg["paths"]["reports_root"] / "stage24e"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "24E",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "standard_name": cfg.get("standard_name"),
        "gate_required_targets": cfg.get("gate_required_targets"),
        "target_weights": cfg.get("target_weights"),
        "train_protocol": {
            "name": "no_holdout_formal_5fold",
            "splits_root": str(NOHOLDOUT_SPLITS.relative_to(ROOT)),
            "holdout": "disabled",
            "aligned_with_standard": "aacdr_stest0_no_holdout",
        },
        "candidates": CANDIDATES,
        "config_path": "configs/round24/eval3.yaml",
        "config_sha256": cfg.get("_config_sha256"),
        "note": "Holdout refs (E-REF*) do not compete for NoHoldout lock ranking.",
    }
    path = out / "candidate_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "candidate_manifest.sha256").write_text(_sha256_file(path) + "\n", encoding="utf-8")
    print("wrote", path)
    return path


def reuse_nh0(cfg: Dict[str, Any]) -> Dict[str, Any]:
    report = ROOT / cfg["paths"]["reports_root"] / "stage24e" / "E-NH0"
    report.mkdir(parents=True, exist_ok=True)
    src = ABLATION_REPORT
    for name in [
        "candidate_summary.json",
        "fold_metrics.csv",
        *[p.name for p in src.glob("ensemble_predictions__*.csv")],
    ]:
        sp = src / name
        if sp.is_file():
            shutil.copy2(sp, report / name)
    payload = json.loads((report / "candidate_summary.json").read_text())
    payload["candidate_id"] = "E-NH0"
    payload["train_source"] = "no_holdout"
    payload["reused_from"] = str(src.relative_to(ROOT))
    gate = evaluate_all_target_gate(
        payload["per_target_fold_mean_auc"],
        gate_table(cfg),
        target_priority=cfg["target_priority"],
        target_weights=cfg["target_weights"],
        gate_required_targets=cfg.get("gate_required_targets"),
    )
    payload["gate"] = gate
    (report / "candidate_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    # copy fold ckpts for lock provenance if present
    run_dst = ROOT / cfg["paths"]["run_root"] / "stage24e" / "E-NH0"
    if ABLATION_RUNS.is_dir():
        run_dst.mkdir(parents=True, exist_ok=True)
        for fold in ABLATION_RUNS.glob("fold_*"):
            if fold.name.endswith("_bak"):
                continue
            d = run_dst / fold.name
            d.mkdir(parents=True, exist_ok=True)
            for f in fold.iterdir():
                if f.is_file():
                    shutil.copy2(f, d / f.name)
    return payload


def reuse_ref(cfg: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    report = ROOT / cfg["paths"]["reports_root"] / "stage24e" / cand["id"]
    report.mkdir(parents=True, exist_ok=True)
    src = ROOT / cand["reuse_from"]
    for p in src.glob("*"):
        if p.is_file():
            shutil.copy2(p, report / p.name)
    payload = json.loads((report / "candidate_summary.json").read_text())
    payload["candidate_id"] = cand["id"]
    payload["train_source"] = "holdout_ref"
    payload["reused_from"] = cand["reuse_from"]
    payload["eligible_for_noholdout_lock"] = False
    gate = evaluate_all_target_gate(
        payload["per_target_fold_mean_auc"],
        gate_table(cfg),
        target_priority=cfg["target_priority"],
        target_weights=cfg["target_weights"],
        gate_required_targets=cfg.get("gate_required_targets"),
    )
    payload["gate"] = gate
    (report / "candidate_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def train_one_fold(
    *,
    cand: Dict[str, Any],
    fold_id: int,
    cfg: Dict[str, Any],
    smoke: bool = False,
    mem_fraction: float = 0.18,
) -> Dict[str, Any]:
    feature_dir = ROOT / cand["feature_path"]
    result_dir = ROOT / cfg["paths"]["run_root"] / "stage24e" / cand["id"] / f"fold_{fold_id}"
    result_dir.mkdir(parents=True, exist_ok=True)
    if _job_done(result_dir) and not smoke:
        return {"status": "skipped_complete", "result_dir": str(result_dir), "candidate_id": cand["id"], "fold_id": fold_id}

    configure_gpu_efficiency(target_utilization=float(cfg.get("target_gpu_utilization", 0.9)))
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(float(mem_fraction), 0)
        except Exception:
            pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smiles = str(ROOT / cfg["paths"]["drug_smiles"])
    dim = _feature_dim(feature_dir)
    context_dim = dim - 64
    tr = cfg["training"]

    # Resume: if train already finished (best.pt) but TCGA infer incomplete, skip retrain.
    resume_infer_only = (result_dir / "best.pt").is_file() and not smoke
    if resume_infer_only:
        print(f"[{cand['id']} fold{fold_id}] resume infer-only from best.pt", flush=True)
        blob = torch.load(result_dir / "best.pt", map_location=device)
        model = BioCDAPredictive(omics_dim=64, context_dim=context_dim).to(device)
        model.load_state_dict(blob["model_state_dict"], strict=True)
        tcga_metrics = _infer_tcga(
            model,
            cand=cand,
            feature_dir=feature_dir,
            smiles=smiles,
            result_dir=result_dir,
            device=device,
            cfg=cfg,
        )
        (result_dir / "tcga_fold_metrics.json").write_text(json.dumps(tcga_metrics, indent=2, default=str) + "\n")
        summary = {
            "status": "complete",
            "candidate_id": cand["id"],
            "fold_id": fold_id,
            "resumed_infer_only": True,
            "best_epoch": blob.get("best_epoch"),
            "tcga_used_in_selection": False,
            "train_source": "no_holdout",
        }
        (result_dir / "fold_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        (result_dir / "status.json").write_text(json.dumps({"status": "complete"}) + "\n")
        return summary

    dev = pd.read_csv(NOHOLDOUT_SPLITS / "development_rows.csv")
    assigns = pd.read_csv(NOHOLDOUT_SPLITS / "formal_5fold_assignments.csv")
    fold_part = assigns[assigns["fold_id"] == fold_id]
    train_ids = fold_part[fold_part["split_role"] == "train"]["_row_id"].astype(int).tolist()
    val_ids = fold_part[fold_part["split_role"] == "val"]["_row_id"].astype(int).tolist()

    r23_cache = ROOT / "outputs/xa_v2_closure"
    graph_cache = load_graph_cache(r23_cache) if (r23_cache / "shared_graph_cache.pkl").is_file() else {}
    print(f"[{cand['id']} fold{fold_id}] NoHoldout dim={dim} context={context_dim}", flush=True)
    dataset = build_xa_dataset(dev, feature_dir=str(feature_dir), drug_smiles_path=smiles, graph_cache=graph_cache)

    # Prefer large micro-batch for GPU occupancy; fall back if OOM at launch site via mem_fraction.
    bs = 64 if smoke else int(tr.get("micro_batch_size", 512))
    train_loader = _subset_loader(dataset, train_ids, batch_size=bs, shuffle=True)
    val_loader = _subset_loader(dataset, val_ids, batch_size=bs, shuffle=False)
    id_to_idx = {int(r): i for i, r in enumerate(dataset.df["_row_id"].astype(int))}
    train_labels = torch.tensor(
        [int(dataset.df.iloc[id_to_idx[r]]["Label"]) for r in train_ids if r in id_to_idx],
        dtype=torch.float32,
    )
    pw = pos_weight_from_labels(train_labels, str(device))

    model = BioCDAPredictive(omics_dim=64, context_dim=context_dim)
    pred_ckpt = ROOT / "result/optimization_runs/round20_unseen_drug_closure/stage20e_release/checkpoints/seed52_fold0.pt"
    if context_dim == 32 and bool(tr.get("predictive_warmstart_gin", True)) and pred_ckpt.is_file():
        from biocda.training.gin_transfer import transfer_e3_gin_into_module

        transfer_e3_gin_into_module(pred_ckpt, model.encoder, strict=True)

    max_epochs = 2 if smoke else int(tr["max_epochs"])
    patience = 2 if smoke else int(tr["early_stopping_patience"])
    result = train_xa_run(
        model,
        train_loader,
        val_loader,
        val_loader,
        run_dir=result_dir,
        max_epochs=max_epochs,
        patience=patience,
        lr=float(tr.get("learning_rate", 3e-4)),
        weight_decay=float(tr.get("weight_decay", 1e-4)),
        grad_clip=float(tr.get("gradient_clip_norm", 1.0)),
        use_amp=bool(tr.get("mixed_precision", True)),
        accumulation_steps=int(tr.get("accumulation_steps", 4)),
        model_type="biocda_predictive",
        architecture_version="biocda-predictive-e3",
        config={"round24e": True, "candidate_id": cand["id"], "fold_id": fold_id, "train_source": "no_holdout"},
        pos_weight=pw,
    )

    blob = torch.load(result_dir / "best.pt", map_location=device)
    model = BioCDAPredictive(omics_dim=64, context_dim=context_dim).to(device)
    model.load_state_dict(blob["model_state_dict"], strict=True)
    tcga_metrics = _infer_tcga(
        model,
        cand=cand,
        feature_dir=feature_dir,
        smiles=smiles,
        result_dir=result_dir,
        device=device,
        cfg=cfg,
    )
    (result_dir / "tcga_fold_metrics.json").write_text(json.dumps(tcga_metrics, indent=2, default=str) + "\n")
    summary = {
        "status": "complete",
        "candidate_id": cand["id"],
        "fold_id": fold_id,
        "best_epoch": result.best_epoch,
        "best_val_DrugMacro_AUC": result.metrics_validation.get("DrugMacro_AUC"),
        "tcga_used_in_selection": False,
        "train_source": "no_holdout",
    }
    (result_dir / "fold_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (result_dir / "status.json").write_text(json.dumps({"status": "complete"}) + "\n")
    return summary


def aggregate_candidate(cfg: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    run_root = ROOT / cfg["paths"]["run_root"] / "stage24e" / cand["id"]
    report_root = ROOT / cfg["paths"]["reports_root"] / "stage24e" / cand["id"]
    report_root.mkdir(parents=True, exist_ok=True)
    n_folds = int(cfg["n_folds"])
    fold_aucs = {t["key"]: [] for t in cfg["targets"]}
    fold_auprcs = {t["key"]: [] for t in cfg["targets"]}
    fold_rows = []
    missing = [i for i in range(n_folds) if not (run_root / f"fold_{i}" / "tcga_fold_metrics.json").is_file()]
    if missing:
        raise FileNotFoundError(f"{cand['id']} missing folds {missing}")
    for fold_id in range(n_folds):
        metrics = json.loads((run_root / f"fold_{fold_id}" / "tcga_fold_metrics.json").read_text())
        for key in fold_aucs:
            auc = metrics[key].get("DrugMacro_AUC")
            auprc = metrics[key].get("DrugMacro_AUPRC")
            fold_aucs[key].append(float(auc) if auc is not None else float("nan"))
            fold_auprcs[key].append(float(auprc) if auprc is not None else float("nan"))
            fold_rows.append(
                {
                    "candidate_id": cand["id"],
                    "fold_id": fold_id,
                    "target_key": key,
                    "DrugMacro_AUC": auc,
                    "DrugMacro_AUPRC": auprc,
                }
            )
    for t in cfg["targets"]:
        key = t["key"]
        frames = [pd.read_csv(run_root / f"fold_{i}" / f"tcga_predictions__{key}.csv") for i in range(n_folds)]
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

    means = {k: float(np.nanmean(v)) for k, v in fold_aucs.items()}
    auprcs = {k: float(np.nanmean(v)) for k, v in fold_auprcs.items()}
    gate = evaluate_all_target_gate(
        means,
        gate_table(cfg),
        target_priority=cfg["target_priority"],
        target_weights=cfg["target_weights"],
        gate_required_targets=cfg.get("gate_required_targets"),
    )
    payload = {
        "candidate_id": cand["id"],
        "architecture": cand["architecture"],
        "feature": cand["feature"],
        "train_source": "no_holdout",
        "eligible_for_noholdout_lock": True,
        "n_folds": n_folds,
        "per_target_fold_mean_auc": means,
        "per_target_fold_mean_auprc": auprcs,
        "per_target_fold_std_auc": {
            k: float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in fold_aucs.items()
        },
        "gate": gate,
        "status": "complete",
    }
    pd.DataFrame(fold_rows).to_csv(report_root / "fold_metrics.csv", index=False)
    (report_root / "candidate_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _launch_job(cand_id: str, fold_id: int, *, mem_fraction: float, config: Path) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/round24/run_stage24e.py"),
        "--config",
        str(config),
        "--single-job",
        json.dumps({"candidate_id": cand_id, "fold_id": fold_id}),
        "--mem-fraction",
        str(mem_fraction),
    ]
    print("+", cand_id, f"fold={fold_id}", flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    status = ROOT / "result/optimization_runs/round24_tcga_recovery/stage24e" / cand_id / f"fold_{fold_id}" / "status.json"
    if status.is_file():
        return json.loads(status.read_text())
    return {"status": "failed", "rc": rc, "candidate_id": cand_id, "fold_id": fold_id}


def finalize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    report_root = ROOT / cfg["paths"]["reports_root"] / "stage24e"
    candidates = []
    for cand in CANDIDATES:
        path = report_root / cand["id"] / "candidate_summary.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        if not payload.get("gate"):
            payload["gate"] = evaluate_all_target_gate(
                payload["per_target_fold_mean_auc"],
                gate_table(cfg),
                target_priority=cfg["target_priority"],
                target_weights=cfg["target_weights"],
                gate_required_targets=cfg.get("gate_required_targets"),
            )
        candidates.append(payload)

    # Only NoHoldout-eligible candidates compete for lock
    lock_pool = [
        c
        for c in candidates
        if c.get("eligible_for_noholdout_lock", c.get("train_source") == "no_holdout")
        and c.get("train_source") != "holdout_ref"
    ]
    ranked = rank_passing_candidates(
        lock_pool,
        target_priority=cfg["target_priority"],
        target_weights=cfg["target_weights"],
    )
    champion = ranked[0] if ranked else None
    decision = {
        "stage": "24E/24F",
        "standard": cfg.get("standard_name"),
        "gate_required_targets": cfg.get("gate_required_targets"),
        "n_candidates": len(candidates),
        "n_lock_pool": len(lock_pool),
        "n_pass": len(ranked),
        "champion_id": champion["candidate_id"] if champion else None,
        "status": "LOCKED" if champion else "NO_LOCK",
        "candidates": [
            {
                "id": c["candidate_id"],
                "train_source": c.get("train_source"),
                "gate": c.get("gate", {}).get("status"),
                "aacdr_gdsc": c["per_target_fold_mean_auc"].get("aacdr_gdsc_intersect"),
                "aacdr_tcga": c["per_target_fold_mean_auc"].get("aacdr_tcga_only"),
                "eligible_for_lock": c.get("eligible_for_noholdout_lock", c.get("train_source") == "no_holdout"),
            }
            for c in candidates
        ],
    }
    (report_root / "stage24e_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    # Lock manifest
    stage24a = ROOT / cfg["paths"]["reports_root"] / "stage24a"
    protocol = json.loads((stage24a / "eval3_manifest.json").read_text()) if (stage24a / "eval3_manifest.json").is_file() else {}
    cand_manifest = json.loads((report_root / "candidate_manifest.json").read_text())
    if champion is None:
        gate = {"status": "NO_LOCK", "n_pass": 0, "n_required": 2, "per_target": {}}
    else:
        gate = champion["gate"]
    lock = build_lock_manifest(
        cfg=cfg,
        gate_result=gate,
        candidate=champion,
        protocol_manifest=protocol,
        candidate_manifest=cand_manifest,
    )
    lock["status"] = decision["status"]
    lock["selection_note"] = (
        "Champion selected among NoHoldout-eligible PASS candidates under AACDR stest0 hard gate; "
        "holdout refs excluded from lock ranking."
    )
    lock_path = ROOT / cfg["locks"]["output"]
    write_lock_manifest(lock_path, lock)

    # Final report
    lines = [
        "# Round 24 — Final Report",
        "",
        f"**Status:** `{decision['status']}`",
        f"**Standard:** AACDR stest0 (no 10% testset) — [`AACDR_drug_macro_auroc_auprc.md`](AACDR_drug_macro_auroc_auprc.md)",
        f"**Hard gate:** `aacdr_gdsc_intersect` > 0.5279 ∧ `aacdr_tcga_only` > 0.4804",
        f"**Champion:** `{decision['champion_id']}`" if decision["champion_id"] else "**Champion:** none (`NO_LOCK`)",
        "",
        "## NoHoldout lock pool",
        "",
        "| ID | Architecture / Feature | aacdr_gdsc | aacdr_tcga | Hard gate |",
        "|----|------------------------|-----------:|-----------:|:---------:|",
    ]
    for c in candidates:
        if c.get("train_source") == "holdout_ref":
            continue
        lines.append(
            f"| {c['candidate_id']} | {c.get('architecture')} × {c.get('feature')} | "
            f"{c['per_target_fold_mean_auc'].get('aacdr_gdsc_intersect', float('nan')):.4f} | "
            f"{c['per_target_fold_mean_auc'].get('aacdr_tcga_only', float('nan')):.4f} | "
            f"{c.get('gate', {}).get('status')} |"
        )
    lines += [
        "",
        "## Holdout references (not ranked for lock)",
        "",
    ]
    for c in candidates:
        if c.get("train_source") != "holdout_ref":
            continue
        lines.append(
            f"- `{c['candidate_id']}`: aacdr_gdsc="
            f"{c['per_target_fold_mean_auc'].get('aacdr_gdsc_intersect', float('nan')):.4f}, "
            f"aacdr_tcga={c['per_target_fold_mean_auc'].get('aacdr_tcga_only', float('nan')):.4f}, "
            f"gate={c.get('gate', {}).get('status')}"
        )
    if champion:
        lines += [
            "",
            "## Champion metrics (5-fold mean DrugMacro)",
            "",
            "| Target | AUROC | AUPRC | Required |",
            "|--------|------:|------:|:--------:|",
        ]
        req = set(cfg.get("gate_required_targets", []))
        for t in cfg["target_priority"]:
            lines.append(
                f"| `{t}` | {champion['per_target_fold_mean_auc'][t]:.4f} | "
                f"{champion['per_target_fold_mean_auprc'][t]:.4f} | "
                f"{'Y' if t in req else 'N'} |"
            )
    lines += [
        "",
        f"**Lock file:** `{cfg['locks']['output']}`",
        f"**Decision:** `reports/round24/stage24e/stage24e_decision.json`",
        "",
    ]
    (ROOT / "docs/round24_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Rebuild vs table
    subprocess.call(
        [sys.executable, str(ROOT / "scripts/round24/rebuild_vs_aacdr_standard.py")],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )

    msg = (
        f"Round24 COMPLETE\n"
        f"status={decision['status']}\n"
        f"champion={decision['champion_id']}\n"
        f"n_pass={decision['n_pass']}/{decision['n_lock_pool']}\n"
        f"standard=aacdr_stest0\n"
        f"lock={cfg['locks']['output']}"
    )
    biocda_notify(msg)
    print(json.dumps(decision, indent=2))
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/round24/eval3.yaml"))
    parser.add_argument("--preregister-only", action="store_true")
    parser.add_argument("--reuse-only", action="store_true", help="Copy NH0/refs only")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--mem-fraction", type=float, default=0.18)
    parser.add_argument("--candidates", nargs="*", default=["E-NH1", "E-NH2"])
    parser.add_argument("--single-job", type=str, default=None)
    args = parser.parse_args()

    cfg = load_eval3_config(args.config)

    if args.single_job:
        spec = json.loads(args.single_job)
        cand = next(c for c in CANDIDATES if c["id"] == spec["candidate_id"])
        train_one_fold(
            cand=cand,
            fold_id=int(spec["fold_id"]),
            cfg=cfg,
            smoke=args.smoke,
            mem_fraction=args.mem_fraction,
        )
        return 0

    write_manifest(cfg)
    if args.preregister_only:
        return 0

    reuse_nh0(cfg)
    for cand in CANDIDATES:
        if cand["action"] == "reuse_ref":
            reuse_ref(cfg, cand)

    if args.reuse_only and not args.train and not args.finalize:
        return 0

    if args.train or (not args.finalize and not args.reuse_only):
        train_cands = [c for c in CANDIDATES if c["id"] in args.candidates and c["action"] == "train"]
        # Build TCGA latents once up-front so parallel workers stay on GPU (train/infer).
        print("Prebuilding TCGA latent caches (serial)...", flush=True)
        prebuild_tcga_caches(cfg, [c["id"] for c in train_cands])
        jobs = []
        for cand in train_cands:
            for fold_id in range(int(cfg["n_folds"])):
                result_dir = ROOT / cfg["paths"]["run_root"] / "stage24e" / cand["id"] / f"fold_{fold_id}"
                if _job_done(result_dir) and not args.smoke:
                    continue
                jobs.append((cand["id"], fold_id))
        print(f"Stage24E jobs to run: {len(jobs)} (max_parallel={args.max_parallel}, mem_fraction={args.mem_fraction})", flush=True)
        if jobs:
            with ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as ex:
                futs = [
                    ex.submit(
                        _launch_job,
                        cid,
                        fid,
                        mem_fraction=args.mem_fraction,
                        config=args.config if args.config.is_absolute() else ROOT / args.config,
                    )
                    for cid, fid in jobs
                ]
                for fut in as_completed(futs):
                    print(fut.result(), flush=True)
        for cand in train_cands:
            aggregate_candidate(cfg, cand)

    if args.finalize or args.train:
        # finalize after train completes
        if all(
            (ROOT / cfg["paths"]["reports_root"] / "stage24e" / cid / "candidate_summary.json").is_file()
            for cid in ["E-NH0", "E-NH1", "E-NH2"]
        ):
            finalize(cfg)
        else:
            print("Not all summaries ready; skip finalize", flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
