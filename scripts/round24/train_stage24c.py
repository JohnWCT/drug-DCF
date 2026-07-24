#!/usr/bin/env python3
"""Stage 24C: feature attribution with fixed pooled-E3 on Round18 formal 5-folds.

Only the feature recipe changes. Architecture = BioCDAPredictive (pooled E3) with
omics_dim=64 and context_dim = feature_dim-64 (Z + remainder).
F3 reuses Stage24B B1 artifacts (same C32 + pooled E3).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from biocda.data.xa_dataset import biocda_collate_fn, build_xa_dataset
from biocda.models.predictive.pooled_e3 import BioCDAPredictive
from biocda.training.graph_cache_io import load_graph_cache
from biocda.training.xa_loop import pos_weight_from_labels, train_xa_run
from biocda.utils.gpu import build_efficient_dataloader_kwargs, configure_gpu_efficiency
from tools.biocda_telegram_notify import biocda_notify
from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics, metrics_to_jsonable
from tools.round18_tcga_dataset import (
    load_tcga_omics_latent_dict,
    prepare_tcga_response_frame,
)
from tools.round19_dataset import Round19ResponseDataset
from tools.round20_tcga import TCGA_TARGETS, build_tcga_o2_features
from biocda.validation.tcga_benchmark import _resolve_round20_lock

FEATURE_RECIPES = [
    {
        "id": "F0",
        "name": "own_plus_summary",
        "path": "result/optimization_runs/round17r_18class/features/r13_exp_008/own_plus_summary",
        "tcga_mode": "pkl",
    },
    {
        "id": "F1",
        "name": "z_plus_summary",
        "path": "result/optimization_runs/round19_factorial/features/z_plus_summary",
        "tcga_mode": "pkl",
    },
    {
        "id": "F2",
        "name": "z_plus_context16",
        "path": "result/optimization_runs/round20_unseen_drug_closure/features/z_plus_context16",
        "tcga_mode": "c16_build",
    },
    {
        "id": "F3",
        "name": "z_plus_context32",
        "path": "result/optimization_runs/round20_unseen_drug_closure/features/z_plus_context32",
        "tcga_mode": "reuse_b1",
    },
    {
        "id": "F4",
        "name": "z_plus_summary_context16",
        "path": "result/optimization_runs/round19_factorial/features/z_plus_summary_context16",
        "tcga_mode": "pkl",
    },
]


def _load_cfg(path: Path) -> Dict[str, Any]:
    p = path if path.is_absolute() else ROOT / path
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _feature_dim(feature_dir: Path) -> int:
    meta = feature_dir / "feature_metadata.json"
    if meta.is_file():
        m = json.loads(meta.read_text())
        d = m.get("response_input_dim") or m.get("feature_dim")
        if d is not None:
            return int(d)
    # fallback: probe ccle pickle
    import pickle

    with (feature_dir / "ccle_latent_proto.pkl").open("rb") as f:
        lat = pickle.load(f)
    return int(np.asarray(next(iter(lat.values()))).reshape(-1).shape[0])


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


def _patient_latent_from_map(latent: Dict[str, Any]) -> Dict[str, Any]:
    patient_latent: Dict[str, Any] = {}
    for key, vec in latent.items():
        parts = str(key).split("-")
        if len(parts) >= 3:
            patient_latent.setdefault("-".join(parts[:3]), vec)
        patient_latent[str(key)] = vec
    return patient_latent


def _load_tcga_latent(recipe: Dict[str, Any], feature_dir: Path) -> Dict[str, Any]:
    mode = recipe["tcga_mode"]
    if mode == "pkl":
        return _patient_latent_from_map(load_tcga_omics_latent_dict(str(feature_dir)))
    if mode in {"c16_build", "c32_build", "reuse_b1"}:
        lock = _resolve_round20_lock()
        lock = dict(lock)
        lock["selected_context"] = dict(lock["selected_context"])
        if "context16" in recipe["name"] or mode == "c16_build":
            lock["selected_context"]["id"] = "C16"
            lock["selected_context"]["omics_dimension"] = 80
        else:
            lock["selected_context"]["id"] = "C32"
            lock["selected_context"]["omics_dimension"] = 96
        lock["selected_context"]["feature_dir"] = str(feature_dir)
        return _patient_latent_from_map(build_tcga_o2_features(lock))
    raise ValueError(mode)


@torch.no_grad()
def _infer_tcga(
    model: nn.Module,
    *,
    recipe: Dict[str, Any],
    feature_dir: Path,
    smiles: str,
    result_dir: Path,
    device: torch.device,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    patient_latent = _load_tcga_latent(recipe, feature_dir)
    out: Dict[str, Any] = {}
    for t in cfg["targets"]:
        key = t["key"]
        path = str(ROOT / t["path"])
        # Prefer Round18 prepare for pkl features; for built latents use frame from prepare + override
        frame, _, _ = prepare_tcga_response_frame(
            path,
            feature_dir=str(feature_dir) if (feature_dir / "tcga_latent_proto.pkl").is_file() else str(
                ROOT / cfg["paths"]["feature_own_plus_summary"]
            ),
            drug_smiles_path=smiles,
            target_key=key,
        )
        # Filter to patients present in this recipe's latent
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
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=biocda_collate_fn)
        model.eval()
        rows = []
        for batch in loader:
            omics = batch["omics"].to(device)
            context = batch["context"].to(device)
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
        print(f"  [{recipe['id']}] {key} AUC={metrics.get('DrugMacro_AUC')} n={len(pred)}", flush=True)
    return out


def _reuse_b1_as_f3(cfg: Dict[str, Any], run_root: Path, report_root: Path) -> None:
    src = ROOT / cfg["paths"]["run_root"] / "stage24b" / "B1"
    dst = run_root / "F3"
    dst.mkdir(parents=True, exist_ok=True)
    for fold_id in range(int(cfg["n_folds"])):
        s = src / f"fold_{fold_id}"
        d = dst / f"fold_{fold_id}"
        d.mkdir(parents=True, exist_ok=True)
        for name in ("best.pt", "tcga_fold_metrics.json", "status.json", "fold_summary.json"):
            if (s / name).is_file():
                shutil.copy2(s / name, d / name)
        for p in s.glob("tcga_predictions__*.csv"):
            shutil.copy2(p, d / p.name)
    # copy candidate summary from B1 reports
    b1rep = ROOT / cfg["paths"]["reports_root"] / "stage24b" / "B1" / "candidate_summary.json"
    if b1rep.is_file():
        payload = json.loads(b1rep.read_text())
        payload["candidate_id"] = "F3"
        payload["feature"] = "z_plus_context32"
        payload["reused_from"] = "stage24b/B1"
        (report_root / "F3").mkdir(parents=True, exist_ok=True)
        (report_root / "F3" / "candidate_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
        for p in (ROOT / cfg["paths"]["reports_root"] / "stage24b" / "B1").glob("ensemble_predictions__*.csv"):
            shutil.copy2(p, report_root / "F3" / p.name)
        if (ROOT / cfg["paths"]["reports_root"] / "stage24b" / "B1" / "fold_metrics.csv").is_file():
            shutil.copy2(
                ROOT / cfg["paths"]["reports_root"] / "stage24b" / "B1" / "fold_metrics.csv",
                report_root / "F3" / "fold_metrics.csv",
            )


def train_one_fold(
    *,
    recipe: Dict[str, Any],
    fold_id: int,
    cfg: Dict[str, Any],
    smoke: bool = False,
    mem_fraction: float = 0.35,
) -> Dict[str, Any]:
    feature_dir = ROOT / recipe["path"]
    result_dir = ROOT / cfg["paths"]["run_root"] / "stage24c" / recipe["id"] / f"fold_{fold_id}"
    result_dir.mkdir(parents=True, exist_ok=True)
    if _job_done(result_dir) and not smoke:
        return {"status": "skipped_complete", "result_dir": str(result_dir), "feature_id": recipe["id"], "fold_id": fold_id}

    configure_gpu_efficiency(target_utilization=float(cfg.get("target_gpu_utilization", 0.9)))
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(float(mem_fraction), 0)
        except Exception:
            pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smiles = str(ROOT / cfg["paths"]["drug_smiles"])
    r18 = ROOT / cfg["paths"]["round18_root"]
    dev = pd.read_csv(r18 / "splits" / "development_rows.csv")
    assigns = pd.read_csv(r18 / "splits" / "formal_5fold_assignments.csv")
    fold_part = assigns[assigns["fold_id"] == fold_id]
    train_ids = fold_part[fold_part["split_role"] == "train"]["_row_id"].astype(int).tolist()
    val_ids = fold_part[fold_part["split_role"] == "val"]["_row_id"].astype(int).tolist()

    dim = _feature_dim(feature_dir)
    if dim <= 64:
        raise ValueError(f"{recipe['id']} dim={dim} expected >64 for Z+context split")
    context_dim = dim - 64

    r23_cache = ROOT / "outputs/xa_v2_closure"
    graph_cache = load_graph_cache(r23_cache) if (r23_cache / "shared_graph_cache.pkl").is_file() else {}
    print(f"[{recipe['id']} fold{fold_id}] dim={dim} context={context_dim}", flush=True)
    dataset = build_xa_dataset(dev, feature_dir=str(feature_dir), drug_smiles_path=smiles, graph_cache=graph_cache)

    tr = cfg["training"]
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
    # Only warm-start GIN when context_dim matches locked C32 teacher (32)
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
        config={"round24c": True, "feature_id": recipe["id"], "fold_id": fold_id, "feature_dim": dim},
        pos_weight=pw,
    )

    blob = torch.load(result_dir / "best.pt", map_location=device)
    model = BioCDAPredictive(omics_dim=64, context_dim=context_dim).to(device)
    model.load_state_dict(blob["model_state_dict"], strict=True)
    tcga_metrics = _infer_tcga(
        model,
        recipe=recipe,
        feature_dir=feature_dir,
        smiles=smiles,
        result_dir=result_dir,
        device=device,
        cfg=cfg,
    )
    (result_dir / "tcga_fold_metrics.json").write_text(json.dumps(tcga_metrics, indent=2, default=str) + "\n")
    summary = {
        "status": "complete",
        "feature_id": recipe["id"],
        "fold_id": fold_id,
        "feature_dim": dim,
        "context_dim": context_dim,
        "best_epoch": result.best_epoch,
        "best_val_DrugMacro_AUC": result.metrics_validation.get("DrugMacro_AUC"),
        "tcga_used_in_selection": False,
    }
    (result_dir / "fold_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (result_dir / "status.json").write_text(json.dumps({"status": "complete"}) + "\n")
    return summary


def aggregate_feature(cfg: Dict[str, Any], recipe: Dict[str, Any]) -> Dict[str, Any]:
    run_root = ROOT / cfg["paths"]["run_root"] / "stage24c" / recipe["id"]
    report_root = ROOT / cfg["paths"]["reports_root"] / "stage24c" / recipe["id"]
    report_root.mkdir(parents=True, exist_ok=True)
    n_folds = int(cfg["n_folds"])
    fold_aucs = {t["key"]: [] for t in cfg["targets"]}
    fold_auprcs = {t["key"]: [] for t in cfg["targets"]}
    fold_rows = []
    missing = [i for i in range(n_folds) if not (run_root / f"fold_{i}" / "tcga_fold_metrics.json").is_file()]
    if missing:
        raise FileNotFoundError(f"{recipe['id']} missing folds {missing}")
    for fold_id in range(n_folds):
        metrics = json.loads((run_root / f"fold_{fold_id}" / "tcga_fold_metrics.json").read_text())
        for key in fold_aucs:
            auc = metrics[key].get("DrugMacro_AUC")
            auprc = metrics[key].get("DrugMacro_AUPRC")
            fold_aucs[key].append(float(auc) if auc is not None else float("nan"))
            fold_auprcs[key].append(float(auprc) if auprc is not None else float("nan"))
            fold_rows.append(
                {
                    "feature_id": recipe["id"],
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

    payload = {
        "candidate_id": recipe["id"],
        "architecture": "biocda_predictive_e3",
        "feature": recipe["name"],
        "n_folds": n_folds,
        "per_target_fold_mean_auc": {k: float(np.nanmean(v)) for k, v in fold_aucs.items()},
        "per_target_fold_mean_auprc": {k: float(np.nanmean(v)) for k, v in fold_auprcs.items()},
        "per_target_fold_std_auc": {
            k: float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in fold_aucs.items()
        },
        "status": "complete",
    }
    pd.DataFrame(fold_rows).to_csv(report_root / "fold_metrics.csv", index=False)
    (report_root / "candidate_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def rank_features(summaries: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    from biocda.validation.round24_gate import evaluate_all_target_gate
    from biocda.validation.round24_protocol import gate_table

    gates = gate_table(cfg)
    rows = []
    for s in summaries:
        gate = evaluate_all_target_gate(
            s["per_target_fold_mean_auc"],
            gates,
            target_priority=cfg["target_priority"],
            target_weights=cfg["target_weights"],
            gate_required_targets=cfg.get("gate_required_targets"),
        )
        deltas = {k: s["per_target_fold_mean_auc"][k] - gates[k]["gate_auroc"] for k in gates}
        rows.append(
            {
                **s,
                "gate": gate,
                "n_pass": gate["n_pass"],
                "min_delta": min(deltas.values()),
                "weighted": gate.get("weighted_DrugMacro_AUC"),
            }
        )
    # Sort: n_pass desc, min_delta desc, weighted desc
    rows.sort(key=lambda r: (r["n_pass"], r["min_delta"], r["weighted"] or 0.0), reverse=True)
    top2 = rows[:2]
    return {"ranked": rows, "top2": [r["candidate_id"] for r in top2], "any_all_pass": any(r["gate"]["status"] == "PASS" for r in rows)}


def _launch_fold_job(recipe_id: str, fold_id: int, *, mem_fraction: float, config: Path) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/round24/train_stage24c.py"),
        "--config",
        str(config),
        "--single-job",
        json.dumps({"feature_id": recipe_id, "fold_id": fold_id}),
        "--mem-fraction",
        str(mem_fraction),
    ]
    print("+", " ".join(cmd[:6]), f"{recipe_id} fold={fold_id}", flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    status = ROOT / "result/optimization_runs/round24_tcga_recovery/stage24c" / recipe_id / f"fold_{fold_id}" / "status.json"
    # resolve via cfg path after load — use default run root
    if status.is_file():
        return json.loads(status.read_text())
    return {"status": "failed", "rc": rc, "feature_id": recipe_id, "fold_id": fold_id}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--features", nargs="*", default=None, help="Subset e.g. F0 F1 F2")
    parser.add_argument("--single-job", type=str, default=None, help='JSON {"feature_id","fold_id"}')
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--mem-fraction", type=float, default=0.32)
    parser.add_argument("--resume", action="store_true", help="Skip Telegram START (resume parallel)")
    args = parser.parse_args()
    cfg = _load_cfg(args.config)
    configure_gpu_efficiency(target_utilization=float(cfg.get("target_gpu_utilization", 0.9)))

    recipes = FEATURE_RECIPES
    if args.features:
        want = set(args.features)
        recipes = [r for r in recipes if r["id"] in want]
    recipe_by_id = {r["id"]: r for r in FEATURE_RECIPES}

    if args.single_job:
        job = json.loads(args.single_job)
        recipe = recipe_by_id[job["feature_id"]]
        summary = train_one_fold(
            recipe=recipe,
            fold_id=int(job["fold_id"]),
            cfg=cfg,
            smoke=args.smoke,
            mem_fraction=float(args.mem_fraction),
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0 if summary.get("status") in {"complete", "skipped_complete"} else 1

    run_root = ROOT / cfg["paths"]["run_root"] / "stage24c"
    report_root = ROOT / cfg["paths"]["reports_root"] / "stage24c"
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    if not args.resume and not args.smoke:
        biocda_notify(f"Round24 Stage24C START features={[r['id'] for r in recipes]} parallel={args.max_parallel}")

    # Reuse F3 from B1 first
    if any(r["id"] == "F3" for r in recipes) and not args.smoke:
        _reuse_b1_as_f3(cfg, run_root, report_root)
        print("F3 reused from Stage24B B1", flush=True)

    train_recipes = [r for r in recipes if r["id"] != "F3" or args.smoke]
    if args.smoke:
        train_recipes = train_recipes[:1]
        for recipe in train_recipes:
            train_one_fold(recipe=recipe, fold_id=0, cfg=cfg, smoke=True, mem_fraction=float(args.mem_fraction))
        print("Stage24C smoke done (no Telegram mid-stage)", flush=True)
        return 0

    # Build pending jobs (skip completed folds)
    jobs: List[Tuple[str, int]] = []
    for recipe in train_recipes:
        for fold_id in range(int(cfg["n_folds"])):
            result_dir = run_root / recipe["id"] / f"fold_{fold_id}"
            if _job_done(result_dir):
                print(f"skip complete {recipe['id']} fold{fold_id}", flush=True)
                continue
            jobs.append((recipe["id"], fold_id))

    print(f"Pending jobs={len(jobs)} max_parallel={args.max_parallel}", flush=True)
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config

    if args.max_parallel <= 1:
        for rid, fold_id in jobs:
            train_one_fold(
                recipe=recipe_by_id[rid],
                fold_id=fold_id,
                cfg=cfg,
                smoke=False,
                mem_fraction=float(args.mem_fraction),
            )
    else:
        with ThreadPoolExecutor(max_workers=int(args.max_parallel)) as ex:
            futs = {
                ex.submit(_launch_fold_job, rid, fold_id, mem_fraction=float(args.mem_fraction), config=cfg_path): (rid, fold_id)
                for rid, fold_id in jobs
            }
            for fut in as_completed(futs):
                rid, fold_id = futs[fut]
                try:
                    res = fut.result()
                    print(f"done {rid} fold{fold_id} status={res.get('status')}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"FAILED {rid} fold{fold_id}: {exc}", flush=True)

    for recipe in train_recipes:
        try:
            payload = aggregate_feature(cfg, recipe)
            print(f"Aggregated {recipe['id']}: {payload['per_target_fold_mean_auc']}", flush=True)
        except FileNotFoundError as e:
            print(e, flush=True)

    if any(r["id"] == "F3" for r in recipes):
        try:
            aggregate_feature(cfg, next(r for r in FEATURE_RECIPES if r["id"] == "F3"))
        except Exception:
            pass

    summaries = []
    for recipe in recipes:
        p = report_root / recipe["id"] / "candidate_summary.json"
        if p.is_file():
            summaries.append(json.loads(p.read_text()))
    ranking = rank_features(summaries, cfg) if summaries else {"ranked": [], "top2": [], "any_all_pass": False}
    (report_root / "feature_attribution_summary.json").write_text(json.dumps(ranking, indent=2, default=str) + "\n")
    biocda_notify(
        f"Round24 Stage24C DONE top2={ranking.get('top2')} any_all_pass={ranking.get('any_all_pass')}"
    )
    print(json.dumps({"top2": ranking.get("top2"), "any_all_pass": ranking.get("any_all_pass")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
