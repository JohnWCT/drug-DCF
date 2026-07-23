#!/usr/bin/env python3
"""Train B0-like pooled_mlp on NoHoldout/AACDR arms; TCGA eval3; compare to Ctrl.

Design:
- Architecture matches Round18 B0: GIN + PooledMLPFusion + Round18ResponseHead
  on own_plus_summary (75-d), full omics vector (no Z/C split).
- Early stop on source validation DrugMacro AUC only (never TCGA).
- Ctrl metrics reused from Stage24A baseline (no retrain).
- Parallel fold jobs with CUDA memory fraction to share GPU with Stage24C.
- Telegram only when the full ablation round finishes.
"""
from __future__ import annotations

import argparse
import json
import os
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
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from biocda.utils.gpu import configure_gpu_efficiency
from biocda.validation.round24_gate import evaluate_all_target_gate
from biocda.validation.round24_protocol import gate_table, load_eval3_config
from drugmodels.ginconv import GINConvNet
from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics, metrics_to_jsonable
from tools.round18_dataset import Round18ResponseDataset, round18_graph_collate_fn
from tools.round18_fusion_models import PooledMLPFusion
from tools.round18_response_head import Round18ResponseHead
from tools.round18_tcga_dataset import (
    Round18TCGADataset,
    prepare_tcga_response_frame,
    round18_tcga_collate_fn,
)

ABLATION_ROOT = ROOT / "result/optimization_runs/round24_train_source_ablation"
REPORT_ROOT = ROOT / "reports/round24/train_source_ablation"
FEATURE_OWN = ROOT / "result/optimization_runs/round17r_18class/features/r13_exp_008/own_plus_summary"
SMILES = ROOT / "data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv"


class PooledMLPModel(nn.Module):
    def __init__(self, omics_dim: int = 75, graph_dim: int = 32):
        super().__init__()
        self.encoder = GINConvNet(
            input_dim=78,
            node_hidden_dim=32,
            graph_output_dim=graph_dim,
            dropout=0.1,
            num_layers=5,
            jk_mode="last",
            pool_type="max",
            use_batch_norm=True,
        )
        self.fusion = PooledMLPFusion(omics_dim=omics_dim, graph_dim=graph_dim)
        self.head = Round18ResponseHead(input_dim=self.fusion.output_dim)

    def forward(self, omics: torch.Tensor, drug_batch) -> torch.Tensor:
        graph_emb = self.encoder(drug_batch)
        fused = self.fusion(omics, graph_emb)
        return self.head(fused).reshape(-1)


def _job_done(result_dir: Path) -> bool:
    return (
        (result_dir / "best.pt").is_file()
        and (result_dir / "tcga_fold_metrics.json").is_file()
        and (result_dir / "status.json").is_file()
    )


def _subset_loader(dataset, row_ids, *, batch_size: int, shuffle: bool) -> DataLoader:
    id_to_idx = {int(r): i for i, r in enumerate(dataset.df["_row_id"].astype(int))}
    indices = [id_to_idx[r] for r in row_ids if r in id_to_idx]
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=round18_graph_collate_fn,
    )


@torch.no_grad()
def _predict(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        omics = batch["omics"].to(device)
        drug = batch["drug_batch"].to(device)
        logits = model(omics, drug)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels = batch["label"].cpu().numpy().astype(int)
        for i in range(len(labels)):
            rows.append(
                {
                    "_row_id": int(batch["_row_id"][i]),
                    "ModelID": str(batch["ModelID"][i]),
                    "DRUG_NAME": str(batch["DRUG_NAME"][i]),
                    "Label": int(labels[i]),
                    "probability": float(probs[i]),
                    "logit": float(logits[i].detach().cpu()),
                }
            )
    return pd.DataFrame(rows)


def _infer_tcga(model, *, cfg, result_dir: Path, device) -> Dict[str, Any]:
    smiles = str(SMILES)
    feature_dir = str(FEATURE_OWN)
    out = {}
    for t in cfg["targets"]:
        key = t["key"]
        frame, tcga_latent, _ = prepare_tcga_response_frame(
            str(ROOT / t["path"]),
            feature_dir=feature_dir,
            drug_smiles_path=smiles,
            target_key=key,
        )
        ds = Round18TCGADataset(frame, tcga_latent=tcga_latent, graph_cache={})
        loader = DataLoader(
            ds, batch_size=256, shuffle=False, num_workers=0, collate_fn=round18_tcga_collate_fn
        )
        pred = _predict(model, loader, device)
        pred.to_csv(result_dir / f"tcga_predictions__{key}.csv", index=False)
        metrics = metrics_to_jsonable(calculate_robust_drug_macro_metrics(pred))
        metrics["n_rows"] = int(len(pred))
        out[key] = metrics
        print(f"  TCGA {key} AUC={metrics.get('DrugMacro_AUC')} n={len(pred)}", flush=True)
    return out


def train_one_fold(
    *,
    arm: str,
    fold_id: int,
    cfg: Dict[str, Any],
    smoke: bool = False,
    mem_fraction: float = 0.4,
) -> Dict[str, Any]:
    result_dir = ABLATION_ROOT / arm / "runs" / f"fold_{fold_id}"
    result_dir.mkdir(parents=True, exist_ok=True)
    if _job_done(result_dir) and not smoke:
        return {"status": "skipped_complete", "arm": arm, "fold_id": fold_id}

    configure_gpu_efficiency(target_utilization=0.9)
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(float(mem_fraction), 0)
        except Exception:
            pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arm_root = ABLATION_ROOT / arm
    dev = pd.read_csv(arm_root / "splits" / "development_rows.csv")
    assigns = pd.read_csv(arm_root / "splits" / "formal_5fold_assignments.csv")
    fold_part = assigns[assigns["fold_id"] == fold_id]
    train_ids = fold_part[fold_part["split_role"] == "train"]["_row_id"].astype(int).tolist()
    val_ids = fold_part[fold_part["split_role"] == "val"]["_row_id"].astype(int).tolist()

    dataset = Round18ResponseDataset(
        dev,
        feature_dir=str(FEATURE_OWN),
        drug_smiles_path=str(SMILES),
        graph_cache={},
    )
    omics_dim = int(dataset.omics_dim)
    bs = 64 if smoke else 512
    train_loader = _subset_loader(dataset, train_ids, batch_size=bs, shuffle=True)
    val_loader = _subset_loader(dataset, val_ids, batch_size=bs, shuffle=False)

    model = PooledMLPModel(omics_dim=omics_dim).to(device)
    id_to_idx = {int(r): i for i, r in enumerate(dataset.df["_row_id"].astype(int))}
    y = np.array(
        [int(dataset.df.iloc[id_to_idx[r]]["Label"]) for r in train_ids if r in id_to_idx],
        dtype=np.float32,
    )
    pos = max(float((y == 1).sum()), 1.0)
    neg = max(float((y == 0).sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = GradScaler(enabled=device.type == "cuda")
    max_epochs = 2 if smoke else 200
    patience = 2 if smoke else 20
    best = float("-inf")
    best_epoch = -1
    bad = 0
    t0 = time.perf_counter()

    print(f"[{arm} fold{fold_id}] start omics_dim={omics_dim} bs={bs} n_train={len(train_ids)} device={device}", flush=True)
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        step = 0
        n_batches = 0
        for batch in train_loader:
            omics = batch["omics"].to(device)
            drug = batch["drug_batch"].to(device)
            labels = batch["label"].to(device)
            with autocast(enabled=scaler.is_enabled()):
                logits = model(omics, drug)
                loss = loss_fn(logits, labels) / 4.0  # accum 4
            scaler.scale(loss).backward()
            step += 1
            n_batches += 1
            if step >= 4:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step = 0
            if smoke and n_batches >= 5:
                break
        if step > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        val_pred = _predict(model, val_loader, device)
        val_metrics = calculate_robust_drug_macro_metrics(val_pred)
        score = float(val_metrics.get("DrugMacro_AUC") or -1.0)
        if score > best:
            best = score
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "omics_dim": omics_dim,
                    "arm": arm,
                    "fold_id": fold_id,
                    "best_val_DrugMacro_AUC": best,
                    "architecture": "pooled_mlp__own_plus_summary",
                },
                result_dir / "best.pt",
            )
            val_pred.to_csv(result_dir / "val_predictions.csv", index=False)
        else:
            bad += 1
            if bad >= patience:
                break

    blob = torch.load(result_dir / "best.pt", map_location=device)
    model = PooledMLPModel(omics_dim=int(blob["omics_dim"])).to(device)
    model.load_state_dict(blob["model_state_dict"], strict=True)
    tcga = _infer_tcga(model, cfg=cfg, result_dir=result_dir, device=device)
    (result_dir / "tcga_fold_metrics.json").write_text(json.dumps(tcga, indent=2, default=str) + "\n")
    summary = {
        "status": "complete",
        "arm": arm,
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_val_DrugMacro_AUC": best,
        "training_time_sec": time.perf_counter() - t0,
        "tcga_used_in_selection": False,
        "smoke": smoke,
    }
    (result_dir / "fold_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (result_dir / "status.json").write_text(json.dumps({"status": "complete"}) + "\n")
    return summary


def aggregate_arm(arm: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_root = ABLATION_ROOT / arm / "runs"
    report = REPORT_ROOT / arm
    report.mkdir(parents=True, exist_ok=True)
    n_folds = int(cfg["n_folds"])
    fold_aucs = {t["key"]: [] for t in cfg["targets"]}
    fold_rows = []
    for fold_id in range(n_folds):
        mpath = run_root / f"fold_{fold_id}" / "tcga_fold_metrics.json"
        if not mpath.is_file():
            raise FileNotFoundError(mpath)
        metrics = json.loads(mpath.read_text())
        for key in fold_aucs:
            auc = metrics[key].get("DrugMacro_AUC")
            fold_aucs[key].append(float(auc) if auc is not None else float("nan"))
            fold_rows.append({"arm": arm, "fold_id": fold_id, "target_key": key, "DrugMacro_AUC": auc,
                              "DrugMacro_AUPRC": metrics[key].get("DrugMacro_AUPRC")})
    # ensemble
    for t in cfg["targets"]:
        key = t["key"]
        frames = [pd.read_csv(run_root / f"fold_{i}" / f"tcga_predictions__{key}.csv") for i in range(n_folds)]
        base = frames[0][["_row_id", "DRUG_NAME", "Label"]].copy()
        for i, df in enumerate(frames):
            base = base.merge(df[["_row_id", "probability"]].rename(columns={"probability": f"p{i}"}), on="_row_id", how="left")
        pcols = [c for c in base.columns if c.startswith("p")]
        base["probability"] = base[pcols].mean(axis=1)
        base.to_csv(report / f"ensemble_predictions__{key}.csv", index=False)

    means = {k: float(np.nanmean(v)) for k, v in fold_aucs.items()}
    stds = {k: float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in fold_aucs.items()}
    gate = evaluate_all_target_gate(means, gate_table(cfg), target_priority=cfg["target_priority"], target_weights=cfg["target_weights"])
    payload = {
        "arm": arm,
        "architecture": "pooled_mlp__own_plus_summary",
        "feature": "own_plus_summary",
        "per_target_fold_mean_auc": means,
        "per_target_fold_std_auc": stds,
        "gate": gate,
    }
    pd.DataFrame(fold_rows).to_csv(report / "fold_metrics.csv", index=False)
    (report / "candidate_summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload


def load_ctrl(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = json.loads((ROOT / "reports/round24/stage24a/baseline_summary.json").read_text())
    means = {k: v["fold_mean_DrugMacro_AUC"] for k, v in base["targets"].items()}
    stds = {k: v.get("fold_std_DrugMacro_AUC", 0.0) for k, v in base["targets"].items()}
    gate = evaluate_all_target_gate(means, gate_table(cfg), target_priority=cfg["target_priority"], target_weights=cfg["target_weights"])
    payload = {
        "arm": "Ctrl",
        "architecture": "pooled_mlp__own_plus_summary",
        "feature": "own_plus_summary",
        "reused_from": "reports/round24/stage24a/baseline_summary.json",
        "per_target_fold_mean_auc": means,
        "per_target_fold_std_auc": stds,
        "gate": gate,
    }
    out = REPORT_ROOT / "Ctrl"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidate_summary.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return payload


def write_final_report(summaries: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Path:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    ctrl = next(s for s in summaries if s["arm"] == "Ctrl")
    rows = []
    for s in summaries:
        for t in cfg["targets"]:
            key = t["key"]
            auc = s["per_target_fold_mean_auc"][key]
            base = ctrl["per_target_fold_mean_auc"][key]
            rows.append({
                "arm": s["arm"],
                "target": key,
                "fold_mean_DrugMacro_AUC": auc,
                "ctrl_auc": base,
                "delta_vs_ctrl": auc - base,
                "gate": t["gate_auroc"],
                "pass_gate": auc > t["gate_auroc"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_ROOT / "per_arm_fold_metrics.csv", index=False)
    summary = {
        "arms": summaries,
        "deltas_vs_ctrl": {
            s["arm"]: {k: s["per_target_fold_mean_auc"][k] - ctrl["per_target_fold_mean_auc"][k]
                       for k in ctrl["per_target_fold_mean_auc"]}
            for s in summaries if s["arm"] != "Ctrl"
        },
        "any_all_target_pass": any(s["gate"]["status"] == "PASS" for s in summaries),
        "diagnostic_only": True,
    }
    (REPORT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    lines = [
        "# Round24 train-source ablation (diagnostic)",
        "",
        "Architecture: **pooled_mlp × own_plus_summary** (B0). Not a formal lock candidate.",
        "",
        "## Fold-mean DrugMacro AUROC",
        "",
        "| Arm | " + " | ".join(t["key"] for t in cfg["targets"]) + " | n_pass |",
        "|-----|" + "|".join(["------"] * len(cfg["targets"])) + "|-------|",
    ]
    for s in summaries:
        vals = " | ".join(f"{s['per_target_fold_mean_auc'][t['key']]:.4f}" for t in cfg["targets"])
        lines.append(f"| {s['arm']} | {vals} | {s['gate']['n_pass']}/5 |")
    lines += ["", "## Δ vs Ctrl", ""]
    for arm, deltas in summary["deltas_vs_ctrl"].items():
        dstr = ", ".join(f"{k} {v:+.4f}" for k, v in deltas.items())
        lines.append(f"- **{arm}**: {dstr}")
    lines += ["", f"any_all_target_pass={summary['any_all_target_pass']}", ""]
    path = REPORT_ROOT / "ablation_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _launch_job(arm: str, fold_id: int, smoke: bool, mem_fraction: float) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["ABLATION_MEM_FRACTION"] = str(mem_fraction)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/round24/run_train_source_ablation.py"),
        "--single-job",
        json.dumps({"arm": arm, "fold_id": fold_id}),
        "--config",
        str(ROOT / "configs/round24/eval3.yaml"),
        "--mem-fraction",
        str(mem_fraction),
    ]
    if smoke:
        cmd.append("--smoke")
    print("+", " ".join(cmd[:6]), f"{arm} fold={fold_id}", flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    status = ABLATION_ROOT / arm / "runs" / f"fold_{fold_id}" / "status.json"
    if status.is_file():
        return json.loads(status.read_text())
    return {"status": "failed", "rc": rc, "arm": arm, "fold_id": fold_id}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/round24/eval3.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--single-job", type=str, default=None)
    parser.add_argument("--mem-fraction", type=float, default=0.4)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--arms", nargs="*", default=["NoHoldout", "AACDR"])
    args = parser.parse_args()
    cfg = load_eval3_config(args.config)

    if args.single_job:
        job = json.loads(args.single_job)
        summary = train_one_fold(
            arm=job["arm"],
            fold_id=int(job["fold_id"]),
            cfg=cfg,
            smoke=args.smoke,
            mem_fraction=float(args.mem_fraction),
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary.get("status") in {"complete", "skipped_complete"} else 1

    # Ensure data prepared
    prep = ROOT / "scripts/round24/prepare_train_source_ablation.py"
    subprocess.check_call([sys.executable, str(prep)], cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    ctrl = load_ctrl(cfg)

    jobs = []
    for arm in args.arms:
        n = 1 if args.smoke else int(cfg["n_folds"])
        for fold_id in range(n):
            jobs.append((arm, fold_id))

    # Prefer parallel subprocesses when formal
    results = []
    if args.smoke or args.max_parallel <= 1:
        for arm, fold_id in jobs:
            results.append(train_one_fold(arm=arm, fold_id=fold_id, cfg=cfg, smoke=args.smoke, mem_fraction=args.mem_fraction))
    else:
        with ThreadPoolExecutor(max_workers=int(args.max_parallel)) as ex:
            futs = {ex.submit(_launch_job, arm, fold_id, False, args.mem_fraction): (arm, fold_id) for arm, fold_id in jobs}
            for fut in as_completed(futs):
                results.append(fut.result())

    if args.smoke:
        print(json.dumps({"smoke_results": results}, indent=2, default=str))
        return 0

    summaries = [ctrl]
    for arm in args.arms:
        summaries.append(aggregate_arm(arm, cfg))
    report_path = write_final_report(summaries, cfg)

    # Telegram only on full round completion
    from tools.biocda_telegram_notify import biocda_notify

    lines = ["Round24 train-source ablation COMPLETE (diagnostic)"]
    for s in summaries:
        means = s["per_target_fold_mean_auc"]
        lines.append(
            f"{s['arm']} n_pass={s['gate']['n_pass']}/5 "
            f"gdsc={means['gdsc_intersect13']:.3f} "
            f"tcga3={means['tcga_only3']:.3f}"
        )
    lines.append(f"report={report_path}")
    biocda_notify("\n".join(lines))
    print(json.dumps({"report": str(report_path), "summaries": [{k: s[k] for k in ('arm','gate','per_target_fold_mean_auc') if k in s} for s in summaries]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
