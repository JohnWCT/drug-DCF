"""GDSC training loop for BioCDA XA validation (Round 21)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from biocda.utils.gpu import configure_gpu_efficiency
from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics, early_stop_score


@dataclass
class TrainRunResult:
    best_epoch: int
    best_val_score: float
    metrics_validation: Dict[str, Any]
    metrics_test: Dict[str, Any]
    training_time: float
    checkpoint_path: str


def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            omics = batch["omics"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            drug_graph = batch["drug_graph"].to(device)
            out = model(omics, context, drug_graph, output_mode="prediction")
            probs = torch.sigmoid(out.logits)
            for i in range(len(labels)):
                rows.append(
                    {
                        "DRUG_NAME": batch["DRUG_NAME"][i],
                        "Label": int(labels[i].item()),
                        "probability": float(probs[i].item()),
                        "logit": float(out.logits[i].item()),
                        "_row_id": batch["_row_id"][i],
                        "ModelID": batch["ModelID"][i],
                    }
                )
    return pd.DataFrame(rows)


def _pos_weight_from_loader(loader: DataLoader, device: str) -> Optional[torch.Tensor]:
    pos = 0.0
    neg = 0.0
    for batch in loader:
        y = batch["labels"]
        pos += float((y == 1).sum())
        neg += float((y == 0).sum())
    if pos <= 0 or neg <= 0:
        return None
    return torch.tensor([neg / pos], device=device)


def pos_weight_from_labels(labels: torch.Tensor, device: str) -> Optional[torch.Tensor]:
    pos = float((labels == 1).sum())
    neg = float((labels == 0).sum())
    if pos <= 0 or neg <= 0:
        return None
    return torch.tensor([neg / pos], device=device)


def train_xa_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    *,
    run_dir: Path,
    max_epochs: int = 200,
    patience: int = 20,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    model_type: str = "biocda_xa_zc",
    architecture_version: str = "biocda-xa-v1",
    config: Optional[Dict[str, Any]] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> TrainRunResult:
    configure_gpu_efficiency(target_utilization=0.9)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    run_dir.mkdir(parents=True, exist_ok=True)

    if pos_weight is None:
        pos_weight = _pos_weight_from_loader(train_loader, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    scaler = GradScaler(enabled=use_amp and device.startswith("cuda"))

    best_score = float("-inf")
    best_epoch = -1
    stale = 0
    ckpt_path = run_dir / "best.pt"
    t0 = time.perf_counter()

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_in_accum = 0
        for batch in train_loader:
            omics = batch["omics"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            drug_graph = batch["drug_graph"].to(device)
            with autocast(enabled=scaler.is_enabled()):
                out = model(omics, context, drug_graph, output_mode="prediction")
                loss = loss_fn(out.logits, labels) / max(int(accumulation_steps), 1)
            scaler.scale(loss).backward()
            step_in_accum += 1
            if step_in_accum >= max(int(accumulation_steps), 1):
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step_in_accum = 0
        if step_in_accum > 0:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        val_pred = _predict(model, val_loader, device)
        val_metrics = calculate_robust_drug_macro_metrics(val_pred)
        score_info = early_stop_score(val_metrics)
        score = float(score_info["score"])
        scheduler.step(score if score == score else 0.0)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            from biocda.training.checkpoint import save_biocda_checkpoint

            save_biocda_checkpoint(
                ckpt_path,
                model=model,
                config=config or {},
                epoch=epoch,
                model_type=model_type,
                architecture_version=architecture_version,
            )
        else:
            stale += 1
            if stale >= patience:
                break

    from biocda.training.checkpoint import load_biocda_checkpoint

    load_biocda_checkpoint(model, ckpt_path, strict=True)
    val_pred = _predict(model, val_loader, device)
    test_pred = _predict(model, test_loader, device)
    val_metrics = calculate_robust_drug_macro_metrics(val_pred)
    test_metrics = calculate_robust_drug_macro_metrics(test_pred)

    val_pred.to_parquet(run_dir / "predictions_validation.parquet", index=False)
    test_pred.to_parquet(run_dir / "predictions_test.parquet", index=False)
    (run_dir / "metrics_by_seed.json").write_text(
        json.dumps(
            {"validation": val_metrics, "test": test_metrics, "best_epoch": best_epoch},
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    return TrainRunResult(
        best_epoch=best_epoch,
        best_val_score=best_score,
        metrics_validation=val_metrics,
        metrics_test=test_metrics,
        training_time=time.perf_counter() - t0,
        checkpoint_path=str(ckpt_path),
    )


def metrics_to_summary_row(
    *,
    model: str,
    seed: int,
    result: TrainRunResult,
) -> Dict[str, Any]:
    vm = result.metrics_validation
    return {
        "model": model,
        "seed": seed,
        "drug_macro_auc": vm.get("DrugMacro_AUC"),
        "drug_macro_auprc": vm.get("DrugMacro_AUPRC"),
        "sample_auc": vm.get("Global_AUC"),
        "sample_auprc": vm.get("Global_AUPRC"),
        "brier": None,
        "ece": None,
        "best_epoch": result.best_epoch,
        "training_time": result.training_time,
    }
