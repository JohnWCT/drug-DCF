"""Phased XA v2 training loop with optional logit distillation."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from biocda.training.distillation import combine_response_kd, export_student_only_state
from biocda.training.freeze_schedule import (
    DEFAULT_PHASES,
    FreezePhase,
    apply_phase,
    build_param_groups,
    phase_for_epoch,
    set_frozen_bn_eval,
)
from biocda.training.xa_loop import TrainRunResult, _predict, pos_weight_from_labels
from biocda.utils.gpu import configure_gpu_efficiency
from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics, early_stop_score


def train_xa_v2_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    *,
    run_dir: Path,
    phases: Optional[List[FreezePhase]] = None,
    patience: int = 20,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    accumulation_steps: int = 1,
    model_type: str = "biocda_xa_transfer",
    architecture_version: str = "biocda-xa-v2",
    config: Optional[Dict[str, Any]] = None,
    pos_weight: Optional[torch.Tensor] = None,
    teacher: Optional[nn.Module] = None,
    lambda_kd: float = 0.5,
    kd_temperature: float = 2.0,
    diagnostics_every: int = 0,
) -> TrainRunResult:
    """Train XA student with warm-up → last-GIN FT → stabilize schedule."""
    configure_gpu_efficiency(target_utilization=0.9)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if teacher is not None:
        teacher = teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    run_dir.mkdir(parents=True, exist_ok=True)
    schedule = phases or DEFAULT_PHASES
    max_epochs = sum(int(p.epochs) for p in schedule)

    if pos_weight is None:
        # collect from one pass
        ys = []
        for batch in train_loader:
            ys.append(batch["labels"])
            if len(ys) > 20:
                break
        pos_weight = pos_weight_from_labels(torch.cat(ys), device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler(enabled=use_amp and device.startswith("cuda"))

    best_score = float("-inf")
    best_epoch = -1
    stale = 0
    ckpt_path = run_dir / "best.pt"
    phase_log: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    current_phase_name = None
    optimizer = None
    scheduler = None

    for epoch in range(max_epochs):
        phase = phase_for_epoch(epoch, schedule)
        if phase.name != current_phase_name:
            info = apply_phase(model, phase)
            groups = build_param_groups(model, phase)
            optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
            )
            current_phase_name = phase.name
            phase_log.append({"epoch": epoch, **info})
            (run_dir / "freeze_phases.json").write_text(
                json.dumps(phase_log, indent=2) + "\n", encoding="utf-8"
            )

        # Keep frozen BN in eval even during model.train()
        model.train()
        frozen = phase.freeze_gin_layers
        set_frozen_bn_eval(model.drug_encoder.gin, frozen)

        assert optimizer is not None and scheduler is not None
        optimizer.zero_grad(set_to_none=True)
        step_in_accum = 0

        for batch in train_loader:
            omics = batch["omics"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            drug_graph = batch["drug_graph"].to(device)
            with autocast(enabled=scaler.is_enabled()):
                out = model(omics, context, drug_graph, output_mode="prediction")
                response = loss_fn(out.logits, labels)
                if teacher is not None:
                    t_logits = teacher(omics, context, drug_graph, output_mode="prediction").logits
                    bundled = combine_response_kd(
                        response,
                        out.logits,
                        t_logits,
                        lambda_kd=lambda_kd,
                        temperature=kd_temperature,
                    )
                    loss = bundled.total / max(int(accumulation_steps), 1)
                else:
                    loss = response / max(int(accumulation_steps), 1)
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
        score = float(early_stop_score(val_metrics)["score"])
        scheduler.step(score if score == score else 0.0)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            payload = {
                "model_state_dict": export_student_only_state(model),
                "epoch": epoch,
                "model_type": model_type,
                "architecture_version": architecture_version,
                "config": config or {},
                "has_teacher": teacher is not None,
                "phase": phase.name,
            }
            torch.save(payload, ckpt_path)
        else:
            stale += 1
            if stale >= patience:
                break

    # Reload best
    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state_dict"], strict=True)
    val_pred = _predict(model, val_loader, device)
    test_pred = _predict(model, test_loader, device)
    val_metrics = calculate_robust_drug_macro_metrics(val_pred)
    test_metrics = calculate_robust_drug_macro_metrics(test_pred)
    val_pred.to_parquet(run_dir / "predictions_validation.parquet", index=False)
    test_pred.to_parquet(run_dir / "predictions_test.parquet", index=False)
    (run_dir / "metrics_by_seed.json").write_text(
        json.dumps(
            {
                "validation": val_metrics,
                "test": test_metrics,
                "best_epoch": best_epoch,
                "phases": phase_log,
            },
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
