"""Round 19 training / eval loop (GIN/GINE/MACCS × P0/P1/P2)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from tools.model_opt import FocalLoss
from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics, early_stop_score
from tools.round18_train_loop import set_round18_seeds


def assert_disjoint_param_groups(*modules: nn.Module) -> None:
    """Every requires_grad param belongs to exactly one module group."""
    seen: Set[int] = set()
    for mod in modules:
        for p in mod.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                raise AssertionError("Duplicate trainable parameter across optimizer groups")
            seen.add(pid)
    # all trainable collected
    all_ids = {id(p) for m in modules for p in m.parameters() if p.requires_grad}
    if seen != all_ids:
        raise AssertionError("Trainable parameter set mismatch across groups")


def build_round19_param_groups(
    encoder: nn.Module,
    fusion: nn.Module,
    head: nn.Module,
    *,
    encoder_lr: float,
    fusion_lr: float,
    head_lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    assert_disjoint_param_groups(encoder, fusion, head)
    return [
        {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": encoder_lr},
        {"params": [p for p in fusion.parameters() if p.requires_grad], "lr": fusion_lr},
        {"params": [p for p in head.parameters() if p.requires_grad], "lr": head_lr},
        # weight_decay applied by AdamW constructor shared; keep groups lr-only
    ]


def forward_round19_batch(
    *,
    encoder: nn.Module,
    fusion: nn.Module,
    encoder_type: str,
    predictor_id: str,
    omics: torch.Tensor,
    batch: Dict[str, Any],
    return_interpretability: bool = False,
    return_attention: bool = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Dict[str, torch.Tensor],
]:
    enc = str(encoder_type).lower()
    pred = str(predictor_id).upper()
    if (return_interpretability or return_attention) and pred != "P2":
        raise ValueError(
            f"{pred} has no atom-level attention; interpretability requires P2"
        )
    if enc == "maccs":
        if pred == "P2":
            raise AssertionError("MACCS incompatible with P2")
        if batch.get("maccs") is None:
            raise ValueError("MACCS batch missing maccs tensor")
        if batch.get("drug_batch") is not None:
            raise AssertionError("MACCS job must not carry PyG drug_batch")
        drug_vec = encoder(batch["maccs"])
        return fusion(
            omics,
            drug_vec,
            return_interpretability=return_interpretability,
            return_attention=return_attention,
        )

    drug_batch = batch["drug_batch"]
    if pred == "P2":
        # Pure atom cross-attention: node embeddings only (no graph residual).
        out = encoder(drug_batch, return_dict=True, return_graph_embedding=False)
        return fusion(
            omics,
            out["node_embeddings"],
            out["batch_index"],
            return_interpretability=return_interpretability,
            return_attention=return_attention,
        )

    out = encoder(drug_batch, return_dict=True, return_graph_embedding=True)
    return fusion(
        omics,
        out["graph_embedding"],
        return_interpretability=return_interpretability,
        return_attention=return_attention,
    )


def train_one_epoch_round19(
    *,
    encoder: nn.Module,
    fusion: nn.Module,
    head: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss_fn: nn.Module,
    device: torch.device,
    encoder_type: str,
    predictor_id: str,
    accumulation_steps: int = 1,
    amp_enabled: bool = True,
    grad_clip_max_norm: float = 1.0,
) -> Dict[str, float]:
    encoder.train()
    fusion.train()
    head.train()
    total_loss = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(dataloader, start=1):
        omics = batch["omics"].to(device)
        labels = batch["label"].to(device).float()
        weights = batch["weight"].to(device).float()
        if batch.get("maccs") is not None:
            batch = {**batch, "maccs": batch["maccs"].to(device), "drug_batch": None}
        elif batch.get("drug_batch") is not None:
            batch = {**batch, "drug_batch": batch["drug_batch"].to(device)}
        with autocast(enabled=amp_enabled and device.type == "cuda"):
            repr_vec = forward_round19_batch(
                encoder=encoder,
                fusion=fusion,
                encoder_type=encoder_type,
                predictor_id=predictor_id,
                omics=omics,
                batch=batch,
            )
            logits = head(repr_vec).view(-1)
            # FocalLoss ignores weights in base; keep path consistent
            loss = loss_fn(logits, labels)
            loss = loss / max(int(accumulation_steps), 1)
        scaler.scale(loss).backward()
        if step % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(encoder.parameters()) + list(fusion.parameters()) + list(head.parameters()) if p.requires_grad],
                grad_clip_max_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu()) * max(int(accumulation_steps), 1)
        n += 1
    if n and (n % accumulation_steps) != 0:
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": total_loss / max(n, 1)}


@torch.no_grad()
def evaluate_predictions_round19(
    *,
    encoder: nn.Module,
    fusion: nn.Module,
    head: nn.Module,
    dataloader,
    device: torch.device,
    encoder_type: str,
    predictor_id: str,
    amp_enabled: bool = True,
) -> Dict[str, Any]:
    encoder.eval()
    fusion.eval()
    head.eval()
    rows = []
    for batch in dataloader:
        omics = batch["omics"].to(device)
        labels = batch["label"].to(device).float()
        local = dict(batch)
        if local.get("maccs") is not None:
            local["maccs"] = local["maccs"].to(device)
            local["drug_batch"] = None
        elif local.get("drug_batch") is not None:
            local["drug_batch"] = local["drug_batch"].to(device)
        with autocast(enabled=amp_enabled and device.type == "cuda"):
            repr_vec = forward_round19_batch(
                encoder=encoder,
                fusion=fusion,
                encoder_type=encoder_type,
                predictor_id=predictor_id,
                omics=omics,
                batch=local,
            )
            logits = head(repr_vec).view(-1)
            probs = torch.sigmoid(logits)
        drugs = batch.get("drug_name", ["NA"] * labels.size(0))
        row_ids = batch.get("_row_id", [-1] * labels.size(0))
        model_ids = batch.get("ModelID", [""] * labels.size(0))
        for i in range(labels.size(0)):
            drug = drugs[i] if isinstance(drugs, (list, tuple)) else str(drugs[i])
            rows.append(
                {
                    "_row_id": int(row_ids[i]),
                    "ModelID": str(model_ids[i]),
                    "DRUG_NAME": str(drug),
                    "Label": int(labels[i].item()),
                    "logit": float(logits[i].item()),
                    "probability": float(probs[i].item()),
                }
            )
    pred_df = pd.DataFrame(rows)
    if pred_df["_row_id"].duplicated().any():
        raise AssertionError("Duplicate _row_id in validation predictions")
    metrics = calculate_robust_drug_macro_metrics(pred_df)
    stop = early_stop_score(metrics)
    return {"predictions": pred_df, "metrics": metrics, "early_stop": stop}


def make_default_loss(device: torch.device, gamma: float = 2.0) -> nn.Module:
    return FocalLoss(gamma=gamma, reduction="mean").to(device)


__all__ = [
    "assert_disjoint_param_groups",
    "build_round19_param_groups",
    "evaluate_predictions_round19",
    "forward_round19_batch",
    "make_default_loss",
    "set_round18_seeds",
    "train_one_epoch_round19",
]
