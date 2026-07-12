"""Round 18 training loop utilities (GIN + fusion + head all in train mode)."""
from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from tools.model_opt import FocalLoss
from tools.round18_cv_metrics import calculate_robust_drug_macro_metrics, early_stop_score


def set_round18_seeds(seed: int = 101) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_param_groups(
    gin: nn.Module,
    fusion: nn.Module,
    head: nn.Module,
    *,
    gin_lr: float,
    fusion_lr: float,
    head_lr: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    return [
        {"params": [p for p in gin.parameters() if p.requires_grad], "lr": gin_lr},
        {"params": [p for p in fusion.parameters() if p.requires_grad], "lr": fusion_lr},
        {"params": [p for p in head.parameters() if p.requires_grad], "lr": head_lr},
    ]


def collect_trainable_parameters(*modules: nn.Module) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for module in modules:
        for p in module.parameters():
            if p.requires_grad:
                params.append(p)
    return params


def _maybe_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def forward_round18_batch(
    *,
    gin: nn.Module,
    fusion: nn.Module,
    architecture_family: str,
    omics: torch.Tensor,
    drug_batch,
    residual_mode: str = "pure",
) -> torch.Tensor:
    """Return fusion representation (not logits)."""
    family = architecture_family.lower()
    needs_nodes = family in {"cross_attention", "atom_cross_attention", "c0", "c1"}
    if needs_nodes:
        gin_out = gin(drug_batch, return_node_embeddings=True, return_graph_embedding=True)
        if residual_mode == "pooled_residual" or family == "c1":
            return fusion(
                omics,
                gin_out["node_embeddings"],
                gin_out["batch_index"],
                graph_embedding=gin_out["graph_embedding"],
            )
        return fusion(omics, gin_out["node_embeddings"], gin_out["batch_index"])

    graph_emb = gin(drug_batch, return_node_embeddings=False)
    return fusion(omics, graph_emb)


def train_one_epoch(
    *,
    gin: nn.Module,
    fusion: nn.Module,
    head: Optional[nn.Module],
    dataloader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss_fn: nn.Module,
    device: torch.device,
    architecture_family: str,
    accumulation_steps: int = 1,
    amp_enabled: bool = True,
    grad_clip_max_norm: float = 1.0,
    residual_mode: str = "pure",
) -> Dict[str, float]:
    """
    Round 18 semantics: gin/fusion/(optional external head) all .train().
    If fusion already includes response head, pass head=None.
    """
    gin.train()
    fusion.train()
    if head is not None:
        head.train()

    trainable = collect_trainable_parameters(gin, fusion, *( [head] if head is not None else [] ))
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    n_steps = 0
    for step, batch in enumerate(dataloader):
        omics = _maybe_to_device(batch["omics"], device)
        labels = _maybe_to_device(batch["label"], device).float()
        weights = batch.get("weight")
        if weights is not None:
            weights = _maybe_to_device(weights, device).float()
        drug_batch = batch["drug_batch"].to(device)

        with autocast(enabled=amp_enabled and device.type == "cuda"):
            fusion_repr = forward_round18_batch(
                gin=gin,
                fusion=fusion,
                architecture_family=architecture_family,
                omics=omics,
                drug_batch=drug_batch,
                residual_mode=residual_mode,
            )
            if head is None:
                raise ValueError("Round 18 requires a separate response_head")
            logits = head(fusion_repr)
            loss = loss_fn(logits, labels, weights)
            loss = loss / max(accumulation_steps, 1)

        if not torch.isfinite(loss):
            raise RuntimeError("numerical_failure: non-finite loss")

        scaler.scale(loss).backward()
        total_loss += float(loss.detach().item()) * max(accumulation_steps, 1)
        n_steps += 1

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    return {"loss": total_loss / max(n_steps, 1), "n_steps": float(n_steps)}


@torch.no_grad()
def evaluate_predictions(
    *,
    gin: nn.Module,
    fusion: nn.Module,
    head: Optional[nn.Module],
    dataloader,
    device: torch.device,
    architecture_family: str,
    residual_mode: str = "pure",
    amp_enabled: bool = True,
) -> Dict[str, Any]:
    gin.eval()
    fusion.eval()
    if head is not None:
        head.eval()

    rows = []
    for batch in dataloader:
        omics = _maybe_to_device(batch["omics"], device)
        labels = _maybe_to_device(batch["label"], device).float()
        drug_batch = batch["drug_batch"].to(device)
        drugs = batch.get("drug_name", ["NA"] * labels.size(0))
        with autocast(enabled=amp_enabled and device.type == "cuda"):
            fusion_repr = forward_round18_batch(
                gin=gin,
                fusion=fusion,
                architecture_family=architecture_family,
                omics=omics,
                drug_batch=drug_batch,
                residual_mode=residual_mode,
            )
            if head is None:
                raise ValueError("Round 18 requires a separate response_head")
            logits = head(fusion_repr)
            probs = torch.sigmoid(logits)
        row_ids = batch.get("_row_id", [None] * labels.size(0))
        model_ids = batch.get("ModelID", [None] * labels.size(0))
        for i in range(labels.size(0)):
            drug = drugs[i] if isinstance(drugs, (list, tuple)) else str(drugs[i])
            rows.append(
                {
                    "_row_id": int(row_ids[i]) if row_ids[i] is not None else -1,
                    "ModelID": str(model_ids[i]) if model_ids[i] is not None else "",
                    "DRUG_NAME": str(drug),
                    "Label": int(labels[i].item()),
                    "logit": float(logits[i].item()),
                    "probability": float(probs[i].item()),
                }
            )

    import pandas as pd

    pred_df = pd.DataFrame(rows)
    metrics = calculate_robust_drug_macro_metrics(pred_df)
    stop = early_stop_score(metrics)
    return {"predictions": pred_df, "metrics": metrics, "early_stop": stop}


def make_default_loss(device: torch.device, gamma: float = 2.0) -> nn.Module:
    return FocalLoss(gamma=gamma, reduction="mean").to(device)


def synthetic_round18_batch(
    *,
    batch_size: int = 4,
    omics_dim: int = 75,
    node_dim: int = 78,
    atoms_per_graph: Sequence[int] = (5, 6, 4, 7),
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Build a tiny in-memory batch for smoke training (no SMILES I/O)."""
    from torch_geometric.data import Batch, Data

    device = device or torch.device("cpu")
    graphs = []
    n = min(batch_size, len(atoms_per_graph))
    for n_atoms in atoms_per_graph[:n]:
        x = torch.randn(n_atoms, node_dim)
        if n_atoms > 1:
            src = torch.arange(0, n_atoms - 1)
            dst = torch.arange(1, n_atoms)
            edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index))
    while len(graphs) < batch_size:
        graphs.append(graphs[len(graphs) % max(len(graphs), 1)])

    labels = torch.randint(0, 2, (batch_size,)).float()
    weights = torch.ones(batch_size)
    omics = torch.randn(batch_size, omics_dim)
    drugs = [f"drug_{i % 3}" for i in range(batch_size)]
    return {
        "omics": omics.to(device),
        "label": labels.to(device),
        "weight": weights.to(device),
        "drug_batch": Batch.from_data_list(graphs[:batch_size]).to(device),
        "drug_name": drugs[:batch_size],
    }


class _SyntheticLoader:
    def __init__(self, batches: List[Dict[str, Any]]):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def run_synthetic_smoke_train(
    architecture_family: str = "pooled_mlp",
    *,
    residual_mode: str = "pure",
    steps: int = 2,
    omics_dim: int = 75,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """End-to-end smoke: build models, train a few steps, evaluate."""
    from drugmodels.ginconv import GINConvNet
    from tools.round18_fusion_models import build_fusion_and_head

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_round18_seeds(101)

    gin = GINConvNet(
        input_dim=78,
        output_dim=32,
        dropout=0.1,
        num_layers=3,
        jk_mode="last",
        pool_type="max",
    ).to(device)

    family = architecture_family.lower()
    if family in {"pooled_transformer", "transformer"}:
        fusion, head = build_fusion_and_head(
            "pooled_transformer",
            omics_dim=omics_dim,
            graph_dim=32,
            transformer_cfg={"d_model": 64, "n_heads": 4, "num_layers": 1, "dim_feedforward": 128},
        )
    elif family in {"cross_attention", "c0", "c1"}:
        mode = "pooled_residual" if family == "c1" or residual_mode == "pooled_residual" else "pure"
        fusion, head = build_fusion_and_head(
            "cross_attention",
            omics_dim=omics_dim,
            residual_mode=mode,
            cross_attn_cfg={"d_model": 64, "n_heads": 4, "num_layers": 1, "dim_feedforward": 128},
        )
        residual_mode = mode
    else:
        fusion, head = build_fusion_and_head("pooled_mlp", omics_dim=omics_dim, graph_dim=32)
    fusion = fusion.to(device)
    head = head.to(device)

    optimizer = torch.optim.AdamW(
        build_param_groups(gin, fusion, head, gin_lr=1e-4, fusion_lr=3e-4, head_lr=3e-4, weight_decay=1e-4),
        weight_decay=1e-4,
    )

    loss_fn = make_default_loss(device)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    loader = _SyntheticLoader(
        [synthetic_round18_batch(batch_size=4, omics_dim=omics_dim, device=device) for _ in range(steps)]
    )

    train_stats = train_one_epoch(
        gin=gin,
        fusion=fusion,
        head=head,
        dataloader=loader,
        optimizer=optimizer,
        scaler=scaler,
        loss_fn=loss_fn,
        device=device,
        architecture_family=architecture_family,
        accumulation_steps=2,
        amp_enabled=(device.type == "cuda"),
        residual_mode=residual_mode,
    )
    assert gin.training  # train_one_epoch leaves models in train mode

    # Force enough class diversity in eval batch labels for metrics
    eval_batch = synthetic_round18_batch(batch_size=12, omics_dim=omics_dim, device=device)
    eval_batch["label"] = torch.tensor([0, 1] * 6, device=device).float()
    eval_batch["drug_name"] = [f"drug_{i % 2}" for i in range(12)]
    eval_out = evaluate_predictions(
        gin=gin,
        fusion=fusion,
        head=head,
        dataloader=_SyntheticLoader([eval_batch]),
        device=device,
        architecture_family=architecture_family,
        residual_mode=residual_mode,
        amp_enabled=(device.type == "cuda"),
    )
    return {
        "train": train_stats,
        "metrics": {
            "Global_AUC": eval_out["metrics"]["Global_AUC"],
            "DrugMacro_AUC": eval_out["metrics"]["DrugMacro_AUC"],
            "n_valid_auc_drugs": eval_out["metrics"]["n_valid_auc_drugs"],
        },
        "early_stop": eval_out["early_stop"],
        "architecture_family": architecture_family,
        "residual_mode": residual_mode,
        "device": str(device),
    }
