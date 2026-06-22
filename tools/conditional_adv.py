"""Conditional adversarial deconfounding modules for Round 10."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

VALID_GLOBAL_ADV_MODES = frozenset(
    {
        "baseline_global_only",
        "conditional_replacement",
        "conditional_plus_weak_global",
    }
)
VALID_CONDITIONAL_ADV_MODES = frozenset({"none", "cancer_embedding"})


class CancerConditionEncoder(nn.Module):
    """Converts cancer type ids into trainable condition embeddings."""

    def __init__(self, num_cancer_types: int, condition_dim: int):
        super().__init__()
        if num_cancer_types <= 1:
            raise ValueError("num_cancer_types must be > 1")
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        self.num_cancer_types = int(num_cancer_types)
        self.condition_dim = int(condition_dim)
        self.embedding = nn.Embedding(self.num_cancer_types, self.condition_dim)

    def forward(self, cancer_type: torch.Tensor) -> torch.Tensor:
        if cancer_type.dtype != torch.long:
            cancer_type = cancer_type.long()
        if cancer_type.numel() > 0:
            min_id = int(cancer_type.min().item())
            max_id = int(cancer_type.max().item())
            if min_id < 0 or max_id >= self.num_cancer_types:
                raise ValueError(
                    f"Invalid cancer_type id in range [{min_id}, {max_id}] "
                    f"for num_cancer_types={self.num_cancer_types}"
                )
        return self.embedding(cancer_type)


class ConditionalDomainCritic(nn.Module):
    """WGAN-GP critic that receives concat(z, cancer_condition)."""

    def __init__(
        self,
        latent_size: int,
        num_cancer_types: int,
        condition_dim: int = 16,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.condition_encoder = CancerConditionEncoder(num_cancer_types, condition_dim)
        layers: list[nn.Module] = []
        in_dim = latent_size + condition_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, cancer_type: torch.Tensor) -> torch.Tensor:
        cond = self.condition_encoder(cancer_type)
        x = torch.cat([z, cond], dim=1)
        return self.net(x).view(-1)


def compute_conditional_gradient_penalty(
    critic: nn.Module,
    real_z: torch.Tensor,
    fake_z: torch.Tensor,
    cancer_type: torch.Tensor,
    device: torch.device,
    gp_weight: float = 10.0,
) -> torch.Tensor:
    """WGAN-GP on interpolated latent z; cancer_type held fixed."""
    min_batch = min(real_z.shape[0], fake_z.shape[0])
    real_z = real_z[:min_batch]
    fake_z = fake_z[:min_batch]
    cancer_type = cancer_type[:min_batch]
    epsilon = torch.rand(min_batch, 1, device=device)
    z_hat = epsilon * real_z + (1.0 - epsilon) * fake_z
    z_hat.requires_grad_(True)
    d_hat = critic(z_hat, cancer_type)
    grad = torch.autograd.grad(
        outputs=d_hat,
        inputs=z_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean() * gp_weight
    return gp


def compute_conditional_gp_by_cancer(
    critic: nn.Module,
    source_z: torch.Tensor,
    target_z: torch.Tensor,
    source_cancer: torch.Tensor,
    target_cancer: torch.Tensor,
    device: torch.device,
    gp_weight: float = 10.0,
    min_pairs: int = 2,
) -> Tuple[torch.Tensor, int, str]:
    """Per-cancer GP when enough paired samples exist; else batch fallback."""
    cancers = torch.unique(torch.cat([source_cancer, target_cancer]))
    gp_terms = []
    skip_count = 0
    for cancer_id in cancers.tolist():
        s_mask = source_cancer == cancer_id
        t_mask = target_cancer == cancer_id
        s_count = int(s_mask.sum().item())
        t_count = int(t_mask.sum().item())
        pair_count = min(s_count, t_count)
        if pair_count < min_pairs:
            skip_count += 1
            continue
        s_z = source_z[s_mask][:pair_count]
        t_z = target_z[t_mask][:pair_count]
        c_ids = source_cancer[s_mask][:pair_count]
        gp_terms.append(
            compute_conditional_gradient_penalty(
                critic, s_z, t_z, c_ids, device=device, gp_weight=gp_weight
            )
        )
    if gp_terms:
        return torch.stack(gp_terms).mean(), skip_count, "per_cancer"
    gp = compute_conditional_gradient_penalty(
        critic,
        source_z.detach(),
        target_z.detach(),
        source_cancer,
        device=device,
        gp_weight=gp_weight,
    )
    return gp, skip_count, "batch_fallback"


def get_cond_adv_lambda_eff(
    epoch: int,
    lambda_cond_adv: float,
    start_epoch: int,
    full_epoch: int,
) -> float:
    """Linear ramp from 0 to lambda_cond_adv between start and full epoch."""
    if lambda_cond_adv <= 0:
        return 0.0
    if epoch < start_epoch:
        return 0.0
    if full_epoch <= start_epoch:
        return float(lambda_cond_adv)
    if epoch >= full_epoch:
        return float(lambda_cond_adv)
    span = float(full_epoch - start_epoch)
    progress = float(epoch - start_epoch) / span
    return float(lambda_cond_adv) * progress


def build_cancer_type_mapping(mapping_int2str: Dict[int, str]) -> Dict[str, Any]:
    cancer_to_id = {str(name): int(idx) for idx, name in mapping_int2str.items()}
    return {
        "num_cancer_types": len(cancer_to_id),
        "cancer_to_id": cancer_to_id,
        "id_to_cancer": {str(k): v for k, v in mapping_int2str.items()},
    }


def resolve_conditional_adv_training_params(param: dict) -> dict:
    conditional_adv_enabled = bool(param.get("conditional_adv_enabled", False))
    conditional_adv_mode = str(param.get("conditional_adv_mode", "none"))
    cancer_condition_dim = int(param.get("cancer_condition_dim", 16))
    lambda_cond_adv = float(param.get("lambda_cond_adv", 0.0))
    cond_adv_start_epoch = int(param.get("cond_adv_start_epoch", 10))
    cond_adv_full_epoch = int(param.get("cond_adv_full_epoch", 60))
    global_adv_mode = str(param.get("global_adv_mode", "baseline_global_only"))
    lambda_global_adv_multiplier = float(param.get("lambda_global_adv_multiplier", 1.0))
    raw_hidden = param.get("cond_critic_hidden_dims", [128, 64])
    cond_critic_hidden_dims = tuple(int(x) for x in raw_hidden)
    cond_critic_dropout = float(param.get("cond_critic_dropout", 0.1))

    if global_adv_mode not in VALID_GLOBAL_ADV_MODES:
        raise ValueError(
            f"Unsupported global_adv_mode={global_adv_mode}. "
            f"Use one of: {sorted(VALID_GLOBAL_ADV_MODES)}"
        )
    if conditional_adv_mode not in VALID_CONDITIONAL_ADV_MODES:
        raise ValueError(
            f"Unsupported conditional_adv_mode={conditional_adv_mode}. "
            f"Use one of: {sorted(VALID_CONDITIONAL_ADV_MODES)}"
        )
    if conditional_adv_enabled and conditional_adv_mode != "cancer_embedding":
        raise ValueError("Round 10 only supports conditional_adv_mode='cancer_embedding'")
    if conditional_adv_enabled and global_adv_mode == "baseline_global_only":
        raise ValueError(
            "conditional_adv_enabled=true requires global_adv_mode "
            "conditional_replacement or conditional_plus_weak_global"
        )
    if not conditional_adv_enabled:
        global_adv_mode = "baseline_global_only"
        conditional_adv_mode = "none"
        lambda_cond_adv = 0.0

    return {
        "conditional_adv_enabled": conditional_adv_enabled,
        "conditional_adv_mode": conditional_adv_mode,
        "cancer_condition_dim": cancer_condition_dim,
        "lambda_cond_adv": lambda_cond_adv,
        "cond_adv_start_epoch": cond_adv_start_epoch,
        "cond_adv_full_epoch": cond_adv_full_epoch,
        "global_adv_mode": global_adv_mode,
        "lambda_global_adv_multiplier": lambda_global_adv_multiplier,
        "cond_critic_hidden_dims": cond_critic_hidden_dims,
        "cond_critic_dropout": cond_critic_dropout,
        "round": str(param.get("round", "")),
        "round10_branch": str(param.get("round10_branch", "")),
        "source_baseline_exp_id": str(param.get("source_baseline_exp_id", "")),
    }


def conditional_adv_metrics_payload(cond_cfg: dict, gan_logs: dict | None = None) -> dict:
    gan_logs = gan_logs or {}
    payload = {
        "round": cond_cfg.get("round", ""),
        "round10_branch": cond_cfg.get("round10_branch", ""),
        "source_baseline_exp_id": cond_cfg.get("source_baseline_exp_id", ""),
        "conditional_adv_enabled": bool(cond_cfg.get("conditional_adv_enabled", False)),
        "conditional_adv_mode": cond_cfg.get("conditional_adv_mode", "none"),
        "cancer_condition_dim": cond_cfg.get("cancer_condition_dim", 16),
        "lambda_cond_adv": cond_cfg.get("lambda_cond_adv", 0.0),
        "cond_adv_start_epoch": cond_cfg.get("cond_adv_start_epoch", 10),
        "cond_adv_full_epoch": cond_cfg.get("cond_adv_full_epoch", 60),
        "effective_lambda_cond_adv_final": gan_logs.get("lambda_cond_eff", 0.0),
        "global_adv_mode": cond_cfg.get("global_adv_mode", "baseline_global_only"),
        "lambda_global_adv_multiplier": cond_cfg.get("lambda_global_adv_multiplier", 1.0),
        "cond_critic_loss_mean": gan_logs.get("cond_critic_loss_mean"),
        "cond_encoder_adv_loss_mean": gan_logs.get("cond_encoder_adv_loss_mean"),
        "cond_gp_mean": gan_logs.get("cond_gp_mean"),
        "cond_gp_skip_count": gan_logs.get("cond_gp_skip_count"),
        "cond_gp_pairing_mode": gan_logs.get("cond_gp_pairing_mode"),
        "num_cancer_types": gan_logs.get("num_cancer_types"),
    }
    if not payload["conditional_adv_enabled"]:
        payload.update(
            {
                "conditional_adv_enabled": False,
                "lambda_cond_adv": 0.0,
                "global_adv_mode": "baseline_global_only",
            }
        )
    return payload
