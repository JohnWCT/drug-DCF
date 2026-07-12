#!/usr/bin/env python3
"""Round 18 CV finetune pipeline: smoke, data_smoke, train/eval/infer modes."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from drugmodels.ginconv import GINConvNet
from tools.round18_dataset import (
    Round18ResponseDataset,
    round18_graph_collate_fn,
    subset_by_assignment,
)
from tools.round18_fusion_models import build_fusion_and_head
from tools.round18_cv_metrics import metrics_to_jsonable
from tools.round18_oom_runner import (
    OOM_EXIT_CODE,
    detect_gpu_job_slots,
    probe_micro_batch,
    write_resource_metadata,
)
from tools.round18_train_loop import (
    build_param_groups,
    evaluate_predictions,
    make_default_loss,
    run_synthetic_smoke_train,
    set_round18_seeds,
    train_one_epoch,
)


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_transformer_cfg(config_id: str) -> Dict[str, Any]:
    screening = _load_json("config/params_round18_screening.json")
    # accept corrected historical name
    lookup = config_id
    if config_id == "P0_historical_hparams_corrected_mask":
        lookup = "P0_historical"
    for cfg in screening.get("pooled_transformer_configs", []):
        if cfg["config_id"] == lookup:
            return dict(cfg)
    for cfg in screening.get("cross_attention_configs", []):
        if cfg["config_id"] == lookup:
            return dict(cfg)
    return {}


def _settings(path: Optional[str] = None) -> dict:
    return _load_json(path or "config/round18_architecture_settings.json")


def run_smoke(args: argparse.Namespace) -> dict:
    families = [
        ("pooled_mlp", "pure"),
        ("pooled_transformer", "pure"),
        ("cross_attention", "pure"),
        ("cross_attention", "pooled_residual"),
    ]
    results = []
    for family, residual in families:
        out = run_synthetic_smoke_train(family, residual_mode=residual, steps=args.steps)
        results.append(out)
        print(json.dumps({"smoke": family, "residual": residual, "train_loss": out["train"]["loss"]}, indent=2))

    probe = probe_micro_batch(
        [512, 256, 128, 64, 32],
        target_effective_batch=1024,
        try_fn=None,
    )
    meta_dir = Path(args.outdir) / "pipeline_smoke_resources"
    write_resource_metadata(
        str(meta_dir),
        probe,
        extra={
            "gpu_slots": detect_gpu_job_slots(1),
            "smoke_architectures": [r["architecture_family"] for r in results],
        },
    )
    summary = {
        "ok": True,
        "n_architectures": len(results),
        "results": results,
        "resource_dir": str(meta_dir),
    }
    out_path = Path(args.outdir) / "pipeline_smoke_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "n_architectures": len(results)}, indent=2))
    return summary


def _build_models(
    *,
    architecture_family: str,
    omics_dim: int,
    residual_mode: str,
    transformer_config_id: str,
    gin_cfg: dict,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, dict]:
    gin = GINConvNet(
        input_dim=int(gin_cfg.get("input_dim", 78)),
        output_dim=int(gin_cfg.get("hidden_dim", 32)),
        dropout=float(gin_cfg.get("dropout", 0.1)),
        num_layers=int(gin_cfg.get("num_layers", 5)),
        jk_mode=str(gin_cfg.get("jk_mode", "last")),
        pool_type=str(gin_cfg.get("pool", "max")),
    ).to(device)

    family = architecture_family.lower()
    tcfg = _load_transformer_cfg(transformer_config_id) if transformer_config_id else {}
    meta = {}
    if family in {"pooled_transformer", "transformer"}:
        fusion, head = build_fusion_and_head(
            "pooled_transformer",
            omics_dim=omics_dim,
            graph_dim=int(gin_cfg.get("hidden_dim", 32)),
            transformer_cfg=tcfg or {"d_model": 128, "n_heads": 4, "num_layers": 1, "dim_feedforward": 128},
        )
        if hasattr(fusion, "metadata"):
            meta = fusion.metadata()
    elif family in {"cross_attention", "atom_cross_attention", "c0", "c1"}:
        fusion, head = build_fusion_and_head(
            "cross_attention",
            omics_dim=omics_dim,
            graph_dim=int(gin_cfg.get("hidden_dim", 32)),
            node_dim=int(gin_cfg.get("hidden_dim", 32)),
            residual_mode=residual_mode,
            cross_attn_cfg=tcfg or {"d_model": 128, "n_heads": 4, "num_layers": 1, "dim_feedforward": 256},
        )
    else:
        fusion, head = build_fusion_and_head(
            "pooled_mlp",
            omics_dim=omics_dim,
            graph_dim=int(gin_cfg.get("hidden_dim", 32)),
        )
    return gin, fusion.to(device), head.to(device), meta


def _make_loaders(
    *,
    response_path: str,
    feature_dir: str,
    drug_smiles_path: str,
    split_assignment: str,
    fold_id: int,
    micro_batch_size: int,
    include_val: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Round18ResponseDataset]:
    eligible = pd.read_csv(response_path)
    assigns = pd.read_csv(split_assignment)
    train_df = subset_by_assignment(eligible, assigns, fold_id=fold_id, split_role="train")
    train_ds = Round18ResponseDataset(
        train_df,
        feature_dir=feature_dir,
        drug_smiles_path=drug_smiles_path,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=round18_graph_collate_fn,
        num_workers=0,
    )
    val_loader = None
    if include_val:
        val_df = subset_by_assignment(eligible, assigns, fold_id=fold_id, split_role="val")
        val_ds = Round18ResponseDataset(
            val_df,
            feature_dir=feature_dir,
            drug_smiles_path=drug_smiles_path,
            graph_cache=train_ds.graph_cache,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=micro_batch_size,
            shuffle=False,
            collate_fn=round18_graph_collate_fn,
            num_workers=0,
        )
    return train_loader, val_loader, train_ds


def run_data_smoke(args: argparse.Namespace) -> dict:
    """Real GDSC + latent + SMILES: a few train batches + BN/param update check."""
    settings = _settings(args.settings)
    outdir = Path(args.outdir)
    response_path = args.response_path or str(outdir / "data" / "round18_eligible_response.csv")
    if not Path(response_path).is_file():
        raise FileNotFoundError(
            f"Missing eligible response: {response_path}. Run config builder --stage 18a first."
        )
    feature_dir = args.feature_dir or str(
        Path(settings["feature_root"]) / settings["feature_model_key"] / "own_plus_summary"
    )
    split_assignment = args.split_assignment or str(
        outdir / "splits" / "screening_3fold_assignments.csv"
    )
    fold_id = int(args.fold_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_round18_seeds(int(args.model_seed))

    train_loader, val_loader, train_ds = _make_loaders(
        response_path=response_path,
        feature_dir=feature_dir,
        drug_smiles_path=args.drug_smiles_path or settings["drug_smiles_path"],
        split_assignment=split_assignment,
        fold_id=fold_id,
        micro_batch_size=int(args.micro_batch_size),
    )
    gin, fusion, head, mask_meta = _build_models(
        architecture_family=args.architecture_family,
        omics_dim=train_ds.omics_dim,
        residual_mode=args.residual_mode,
        transformer_config_id=args.transformer_config_id,
        gin_cfg=settings["gin"],
        device=device,
    )

    # Snapshot BatchNorm running stats
    bn_before = []
    for m in gin.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            bn_before.append(m.running_mean.detach().cpu().clone())
            break

    opt_cfg = settings["optimizer"]
    gin_lr = float(args.global_lr) if args.global_lr is not None else float(opt_cfg["gin_lr"])
    fusion_lr = float(args.global_lr) if args.global_lr is not None else float(opt_cfg["fusion_lr"])
    head_lr = float(args.global_lr) if args.global_lr is not None else float(opt_cfg["head_lr"])
    optimizer = torch.optim.AdamW(
        build_param_groups(
            gin,
            fusion,
            head,
            gin_lr=gin_lr,
            fusion_lr=fusion_lr,
            head_lr=head_lr,
            weight_decay=float(opt_cfg["weight_decay"]),
        )
    )
    # verify no param overlap
    ids = (
        {id(p) for p in gin.parameters()}
        | {id(p) for p in fusion.parameters()}
        | {id(p) for p in head.parameters()}
    )
    assert len(ids) == (
        sum(1 for _ in gin.parameters()) + sum(1 for _ in fusion.parameters()) + sum(1 for _ in head.parameters())
    )

    loss_fn = make_default_loss(device)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    # limit batches
    max_batches = int(args.max_batches or 2)
    limited = []
    for i, batch in enumerate(train_loader):
        if i >= max_batches:
            break
        limited.append(batch)

    class _L:
        def __init__(self, batches):
            self.batches = batches

        def __iter__(self):
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)

    train_stats = train_one_epoch(
        gin=gin,
        fusion=fusion,
        head=head,
        dataloader=_L(limited),
        optimizer=optimizer,
        scaler=scaler,
        loss_fn=loss_fn,
        device=device,
        architecture_family=args.architecture_family,
        accumulation_steps=int(args.accumulation_steps),
        amp_enabled=(device.type == "cuda"),
        residual_mode=args.residual_mode,
    )

    bn_updated = False
    for m in gin.modules():
        if isinstance(m, torch.nn.BatchNorm1d) and bn_before:
            bn_updated = not torch.allclose(bn_before[0], m.running_mean.detach().cpu())
            break

    gin_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in gin.parameters() if p.requires_grad)
    # after optimizer step grads may be None; check param change instead
    # re-run one tiny step to confirm grads flow
    batch0 = limited[0]
    gin.train()
    fusion.train()
    head.train()
    optimizer.zero_grad(set_to_none=True)
    from tools.round18_train_loop import forward_round18_batch

    omics = batch0["omics"].to(device)
    labels = batch0["label"].to(device).float()
    drug_batch = batch0["drug_batch"].to(device)
    repr_vec = forward_round18_batch(
        gin=gin,
        fusion=fusion,
        architecture_family=args.architecture_family,
        omics=omics,
        drug_batch=drug_batch,
        residual_mode=args.residual_mode,
    )
    loss = loss_fn(head(repr_vec), labels, batch0["weight"].to(device))
    loss.backward()
    gin_has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0
        for p in gin.parameters()
        if p.requires_grad
    )

    summary = {
        "ok": True,
        "mode": "data_smoke",
        "n_train_rows": len(train_ds),
        "omics_dim": train_ds.omics_dim,
        "train_loss": train_stats["loss"],
        "bn_running_mean_updated": bool(bn_updated),
        "gin_has_grad": bool(gin_has_grad),
        "architecture_family": args.architecture_family,
        "fold_id": fold_id,
        "mask_meta": mask_meta,
        "device": str(device),
    }
    out_path = Path(args.result_dir or (outdir / "data_smoke")) / "data_smoke_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not gin_has_grad:
        raise RuntimeError("GIN parameters did not receive gradients in data_smoke")
    return summary


def train_fold(args: argparse.Namespace) -> dict:
    settings = _settings(args.settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_round18_seeds(int(args.model_seed))
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_loader, val_loader, train_ds = _make_loaders(
            response_path=args.response_path,
            feature_dir=args.feature_dir,
            drug_smiles_path=args.drug_smiles_path or settings["drug_smiles_path"],
            split_assignment=args.split_assignment,
            fold_id=int(args.fold_id),
            micro_batch_size=int(args.micro_batch_size),
        )
        gin, fusion, head, mask_meta = _build_models(
            architecture_family=args.architecture_family,
            omics_dim=train_ds.omics_dim,
            residual_mode=args.residual_mode,
            transformer_config_id=args.transformer_config_id,
            gin_cfg=settings["gin"],
            device=device,
        )
        opt_cfg = settings["optimizer"]
        if args.global_lr is not None:
            gin_lr = fusion_lr = head_lr = float(args.global_lr)
        else:
            gin_lr = float(opt_cfg["gin_lr"])
            fusion_lr = float(opt_cfg["fusion_lr"])
            head_lr = float(opt_cfg["head_lr"])
        optimizer = torch.optim.AdamW(
            build_param_groups(
                gin, fusion, head, gin_lr=gin_lr, fusion_lr=fusion_lr, head_lr=head_lr, weight_decay=float(opt_cfg["weight_decay"])
            )
        )
        loss_fn = make_default_loss(device)
        scaler = GradScaler(enabled=(device.type == "cuda" and not args.disable_amp))
        max_epochs = int(args.max_epochs or settings["screening_cv"]["max_epochs"])
        patience = int(args.early_stop_patience or settings["screening_cv"]["early_stop_patience"])
        start_epoch = int(args.early_stop_start_epoch or settings["screening_cv"]["early_stop_start_epoch"])
        best_score = float("-inf")
        best_epoch = -1
        wait = 0
        history = []

        for epoch in range(max_epochs):
            # optional short run for smoke jobs
            if args.max_batches:
                batches = []
                for i, b in enumerate(train_loader):
                    if i >= int(args.max_batches):
                        break
                    batches.append(b)

                class _L:
                    def __init__(self, xs):
                        self.xs = xs

                    def __iter__(self):
                        return iter(self.xs)

                    def __len__(self):
                        return len(self.xs)

                loader = _L(batches)
            else:
                loader = train_loader

            train_stats = train_one_epoch(
                gin=gin,
                fusion=fusion,
                head=head,
                dataloader=loader,
                optimizer=optimizer,
                scaler=scaler,
                loss_fn=loss_fn,
                device=device,
                architecture_family=args.architecture_family,
                accumulation_steps=int(args.accumulation_steps),
                amp_enabled=(device.type == "cuda" and not args.disable_amp),
                residual_mode=args.residual_mode,
            )
            val_out = evaluate_predictions(
                gin=gin,
                fusion=fusion,
                head=head,
                dataloader=val_loader,
                device=device,
                architecture_family=args.architecture_family,
                residual_mode=args.residual_mode,
                amp_enabled=(device.type == "cuda" and not args.disable_amp),
            )
            score = float(val_out["early_stop"]["score"])
            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "DrugMacro_AUC": val_out["metrics"]["DrugMacro_AUC"],
                "Global_AUC": val_out["metrics"]["Global_AUC"],
                "n_valid_auc_drugs": val_out["metrics"]["n_valid_auc_drugs"],
                "early_stop_score": score,
                "fallback_used": val_out["early_stop"].get("fallback_used"),
            }
            history.append(row)
            pd.DataFrame(history).to_csv(result_dir / "train_history.csv", index=False)
            improved = score > best_score
            if improved:
                best_score = score
                best_epoch = epoch
                wait = 0
                torch.save(
                    {
                        "gin": gin.state_dict(),
                        "fusion": fusion.state_dict(),
                        "head": head.state_dict(),
                        "epoch": epoch,
                        "metrics": val_out["metrics"],
                        "mask_meta": mask_meta,
                    },
                    result_dir / "checkpoint.pt",
                )
                val_out["predictions"].to_csv(result_dir / "val_predictions.csv", index=False)
                Path(result_dir / "val_metrics.json").write_text(
                    json.dumps(metrics_to_jsonable(val_out["metrics"]), indent=2), encoding="utf-8"
                )
            else:
                if epoch >= start_epoch:
                    wait += 1
            if wait >= patience:
                break
            if args.max_epochs_cap and epoch + 1 >= int(args.max_epochs_cap):
                break

        summary = {
            "ok": True,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "n_epochs": len(history),
            "history_tail": history[-5:],
            "mask_meta": mask_meta,
            "micro_batch_size": int(args.micro_batch_size),
            "accumulation_steps": int(args.accumulation_steps),
        }
        Path(result_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.DataFrame(history).to_csv(result_dir / "train_history.csv", index=False)
        peak = None
        if device.type == "cuda":
            peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        Path(result_dir / "runtime_resource_summary.json").write_text(
            json.dumps(
                {
                    "micro_batch_size": int(args.micro_batch_size),
                    "accumulation_steps": int(args.accumulation_steps),
                    "peak_gpu_mem_mb": peak,
                    "device": str(device),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2))
        return summary
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(json.dumps({"oom": True, "detail": str(exc)}), flush=True)
            raise SystemExit(OOM_EXIT_CODE)
        traceback.print_exc()
        raise


def evaluate_fold(args: argparse.Namespace) -> dict:
    settings = _settings(args.settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = Path(args.result_dir)
    ckpt_path = result_dir / "checkpoint.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    _, val_loader, train_ds = _make_loaders(
        response_path=args.response_path,
        feature_dir=args.feature_dir,
        drug_smiles_path=args.drug_smiles_path or settings["drug_smiles_path"],
        split_assignment=args.split_assignment,
        fold_id=int(args.fold_id),
        micro_batch_size=int(args.micro_batch_size),
    )
    gin, fusion, head, _ = _build_models(
        architecture_family=args.architecture_family,
        omics_dim=train_ds.omics_dim,
        residual_mode=args.residual_mode,
        transformer_config_id=args.transformer_config_id,
        gin_cfg=settings["gin"],
        device=device,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    gin.load_state_dict(ckpt["gin"])
    fusion.load_state_dict(ckpt["fusion"])
    head.load_state_dict(ckpt["head"])
    out = evaluate_predictions(
        gin=gin,
        fusion=fusion,
        head=head,
        dataloader=val_loader,
        device=device,
        architecture_family=args.architecture_family,
        residual_mode=args.residual_mode,
    )
    out["predictions"].to_csv(result_dir / "evaluate_fold_predictions.csv", index=False)
    Path(result_dir / "evaluate_fold_metrics.json").write_text(
        json.dumps(metrics_to_jsonable(out["metrics"]), indent=2), encoding="utf-8"
    )
    print(json.dumps({"ok": True, "metrics": metrics_to_jsonable(out["metrics"])}, indent=2))
    return out["metrics"]


def infer_split(args: argparse.Namespace, *, split_role: str, out_name: str) -> dict:
    """Infer on internal_test rows listed in assignment or dedicated split CSV."""
    settings = _settings(args.settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = Path(args.result_dir)
    ckpt = torch.load(result_dir / "checkpoint.pt", map_location=device)

    eligible = pd.read_csv(args.response_path)
    if args.split_assignment and Path(args.split_assignment).is_file():
        # Prefer dedicated internal_test_split.csv when role is internal_test
        if split_role == "internal_test":
            it_path = Path(args.outdir) / "splits" / "internal_test_split.csv"
            if it_path.is_file():
                subset = pd.read_csv(it_path)
            else:
                assigns = pd.read_csv(args.split_assignment)
                subset = subset_by_assignment(
                    eligible, assigns, fold_id=int(args.fold_id), split_role="val"
                )
        else:
            assigns = pd.read_csv(args.split_assignment)
            subset = subset_by_assignment(
                eligible, assigns, fold_id=int(args.fold_id), split_role=split_role
            )
    else:
        subset = eligible

    ds = Round18ResponseDataset(
        subset,
        feature_dir=args.feature_dir,
        drug_smiles_path=args.drug_smiles_path or settings["drug_smiles_path"],
    )
    loader = DataLoader(
        ds,
        batch_size=int(args.micro_batch_size),
        shuffle=False,
        collate_fn=round18_graph_collate_fn,
    )
    gin, fusion, head, _ = _build_models(
        architecture_family=args.architecture_family,
        omics_dim=ds.omics_dim,
        residual_mode=args.residual_mode,
        transformer_config_id=args.transformer_config_id,
        gin_cfg=settings["gin"],
        device=device,
    )
    gin.load_state_dict(ckpt["gin"])
    fusion.load_state_dict(ckpt["fusion"])
    head.load_state_dict(ckpt["head"])
    out = evaluate_predictions(
        gin=gin,
        fusion=fusion,
        head=head,
        dataloader=loader,
        device=device,
        architecture_family=args.architecture_family,
        residual_mode=args.residual_mode,
    )
    pred_path = result_dir / out_name
    out["predictions"].to_csv(pred_path, index=False)
    print(json.dumps({"ok": True, "wrote": str(pred_path), "n": len(out["predictions"])}, indent=2))
    return {"wrote": str(pred_path), "metrics": out["metrics"]}


def infer_tcga(args: argparse.Namespace) -> dict:
    raise NotImplementedError(
        "infer_tcga requires locked selection + TCGA feature alignment; "
        "Stage 18E wiring comes after 18D lock file exists."
    )


def export_attention(args: argparse.Namespace) -> dict:
    raise NotImplementedError(
        "export_attention is Stage 18F; requires locked cross-attention checkpoint."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 18 CV pipeline")
    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "data_smoke",
            "train_fold",
            "evaluate_fold",
            "infer_internal_test",
            "infer_tcga",
            "export_attention",
        ],
        default="smoke",
    )
    parser.add_argument("--outdir", default="result/optimization_runs/round18_architecture")
    parser.add_argument("--settings", default="config/round18_architecture_settings.json")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--manifest-row", default=None)
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--architecture-family", default="pooled_mlp")
    parser.add_argument("--omics-mode", default="own_plus_summary")
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--split-assignment", default=None)
    parser.add_argument("--response-path", default=None)
    parser.add_argument("--drug-smiles-path", default=None)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--transformer-config-id", default="")
    parser.add_argument("--residual-mode", default="pure")
    parser.add_argument("--global-lr", type=float, default=None)
    parser.add_argument("--model-seed", type=int, default=101)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-epochs-cap", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-start-epoch", type=int, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()

    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "data_smoke":
        run_data_smoke(args)
    elif args.mode == "train_fold":
        if not args.response_path or not args.feature_dir or not args.split_assignment or not args.result_dir:
            raise SystemExit(
                "train_fold requires --response-path --feature-dir --split-assignment --result-dir"
            )
        train_fold(args)
    elif args.mode == "evaluate_fold":
        evaluate_fold(args)
    elif args.mode == "infer_internal_test":
        infer_split(args, split_role="internal_test", out_name="internal_test_predictions.csv")
    elif args.mode == "infer_tcga":
        infer_tcga(args)
    elif args.mode == "export_attention":
        export_attention(args)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
