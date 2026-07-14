#!/usr/bin/env python3
"""Round 19 pipeline entry (data_smoke / train_fold)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round18_cv_metrics import metrics_to_jsonable
from tools.round18_dataset import subset_by_assignment
from tools.round18_response_head import Round18ResponseHead
from tools.round19_dataset import Round19ResponseDataset, round19_collate_fn
from tools.round19_drug_encoders import assert_no_hybrid, build_drug_encoder
from tools.round19_feature_builder import OMICS_ALIAS, resolve_omics_dim
from tools.round19_fusion_models import assert_compatible, build_predictor
from tools.round19_graph_features import BOND_FEATURE_DIM, bond_schema_hash
from tools.round19_train_loop import (
    build_round19_param_groups,
    evaluate_predictions_round19,
    make_default_loss,
    set_round18_seeds,
    train_one_epoch_round19,
)

OOM_EXIT_CODE = 42


def _load_settings(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _feature_dir(settings: dict, omics_id: str) -> str:
    root = settings.get(
        "round19_feature_out_root", "result/optimization_runs/round19_factorial/features"
    )
    return str(Path(root) / OMICS_ALIAS[omics_id])


def _metrics_jsonable(metrics: dict) -> dict:
    return metrics_to_jsonable(metrics)


def _assert_no_internal_overlap(val_df: pd.DataFrame, internal_test_path: Optional[str]) -> None:
    if not internal_test_path or not Path(internal_test_path).is_file():
        return
    internal = pd.read_csv(internal_test_path)
    if "ModelID" not in internal.columns:
        return
    overlap = set(val_df["ModelID"].astype(str)) & set(internal["ModelID"].astype(str))
    if overlap:
        raise AssertionError(f"Validation ModelIDs overlap internal-test: {sorted(list(overlap))[:5]}")


def _assert_val_row_ids(pred_df: pd.DataFrame, assignment_df: pd.DataFrame, fold_id: int) -> None:
    expected = set(
        assignment_df[
            (assignment_df["fold_id"].astype(int) == int(fold_id))
            & (assignment_df["split_role"].astype(str) == "val")
        ]["_row_id"].astype(int)
    )
    got = set(pred_df["_row_id"].astype(int))
    if got != expected:
        raise AssertionError(
            f"val _row_id mismatch: missing={len(expected - got)} extra={len(got - expected)}"
        )


def _build_encoder_fusion_head(settings: dict, *, drug_id: str, predictor_id: str, omics_dim: int, device):
    drug_cfg = settings["drug_reps"][drug_id]
    enc_type = drug_cfg["type"]
    assert_compatible(drug_id, predictor_id)
    assert_no_hybrid(enc_type, has_maccs=(enc_type == "maccs"), has_graph=(enc_type in {"gin", "gine"}))
    if enc_type == "maccs":
        encoder = build_drug_encoder("maccs", maccs_output_dim=int(drug_cfg["output_dim"]))
        drug_dim = int(drug_cfg["output_dim"])
        node_dim = 32
    else:
        encoder = build_drug_encoder(
            enc_type,
            node_hidden_dim=int(drug_cfg["node_hidden_dim"]),
            graph_output_dim=int(drug_cfg["graph_output_dim"]),
            edge_dim=int(drug_cfg.get("edge_dim", BOND_FEATURE_DIM)),
        )
        drug_dim = int(drug_cfg["graph_output_dim"])
        node_dim = int(drug_cfg["node_hidden_dim"])
    fusion = build_predictor(predictor_id, omics_dim=omics_dim, drug_dim=drug_dim, node_dim=node_dim)
    head = Round18ResponseHead(input_dim=fusion.output_dim)
    return encoder.to(device), fusion.to(device), head.to(device), enc_type, drug_cfg


def _make_loaders(
    *,
    response_path: str,
    feature_dir: str,
    drug_smiles_path: str,
    split_assignment: str,
    fold_id: int,
    encoder_type: str,
    with_bonds: bool,
    micro_batch_size: int,
    max_rows: Optional[int] = None,
):
    eligible = pd.read_csv(response_path)
    assignments = pd.read_csv(split_assignment)
    train_df = subset_by_assignment(eligible, assignments, fold_id=fold_id, split_role="train")
    val_df = subset_by_assignment(eligible, assignments, fold_id=fold_id, split_role="val")
    if max_rows is not None:
        train_df = train_df.head(int(max_rows)).copy()
        val_df = val_df.head(int(max_rows)).copy()
    train_ds = Round19ResponseDataset(
        train_df,
        feature_dir=feature_dir,
        drug_smiles_path=drug_smiles_path,
        encoder_type=encoder_type,
        with_bonds=with_bonds,
    )
    val_ds = Round19ResponseDataset(
        val_df,
        feature_dir=feature_dir,
        drug_smiles_path=drug_smiles_path,
        encoder_type=encoder_type,
        with_bonds=with_bonds,
        graph_cache=train_ds.graph_cache,
        maccs_by_drug=train_ds.maccs_by_drug,
    )
    train_loader = DataLoader(
        train_ds, batch_size=micro_batch_size, shuffle=True, num_workers=0, collate_fn=round19_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=micro_batch_size, shuffle=False, num_workers=0, collate_fn=round19_collate_fn
    )
    return train_loader, val_loader, train_ds, val_ds, assignments


def run_data_smoke(args: argparse.Namespace) -> dict:
    settings = _load_settings(args.settings)
    result_dir = Path(args.result_dir or args.outdir)
    result_dir.mkdir(parents=True, exist_ok=True)
    drug_cfg = settings["drug_reps"][args.drug_id]
    enc_type = drug_cfg["type"]
    with_bonds = bool(drug_cfg.get("edge_features"))
    train_loader, _, train_ds, _, _ = _make_loaders(
        response_path=args.response_path,
        feature_dir=_feature_dir(settings, args.omics_id),
        drug_smiles_path=settings["drug_smiles_path"],
        split_assignment=args.split_assignment,
        fold_id=int(args.fold_id),
        encoder_type=enc_type,
        with_bonds=with_bonds,
        micro_batch_size=int(args.micro_batch_size),
        max_rows=int(args.max_rows),
    )
    assert train_ds.omics_dim == resolve_omics_dim(args.omics_id)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    encoder, fusion, head, enc_type, _ = _build_encoder_fusion_head(
        settings, drug_id=args.drug_id, predictor_id=args.predictor_id, omics_dim=train_ds.omics_dim, device=device
    )
    loss_fn = make_default_loss(device)
    opt = torch.optim.AdamW(
        build_round19_param_groups(
            encoder,
            fusion,
            head,
            encoder_lr=1e-4,
            fusion_lr=3e-4,
            head_lr=3e-4,
            weight_decay=1e-4,
        )
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))
    n_batches = 0
    last_loss = None
    for batch in train_loader:
        if n_batches >= int(args.max_batches):
            break
        # one-step via train_one_epoch style
        class _One:
            def __init__(self, b):
                self.b = b

            def __iter__(self):
                yield self.b

            def __len__(self):
                return 1

        stats = train_one_epoch_round19(
            encoder=encoder,
            fusion=fusion,
            head=head,
            dataloader=_One(batch),
            optimizer=opt,
            scaler=scaler,
            loss_fn=loss_fn,
            device=device,
            encoder_type=enc_type,
            predictor_id=args.predictor_id,
            accumulation_steps=1,
            amp_enabled=(device.type == "cuda"),
        )
        last_loss = stats["loss"]
        n_batches += 1
    summary = {
        "mode": "data_smoke",
        "drug_id": args.drug_id,
        "predictor_id": args.predictor_id,
        "omics_id": args.omics_id,
        "n_rows": int(len(train_ds)),
        "n_batches": int(n_batches),
        "omics_dim": int(train_ds.omics_dim),
        "encoder_type": enc_type,
        "last_loss": last_loss,
        "device": str(device),
    }
    (result_dir / "data_smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def train_fold(args: argparse.Namespace) -> dict:
    settings = _load_settings(args.settings)
    set_round18_seeds(int(args.model_seed or settings.get("model_seed", 101)))
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    drug_cfg = settings["drug_reps"][args.drug_id]
    enc_type = drug_cfg["type"]
    with_bonds = bool(drug_cfg.get("edge_features"))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Forbid max_batches on formal/pilot full training unless explicitly allowed.
    if args.max_batches and not args.allow_max_batches:
        raise SystemExit("train_fold forbids --max-batches unless --allow-max-batches")

    try:
        train_loader, val_loader, train_ds, val_ds, assignments = _make_loaders(
            response_path=args.response_path,
            feature_dir=_feature_dir(settings, args.omics_id),
            drug_smiles_path=settings["drug_smiles_path"],
            split_assignment=args.split_assignment,
            fold_id=int(args.fold_id),
            encoder_type=enc_type,
            with_bonds=with_bonds,
            micro_batch_size=int(args.micro_batch_size),
            max_rows=None,
        )
        assert train_ds.omics_dim == resolve_omics_dim(args.omics_id)
        encoder, fusion, head, enc_type, drug_cfg = _build_encoder_fusion_head(
            settings,
            drug_id=args.drug_id,
            predictor_id=args.predictor_id,
            omics_dim=train_ds.omics_dim,
            device=device,
        )

        # Representation semantic asserts
        if args.drug_id == "D1":
            assert int(drug_cfg["node_hidden_dim"]) == 64 and int(drug_cfg["graph_output_dim"]) == 32
        if args.drug_id == "D2":
            assert int(drug_cfg["node_hidden_dim"]) == 64 and int(drug_cfg["graph_output_dim"]) == 64
        if args.predictor_id == "P2" and hasattr(fusion, "residual_mode"):
            raise AssertionError("P2 fusion must remain pure (no residual_mode attr)")

        opt_cfg = settings["optimizer"]
        optimizer = torch.optim.AdamW(
            build_round19_param_groups(
                encoder,
                fusion,
                head,
                encoder_lr=float(opt_cfg.get("encoder_lr", opt_cfg.get("gin_lr", 1e-4))),
                fusion_lr=float(opt_cfg["fusion_lr"]),
                head_lr=float(opt_cfg["head_lr"]),
                weight_decay=float(opt_cfg["weight_decay"]),
            ),
            weight_decay=float(opt_cfg["weight_decay"]),
        )
        loss_fn = make_default_loss(device)
        scaler = GradScaler(enabled=(device.type == "cuda" and not args.disable_amp))
        max_epochs = int(args.max_epochs)
        patience = int(args.early_stop_patience)
        start_epoch = int(args.early_stop_start_epoch)
        best_score = float("-inf")
        best_epoch = -1
        wait = 0
        history = []

        for epoch in range(max_epochs):
            train_stats = train_one_epoch_round19(
                encoder=encoder,
                fusion=fusion,
                head=head,
                dataloader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                loss_fn=loss_fn,
                device=device,
                encoder_type=enc_type,
                predictor_id=args.predictor_id,
                accumulation_steps=int(args.accumulation_steps),
                amp_enabled=(device.type == "cuda" and not args.disable_amp),
                grad_clip_max_norm=float(opt_cfg.get("grad_clip_max_norm", 1.0)),
            )
            val_out = evaluate_predictions_round19(
                encoder=encoder,
                fusion=fusion,
                head=head,
                dataloader=val_loader,
                device=device,
                encoder_type=enc_type,
                predictor_id=args.predictor_id,
                amp_enabled=(device.type == "cuda" and not args.disable_amp),
            )
            _assert_val_row_ids(val_out["predictions"], assignments, int(args.fold_id))
            _assert_no_internal_overlap(val_out["predictions"], args.internal_test_path)
            score = float(val_out["early_stop"]["score"])
            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "DrugMacro_AUC": val_out["metrics"].get("DrugMacro_AUC"),
                "Global_AUC": val_out["metrics"].get("Global_AUC"),
                "n_valid_auc_drugs": val_out["metrics"].get("n_valid_auc_drugs"),
                "early_stop_score": score,
                "fallback_used": val_out["early_stop"].get("fallback_used"),
                "fallback_reason": val_out["early_stop"].get("fallback_reason"),
            }
            history.append(row)
            pd.DataFrame(history).to_csv(result_dir / "train_history.csv", index=False)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                wait = 0
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "fusion": fusion.state_dict(),
                        "head": head.state_dict(),
                        "epoch": epoch,
                        "metrics": val_out["metrics"],
                        "drug_id": args.drug_id,
                        "predictor_id": args.predictor_id,
                        "omics_id": args.omics_id,
                        "encoder_type": enc_type,
                        "node_hidden_dim": drug_cfg.get("node_hidden_dim"),
                        "graph_output_dim": drug_cfg.get("graph_output_dim") or drug_cfg.get("output_dim"),
                        "bond_schema_hash": bond_schema_hash() if with_bonds else None,
                        "edge_feature_dim": int(drug_cfg.get("edge_dim", BOND_FEATURE_DIM)) if with_bonds else None,
                    },
                    result_dir / "checkpoint.pt",
                )
                val_out["predictions"].to_csv(result_dir / "val_predictions.csv", index=False)
                (result_dir / "val_metrics.json").write_text(
                    json.dumps(_metrics_jsonable(val_out["metrics"]), indent=2), encoding="utf-8"
                )
            else:
                if epoch >= start_epoch:
                    wait += 1
            if wait >= patience:
                break

        summary = {
            "ok": True,
            "mode": "train_fold",
            "pilot": bool(args.pilot),
            "drug_id": args.drug_id,
            "predictor_id": args.predictor_id,
            "omics_id": args.omics_id,
            "encoder_type": enc_type,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "n_epochs": len(history),
            "n_train": int(len(train_ds)),
            "n_val": int(len(val_ds)),
            "micro_batch_size": int(args.micro_batch_size),
            "accumulation_steps": int(args.accumulation_steps),
            "effective_batch": int(args.micro_batch_size) * int(args.accumulation_steps),
        }
        (result_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        peak = float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else None
        (result_dir / "runtime_resource_summary.json").write_text(
            json.dumps(
                {
                    "micro_batch_size": int(args.micro_batch_size),
                    "accumulation_steps": int(args.accumulation_steps),
                    "effective_batch": int(args.micro_batch_size) * int(args.accumulation_steps),
                    "peak_gpu_mem_mb": peak,
                    "device": str(device),
                    "encoder_type": enc_type,
                    "bond_schema_hash": bond_schema_hash() if with_bonds else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Pilot must NOT write formal stage19b job_status that dispatcher would skip.
        status_name = "pilot_job_status.json" if args.pilot else "job_status.json"
        (result_dir / status_name).write_text(
            json.dumps({"status": "done", "summary": summary}, indent=2), encoding="utf-8"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19 pipeline")
    parser.add_argument("--mode", default="data_smoke", choices=["data_smoke", "train_fold"])
    parser.add_argument("--settings", default="config/round19_factorial_settings.json")
    parser.add_argument("--outdir", default="result/optimization_runs/round19_factorial/data_smoke")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument(
        "--response-path",
        default="result/optimization_runs/round19_factorial/data/round19_eligible_response.csv",
    )
    parser.add_argument(
        "--split-assignment",
        default="result/optimization_runs/round19_factorial/splits/screening_3fold_assignments.csv",
    )
    parser.add_argument(
        "--internal-test-path",
        default="result/optimization_runs/round19_factorial/splits/internal_test_split.csv",
    )
    parser.add_argument("--drug-id", default="D0")
    parser.add_argument("--predictor-id", default="P0")
    parser.add_argument("--omics-id", default="O1")
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--allow-max-batches", action="store_true")
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-start-epoch", type=int, default=10)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.mode == "data_smoke":
        if not args.max_batches:
            args.max_batches = 2
        run_data_smoke(args)
    elif args.mode == "train_fold":
        if not args.result_dir:
            raise SystemExit("train_fold requires --result-dir")
        train_fold(args)
    else:
        raise SystemExit(f"Unsupported mode {args.mode}")


if __name__ == "__main__":
    main()
