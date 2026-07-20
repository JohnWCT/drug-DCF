#!/usr/bin/env python3
"""Round 20 CV training runner (Stage 20A / 20B single job).

This reuses the Round 19 training building blocks so that a Round 20 job is
byte-for-byte comparable to the resolved E3 contract, differing ONLY by:

  * omics feature store (C16 = 80-d, C32 = 96-d), and
  * (Stage 20B) predictor architecture (pooled E3 vs gated pooled fusion).

It never re-fits PCA, never mutates a feature store, and reads the E3 exact
contract from ``resolved_e3.json`` (no guessing by name).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round18_response_head import Round18ResponseHead  # noqa: E402
from tools.round19_dataset import Round19ResponseDataset, round19_collate_fn  # noqa: E402
from tools.round19_drug_encoders import assert_no_hybrid, build_drug_encoder  # noqa: E402
from tools.round19_graph_features import BOND_FEATURE_DIM, bond_schema_hash  # noqa: E402
from tools.round19_train_loop import (  # noqa: E402
    build_round19_param_groups,
    evaluate_predictions_round19,
    make_default_loss,
    set_round18_seeds,
    train_one_epoch_round19,
)
from tools.round18_cv_metrics import metrics_to_jsonable  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

OOM_EXIT_CODE = 42


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_status(result_dir: Path, payload: dict) -> None:
    _write_json(result_dir / "status.json", payload)


def _build_predictor(
    *,
    predictor_kind: str,
    settings: dict,
    e3: dict,
    omics_dim: int,
    device: torch.device,
):
    """Return (encoder, fusion, head, enc_type, drug_cfg).

    predictor_kind:
      * ``pooled_e3``            -> Round 19 P0 AdapterMLPFusion + ResponseHead
      * ``gated_pooled_fusion``  -> Round 20 GatedPooledFusion (self-contained head)
    """
    drug_cfg = settings["drug_reps"]["D0"]
    enc_type = drug_cfg["type"]
    assert enc_type == "gin", f"Round 20 D0 must be gin, got {enc_type}"
    assert_no_hybrid(enc_type, has_maccs=False, has_graph=True)
    encoder = build_drug_encoder(
        enc_type,
        node_hidden_dim=int(drug_cfg["node_hidden_dim"]),
        graph_output_dim=int(drug_cfg["graph_output_dim"]),
        edge_dim=int(drug_cfg.get("edge_dim", BOND_FEATURE_DIM)),
    )
    drug_dim = int(drug_cfg["graph_output_dim"])
    node_dim = int(drug_cfg["node_hidden_dim"])

    if predictor_kind == "pooled_e3":
        from tools.round19_fusion_models import assert_compatible, build_predictor

        assert_compatible("D0", "P0")
        fusion = build_predictor("P0", omics_dim=omics_dim, drug_dim=drug_dim, node_dim=node_dim)
        head = Round18ResponseHead(input_dim=fusion.output_dim)
        return encoder.to(device), fusion.to(device), head.to(device), enc_type, drug_cfg

    if predictor_kind == "gated_pooled_fusion":
        from tools.round20_gated_fusion import GatedPooledFusionPredictor

        fusion = GatedPooledFusionPredictor(
            omics_dim=omics_dim,
            drug_dim=drug_dim,
            hidden_dim=128,
            head_dim=64,
            dropout=0.20,
        )
        head = torch.nn.Identity()
        return encoder.to(device), fusion.to(device), head.to(device), enc_type, drug_cfg

    raise ValueError(f"Unknown predictor_kind={predictor_kind}")


def _make_loaders(*, response_path, feature_dir, drug_smiles_path, split_assignment, fold_id, micro_batch_size, num_workers=0):
    from tools.round18_dataset import subset_by_assignment

    eligible = pd.read_csv(response_path)
    assignments = pd.read_csv(split_assignment)
    train_df = subset_by_assignment(eligible, assignments, fold_id=fold_id, split_role="train")
    val_df = subset_by_assignment(eligible, assignments, fold_id=fold_id, split_role="val")
    train_ds = Round19ResponseDataset(
        train_df, feature_dir=feature_dir, drug_smiles_path=drug_smiles_path,
        encoder_type="gin", with_bonds=False, omics_id="O2",
    )
    val_ds = Round19ResponseDataset(
        val_df, feature_dir=feature_dir, drug_smiles_path=drug_smiles_path,
        encoder_type="gin", with_bonds=False, graph_cache=train_ds.graph_cache, omics_id="O2",
    )
    loader_kwargs = dict(num_workers=int(num_workers), collate_fn=round19_collate_fn)
    if int(num_workers) > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=micro_batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=micro_batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_ds, val_ds, assignments


def _predictor_param_count(fusion, head) -> int:
    return int(sum(p.numel() for p in list(fusion.parameters()) + list(head.parameters())))


def train_job(args: argparse.Namespace) -> dict:
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    settings = _load_json(args.settings)
    e3 = _load_json(args.e3_contract)
    e3 = e3.get("resolved_e3", e3)

    model_seed = int(args.model_seed or e3["training"]["model_seed"])
    set_round18_seeds(model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    _write_status(result_dir, {
        "job_id": args.job_id, "status": "RUNNING", "started_at": _utc_now(),
        "completed_at": None, "best_epoch": None, "error": None,
    })

    started = time.time()
    try:
        train_loader, val_loader, train_ds, val_ds, assignments = _make_loaders(
            response_path=args.response_path,
            feature_dir=args.feature_dir,
            drug_smiles_path=settings["drug_smiles_path"],
            split_assignment=args.split_assignment,
            fold_id=int(args.fold_id),
            micro_batch_size=int(args.micro_batch_size),
            num_workers=int(args.num_workers),
        )
        omics_dim = int(train_ds.omics_dim)
        if omics_dim != int(args.expected_omics_dim):
            raise AssertionError(
                f"omics dim mismatch: store={omics_dim} expected={args.expected_omics_dim}"
            )

        encoder, fusion, head, enc_type, drug_cfg = _build_predictor(
            predictor_kind=args.predictor_kind,
            settings=settings,
            e3=e3,
            omics_dim=omics_dim,
            device=device,
        )
        predictor_id = "P0" if args.predictor_kind == "pooled_e3" else "P0_GATED"

        opt_cfg = e3["optimizer"]
        param_groups = build_round19_param_groups(
            encoder, fusion, head,
            encoder_lr=float(opt_cfg["encoder_lr"]),
            fusion_lr=float(opt_cfg["fusion_lr"]),
            head_lr=float(opt_cfg["head_lr"]),
            weight_decay=float(opt_cfg["weight_decay"]),
        )
        # Identity head (gated fusion) has no parameters; drop empty groups.
        param_groups = [g for g in param_groups if list(g["params"])]
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=float(opt_cfg["weight_decay"]),
        )
        loss_fn = make_default_loss(device)
        amp_enabled = device.type == "cuda" and not args.disable_amp
        scaler = GradScaler(enabled=amp_enabled)

        max_epochs = int(args.max_epochs or e3["training"]["max_epochs"])
        patience = int(args.early_stop_patience or e3["training"]["early_stop_patience"])
        start_epoch = int(args.early_stop_start_epoch or e3["training"]["early_stop_start_epoch"])
        if args.smoke:
            max_epochs = min(max_epochs, int(args.smoke_epochs))
            patience = max_epochs

        best_score = float("-inf")
        best_epoch = -1
        best_metrics = None
        wait = 0
        history = []
        # gate_fusion uses predictor_id P0-style forward path already handled by fusion forward
        train_predictor_id = "P0"

        for epoch in range(max_epochs):
            train_stats = train_one_epoch_round19(
                encoder=encoder, fusion=fusion, head=head, dataloader=train_loader,
                optimizer=optimizer, scaler=scaler, loss_fn=loss_fn, device=device,
                encoder_type=enc_type, predictor_id=train_predictor_id,
                accumulation_steps=int(args.accumulation_steps),
                amp_enabled=amp_enabled,
                grad_clip_max_norm=float(opt_cfg.get("grad_clip_max_norm", 1.0)),
            )
            val_out = evaluate_predictions_round19(
                encoder=encoder, fusion=fusion, head=head, dataloader=val_loader,
                device=device, encoder_type=enc_type, predictor_id=train_predictor_id,
                amp_enabled=amp_enabled,
            )
            score = float(val_out["early_stop"]["score"])
            history.append({
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "DrugMacro_AUC": val_out["metrics"].get("DrugMacro_AUC"),
                "DrugMacro_AUPRC": val_out["metrics"].get("DrugMacro_AUPRC"),
                "Global_AUC": val_out["metrics"].get("Global_AUC"),
                "early_stop_score": score,
            })
            pd.DataFrame(history).to_csv(result_dir / "train_history.csv", index=False)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_metrics = dict(val_out["metrics"])
                wait = 0
                torch.save({
                    "model_state_dict": {
                        "encoder": encoder.state_dict(),
                        "fusion": fusion.state_dict(),
                        "head": head.state_dict(),
                    },
                    "epoch": epoch,
                    "metrics": metrics_to_jsonable(val_out["metrics"]),
                    "candidate_id": args.candidate_id,
                    "context_id": args.context_id,
                    "predictor_kind": args.predictor_kind,
                    "omics_dim": omics_dim,
                    "split_seed": int(args.split_seed),
                    "fold_id": int(args.fold_id),
                    "model_seed": model_seed,
                    "graph_output_dim": int(drug_cfg["graph_output_dim"]),
                    "node_hidden_dim": int(drug_cfg["node_hidden_dim"]),
                }, result_dir / "best_checkpoint.pt")
                val_out["predictions"].to_csv(result_dir / "validation_predictions.csv", index=False)
            elif epoch >= start_epoch:
                wait += 1
            if wait >= patience:
                break

        elapsed = time.time() - started
        n_val_drugs = int(val_ds.df["DRUG_NAME"].nunique())
        metrics_payload = {
            "job_id": args.job_id,
            "candidate_id": args.candidate_id,
            "context_id": args.context_id,
            "predictor_kind": args.predictor_kind,
            "split_seed": int(args.split_seed),
            "fold": int(args.fold_id),
            "metrics": {
                "DrugMacro_AUC": best_metrics.get("DrugMacro_AUC"),
                "DrugMacro_AUPRC": best_metrics.get("DrugMacro_AUPRC"),
                "Global_AUC": best_metrics.get("Global_AUC"),
                "Global_AUPRC": best_metrics.get("Global_AUPRC"),
            },
            "counts": {
                "rows": int(len(val_ds)),
                "drugs": n_val_drugs,
                "valid_auc_drugs": best_metrics.get("n_valid_auc_drugs"),
            },
            "best_epoch": best_epoch,
            "n_epochs": len(history),
            "training_time_seconds": round(elapsed, 2),
            "predictor_param_count": _predictor_param_count(fusion, head),
            "effective_batch": int(args.micro_batch_size) * int(args.accumulation_steps),
            "omics_dim": omics_dim,
            "status": "COMPLETE",
            "smoke": bool(args.smoke),
        }
        _write_json(result_dir / "metrics.json", metrics_payload)
        _write_json(result_dir / "job_config.json", vars(args))
        peak = float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else None
        _write_json(result_dir / "environment.json", {
            "device": str(device), "peak_gpu_mem_mb": peak,
            "torch": torch.__version__, "bond_schema_hash": bond_schema_hash(),
        })
        _write_status(result_dir, {
            "job_id": args.job_id, "status": "COMPLETE", "started_at": None,
            "completed_at": _utc_now(), "best_epoch": best_epoch, "error": None,
        })
        print(json.dumps({"ok": True, "job_id": args.job_id, "best_epoch": best_epoch,
                          "DrugMacro_AUC": best_metrics.get("DrugMacro_AUC"),
                          "elapsed_s": round(elapsed, 1)}, indent=2))
        return metrics_payload

    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or ("cuda" in msg and "memory" in msg):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _write_status(result_dir, {"job_id": args.job_id, "status": "FAILED",
                                       "completed_at": _utc_now(), "error": "OOM"})
            print(json.dumps({"oom": True, "detail": str(exc)}), flush=True)
            raise SystemExit(OOM_EXIT_CODE)
        _write_status(result_dir, {"job_id": args.job_id, "status": "FAILED",
                                   "completed_at": _utc_now(), "error": str(exc)})
        traceback.print_exc()
        raise
    except Exception as exc:  # noqa: BLE001
        _write_status(result_dir, {"job_id": args.job_id, "status": "FAILED",
                                   "completed_at": _utc_now(), "error": str(exc)})
        traceback.print_exc()
        raise


def main() -> None:
    p = argparse.ArgumentParser(description="Round 20 CV training runner")
    p.add_argument("--job-id", required=True)
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--context-id", required=True)
    p.add_argument("--predictor-kind", default="pooled_e3",
                   choices=["pooled_e3", "gated_pooled_fusion"])
    p.add_argument("--settings", default="config/round19_factorial_settings.json")
    p.add_argument("--e3-contract",
                   default="result/optimization_runs/round20_unseen_drug_closure/stage20_0/resolved_e3.json")
    p.add_argument("--response-path",
                   default="result/optimization_runs/round19_factorial/splits/development_rows.csv")
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--expected-omics-dim", type=int, required=True)
    p.add_argument("--split-assignment", required=True)
    p.add_argument("--split-seed", type=int, required=True)
    p.add_argument("--fold-id", type=int, required=True)
    p.add_argument("--model-seed", type=int, default=None)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--micro-batch-size", type=int, default=256)
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--early-stop-start-epoch", type=int, default=None)
    p.add_argument("--disable-amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-epochs", type=int, default=2)
    args = p.parse_args()
    train_job(args)


if __name__ == "__main__":
    main()
