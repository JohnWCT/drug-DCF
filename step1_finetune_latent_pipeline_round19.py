#!/usr/bin/env python3
"""Round 19 training and native post-hoc inference pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round18_cv_metrics import metrics_to_jsonable
from tools.round18_dataset import subset_by_assignment
from tools.round18_eligible_data import load_smiles_lookup
from tools.round18_response_head import Round18ResponseHead
from tools.finetune_tcga_eval import load_tcga_response_csv
from tools.round19_context_controls import build_partition_permutation, validate_context_shuffle
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
TCGA_TARGETS = {
    "gdsc_intersect13": "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv",
    "tcga_only3": "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv",
    "dapl": "data/TCGA/TCGA_drug_response_from_DAPL.csv",
    "aacdr_tcga_only": "data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv",
    "aacdr_gdsc_intersect": "data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv",
}
SOURCE_SUMMARY_NAMES = (
    "proto_own_source_cosine_dist",
    "proto_own_source_l2_dist",
    "proto_source_anchor_initialized",
    "proto_source_min_dist",
    "proto_source_top1_margin",
    "proto_source_mean_dist",
    "proto_source_std_dist",
)


def _load_settings(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _load_feature_names(path: Path) -> list:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"Expected feature-name list: {path}")
    return [str(item) for item in value]


def _load_latent_pickle(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Expected non-empty latent dict: {path}")
    return {
        str(key): np.asarray(vector, dtype=np.float32).reshape(-1)
        for key, vector in value.items()
    }


def _validate_latent_dict(latent: Dict[str, np.ndarray], expected_dim: int, source: Path) -> None:
    bad = [
        key
        for key, vector in latent.items()
        if vector.shape != (expected_dim,) or not np.isfinite(vector).all()
    ]
    if bad:
        raise ValueError(
            f"Invalid {expected_dim}-d finite TCGA vectors in {source}: {bad[:5]}"
        )


def _infer_training_split_seed(args: argparse.Namespace, settings: dict) -> int:
    if args.split_seed is not None:
        return int(args.split_seed)
    assignments = pd.read_csv(args.split_assignment, nrows=1000)
    if "split_seed" in assignments:
        values = pd.to_numeric(assignments["split_seed"], errors="raise").dropna().astype(int).unique()
        if len(values) == 1:
            return int(values[0])
    match = re.search(r"seed[_-]?(\d+)", Path(args.split_assignment).name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return int(settings["screening_split_seed"])


def _feature_dir(settings: dict, omics_id: str) -> str:
    root = settings.get(
        "round19_feature_out_root", "result/optimization_runs/round19_factorial/features"
    )
    return str(Path(root) / OMICS_ALIAS[omics_id])


def preflight_tcga_features(settings: dict, omics_id: str) -> Tuple[Dict[str, np.ndarray], dict]:
    """Resolve TCGA features without fitting or reading TCGA response labels."""
    oid = str(omics_id).upper()
    feature_root = Path(
        settings.get(
            "round19_feature_out_root",
            "result/optimization_runs/round19_factorial/features",
        )
    )
    feature_dir = feature_root / OMICS_ALIAS[oid]
    report = {
        "ok": False,
        "omics_id": oid,
        "feature_dir": str(feature_dir),
        "fit_on_tcga_labels": False,
        "checks": [],
    }

    def require(path: Path, description: str) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"{description}: {path}")
        report["checks"].append({"check": description, "path": str(path), "ok": True})
        return path

    if oid in {"O0", "O1", "O3"}:
        tcga_path = require(feature_dir / "tcga_latent_proto.pkl", "existing TCGA feature artifact")
        feature_meta_path = require(
            feature_dir / "feature_metadata.json", "TCGA transform provenance"
        )
        feature_meta = _load_json(feature_meta_path)
        if int(feature_meta.get("n_tcga_samples", 0)) <= 0:
            raise AssertionError(
                f"{oid} metadata does not prove TCGA was transformed"
            )
        if oid == "O3":
            projection_path = require(
                feature_dir / "projection_model.pkl", "O3 development projection"
            )
            projection_meta_path = require(
                feature_dir / "projection_metadata.json", "O3 projection provenance"
            )
            if _load_json(projection_meta_path).get("fit_domain") != "source_only":
                raise AssertionError("O3 projection must declare fit_domain=source_only")
            report["projection_sha256"] = _sha256_file(projection_path)
        latent = _load_latent_pickle(tcga_path)
        _validate_latent_dict(latent, resolve_omics_dim(oid), tcga_path)
        report["derivation"] = "existing_development_fitted_artifact"
        report["source_artifacts"] = [str(tcga_path), str(feature_meta_path)]
    elif oid == "O2":
        o3_dir = feature_root / OMICS_ALIAS["O3"]
        tcga_path = require(o3_dir / "tcga_latent_proto.pkl", "O3 TCGA source artifact")
        o2_names_path = require(feature_dir / "feature_names.json", "O2 feature names")
        o3_names_path = require(o3_dir / "feature_names.json", "O3 feature names")
        o2_projection = require(feature_dir / "projection_model.pkl", "O2 development projection")
        o3_projection = require(o3_dir / "projection_model.pkl", "O3 development projection")
        projection_meta_path = require(
            o3_dir / "projection_metadata.json", "O3 projection provenance"
        )
        projection_meta = _load_json(projection_meta_path)
        if projection_meta.get("fit_domain") != "source_only":
            raise AssertionError("O3 projection must declare fit_domain=source_only")
        if _sha256_file(o2_projection) != _sha256_file(o3_projection):
            raise AssertionError("O2/O3 development projection hashes differ")
        o2_names, o3_names = _load_feature_names(o2_names_path), _load_feature_names(o3_names_path)
        if len(o2_names) != 80 or o3_names[:80] != o2_names:
            raise AssertionError("O2 names must exactly equal the first 80 O3 feature names")
        o3_latent = _load_latent_pickle(tcga_path)
        _validate_latent_dict(o3_latent, 91, tcga_path)
        latent = {key: value[:80].copy() for key, value in o3_latent.items()}
        report["checks"].append(
            {
                "check": "reuse identical source-only projection",
                "ok": True,
                "sha256": _sha256_file(o3_projection),
            }
        )
        report["derivation"] = "slice_O3_z_plus_context16_using_development_projection"
        report["source_artifacts"] = [
            str(tcga_path),
            str(o3_projection),
            str(projection_meta_path),
        ]
    elif oid == "O4":
        o1_dir = feature_root / OMICS_ALIAS["O1"]
        tcga_path = require(o1_dir / "tcga_latent_proto.pkl", "O1 TCGA source artifact")
        o1_ccle_path = require(o1_dir / "ccle_latent_proto.pkl", "O1 development feature artifact")
        o4_ccle_path = require(feature_dir / "ccle_latent_proto.pkl", "O4 development feature artifact")
        o1_names_path = require(o1_dir / "feature_names.json", "O1 feature names")
        o4_names_path = require(feature_dir / "feature_names.json", "O4 feature names")
        o1_meta_path = require(o1_dir / "feature_metadata.json", "O1 transform provenance")
        o1_meta = _load_json(o1_meta_path)
        if int(o1_meta.get("n_tcga_samples", 0)) <= 0:
            raise AssertionError("O1 metadata does not prove TCGA was transformed")
        o1_names = _load_feature_names(o1_names_path)
        summary_names = o1_names[64:] if len(o1_names) == 75 else o1_names
        if len(summary_names) != 11:
            raise AssertionError(f"Expected 11 O1 summary names, got {len(summary_names)}")
        indexes = []
        for name in SOURCE_SUMMARY_NAMES:
            if name not in summary_names:
                raise KeyError(f"O1 source feature missing: {name}")
            indexes.append(64 + summary_names.index(name))
        expected_o4_names = [f"z_dim{i:03d}" for i in range(64)] + list(SOURCE_SUMMARY_NAMES)
        if _load_feature_names(o4_names_path) != expected_o4_names:
            raise AssertionError("O4 feature names do not match the source-only contract")

        o1_ccle = _load_latent_pickle(o1_ccle_path)
        o4_ccle = _load_latent_pickle(o4_ccle_path)
        if set(o1_ccle) != set(o4_ccle):
            raise AssertionError("O1/O4 development ModelID sets differ")

        def derive(vector: np.ndarray) -> np.ndarray:
            if vector.shape != (75,):
                raise ValueError(f"Expected O1 75-d vector, got {vector.shape}")
            return np.concatenate([vector[:64], vector[indexes]]).astype(np.float32)

        mismatches = [
            key
            for key in o1_ccle
            if not np.array_equal(derive(o1_ccle[key]), o4_ccle[key])
        ]
        if mismatches:
            raise AssertionError(
                f"O4 development features are not exact O1 source-only selections: {mismatches[:5]}"
            )
        o1_tcga = _load_latent_pickle(tcga_path)
        latent = {key: derive(value) for key, value in o1_tcga.items()}
        report["checks"].append(
            {"check": "O4 exactly reproduced from development O1 columns", "ok": True}
        )
        report["derivation"] = "select_O1_development_transformed_source_only_features"
        report["source_artifacts"] = [str(tcga_path), str(o1_meta_path)]
    else:
        raise ValueError(f"Unsupported omics_id={omics_id}")

    _validate_latent_dict(latent, resolve_omics_dim(oid), feature_dir)
    report["ok"] = True
    report["n_tcga_latents"] = len(latent)
    report["omics_dim"] = resolve_omics_dim(oid)
    return latent, report


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
    omics_id: Optional[str] = None,
    train_context_permutation: Optional[Dict[str, str]] = None,
    val_context_permutation: Optional[Dict[str, str]] = None,
):
    eligible = pd.read_csv(response_path)
    assignments = pd.read_csv(split_assignment)
    train_df = subset_by_assignment(eligible, assignments, fold_id=fold_id, split_role="train")
    val_df = subset_by_assignment(eligible, assignments, fold_id=fold_id, split_role="val")
    if max_rows is not None:
        train_df = train_df.head(int(max_rows)).copy()
        val_df = val_df.head(int(max_rows)).copy()
    # For drug-/scaffold-heldout, train and val drug sets are disjoint.
    # Preload MACCS for the union so val does not inherit a train-only map.
    maccs_by_drug = None
    if str(encoder_type).lower() == "maccs":
        from tools.round19_drug_features import load_maccs_by_drug_name

        drug_col = "DRUG_NAME" if "DRUG_NAME" in train_df.columns else "mapped_name"
        union_drugs = sorted(
            set(train_df[drug_col].astype(str)).union(set(val_df[drug_col].astype(str)))
        )
        maccs_by_drug = load_maccs_by_drug_name(drug_smiles_path, drug_names=union_drugs)

    train_ds = Round19ResponseDataset(
        train_df,
        feature_dir=feature_dir,
        drug_smiles_path=drug_smiles_path,
        encoder_type=encoder_type,
        with_bonds=with_bonds,
        maccs_by_drug=maccs_by_drug,
        context_permutation=train_context_permutation,
        omics_id=omics_id,
    )
    val_ds = Round19ResponseDataset(
        val_df,
        feature_dir=feature_dir,
        drug_smiles_path=drug_smiles_path,
        encoder_type=encoder_type,
        with_bonds=with_bonds,
        graph_cache=train_ds.graph_cache,
        maccs_by_drug=train_ds.maccs_by_drug,
        context_permutation=val_context_permutation,
        omics_id=omics_id,
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


def _build_context_perms(
    assignments: pd.DataFrame,
    fold_id: int,
    train_seed: int,
    val_seed: int,
) -> Tuple[Dict[str, str], Dict[str, str], dict]:
    fold = int(fold_id)
    train_mids = sorted(
        set(
            assignments[
                (assignments["fold_id"].astype(int) == fold)
                & (assignments["split_role"].astype(str) == "train")
            ]["ModelID"].astype(str)
        )
    )
    val_mids = sorted(
        set(
            assignments[
                (assignments["fold_id"].astype(int) == fold)
                & (assignments["split_role"].astype(str) == "val")
            ]["ModelID"].astype(str)
        )
    )
    train_perm = build_partition_permutation(train_mids, int(train_seed))
    val_perm = build_partition_permutation(val_mids, int(val_seed))
    validate_context_shuffle(train_perm, train_mids)
    validate_context_shuffle(val_perm, val_mids)
    meta = {
        "context_control": "shuffled",
        "shuffle_unit": "ModelID",
        "shuffle_scope": "within_partition",
        "train_shuffle_seed": int(train_seed),
        "validation_shuffle_seed": int(val_seed),
        "derangement": True,
        "n_train_modelids": len(train_mids),
        "n_val_modelids": len(val_mids),
    }
    return train_perm, val_perm, meta


def train_fold(args: argparse.Namespace) -> dict:
    settings = _load_settings(args.settings)
    split_seed = _infer_training_split_seed(args, settings)
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
        assignments = pd.read_csv(args.split_assignment)
        train_context_perm = None
        val_context_perm = None
        shuffle_meta = None
        control_type = str(getattr(args, "control_type", "none") or "none")
        feature_dir = _feature_dir(settings, args.omics_id)
        if control_type == "context_shuffle":
            if args.omics_id not in {"O2", "O3"}:
                raise SystemExit("context_shuffle requires omics_id O2 or O3")
            if args.train_shuffle_seed is None or args.val_shuffle_seed is None:
                raise SystemExit("context_shuffle requires --train-shuffle-seed and --val-shuffle-seed")
            train_context_perm, val_context_perm, shuffle_meta = _build_context_perms(
                assignments,
                int(args.fold_id),
                int(args.train_shuffle_seed),
                int(args.val_shuffle_seed),
            )

        train_loader, val_loader, train_ds, val_ds, assignments = _make_loaders(
            response_path=args.response_path,
            feature_dir=feature_dir,
            drug_smiles_path=settings["drug_smiles_path"],
            split_assignment=args.split_assignment,
            fold_id=int(args.fold_id),
            encoder_type=enc_type,
            with_bonds=with_bonds,
            micro_batch_size=int(args.micro_batch_size),
            max_rows=None,
            omics_id=args.omics_id,
            train_context_permutation=train_context_perm,
            val_context_permutation=val_context_perm,
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
                        "split_seed": split_seed,
                        "fold_id": int(args.fold_id),
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
            "split_seed": split_seed,
            "fold_id": int(args.fold_id),
            "control_type": control_type,
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
        if shuffle_meta is not None:
            (result_dir / "context_shuffle_metadata.json").write_text(
                json.dumps(shuffle_meta, indent=2), encoding="utf-8"
            )
            summary["context_shuffle_metadata"] = shuffle_meta
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


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    path = Path(args.checkpoint_path) if args.checkpoint_path else Path(args.result_dir) / "checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Round19 checkpoint: {path}")
    return path


def _strict_load_inference_models(
    args: argparse.Namespace,
    settings: dict,
    *,
    omics_dim: int,
    device: torch.device,
):
    if args.split_seed is None:
        raise ValueError("post-hoc inference requires --split-seed")
    checkpoint_path = _resolve_checkpoint(args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Round19 checkpoint must be a dict: {checkpoint_path}")
    required = {
        "encoder",
        "fusion",
        "head",
        "drug_id",
        "predictor_id",
        "omics_id",
    }
    missing = required - set(checkpoint)
    if missing:
        raise KeyError(
            f"Round19 checkpoint missing required keys {sorted(missing)}: {checkpoint_path}"
        )
    expected = {
        "drug_id": str(args.drug_id),
        "predictor_id": str(args.predictor_id),
        "omics_id": str(args.omics_id),
        "split_seed": int(args.split_seed),
        "fold_id": int(args.fold_id),
    }
    identity = dict(checkpoint)
    if not {"split_seed", "fold_id"} <= set(checkpoint):
        inventory_path = Path(args.checkpoint_inventory)
        if not inventory_path.is_file():
            raise KeyError(
                "Legacy Round19 checkpoint lacks split_seed/fold_id and the "
                f"authoritative checkpoint inventory is missing: {inventory_path}"
            )
        inventory = pd.read_csv(inventory_path)
        inventory_required = {
            "candidate_id",
            "drug_id",
            "predictor_id",
            "omics_id",
            "split_seed",
            "fold_id",
        }
        inventory_missing = inventory_required - set(inventory)
        if inventory_missing:
            raise KeyError(
                f"checkpoint inventory missing columns: {sorted(inventory_missing)}"
            )
        requested_path = checkpoint_path.resolve()

        def inventory_checkpoint_path(row: pd.Series) -> Path:
            if "checkpoint_path" in row.index and pd.notna(row["checkpoint_path"]):
                return Path(str(row["checkpoint_path"])).resolve()
            if "result_dir" not in row.index or pd.isna(row["result_dir"]):
                raise KeyError("checkpoint inventory needs checkpoint_path or result_dir")
            return (Path(str(row["result_dir"])) / "checkpoint.pt").resolve()

        matches = inventory[
            inventory.apply(
                lambda row: inventory_checkpoint_path(row) == requested_path, axis=1
            )
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"Checkpoint must match exactly one authoritative inventory row; got {len(matches)}"
            )
        inventory_row = matches.iloc[0]
        source_candidate_id = args.source_candidate_id or args.candidate_id
        if not source_candidate_id:
            raise ValueError(
                "Legacy checkpoint attestation requires --candidate-id or --source-candidate-id"
            )
        if str(inventory_row["candidate_id"]) != str(source_candidate_id):
            raise AssertionError(
                "checkpoint inventory candidate mismatch: "
                f"inventory={inventory_row['candidate_id']} requested={source_candidate_id}"
            )
        for key in expected:
            identity[key] = inventory_row[key]
    mismatches = {}
    for key, wanted in expected.items():
        got = identity[key]
        got = int(got) if key in {"split_seed", "fold_id"} else str(got)
        if got != wanted:
            mismatches[key] = {"checkpoint": got, "requested": wanted}
    if mismatches:
        raise AssertionError(f"Round19 checkpoint identity mismatch: {mismatches}")

    encoder, fusion, head, encoder_type, drug_cfg = _build_encoder_fusion_head(
        settings,
        drug_id=args.drug_id,
        predictor_id=args.predictor_id,
        omics_dim=omics_dim,
        device=device,
    )
    if checkpoint.get("encoder_type") not in (None, encoder_type):
        raise AssertionError(
            f"encoder_type mismatch: checkpoint={checkpoint.get('encoder_type')} requested={encoder_type}"
        )
    with_bonds = bool(drug_cfg.get("edge_features"))
    if with_bonds:
        if checkpoint.get("bond_schema_hash") != bond_schema_hash():
            raise AssertionError("GINE checkpoint bond_schema_hash mismatch")
        if int(checkpoint.get("edge_feature_dim", -1)) != int(
            drug_cfg.get("edge_dim", BOND_FEATURE_DIM)
        ):
            raise AssertionError("GINE checkpoint edge_feature_dim mismatch")
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    fusion.load_state_dict(checkpoint["fusion"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    return encoder, fusion, head, encoder_type, drug_cfg, checkpoint_path


def _member_id(split_seed: int, fold_id: int) -> str:
    return f"seed{int(split_seed)}_fold{int(fold_id)}"


def _annotate_posthoc_predictions(
    predictions: pd.DataFrame,
    source_frame: pd.DataFrame,
    args: argparse.Namespace,
    *,
    target_key: str,
    checkpoint_path: Path,
) -> pd.DataFrame:
    if not args.candidate_id:
        raise ValueError("post-hoc inference requires --candidate-id")
    source_candidate_id = args.source_candidate_id or args.candidate_id
    metadata = source_frame.copy()
    if "_row_id" not in metadata:
        raise KeyError("inference source frame missing _row_id")
    if "eval_row_id" not in metadata:
        metadata["eval_row_id"] = [
            f"{target_key}|{int(row_id)}" for row_id in metadata["_row_id"]
        ]
    keep = ["_row_id", "eval_row_id"]
    for column in ("Patient_id",):
        if column in metadata:
            keep.append(column)
    merged = predictions.merge(
        metadata[keep],
        on="_row_id",
        how="left",
        validate="one_to_one",
    )
    if merged["eval_row_id"].isna().any():
        raise AssertionError("Failed to reattach eval_row_id to predictions")
    merged["candidate_id"] = str(args.candidate_id)
    merged["source_candidate_id"] = str(source_candidate_id)
    merged["split_seed"] = int(args.split_seed)
    merged["fold_id"] = int(args.fold_id)
    merged["member_id"] = _member_id(args.split_seed, args.fold_id)
    merged["drug_name"] = merged["DRUG_NAME"].astype(str)
    merged["target_key"] = str(target_key)
    merged["drug_id"] = str(args.drug_id)
    merged["predictor_id"] = str(args.predictor_id)
    merged["omics_id"] = str(args.omics_id)
    merged["checkpoint_path"] = str(checkpoint_path)
    required = [
        "candidate_id",
        "source_candidate_id",
        "eval_row_id",
        "split_seed",
        "fold_id",
        "member_id",
        "Label",
        "ModelID",
        "drug_name",
        "target_key",
        "probability",
        "drug_id",
        "predictor_id",
        "omics_id",
    ]
    if merged[required].isna().any().any():
        bad = [column for column in required if merged[column].isna().any()]
        raise AssertionError(f"Null post-hoc identity fields: {bad}")
    return merged


def _run_posthoc_dataset(
    args: argparse.Namespace,
    settings: dict,
    frame: pd.DataFrame,
    *,
    target_key: str,
    latent_by_id: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, dict, Path]:
    drug_cfg = settings["drug_reps"][args.drug_id]
    encoder_type = str(drug_cfg["type"])
    maccs_by_drug = None
    if encoder_type.lower() == "maccs":
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys
        from tools.round19_drug_features import load_maccs_by_drug_name

        drugs = sorted(set(frame["DRUG_NAME"].astype(str)))
        maccs_by_drug = load_maccs_by_drug_name(
            settings["drug_smiles_path"], drug_names=drugs
        )
        for _, row in frame.iterrows():
            drug_name = str(row["DRUG_NAME"])
            if drug_name in maccs_by_drug:
                continue
            smiles = str(row.get("smiles", "")).strip()
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None:
                raise ValueError(f"Cannot build MACCS for TCGA drug {drug_name!r}")
            bits = np.asarray(list(MACCSkeys.GenMACCSKeys(molecule)), dtype=np.float32)
            if bits.shape == (167,):
                bits = bits[1:]
            if bits.shape != (166,):
                raise AssertionError(f"Unexpected MACCS shape for {drug_name}: {bits.shape}")
            maccs_by_drug[drug_name] = bits
    dataset = Round19ResponseDataset(
        frame,
        feature_dir=_feature_dir(settings, args.omics_id),
        drug_smiles_path=settings["drug_smiles_path"],
        encoder_type=encoder_type,
        with_bonds=bool(drug_cfg.get("edge_features")),
        maccs_by_drug=maccs_by_drug,
        omics_id=args.omics_id,
        latent_by_id=latent_by_id,
    )
    expected_dim = resolve_omics_dim(args.omics_id)
    if dataset.omics_dim != expected_dim:
        raise AssertionError(f"{args.omics_id} dim {dataset.omics_dim} != {expected_dim}")
    loader = DataLoader(
        dataset,
        batch_size=int(args.micro_batch_size),
        shuffle=False,
        num_workers=int(os.environ.get("ROUND19_NUM_WORKERS", "0")),
        collate_fn=round19_collate_fn,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    encoder, fusion, head, encoder_type, _, checkpoint_path = _strict_load_inference_models(
        args, settings, omics_dim=dataset.omics_dim, device=device
    )
    output = evaluate_predictions_round19(
        encoder=encoder,
        fusion=fusion,
        head=head,
        dataloader=loader,
        device=device,
        encoder_type=encoder_type,
        predictor_id=args.predictor_id,
        amp_enabled=(device.type == "cuda" and not args.disable_amp),
    )
    annotated = _annotate_posthoc_predictions(
        output["predictions"],
        dataset.df,
        args,
        target_key=target_key,
        checkpoint_path=checkpoint_path,
    )
    return annotated, output["metrics"], checkpoint_path


def infer_internal_test(args: argparse.Namespace) -> dict:
    settings = _load_settings(args.settings)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(args.internal_test_path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing locked Round19 internal split: {split_path}")
    frame = pd.read_csv(split_path).copy()
    if frame.empty:
        raise ValueError(f"Empty locked Round19 internal split: {split_path}")
    if "_row_id" not in frame:
        raise KeyError("Round19 internal_test_split.csv must preserve _row_id")
    if frame["_row_id"].duplicated().any():
        raise AssertionError("Duplicate _row_id in Round19 internal_test_split.csv")
    frame["eval_row_id"] = [
        f"internal_test|{int(row_id)}" for row_id in frame["_row_id"]
    ]
    predictions, metrics, checkpoint_path = _run_posthoc_dataset(
        args, settings, frame, target_key="internal_test"
    )
    output_path = result_dir / "internal_test_predictions.csv"
    predictions.to_csv(output_path, index=False)
    (result_dir / "internal_test_metrics.json").write_text(
        json.dumps(_metrics_jsonable(metrics), indent=2), encoding="utf-8"
    )
    summary = {
        "ok": True,
        "mode": "infer_internal_test",
        "wrote": str(output_path),
        "n_rows": len(predictions),
        "checkpoint_path": str(checkpoint_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


def _prepare_tcga_frame(
    target_path: Path,
    *,
    target_key: str,
    latent: Dict[str, np.ndarray],
    drug_smiles_path: str,
) -> pd.DataFrame:
    raw = load_tcga_response_csv(str(target_path)).reset_index(drop=True).copy()
    drug_column = "drug_name" if "drug_name" in raw else "DRUG_NAME"
    required = {"Patient_id", "Label", drug_column}
    missing = required - set(raw)
    if missing:
        raise KeyError(f"TCGA target missing columns {sorted(missing)}: {target_path}")
    patient_to_latent = {}
    for latent_key in latent:
        parts = str(latent_key).split("-")
        if len(parts) >= 3:
            patient_to_latent.setdefault("-".join(parts[:3]), latent_key)
    smiles_lookup = load_smiles_lookup(drug_smiles_path)
    rows = []
    miss_latent = miss_smiles = 0
    for source_row_id, row in raw.iterrows():
        patient_id = str(row["Patient_id"])
        latent_key = patient_to_latent.get(patient_id)
        if latent_key is None:
            miss_latent += 1
            continue
        drug_name = str(row[drug_column]).strip()
        inline_smiles = str(row.get("smiles", "")).strip()
        if inline_smiles.lower() == "nan":
            inline_smiles = ""
        smiles = inline_smiles or smiles_lookup.get(drug_name.lower(), "")
        if not smiles:
            miss_smiles += 1
            continue
        label = int(row["Label"])
        eval_row_id = (
            f"{target_key}|{patient_id}|{drug_name}|{label}|{int(source_row_id)}"
        )
        rows.append(
            {
                "_row_id": len(rows),
                "ModelID": patient_id,
                "_latent_key": str(latent_key),
                "Patient_id": patient_id,
                "DRUG_NAME": drug_name,
                "drug_name": drug_name,
                "Label": label,
                "eval_row_id": eval_row_id,
                "target_key": target_key,
                "drug_smiles_key": drug_name.lower(),
                "smiles": smiles,
            }
        )
    if not rows:
        raise RuntimeError(
            f"No aligned TCGA rows for {target_key}; "
            f"missing_latent={miss_latent}, missing_smiles={miss_smiles}"
        )
    frame = pd.DataFrame(rows)
    frame.attrs["n_miss_latent"] = miss_latent
    frame.attrs["n_miss_smiles"] = miss_smiles
    return frame


def infer_tcga(args: argparse.Namespace) -> dict:
    settings = _load_settings(args.settings)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if not args.target_key or args.target_key not in TCGA_TARGETS:
        raise ValueError(
            f"infer_tcga requires one of five --target-key values: {sorted(TCGA_TARGETS)}"
        )
    target_path = Path(args.target_path or TCGA_TARGETS[args.target_key])
    preflight_path = result_dir / "tcga_feature_preflight.json"
    try:
        latent, preflight = preflight_tcga_features(settings, args.omics_id)
        preflight.update(
            {
                "target_key": args.target_key,
                "target_path": str(target_path),
                "checkpoint_path": str(_resolve_checkpoint(args)),
            }
        )
        preflight_path.write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    except Exception as exc:
        failure = {
            "ok": False,
            "omics_id": args.omics_id,
            "target_key": args.target_key,
            "target_path": str(target_path),
            "fit_on_tcga_labels": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        preflight_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"TCGA feature preflight failed closed; see {preflight_path}: {exc}"
        ) from exc
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing TCGA target: {target_path}")
    frame = _prepare_tcga_frame(
        target_path,
        target_key=args.target_key,
        latent=latent,
        drug_smiles_path=settings["drug_smiles_path"],
    )
    patient_latent = {
        str(row["Patient_id"]): latent[str(row["_latent_key"])]
        for _, row in frame.iterrows()
    }
    predictions, metrics, checkpoint_path = _run_posthoc_dataset(
        args,
        settings,
        frame,
        target_key=args.target_key,
        latent_by_id=patient_latent,
    )
    output_path = result_dir / "tcga_predictions.csv"
    predictions.to_csv(output_path, index=False)
    (result_dir / "tcga_metrics.json").write_text(
        json.dumps(_metrics_jsonable(metrics), indent=2), encoding="utf-8"
    )
    summary = {
        "ok": True,
        "mode": "infer_tcga",
        "target_key": args.target_key,
        "target_path": str(target_path),
        "wrote": str(output_path),
        "n_rows": len(predictions),
        "n_miss_latent": int(frame.attrs["n_miss_latent"]),
        "n_miss_smiles": int(frame.attrs["n_miss_smiles"]),
        "checkpoint_path": str(checkpoint_path),
        "preflight": str(preflight_path),
    }
    (result_dir / "tcga_infer_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 19 pipeline")
    parser.add_argument(
        "--mode",
        default="data_smoke",
        choices=["data_smoke", "train_fold", "infer_internal_test", "infer_tcga"],
    )
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
    parser.add_argument("--split-seed", type=int, default=None)
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
    parser.add_argument("--control-type", default="none", choices=["none", "context_shuffle"])
    parser.add_argument("--train-shuffle-seed", type=int, default=None)
    parser.add_argument("--val-shuffle-seed", type=int, default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument(
        "--checkpoint-inventory",
        default="result/optimization_runs/round19_factorial/manifests/stage19d_manifest.csv",
    )
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--source-candidate-id", default=None)
    parser.add_argument("--target-key", default=None)
    parser.add_argument("--target-path", default=None)
    args = parser.parse_args()
    if args.mode == "data_smoke":
        if not args.max_batches:
            args.max_batches = 2
        run_data_smoke(args)
    elif args.mode == "train_fold":
        if not args.result_dir:
            raise SystemExit("train_fold requires --result-dir")
        train_fold(args)
    elif args.mode == "infer_internal_test":
        if not args.result_dir:
            raise SystemExit("infer_internal_test requires --result-dir")
        infer_internal_test(args)
    elif args.mode == "infer_tcga":
        if not args.result_dir:
            raise SystemExit("infer_tcga requires --result-dir")
        infer_tcga(args)
    else:
        raise SystemExit(f"Unsupported mode {args.mode}")


if __name__ == "__main__":
    main()
