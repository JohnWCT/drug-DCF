#!/usr/bin/env python3
"""Build combined latent + prototype-distance feature dicts for Round 13 finetune."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.extract_round12_prototypes import extract_prototypes_from_checkpoint
from tools.prototype_response_features import compute_proto_distance_features, concat_latent_and_proto_features
from tools.round9_diagnostics_common import (
    _load_cancer_maps,
    _load_latent_dict,
    find_latent_paths,
    resolve_path,
    tcga_three_segment_key,
)


def _build_scaler(name: str):
    name = str(name).lower()
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "none":
        return None
    raise ValueError(f"Unsupported proto_feature_scaler={name!r}")


def _sample_cancer_id(sample_id: str, domain: str, ccle_map: pd.Series, tcga_map: pd.Series, name_to_id: Dict[str, int]) -> int:
    sid = str(sample_id)
    if domain == "source":
        if sid not in ccle_map.index:
            return -1
        cancer = str(ccle_map.loc[sid])
    else:
        patient = tcga_three_segment_key(sid)
        if patient not in tcga_map.index:
            return -1
        cancer = str(tcga_map.loc[patient])
    return int(name_to_id.get(cancer, -1))


def _load_or_extract_prototypes(checkpoint_dir: str, proto_cache_dir: str, strict: bool) -> Dict:
    cache_dir = resolve_path(proto_cache_dir)
    required = [
        "source_anchor_prototypes.pt",
        "target_prototypes.pt",
        "cancer_type_mapping.json",
    ]
    if all(os.path.isfile(os.path.join(cache_dir, f)) for f in required):
        import torch

        src = torch.load(os.path.join(cache_dir, "source_anchor_prototypes.pt"), map_location="cpu")
        tgt = torch.load(os.path.join(cache_dir, "target_prototypes.pt"), map_location="cpu")
        with open(os.path.join(cache_dir, "cancer_type_mapping.json"), encoding="utf-8") as f:
            mapping = json.load(f)
        return {
            "source_anchor_prototypes": src["prototypes"].numpy(),
            "target_prototypes": tgt["prototypes"].numpy(),
            "source_initialized": src["initialized"].numpy().astype(bool),
            "target_initialized": tgt["initialized"].numpy().astype(bool),
            "cancer_type_mapping": mapping,
        }
    payload = extract_prototypes_from_checkpoint(checkpoint_dir, outdir=cache_dir)
    if strict and int(payload["prototype_metrics"]["source_initialized_count"]) == 0:
        raise ValueError(f"No initialized source anchors for {checkpoint_dir}")
    return payload


def build_combined_latent_dicts(
    checkpoint_dir: str,
    feature_mode: str,
    outdir: str,
    metric: str = "cosine",
    include_l2_distance: bool = False,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = True,
    proto_feature_scaler: str = "standard",
    strict: bool = False,
    proto_cache_dir: Optional[str] = None,
) -> Dict:
    checkpoint_dir = resolve_path(checkpoint_dir)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)

    source_pkl, target_pkl = find_latent_paths(checkpoint_dir)
    if not source_pkl:
        raise FileNotFoundError(f"Missing CCLE latent dict under {checkpoint_dir}")
    ccle_latent = _load_latent_dict(source_pkl)
    tcga_latent = _load_latent_dict(target_pkl) if target_pkl and os.path.isfile(target_pkl) else {}

    cache_dir = proto_cache_dir or os.path.join(outdir, "_proto_cache")
    proto = _load_or_extract_prototypes(checkpoint_dir, cache_dir, strict=strict)
    mapping = proto["cancer_type_mapping"]
    name_to_id = mapping.get("name_to_id", {})
    ccle_map, tcga_map = _load_cancer_maps()

    feature_mode = str(feature_mode).lower()
    if feature_mode == "none":
        combined_ccle = {k: np.asarray(v, dtype=np.float32) for k, v in ccle_latent.items()}
        combined_tcga = {k: np.asarray(v, dtype=np.float32) for k, v in tcga_latent.items()}
        feature_names: List[str] = []
        proto_dim = 0
        scaler_payload = None
    else:
        ccle_ids = [
            _sample_cancer_id(sid, "source", ccle_map, tcga_map, name_to_id) for sid in ccle_latent.keys()
        ]
        ccle_z = np.stack([ccle_latent[sid] for sid in ccle_latent.keys()], axis=0)
        proto_ccle = compute_proto_distance_features(
            ccle_z,
            ccle_ids,
            proto["source_anchor_prototypes"],
            target_prototypes=proto["target_prototypes"],
            cancer_type_mapping=mapping,
            mode=feature_mode,
            metric=metric,
            include_l2_distance=include_l2_distance,
            include_same_cancer_gap=include_same_cancer_gap,
            include_initialized_flag=include_initialized_flag,
            strict=strict,
            source_initialized=proto["source_initialized"],
            target_initialized=proto["target_initialized"],
        )
        feature_names = list(proto_ccle["feature_names"])
        proto_dim = len(feature_names)

        scaler = _build_scaler(proto_feature_scaler)
        proto_mat = np.asarray(proto_ccle["features"], dtype=np.float32)
        if scaler is not None and len(proto_mat) > 0:
            train_idx = np.arange(len(proto_mat))
            if len(proto_mat) >= 20:
                labels = np.array([max(cid, 0) for cid in ccle_ids])
                try:
                    train_idx, _ = train_test_split(
                        np.arange(len(proto_mat)),
                        test_size=0.1,
                        random_state=42,
                        stratify=labels,
                    )
                except ValueError:
                    train_idx, _ = train_test_split(
                        np.arange(len(proto_mat)), test_size=0.1, random_state=42
                    )
            scaler.fit(proto_mat[train_idx])
            proto_mat = scaler.transform(proto_mat).astype(np.float32)
            scaler_payload = {
                "type": proto_feature_scaler,
                "mean": getattr(scaler, "mean_", None),
                "scale": getattr(scaler, "scale_", None),
                "center": getattr(scaler, "center_", None),
            }
        else:
            scaler_payload = {"type": "none"}

        combined_ccle = {}
        for i, sid in enumerate(ccle_latent.keys()):
            combined_ccle[sid] = concat_latent_and_proto_features(ccle_z[i], {"features": proto_mat[i]})

        combined_tcga = {}
        if tcga_latent:
            tcga_ids = [
                _sample_cancer_id(sid, "target", ccle_map, tcga_map, name_to_id) for sid in tcga_latent.keys()
            ]
            tcga_z = np.stack([tcga_latent[sid] for sid in tcga_latent.keys()], axis=0)
            proto_tcga = compute_proto_distance_features(
                tcga_z,
                tcga_ids,
                proto["source_anchor_prototypes"],
                target_prototypes=proto["target_prototypes"],
                cancer_type_mapping=mapping,
                mode=feature_mode,
                metric=metric,
                include_l2_distance=include_l2_distance,
                include_same_cancer_gap=include_same_cancer_gap,
                include_initialized_flag=include_initialized_flag,
                strict=strict,
                source_initialized=proto["source_initialized"],
                target_initialized=proto["target_initialized"],
            )
            proto_tcga_mat = np.asarray(proto_tcga["features"], dtype=np.float32)
            if scaler is not None and scaler_payload and scaler_payload.get("type") != "none":
                proto_tcga_mat = scaler.transform(proto_tcga_mat).astype(np.float32)
            for i, sid in enumerate(tcga_latent.keys()):
                combined_tcga[sid] = concat_latent_and_proto_features(tcga_z[i], {"features": proto_tcga_mat[i]})

    latent_dim = len(next(iter(ccle_latent.values())))
    response_input_dim = len(next(iter(combined_ccle.values())))

    ccle_out = os.path.join(outdir, "ccle_latent_proto.pkl")
    tcga_out = os.path.join(outdir, "tcga_latent_proto.pkl")
    with open(ccle_out, "wb") as f:
        pickle.dump(combined_ccle, f)
    with open(tcga_out, "wb") as f:
        pickle.dump(combined_tcga, f)

    metadata = {
        "checkpoint_dir": checkpoint_dir,
        "prototype_feature_mode": feature_mode,
        "response_input_mode": "z_only" if feature_mode == "none" else "z_plus_proto_features",
        "proto_feature_dim": proto_dim,
        "latent_dim": latent_dim,
        "response_input_dim": response_input_dim,
        "proto_feature_scaler": proto_feature_scaler,
        "metric": metric,
        "include_l2_distance": include_l2_distance,
        "include_same_cancer_gap": include_same_cancer_gap,
        "include_initialized_flag": include_initialized_flag,
        "n_ccle_samples": len(combined_ccle),
        "n_tcga_samples": len(combined_tcga),
        "scaler": scaler_payload,
    }
    with open(os.path.join(outdir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    with open(os.path.join(outdir, "feature_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata


def _resolve_feature_outdir(row: pd.Series, outdir: str, feature_mode: str) -> str:
    manifest_dir = row.get("combined_latent_dir")
    if manifest_dir is not None and str(manifest_dir).strip() not in ("", "nan"):
        return resolve_path(str(manifest_dir))
    return os.path.join(resolve_path(outdir), str(row["source_model_id"]), feature_mode)


def extract_from_manifest(manifest_path: str, outdir: str, strict: bool = True) -> pd.DataFrame:
    manifest = pd.read_csv(resolve_path(manifest_path))
    rows = []
    for _, row in manifest.iterrows():
        feature_mode = str(row.get("prototype_feature_mode", row.get("feature_mode", "none")))
        target_out = _resolve_feature_outdir(row, outdir, feature_mode)
        meta = build_combined_latent_dicts(
            checkpoint_dir=str(row["checkpoint_dir"]),
            feature_mode=feature_mode,
            outdir=target_out,
            metric=str(row.get("prototype_distance_metric", "cosine")),
            include_l2_distance=bool(row.get("include_l2_distance", False)),
            include_same_cancer_gap=bool(row.get("include_same_cancer_gap", True)),
            include_initialized_flag=bool(row.get("include_initialized_flag", True)),
            proto_feature_scaler=str(row.get("proto_feature_scaler", "standard")),
            strict=strict,
        )
        meta["job_id"] = row.get("job_id", "")
        meta["source_model_id"] = row.get("source_model_id", "")
        meta["combined_latent_dir"] = target_out
        rows.append(meta)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Round 13 prototype response features")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args()
    summary = extract_from_manifest(args.manifest, args.outdir, strict=args.strict)
    summary_path = os.path.join(resolve_path(args.outdir), "feature_extraction_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote feature artifacts under {resolve_path(args.outdir)} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
