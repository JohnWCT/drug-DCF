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
from tools.prototype_response_features import (
    build_projection_raw_row,
    build_raw_context_vector,
    build_raw_delta_vector,
    compute_own_proto_context_features_batch,
    compute_own_proto_delta_replacement_features_batch,
    compute_round17_standalone_features_batch,
    compute_proto_distance_features,
    concat_latent_and_proto_features,
    fit_context_projection,
    get_own_source_target_vectors,
    get_projected_context_dim,
    get_projected_delta_dim,
    is_own_proto_context_mode,
    is_own_proto_delta_replacement_mode,
    is_round17_standalone_mode,
    resolve_feature_mode_options,
)
from tools.round9_diagnostics_common import (
    _load_cancer_maps,
    _load_latent_dict,
    filter_latent_dict_by_cancer_types,
    find_latent_paths,
    load_checkpoint_cancer_type_mapping,
    normalize_proto_cancer_type_mapping,
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


def _filter_latents_to_trainable(
    ccle_latent: Dict[str, np.ndarray],
    tcga_latent: Dict[str, np.ndarray],
    mapping: Dict,
    ccle_map: Optional[pd.Series] = None,
    tcga_map: Optional[pd.Series] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    allowed = list(mapping.get("cancer_names") or mapping.get("name_to_id", {}).keys())
    if not allowed:
        raise ValueError("cancer_type_mapping missing cancer_names/name_to_id")
    if ccle_map is None or tcga_map is None:
        ccle_map, tcga_map = _load_cancer_maps()
    filtered_ccle = filter_latent_dict_by_cancer_types(
        ccle_latent, "source", allowed, ccle_map=ccle_map, tcga_map=tcga_map
    )
    filtered_tcga = filter_latent_dict_by_cancer_types(
        tcga_latent, "target", allowed, ccle_map=ccle_map, tcga_map=tcga_map
    )
    if not filtered_ccle:
        raise ValueError("No CCLE latents remain after filtering to trainable cancer types")
    return filtered_ccle, filtered_tcga


DEFAULT_REQUIRE_N_TRAINABLE_CANCER_TYPES = 18


def _assert_18class_mapping(
    mapping: Dict,
    *,
    require_n: int = DEFAULT_REQUIRE_N_TRAINABLE_CANCER_TYPES,
    context: str = "",
) -> None:
    n = int(mapping.get("num_cancer_types", len(mapping.get("cancer_names", []))))
    source = str(mapping.get("mapping_source", ""))
    if source and source != "checkpoint_metadata":
        raise ValueError(
            f"Prototype mapping must come from checkpoint_metadata, got {source!r}"
            + (f" ({context})" if context else "")
        )
    if n != int(require_n):
        raise ValueError(
            f"n_trainable_cancer_types={n} but require_n_trainable_cancer_types={require_n}"
            + (f" ({context})" if context else "")
        )


def _prototype_qc_fields(
    mapping: Dict,
    source_initialized: Optional[np.ndarray],
    target_initialized: Optional[np.ndarray],
) -> Dict:
    src_init = (
        np.asarray(source_initialized, dtype=bool)
        if source_initialized is not None
        else np.array([], dtype=bool)
    )
    tgt_init = (
        np.asarray(target_initialized, dtype=bool)
        if target_initialized is not None
        else np.array([], dtype=bool)
    )
    return {
        "prototype_class_source": "checkpoint_metadata",
        "n_trainable_cancer_types": int(mapping.get("num_cancer_types", len(mapping.get("cancer_names", [])))),
        "source_prototypes_used": int(src_init.sum()) if len(src_init) else 0,
        "target_prototypes_used": int(tgt_init.sum()) if len(tgt_init) else 0,
        "uses_legacy_28class_cache": False,
    }


def _write_prototype_qc_artifacts(
    outdir: str,
    mapping: Dict,
    source_initialized: Optional[np.ndarray],
    target_initialized: Optional[np.ndarray],
) -> None:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "cancer_type_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    cancer_names = list(mapping.get("cancer_names", []))
    src_init = (
        np.asarray(source_initialized, dtype=bool)
        if source_initialized is not None
        else np.zeros(len(cancer_names), dtype=bool)
    )
    tgt_init = (
        np.asarray(target_initialized, dtype=bool)
        if target_initialized is not None
        else np.zeros(len(cancer_names), dtype=bool)
    )
    rows = []
    for i, name in enumerate(cancer_names):
        rows.append(
            {
                "cancer_id": i,
                "cancer_type": name,
                "source_initialized": bool(src_init[i]) if i < len(src_init) else False,
                "target_initialized": bool(tgt_init[i]) if i < len(tgt_init) else False,
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "prototype_coverage.csv"), index=False)


def _load_or_extract_prototypes(
    checkpoint_dir: str,
    proto_cache_dir: str,
    strict: bool,
    require_n_trainable_cancer_types: int = DEFAULT_REQUIRE_N_TRAINABLE_CANCER_TYPES,
) -> Dict:
    cache_dir = resolve_path(proto_cache_dir)
    required = [
        "source_anchor_prototypes.pt",
        "target_prototypes.pt",
        "cancer_type_mapping.json",
    ]
    checkpoint_mapping = load_checkpoint_cancer_type_mapping(checkpoint_dir)
    _assert_18class_mapping(
        checkpoint_mapping,
        require_n=require_n_trainable_cancer_types,
        context=checkpoint_dir,
    )
    expected_n = int(checkpoint_mapping["num_cancer_types"])
    if all(os.path.isfile(os.path.join(cache_dir, f)) for f in required):
        with open(os.path.join(cache_dir, "cancer_type_mapping.json"), encoding="utf-8") as f:
            mapping = normalize_proto_cancer_type_mapping(json.load(f))
        cache_ok = (
            int(mapping.get("num_cancer_types", 0)) == expected_n
            and mapping.get("mapping_source") == "checkpoint_metadata"
        )
        if cache_ok:
            import torch

            src = torch.load(os.path.join(cache_dir, "source_anchor_prototypes.pt"), map_location="cpu")
            tgt = torch.load(os.path.join(cache_dir, "target_prototypes.pt"), map_location="cpu")
            return {
                "source_anchor_prototypes": src["prototypes"].numpy(),
                "target_prototypes": tgt["prototypes"].numpy(),
                "source_initialized": src["initialized"].numpy().astype(bool),
                "target_initialized": tgt["initialized"].numpy().astype(bool),
                "cancer_type_mapping": mapping,
            }
    payload = extract_prototypes_from_checkpoint(checkpoint_dir, outdir=cache_dir)
    _assert_18class_mapping(
        payload["cancer_type_mapping"],
        require_n=require_n_trainable_cancer_types,
        context=checkpoint_dir,
    )
    if strict and int(payload["prototype_metrics"]["source_initialized_count"]) == 0:
        raise ValueError(f"No initialized source anchors for {checkpoint_dir}")
    return payload


def build_combined_latent_dicts_own_proto(
    checkpoint_dir: str,
    feature_mode: str,
    outdir: str,
    metric: str = "cosine",
    include_l2_distance: bool = True,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = True,
    proto_feature_scaler: str = "standard",
    strict: bool = False,
    proto_cache_dir: Optional[str] = None,
) -> Dict:
    checkpoint_dir = resolve_path(checkpoint_dir)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)
    feature_mode = str(feature_mode).lower()

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
    ccle_latent, tcga_latent = _filter_latents_to_trainable(
        ccle_latent, tcga_latent, mapping, ccle_map=ccle_map, tcga_map=tcga_map
    )

    ccle_ids = [_sample_cancer_id(sid, "source", ccle_map, tcga_map, name_to_id) for sid in ccle_latent.keys()]
    ccle_z = np.stack([ccle_latent[sid] for sid in ccle_latent.keys()], axis=0)
    latent_dim = int(ccle_z.shape[1])

    projection_model = None
    projection_metadata = None
    proj_dim = get_projected_context_dim(feature_mode)
    if proj_dim > 0:
        raw_rows = []
        for vec, cid in zip(ccle_z, ccle_ids):
            vecs = get_own_source_target_vectors(
                int(cid),
                proto["source_anchor_prototypes"],
                proto["target_prototypes"],
                source_initialized=proto["source_initialized"],
                target_initialized=proto["target_initialized"],
                strict=strict,
                latent_dim=latent_dim,
            )
            raw_rows.append(build_raw_context_vector(vec, vecs["source_anchor"], vecs["target_proto"]))
        raw_mat = np.stack(raw_rows, axis=0)
        projection_model = fit_context_projection(raw_mat, proj_dim)
        actual_proj_dim = int(getattr(projection_model, "n_components_", proj_dim))
        projection_metadata = {
            "projection_type": "pca",
            "fit_domain": "source_only",
            "input_dim": int(raw_mat.shape[1]),
            "requested_output_dim": int(proj_dim),
            "output_dim": actual_proj_dim,
            "explained_variance_ratio_sum": float(np.sum(projection_model.explained_variance_ratio_)),
        }
        with open(os.path.join(outdir, "projection_model.pkl"), "wb") as f:
            pickle.dump(projection_model, f)
        with open(os.path.join(outdir, "projection_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(projection_metadata, f, indent=2)

    proto_ccle = compute_own_proto_context_features_batch(
        ccle_z,
        ccle_ids,
        proto["source_anchor_prototypes"],
        target_prototypes=proto["target_prototypes"],
        mode=feature_mode,
        cancer_type_mapping=mapping,
        metric=metric,
        include_l2_distance=include_l2_distance,
        include_same_cancer_gap=include_same_cancer_gap,
        include_initialized_flag=include_initialized_flag,
        source_initialized=proto["source_initialized"],
        target_initialized=proto["target_initialized"],
        projection_model=projection_model,
        strict=strict,
    )
    feature_names = list(proto_ccle["feature_names"])
    proto_mat = np.asarray(proto_ccle["features"], dtype=np.float32)

    scaler = _build_scaler(proto_feature_scaler)
    scaler_payload = {"type": proto_feature_scaler}
    if scaler is not None and len(proto_mat) > 0:
        train_idx = np.arange(len(proto_mat))
        if len(proto_mat) >= 20:
            labels = np.array([max(cid, 0) for cid in ccle_ids])
            try:
                train_idx, _ = train_test_split(
                    np.arange(len(proto_mat)), test_size=0.1, random_state=42, stratify=labels
                )
            except ValueError:
                train_idx, _ = train_test_split(np.arange(len(proto_mat)), test_size=0.1, random_state=42)
        scaler.fit(proto_mat[train_idx])
        proto_mat = scaler.transform(proto_mat).astype(np.float32)
        scaler_payload = {
            "type": proto_feature_scaler,
            "mean": getattr(scaler, "mean_", None),
            "scale": getattr(scaler, "scale_", None),
            "center": getattr(scaler, "center_", None),
        }

    combined_ccle = {}
    for i, sid in enumerate(ccle_latent.keys()):
        combined_ccle[sid] = concat_latent_and_proto_features(ccle_z[i], {"features": proto_mat[i]})

    combined_tcga = {}
    if tcga_latent:
        tcga_ids = [_sample_cancer_id(sid, "target", ccle_map, tcga_map, name_to_id) for sid in tcga_latent.keys()]
        tcga_z = np.stack([tcga_latent[sid] for sid in tcga_latent.keys()], axis=0)
        proto_tcga = compute_own_proto_context_features_batch(
            tcga_z,
            tcga_ids,
            proto["source_anchor_prototypes"],
            target_prototypes=proto["target_prototypes"],
            mode=feature_mode,
            cancer_type_mapping=mapping,
            metric=metric,
            include_l2_distance=include_l2_distance,
            include_same_cancer_gap=include_same_cancer_gap,
            include_initialized_flag=include_initialized_flag,
            source_initialized=proto["source_initialized"],
            target_initialized=proto["target_initialized"],
            projection_model=projection_model,
            strict=strict,
        )
        proto_tcga_mat = np.asarray(proto_tcga["features"], dtype=np.float32)
        if scaler is not None and scaler_payload.get("type") != "none":
            proto_tcga_mat = scaler.transform(proto_tcga_mat).astype(np.float32)
        for i, sid in enumerate(tcga_latent.keys()):
            combined_tcga[sid] = concat_latent_and_proto_features(tcga_z[i], {"features": proto_tcga_mat[i]})

    latent_dim = len(next(iter(ccle_latent.values())))
    response_input_dim = len(next(iter(combined_ccle.values())))
    proto_dim = len(feature_names)

    ccle_out = os.path.join(outdir, "ccle_latent_proto.pkl")
    tcga_out = os.path.join(outdir, "tcga_latent_proto.pkl")
    with open(ccle_out, "wb") as f:
        pickle.dump(combined_ccle, f)
    with open(tcga_out, "wb") as f:
        pickle.dump(combined_tcga, f)

    # z column names prepended for metadata clarity
    z_names = [f"z_dim{i:03d}" for i in range(latent_dim)]
    full_feature_names = z_names + feature_names

    metadata = {
        "checkpoint_dir": checkpoint_dir,
        "prototype_feature_mode": feature_mode,
        "response_input_mode": "z_plus_proto_features",
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
        "requires_projection": proj_dim > 0,
        "projection_dim": proj_dim,
        "projection_fit_domain": "source_only" if proj_dim > 0 else None,
        "projection_metadata": projection_metadata,
    }
    metadata.update(
        _prototype_qc_fields(
            mapping,
            proto.get("source_initialized"),
            proto.get("target_initialized"),
        )
    )
    _assert_18class_mapping(mapping, context=outdir)
    _write_prototype_qc_artifacts(
        outdir,
        mapping,
        proto.get("source_initialized"),
        proto.get("target_initialized"),
    )
    with open(os.path.join(outdir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(full_feature_names, f, indent=2)
    with open(os.path.join(outdir, "feature_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata


def build_combined_latent_dicts_delta_replacement(
    checkpoint_dir: str,
    feature_mode: str,
    outdir: str,
    metric: str = "cosine",
    include_l2_distance: bool = True,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = True,
    proto_feature_scaler: str = "standard",
    strict: bool = False,
    proto_cache_dir: Optional[str] = None,
) -> Dict:
    checkpoint_dir = resolve_path(checkpoint_dir)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)
    feature_mode = str(feature_mode).lower()

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
    ccle_latent, tcga_latent = _filter_latents_to_trainable(
        ccle_latent, tcga_latent, mapping, ccle_map=ccle_map, tcga_map=tcga_map
    )

    ccle_ids = [_sample_cancer_id(sid, "source", ccle_map, tcga_map, name_to_id) for sid in ccle_latent.keys()]
    ccle_z = np.stack([ccle_latent[sid] for sid in ccle_latent.keys()], axis=0)
    latent_dim = int(ccle_z.shape[1])

    projection_model = None
    projection_metadata = None
    proj_dim = get_projected_delta_dim(feature_mode)
    if proj_dim > 0:
        raw_rows = []
        for vec, cid in zip(ccle_z, ccle_ids):
            vecs = get_own_source_target_vectors(
                int(cid),
                proto["source_anchor_prototypes"],
                proto["target_prototypes"],
                source_initialized=proto["source_initialized"],
                target_initialized=proto["target_initialized"],
                strict=strict,
                latent_dim=latent_dim,
            )
            raw_rows.append(
                build_projection_raw_row(
                    vec,
                    vecs["source_anchor"],
                    vecs["target_proto"],
                    feature_mode,
                    target_available=bool(vecs.get("target_initialized", True)),
                )
            )
        raw_mat = np.stack(raw_rows, axis=0)
        projection_model = fit_context_projection(raw_mat, proj_dim)
        actual_proj_dim = int(getattr(projection_model, "n_components_", proj_dim))
        projection_metadata = {
            "projection_type": "pca",
            "fit_domain": "source_only",
            "input_dim": int(raw_mat.shape[1]),
            "requested_output_dim": int(proj_dim),
            "output_dim": actual_proj_dim,
            "feature_mode": feature_mode,
            "explained_variance_ratio_sum": float(np.sum(projection_model.explained_variance_ratio_)),
        }
        with open(os.path.join(outdir, "projection_model.pkl"), "wb") as f:
            pickle.dump(projection_model, f)
        with open(os.path.join(outdir, "projection_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(projection_metadata, f, indent=2)

    proto_ccle = compute_own_proto_delta_replacement_features_batch(
        ccle_z,
        ccle_ids,
        proto["source_anchor_prototypes"],
        target_prototypes=proto["target_prototypes"],
        mode=feature_mode,
        cancer_type_mapping=mapping,
        metric=metric,
        include_l2_distance=include_l2_distance,
        include_same_cancer_gap=include_same_cancer_gap,
        include_initialized_flag=include_initialized_flag,
        source_initialized=proto["source_initialized"],
        target_initialized=proto["target_initialized"],
        projection_model=projection_model,
        strict=strict,
    )
    feature_names = list(proto_ccle["feature_names"])
    proto_mat = np.asarray(proto_ccle["features"], dtype=np.float32)
    row_meta = proto_ccle.get("metadata", {})

    scaler = _build_scaler(proto_feature_scaler)
    scaler_payload = {"type": proto_feature_scaler}
    if scaler is not None and len(proto_mat) > 0:
        train_idx = np.arange(len(proto_mat))
        if len(proto_mat) >= 20:
            labels = np.array([max(cid, 0) for cid in ccle_ids])
            try:
                train_idx, _ = train_test_split(
                    np.arange(len(proto_mat)), test_size=0.1, random_state=42, stratify=labels
                )
            except ValueError:
                train_idx, _ = train_test_split(np.arange(len(proto_mat)), test_size=0.1, random_state=42)
        scaler.fit(proto_mat[train_idx])
        proto_mat = scaler.transform(proto_mat).astype(np.float32)
        scaler_payload = {
            "type": proto_feature_scaler,
            "mean": getattr(scaler, "mean_", None),
            "scale": getattr(scaler, "scale_", None),
            "center": getattr(scaler, "center_", None),
        }

    combined_ccle = {}
    for i, sid in enumerate(ccle_latent.keys()):
        combined_ccle[sid] = concat_latent_and_proto_features(ccle_z[i], {"features": proto_mat[i]})

    combined_tcga = {}
    if tcga_latent:
        tcga_ids = [_sample_cancer_id(sid, "target", ccle_map, tcga_map, name_to_id) for sid in tcga_latent.keys()]
        tcga_z = np.stack([tcga_latent[sid] for sid in tcga_latent.keys()], axis=0)
        proto_tcga = compute_own_proto_delta_replacement_features_batch(
            tcga_z,
            tcga_ids,
            proto["source_anchor_prototypes"],
            target_prototypes=proto["target_prototypes"],
            mode=feature_mode,
            cancer_type_mapping=mapping,
            metric=metric,
            include_l2_distance=include_l2_distance,
            include_same_cancer_gap=include_same_cancer_gap,
            include_initialized_flag=include_initialized_flag,
            source_initialized=proto["source_initialized"],
            target_initialized=proto["target_initialized"],
            projection_model=projection_model,
            strict=strict,
        )
        proto_tcga_mat = np.asarray(proto_tcga["features"], dtype=np.float32)
        if scaler is not None and scaler_payload.get("type") != "none":
            proto_tcga_mat = scaler.transform(proto_tcga_mat).astype(np.float32)
        for i, sid in enumerate(tcga_latent.keys()):
            combined_tcga[sid] = concat_latent_and_proto_features(tcga_z[i], {"features": proto_tcga_mat[i]})

    ccle_out = os.path.join(outdir, "ccle_latent_proto.pkl")
    tcga_out = os.path.join(outdir, "tcga_latent_proto.pkl")
    with open(ccle_out, "wb") as f:
        pickle.dump(combined_ccle, f)
    with open(tcga_out, "wb") as f:
        pickle.dump(combined_tcga, f)

    response_input_dim = len(next(iter(combined_ccle.values())))
    proto_dim = len(feature_names)
    z_names = [f"z_dim{i:03d}" for i in range(latent_dim)]
    full_feature_names = z_names + feature_names

    metadata = {
        "checkpoint_dir": checkpoint_dir,
        "feature_mode": feature_mode,
        "prototype_feature_mode": feature_mode,
        "response_input_mode": "z_plus_proto_features",
        "base_latent_dim": latent_dim,
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
        "uses_own_plus_summary": bool(row_meta.get("uses_own_plus_summary", False)),
        "uses_delta": bool(row_meta.get("uses_delta", False)),
        "uses_projection": bool(row_meta.get("uses_projection", False)),
        "projection_dim": int(row_meta.get("projection_dim", 0)),
        "projection_fit_domain": "source_only" if row_meta.get("uses_projection") else None,
        "projection_metadata": projection_metadata,
    }
    metadata.update(
        _prototype_qc_fields(
            mapping,
            proto.get("source_initialized"),
            proto.get("target_initialized"),
        )
    )
    _assert_18class_mapping(mapping, context=outdir)
    _write_prototype_qc_artifacts(
        outdir,
        mapping,
        proto.get("source_initialized"),
        proto.get("target_initialized"),
    )
    with open(os.path.join(outdir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(full_feature_names, f, indent=2)
    with open(os.path.join(outdir, "feature_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata


def build_combined_latent_dicts_round17_standalone(
    checkpoint_dir: str,
    feature_mode: str,
    outdir: str,
    metric: str = "cosine",
    include_l2_distance: bool = True,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = False,
    proto_feature_scaler: str = "standard",
    strict: bool = False,
    proto_cache_dir: Optional[str] = None,
) -> Dict:
    checkpoint_dir = resolve_path(checkpoint_dir)
    outdir = resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)
    feature_mode = str(feature_mode).lower()

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
    ccle_latent, tcga_latent = _filter_latents_to_trainable(
        ccle_latent, tcga_latent, mapping, ccle_map=ccle_map, tcga_map=tcga_map
    )

    ccle_ids = [_sample_cancer_id(sid, "source", ccle_map, tcga_map, name_to_id) for sid in ccle_latent.keys()]
    ccle_z = np.stack([ccle_latent[sid] for sid in ccle_latent.keys()], axis=0)
    latent_dim = int(ccle_z.shape[1])

    projection_model = None
    projection_metadata = None
    proj_dim = get_projected_context_dim(feature_mode)
    if proj_dim > 0:
        raw_rows = []
        for vec, cid in zip(ccle_z, ccle_ids):
            vecs = get_own_source_target_vectors(
                int(cid),
                proto["source_anchor_prototypes"],
                proto["target_prototypes"],
                source_initialized=proto["source_initialized"],
                target_initialized=proto["target_initialized"],
                strict=strict,
                latent_dim=latent_dim,
            )
            raw_rows.append(
                build_projection_raw_row(
                    vec,
                    vecs["source_anchor"],
                    vecs["target_proto"],
                    feature_mode,
                    target_available=bool(vecs.get("target_initialized", True)),
                )
            )
        raw_mat = np.stack(raw_rows, axis=0)
        projection_model = fit_context_projection(raw_mat, proj_dim)
        projection_metadata = {
            "projection_type": "pca",
            "fit_domain": "source_only",
            "input_dim": int(raw_mat.shape[1]),
            "requested_output_dim": int(proj_dim),
            "output_dim": int(getattr(projection_model, "n_components_", proj_dim)),
            "feature_mode": feature_mode,
        }
        with open(os.path.join(outdir, "projection_model.pkl"), "wb") as f:
            pickle.dump(projection_model, f)
        with open(os.path.join(outdir, "projection_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(projection_metadata, f, indent=2)

    proto_ccle = compute_round17_standalone_features_batch(
        ccle_z,
        ccle_ids,
        proto["source_anchor_prototypes"],
        target_prototypes=proto["target_prototypes"],
        mode=feature_mode,
        cancer_type_mapping=mapping,
        metric=metric,
        include_l2_distance=include_l2_distance,
        include_same_cancer_gap=include_same_cancer_gap,
        include_initialized_flag=include_initialized_flag,
        source_initialized=proto["source_initialized"],
        target_initialized=proto["target_initialized"],
        projection_model=projection_model,
        strict=strict,
    )
    feature_names = list(proto_ccle["feature_names"])
    proto_mat = np.asarray(proto_ccle["features"], dtype=np.float32)
    row_meta = proto_ccle.get("metadata", {})

    scaler = _build_scaler(proto_feature_scaler)
    scaler_payload = {"type": proto_feature_scaler}
    if scaler is not None and len(proto_mat) > 0:
        scaler.fit(proto_mat)
        proto_mat = scaler.transform(proto_mat).astype(np.float32)
        scaler_payload = {"type": proto_feature_scaler, "mean": getattr(scaler, "mean_", None)}

    combined_ccle = {
        sid: concat_latent_and_proto_features(ccle_z[i], {"features": proto_mat[i]})
        for i, sid in enumerate(ccle_latent.keys())
    }
    combined_tcga = {}
    if tcga_latent:
        tcga_ids = [_sample_cancer_id(sid, "target", ccle_map, tcga_map, name_to_id) for sid in tcga_latent.keys()]
        tcga_z = np.stack([tcga_latent[sid] for sid in tcga_latent.keys()], axis=0)
        proto_tcga = compute_round17_standalone_features_batch(
            tcga_z,
            tcga_ids,
            proto["source_anchor_prototypes"],
            target_prototypes=proto["target_prototypes"],
            mode=feature_mode,
            cancer_type_mapping=mapping,
            metric=metric,
            include_l2_distance=include_l2_distance,
            include_same_cancer_gap=include_same_cancer_gap,
            include_initialized_flag=include_initialized_flag,
            source_initialized=proto["source_initialized"],
            target_initialized=proto["target_initialized"],
            projection_model=projection_model,
            strict=strict,
        )
        proto_tcga_mat = np.asarray(proto_tcga["features"], dtype=np.float32)
        if scaler is not None and scaler_payload.get("type") != "none":
            proto_tcga_mat = scaler.transform(proto_tcga_mat).astype(np.float32)
        for i, sid in enumerate(tcga_latent.keys()):
            combined_tcga[sid] = concat_latent_and_proto_features(tcga_z[i], {"features": proto_tcga_mat[i]})

    response_input_dim = len(next(iter(combined_ccle.values())))
    z_names = [f"z_dim{i:03d}" for i in range(latent_dim)]
    full_feature_names = z_names + feature_names
    ccle_out = os.path.join(outdir, "ccle_latent_proto.pkl")
    tcga_out = os.path.join(outdir, "tcga_latent_proto.pkl")
    with open(ccle_out, "wb") as f:
        pickle.dump(combined_ccle, f)
    with open(tcga_out, "wb") as f:
        pickle.dump(combined_tcga, f)

    metadata = {
        "checkpoint_dir": checkpoint_dir,
        "feature_mode": feature_mode,
        "prototype_feature_mode": feature_mode,
        "response_input_mode": "z_plus_proto_features",
        "base_latent_dim": latent_dim,
        "proto_feature_dim": len(feature_names),
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
        "uses_own_plus_summary": bool(row_meta.get("uses_own_plus_summary", False)),
        "uses_projection": bool(row_meta.get("uses_projection", False)),
        "projection_dim": int(row_meta.get("projection_dim", 0)),
        "projection_fit_domain": "source_only" if row_meta.get("uses_projection") else None,
        "projection_metadata": projection_metadata,
    }
    metadata.update(
        _prototype_qc_fields(
            mapping,
            proto.get("source_initialized"),
            proto.get("target_initialized"),
        )
    )
    _assert_18class_mapping(mapping, context=outdir)
    _write_prototype_qc_artifacts(
        outdir,
        mapping,
        proto.get("source_initialized"),
        proto.get("target_initialized"),
    )
    with open(os.path.join(outdir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(full_feature_names, f, indent=2)
    with open(os.path.join(outdir, "feature_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata


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
    ccle_latent, tcga_latent = _filter_latents_to_trainable(
        ccle_latent, tcga_latent, mapping, ccle_map=ccle_map, tcga_map=tcga_map
    )

    feature_mode = str(feature_mode).lower()
    if is_round17_standalone_mode(feature_mode):
        return build_combined_latent_dicts_round17_standalone(
            checkpoint_dir=checkpoint_dir,
            feature_mode=feature_mode,
            outdir=outdir,
            metric=metric,
            include_l2_distance=include_l2_distance,
            include_same_cancer_gap=include_same_cancer_gap,
            include_initialized_flag=include_initialized_flag,
            proto_feature_scaler=proto_feature_scaler,
            strict=strict,
            proto_cache_dir=proto_cache_dir,
        )
    if is_own_proto_delta_replacement_mode(feature_mode):
        return build_combined_latent_dicts_delta_replacement(
            checkpoint_dir=checkpoint_dir,
            feature_mode=feature_mode,
            outdir=outdir,
            metric=metric,
            include_l2_distance=include_l2_distance,
            include_same_cancer_gap=include_same_cancer_gap,
            include_initialized_flag=include_initialized_flag,
            proto_feature_scaler=proto_feature_scaler,
            strict=strict,
            proto_cache_dir=proto_cache_dir,
        )
    if is_own_proto_context_mode(feature_mode):
        return build_combined_latent_dicts_own_proto(
            checkpoint_dir=checkpoint_dir,
            feature_mode=feature_mode,
            outdir=outdir,
            metric=metric,
            include_l2_distance=include_l2_distance,
            include_same_cancer_gap=include_same_cancer_gap,
            include_initialized_flag=include_initialized_flag,
            proto_feature_scaler=proto_feature_scaler,
            strict=strict,
            proto_cache_dir=proto_cache_dir,
        )

    mode_opts = resolve_feature_mode_options(
        feature_mode,
        include_l2_distance=include_l2_distance,
        include_same_cancer_gap=include_same_cancer_gap,
        include_initialized_flag=include_initialized_flag,
        proto_feature_scaler=proto_feature_scaler,
    )
    compute_mode = mode_opts["mode"]
    if compute_mode == "none":
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
            mode=compute_mode,
            metric=metric,
            include_l2_distance=mode_opts["include_l2_distance"],
            include_same_cancer_gap=mode_opts["include_same_cancer_gap"],
            include_initialized_flag=mode_opts["include_initialized_flag"],
            strict=strict,
            source_initialized=proto["source_initialized"],
            target_initialized=proto["target_initialized"],
        )
        feature_names = list(proto_ccle["feature_names"])
        proto_dim = len(feature_names)

        scaler = _build_scaler(mode_opts["proto_feature_scaler"])
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
                "type": mode_opts["proto_feature_scaler"],
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
                mode=compute_mode,
                metric=metric,
                include_l2_distance=mode_opts["include_l2_distance"],
                include_same_cancer_gap=mode_opts["include_same_cancer_gap"],
                include_initialized_flag=mode_opts["include_initialized_flag"],
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
        "prototype_feature_mode": mode_opts["feature_mode_label"],
        "response_input_mode": "z_only" if compute_mode == "none" else "z_plus_proto_features",
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
    metadata.update(
        _prototype_qc_fields(
            mapping,
            proto.get("source_initialized"),
            proto.get("target_initialized"),
        )
    )
    _assert_18class_mapping(mapping, context=outdir)
    _write_prototype_qc_artifacts(
        outdir,
        mapping,
        proto.get("source_initialized"),
        proto.get("target_initialized"),
    )
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
