#!/usr/bin/env python3
"""Prototype-aware tSNE visualization for Round 17."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.round9_diagnostics_common import (
    _load_cancer_maps,
    load_json,
    normalize_proto_cancer_type_mapping,
    resolve_path,
)


def _load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _subsample_indices(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def _collect_points(
    source_latent: Dict[str, np.ndarray],
    target_latent: Dict[str, np.ndarray],
    source_prototypes: np.ndarray,
    target_prototypes: np.ndarray,
    source_initialized: Optional[np.ndarray],
    target_initialized: Optional[np.ndarray],
    id_to_name: Dict[int, str],
    allowed_cancer_types: Sequence[str],
    ccle_map: pd.Series,
    tcga_map: pd.Series,
    max_source_samples: int,
    max_target_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[dict]]:
    from tools.pretrain_common import tcga_three_segment_key

    allowed = {str(x) for x in allowed_cancer_types}
    rows: List[dict] = []
    feats: List[np.ndarray] = []

    source_ids = [
        sid
        for sid in source_latent.keys()
        if sid in ccle_map.index and str(ccle_map.loc[sid]) in allowed
    ]
    src_idx = _subsample_indices(len(source_ids), max_source_samples, rng)
    for i in src_idx:
        sid = source_ids[i]
        vec = np.asarray(source_latent[sid], dtype=np.float64).reshape(-1)
        feats.append(vec)
        rows.append(
            {
                "point_id": f"source_sample::{sid}",
                "point_type": "sample",
                "domain": "source",
                "cancer_type": str(ccle_map.loc[sid]),
                "sample_id": sid,
                "is_source_prototype": 0,
                "is_target_prototype": 0,
                "prototype_initialized": 1,
            }
        )

    target_ids = []
    for sid in target_latent.keys():
        patient = tcga_three_segment_key(sid)
        if patient not in tcga_map.index:
            continue
        if str(tcga_map.loc[patient]) in allowed:
            target_ids.append(sid)
    tgt_idx = _subsample_indices(len(target_ids), max_target_samples, rng)
    for i in tgt_idx:
        sid = target_ids[i]
        vec = np.asarray(target_latent[sid], dtype=np.float64).reshape(-1)
        patient = tcga_three_segment_key(sid)
        feats.append(vec)
        rows.append(
            {
                "point_id": f"target_sample::{sid}",
                "point_type": "sample",
                "domain": "target",
                "cancer_type": str(tcga_map.loc[patient]),
                "sample_id": sid,
                "is_source_prototype": 0,
                "is_target_prototype": 0,
                "prototype_initialized": 1,
            }
        )

    src_init = (
        np.asarray(source_initialized, dtype=bool)
        if source_initialized is not None
        else np.ones(len(source_prototypes), dtype=bool)
    )
    tgt_init = (
        np.asarray(target_initialized, dtype=bool)
        if target_initialized is not None
        else np.ones(len(target_prototypes), dtype=bool)
    )

    for cid in range(len(source_prototypes)):
        if not bool(src_init[cid]):
            continue
        vec = np.asarray(source_prototypes[cid], dtype=np.float64).reshape(-1)
        feats.append(vec)
        cname = id_to_name.get(cid, str(cid))
        rows.append(
            {
                "point_id": f"source_prototype::{cname}",
                "point_type": "source_prototype",
                "domain": "prototype_source",
                "cancer_type": cname,
                "sample_id": "",
                "is_source_prototype": 1,
                "is_target_prototype": 0,
                "prototype_initialized": 1,
            }
        )

    for cid in range(len(target_prototypes)):
        if not bool(tgt_init[cid]):
            continue
        vec = np.asarray(target_prototypes[cid], dtype=np.float64).reshape(-1)
        feats.append(vec)
        cname = id_to_name.get(cid, str(cid))
        rows.append(
            {
                "point_id": f"target_prototype::{cname}",
                "point_type": "target_prototype",
                "domain": "prototype_target",
                "cancer_type": cname,
                "sample_id": "",
                "is_source_prototype": 0,
                "is_target_prototype": 1,
                "prototype_initialized": 1,
            }
        )

    if not feats:
        raise ValueError("No points collected for tSNE")
    return np.stack(feats, axis=0), rows


def run_prototype_tsne(
    *,
    source_latent: Dict[str, np.ndarray],
    target_latent: Dict[str, np.ndarray],
    source_prototypes: np.ndarray,
    target_prototypes: np.ndarray,
    source_initialized: Optional[np.ndarray],
    target_initialized: Optional[np.ndarray],
    cancer_type_mapping: Optional[dict],
    outdir: str,
    title: str,
    tsne_cfg: Optional[dict] = None,
    max_source_samples: int = 3000,
    max_target_samples: int = 3000,
) -> dict:
    os.makedirs(outdir, exist_ok=True)
    tsne_cfg = tsne_cfg or {}
    rng = np.random.default_rng(int(tsne_cfg.get("random_state", 17)))
    mapping = normalize_proto_cancer_type_mapping(cancer_type_mapping or {})
    id_to_name = {int(k): str(v) for k, v in mapping.get("id_to_name", {}).items()}
    allowed_cancer_types = list(mapping.get("cancer_names", []))
    if not allowed_cancer_types:
        allowed_cancer_types = [id_to_name[i] for i in sorted(id_to_name.keys())]
    ccle_map, tcga_map = _load_cancer_maps()

    feats, rows = _collect_points(
        source_latent,
        target_latent,
        source_prototypes,
        target_prototypes,
        source_initialized,
        target_initialized,
        id_to_name,
        allowed_cancer_types,
        ccle_map,
        tcga_map,
        max_source_samples,
        max_target_samples,
        rng,
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    tsne_kwargs = {
        "n_components": 2,
        "random_state": int(tsne_cfg.get("random_state", 17)),
        "perplexity": min(int(tsne_cfg.get("perplexity", 30)), max(2, len(feats) - 1)),
        "init": str(tsne_cfg.get("init", "pca")),
        "learning_rate": tsne_cfg.get("learning_rate", "auto"),
    }
    max_iter = int(tsne_cfg.get("max_iter", 1000))
    try:
        tsne = TSNE(**tsne_kwargs, max_iter=max_iter)
    except TypeError:
        tsne = TSNE(**tsne_kwargs, n_iter=max_iter)
    emb = tsne.fit_transform(feats)

    coord_df = pd.DataFrame(rows)
    coord_df["tsne_1"] = emb[:, 0]
    coord_df["tsne_2"] = emb[:, 1]
    coord_path = os.path.join(outdir, "prototype_tsne_coordinates.csv")
    coord_df.to_csv(coord_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    samples = coord_df[coord_df["point_type"] == "sample"]
    src = samples[samples["domain"] == "source"]
    tgt = samples[samples["domain"] == "target"]
    ax.scatter(src["tsne_1"], src["tsne_2"], s=12, alpha=0.25, c="#1f77b4", marker="o", label="source samples")
    ax.scatter(tgt["tsne_1"], tgt["tsne_2"], s=12, alpha=0.25, c="#ff7f0e", marker="^", label="target samples")

    src_p = coord_df[coord_df["point_type"] == "source_prototype"]
    tgt_p = coord_df[coord_df["point_type"] == "target_prototype"]
    if not src_p.empty:
        ax.scatter(
            src_p["tsne_1"],
            src_p["tsne_2"],
            s=180,
            c="#000000",
            marker="*",
            edgecolors="white",
            linewidths=0.8,
            label="source prototype",
        )
        for _, row in src_p.iterrows():
            ax.annotate(row["cancer_type"], (row["tsne_1"], row["tsne_2"]), fontsize=7)
    if not tgt_p.empty:
        ax.scatter(
            tgt_p["tsne_1"],
            tgt_p["tsne_2"],
            s=180,
            c="#d62728",
            marker="*",
            edgecolors="white",
            linewidths=0.8,
            label="target prototype",
        )
        for _, row in tgt_p.iterrows():
            ax.annotate(row["cancer_type"], (row["tsne_1"], row["tsne_2"]), fontsize=7)

    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    png_path = os.path.join(outdir, "prototype_tsne_samples_and_prototypes.png")
    pdf_path = os.path.join(outdir, "prototype_tsne_samples_and_prototypes.pdf")
    fig.tight_layout()
    fig.savefig(png_path, dpi=250)
    fig.savefig(pdf_path)
    plt.close(fig)

    meta = {
        "title": title,
        "n_points": int(len(coord_df)),
        "n_source_prototypes": int(len(src_p)),
        "n_target_prototypes": int(len(tgt_p)),
        "num_trainable_cancer_types": len(allowed_cancer_types),
        "trainable_cancer_types": allowed_cancer_types,
        "missing_target_prototypes_skipped": int(len(target_prototypes) - len(tgt_p)),
        "coordinates_csv": coord_path,
        "png": png_path,
        "pdf": pdf_path,
        "tsne_config": tsne_cfg,
    }
    meta_path = os.path.join(outdir, "prototype_tsne_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def _resolve_checkpoint_from_manifest_row(row: pd.Series, settings: dict) -> str:
    checkpoint = str(row.get("checkpoint_dir", row.get("pretrain_dir", ""))).strip()
    if checkpoint and os.path.isdir(resolve_path(checkpoint)):
        return resolve_path(checkpoint)
    model_key = str(row.get("model_id", row.get("model_key", ""))).strip()
    try:
        from tools.round17r_18class_config_builder import ROUND17R_MODEL_SPECS

        if model_key in ROUND17R_MODEL_SPECS:
            spec = ROUND17R_MODEL_SPECS[model_key]
            return os.path.join(resolve_path(settings[spec["checkpoint_root_key"]]), "pretrain", spec["checkpoint_subdir"])
    except ImportError:
        pass
    from tools.round17_direct_proto_config_builder import ROUND17_MODEL_SPECS

    if model_key in ROUND17_MODEL_SPECS:
        spec = ROUND17_MODEL_SPECS[model_key]
        return os.path.join(resolve_path(settings[spec["checkpoint_root_key"]]), "pretrain", spec["checkpoint_subdir"])
    raise FileNotFoundError(f"Cannot resolve checkpoint for model_id={model_key}")


def run_batch_from_manifest(
    settings_path: str,
    manifest_path: str,
    outdir: str,
    models: Sequence[str],
    force: bool = False,
    max_source_samples: int = 3000,
    max_target_samples: int = 3000,
) -> List[dict]:
    settings = load_json(settings_path)
    manifest = pd.read_csv(resolve_path(manifest_path))
    tsne_cfg = settings.get("tsne", {})
    outputs = []
    for model_key in models:
        sub = manifest[manifest["model_id"] == model_key]
        if sub.empty:
            sub = manifest[manifest["model_id"].astype(str).str.startswith(model_key)]
        if sub.empty:
            print(f"skip {model_key}: not in manifest")
            continue
        row = sub.iloc[0]
        model_out = os.path.join(resolve_path(outdir), model_key)
        if not force and os.path.isfile(os.path.join(model_out, "prototype_tsne_coordinates.csv")):
            print(f"skip {model_key}: exists")
            continue
        checkpoint_dir = _resolve_checkpoint_from_manifest_row(row, settings)
        from tools.extract_round13_proto_features import find_latent_paths
        from tools.extract_round12_prototypes import extract_prototypes_from_checkpoint

        proto_cache_dir = os.path.join(model_out, "_proto_cache")
        if os.path.isdir(proto_cache_dir):
            import shutil

            shutil.rmtree(proto_cache_dir)
        source_pkl, target_pkl = find_latent_paths(checkpoint_dir)
        source_latent = _load_pickle(source_pkl)
        target_latent = _load_pickle(target_pkl) if target_pkl and os.path.isfile(target_pkl) else {}
        proto = extract_prototypes_from_checkpoint(checkpoint_dir, outdir=proto_cache_dir)
        meta = run_prototype_tsne(
            source_latent=source_latent,
            target_latent=target_latent,
            source_prototypes=proto["source_anchor_prototypes"],
            target_prototypes=proto["target_prototypes"],
            source_initialized=proto.get("source_initialized"),
            target_initialized=proto.get("target_initialized"),
            cancer_type_mapping=proto.get("cancer_type_mapping"),
            outdir=model_out,
            title=f"Round17 {model_key} prototype tSNE",
            tsne_cfg=tsne_cfg,
            max_source_samples=max_source_samples,
            max_target_samples=max_target_samples,
        )
        outputs.append(meta)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 17 prototype-aware tSNE")
    parser.add_argument("--settings", default="config/round17_direct_proto_settings.json")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--max-source-samples", type=int, default=3000)
    parser.add_argument("--max-target-samples", type=int, default=3000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--source-latent", default=None)
    parser.add_argument("--target-latent", default=None)
    parser.add_argument("--source-prototypes", default=None)
    parser.add_argument("--target-prototypes", default=None)
    parser.add_argument("--title", default="Round 17 prototype tSNE")
    args = parser.parse_args()

    if args.manifest and args.models:
        run_batch_from_manifest(
            args.settings,
            args.manifest,
            args.outdir,
            args.models,
            force=args.force,
            max_source_samples=args.max_source_samples,
            max_target_samples=args.max_target_samples,
        )
        return

    if not all([args.source_latent, args.source_prototypes]):
        raise SystemExit("Provide --manifest + --models, or single-model latent/prototype paths")
    source_latent = _load_pickle(resolve_path(args.source_latent))
    target_latent = _load_pickle(resolve_path(args.target_latent)) if args.target_latent else {}
    proto_pack = _load_pickle(resolve_path(args.source_prototypes))
    target_protos = proto_pack.get("target_prototypes", proto_pack)
    source_protos = proto_pack.get("source_anchor_prototypes", proto_pack)
    run_prototype_tsne(
        source_latent=source_latent,
        target_latent=target_latent,
        source_prototypes=np.asarray(source_protos),
        target_prototypes=np.asarray(target_protos),
        source_initialized=proto_pack.get("source_initialized") if isinstance(proto_pack, dict) else None,
        target_initialized=proto_pack.get("target_initialized") if isinstance(proto_pack, dict) else None,
        cancer_type_mapping=proto_pack.get("cancer_type_mapping") if isinstance(proto_pack, dict) else None,
        outdir=resolve_path(args.outdir),
        title=args.title,
        max_source_samples=args.max_source_samples,
        max_target_samples=args.max_target_samples,
    )


if __name__ == "__main__":
    main()
