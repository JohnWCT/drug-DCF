#!/usr/bin/env python3
"""Extract source/target prototypes from Round 12 (or Round 11) pretrain checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.analyze_cancer_prototypes import analyze_model
from tools.round9_diagnostics_common import find_latent_paths, load_latent_domain_frame, resolve_path


def _prototype_matrix(vectors: List[Tuple[str, np.ndarray]], cancer_names: List[str], dim: int) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.zeros((len(cancer_names), dim), dtype=np.float32)
    init = np.zeros(len(cancer_names), dtype=bool)
    name_to_idx = {name: i for i, name in enumerate(cancer_names)}
    for name, vec in vectors:
        if name not in name_to_idx:
            continue
        idx = name_to_idx[name]
        mat[idx] = np.asarray(vec, dtype=np.float32)
        init[idx] = True
    return mat, init


def extract_prototypes_from_checkpoint(
    checkpoint_dir: str,
    outdir: Optional[str] = None,
    min_source: int = 2,
    min_target: int = 2,
) -> Dict:
    checkpoint_dir = resolve_path(checkpoint_dir)
    model = {"model_id": os.path.basename(checkpoint_dir), "checkpoint_dir": checkpoint_dir}
    by_cancer, summary, src_df, tgt_df = analyze_model(
        model, metrics=["cosine", "euclidean"], min_source=min_source, min_target=min_target
    )

    frame = load_latent_domain_frame(checkpoint_dir)
    z_cols = [c for c in frame.columns if c.startswith("z")]
    latent_dim = len(z_cols)

    cancer_names = sorted(by_cancer["cancer_type"].astype(str).unique().tolist())
    if not cancer_names:
        cancer_names = sorted(frame["cancer_type"].astype(str).unique().tolist())

    source_vectors: List[Tuple[str, np.ndarray]] = []
    target_vectors: List[Tuple[str, np.ndarray]] = []
    for cancer_type, sub in frame.groupby("cancer_type"):
        source = sub[sub["domain"] == "source"][z_cols].to_numpy(dtype=np.float32)
        target = sub[sub["domain"] == "target"][z_cols].to_numpy(dtype=np.float32)
        if len(source) >= min_source:
            source_vectors.append((str(cancer_type), source.mean(axis=0)))
        if len(target) >= min_target:
            target_vectors.append((str(cancer_type), target.mean(axis=0)))

    source_mat, source_init = _prototype_matrix(source_vectors, cancer_names, latent_dim)
    target_mat, target_init = _prototype_matrix(target_vectors, cancer_names, latent_dim)

    id_to_name = {i: name for i, name in enumerate(cancer_names)}
    name_to_id = {name: i for i, name in enumerate(cancer_names)}
    mapping = {"id_to_name": id_to_name, "name_to_id": name_to_id, "cancer_names": cancer_names}

    metrics = {
        "model_id": os.path.basename(checkpoint_dir),
        "latent_dim": latent_dim,
        "num_cancer_types": len(cancer_names),
        "source_initialized_count": int(source_init.sum()),
        "target_initialized_count": int(target_init.sum()),
        "mean_same_cancer_proto_distance": summary.get(
            "mean_same_cancer_source_target_cosine_distance", np.nan
        ),
        "mean_inter_cancer_source_margin": summary.get("mean_inter_cancer_source_margin", np.nan),
        "mean_inter_cancer_target_margin": summary.get("mean_inter_cancer_target_margin", np.nan),
    }

    if outdir:
        outdir = resolve_path(outdir)
        os.makedirs(outdir, exist_ok=True)
        torch.save(
            {"prototypes": torch.from_numpy(source_mat), "initialized": torch.from_numpy(source_init)},
            os.path.join(outdir, "source_anchor_prototypes.pt"),
        )
        torch.save(
            {"prototypes": torch.from_numpy(target_mat), "initialized": torch.from_numpy(target_init)},
            os.path.join(outdir, "target_prototypes.pt"),
        )
        with open(os.path.join(outdir, "cancer_type_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        with open(os.path.join(outdir, "prototype_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        if not by_cancer.empty:
            by_cancer.to_csv(os.path.join(outdir, "per_cancer_prototype_gap.csv"), index=False)

    return {
        "source_anchor_prototypes": source_mat,
        "target_prototypes": target_mat,
        "source_initialized": source_init,
        "target_initialized": target_init,
        "cancer_type_mapping": mapping,
        "prototype_metrics": metrics,
        "per_cancer": by_cancer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract prototype artifacts from a pretrain checkpoint")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--min-source", type=int, default=2)
    parser.add_argument("--min-target", type=int, default=2)
    args = parser.parse_args()
    payload = extract_prototypes_from_checkpoint(
        args.checkpoint_dir,
        outdir=args.outdir,
        min_source=args.min_source,
        min_target=args.min_target,
    )
    print(f"Wrote prototypes to {resolve_path(args.outdir)}")
    print(json.dumps(payload["prototype_metrics"], indent=2))


if __name__ == "__main__":
    main()
