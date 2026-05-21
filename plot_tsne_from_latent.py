"""
Re-draw dual-panel t-SNE from saved pretrain latent pickles (no retraining).

Uses the same test split & labels as pretrain_VAEwC, and the same 2-panel layout
as tools.pretrain_tsne (A: domain, B: cancer type).

Example (Docker):
  docker exec -w /workspace/DAPL DAPL python3 plot_tsne_from_latent.py \
    --exp_dir result/pretrain_vaewc/exp_011

Output (default): <exp_dir>/tsne_gan_best.png
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from evaluate_raw_data import DEFAULT_SOURCE_CSV, load_labeled_feature_splits
from tools.dataprocess import safemakedirs
from tools.pretrain_common import TARGET_DOMAIN_CONFIG, tcga_three_segment_key
from tools.pretrain_tsne import plot_latent_tsne_dual

plt.switch_backend("Agg")


def _load_latent_dict(path: str) -> Dict[str, List[float]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Latent pickle not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data)}")
    return {str(k): v for k, v in data.items()}


def _resolve_latent_vector(latent_dict: Dict[str, List[float]], sample_id: str) -> Tuple[List[float] | None, str]:
    """Lookup latent by exact key, then TCGA patient key fallback."""
    sid = str(sample_id)
    if sid in latent_dict:
        return latent_dict[sid], sid
    patient_key = tcga_three_segment_key(sid)
    if patient_key in latent_dict:
        return latent_dict[patient_key], patient_key
    return None, sid


def _align_test_latents(
    test_df,
    label_int: np.ndarray,
    latent_dict: Dict[str, List[float]],
    domain_name: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Map test-split rows to latent vectors (same samples as pretrain t-SNE).
    """
    feats: List[np.ndarray] = []
    labels: List[int] = []
    missing: List[str] = []

    for i, sid in enumerate(test_df.index.astype(str)):
        vec, _ = _resolve_latent_vector(latent_dict, sid)
        if vec is None:
            missing.append(sid)
            continue
        feats.append(np.asarray(vec, dtype=np.float32))
        labels.append(int(label_int[i]))

    if missing:
        print(f"[{domain_name}] warning: {len(missing)} test samples missing in latent dict (skipped)")
        if len(missing) <= 10:
            print(f"  missing ids: {missing}")
        else:
            print(f"  missing ids (first 10): {missing[:10]}")

    if not feats:
        raise ValueError(f"No {domain_name} test samples matched latent dict.")

    return np.vstack(feats), np.asarray(labels, dtype=np.int64), missing


def plot_tsne_from_exp_dir(
    exp_dir: str,
    source_csv: str = DEFAULT_SOURCE_CSV,
    target_csv: str | None = None,
    target_domain: str = "tcga",
    target_cancer_reference: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_points: int = 3000,
    output_path: str | None = None,
    suptitle: str | None = None,
) -> str:
    exp_dir = os.path.abspath(exp_dir)
    ccle_pkl = os.path.join(exp_dir, "ccle_latent_dict.pkl")
    tcga_pkl = os.path.join(exp_dir, "tcga_latent_dict.pkl")

    ccle_latent = _load_latent_dict(ccle_pkl)
    tcga_latent = _load_latent_dict(tcga_pkl)

    domain_cfg = TARGET_DOMAIN_CONFIG[target_domain]
    resolved_target = target_csv or domain_cfg["target_expression"]
    resolved_cancer_ref = target_cancer_reference or domain_cfg["target_cancer_reference"]

    (
        ccle_test,
        tcga_test,
        ccle_test_labels,
        tcga_test_labels,
        mapping_int2str,
        _num_classes,
    ) = load_labeled_feature_splits(
        ccle_path=source_csv,
        xena_path=resolved_target,
        target_domain=target_domain,
        target_cancer_reference_path=resolved_cancer_ref,
        test_size=test_size,
        random_state=random_state,
    )

    source_z, source_labels, _ = _align_test_latents(
        ccle_test, ccle_test_labels, ccle_latent, "CCLE"
    )
    target_z, target_labels, _ = _align_test_latents(
        tcga_test, tcga_test_labels, tcga_latent, "TCGA"
    )

    if output_path is None:
        output_path = os.path.join(exp_dir, "tsne_gan_best.png")
    safemakedirs(os.path.dirname(output_path) or ".")

    exp_name = os.path.basename(exp_dir.rstrip(os.sep))
    if suptitle is None:
        suptitle = f"GAN Best Latent t-SNE (Test Split) — {exp_name}"

    plot_latent_tsne_dual(
        source_z,
        target_z,
        source_labels,
        target_labels,
        mapping_int2str,
        output_path,
        suptitle=suptitle,
        max_points=max_points,
    )
    print(f"[tsne] saved {output_path}")
    print(f"  CCLE test points: {len(source_z)} | TCGA test points: {len(target_z)}")

    summary = {
        "exp_dir": exp_dir,
        "ccle_latent_pkl": ccle_pkl,
        "tcga_latent_pkl": tcga_pkl,
        "source_test_count": int(len(source_z)),
        "target_test_count": int(len(target_z)),
        "test_size": test_size,
        "random_state": random_state,
        "output": output_path,
    }
    summary_path = os.path.join(exp_dir, "tsne_replay_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[tsne] summary {summary_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Re-plot dual-panel t-SNE from ccle/tcga latent pickles in a pretrain exp folder."
    )
    parser.add_argument(
        "--exp_dir",
        required=True,
        type=str,
        help="Pretrain experiment folder containing ccle_latent_dict.pkl & tcga_latent_dict.pkl",
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE_CSV, type=str, help="CCLE expression csv")
    parser.add_argument("--target", default=None, type=str, help="TCGA expression csv (auto if omitted)")
    parser.add_argument("--target_domain", default="tcga", choices=["tcga", "pdx"], type=str)
    parser.add_argument("--target_cancer_reference", default=None, type=str)
    parser.add_argument("--test_size", default=0.2, type=float, help="Must match pretrain split (default 0.2)")
    parser.add_argument("--random_state", default=42, type=int, help="Must match pretrain split (default 42)")
    parser.add_argument("--max_points", default=3000, type=int, help="Subsample cap for t-SNE")
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Output png path (default: <exp_dir>/tsne_gan_best.png)",
    )
    parser.add_argument("--suptitle", default=None, type=str, help="Figure suptitle override")
    args = parser.parse_args()

    plot_tsne_from_exp_dir(
        exp_dir=args.exp_dir,
        source_csv=args.source,
        target_csv=args.target,
        target_domain=args.target_domain,
        target_cancer_reference=args.target_cancer_reference,
        test_size=args.test_size,
        random_state=args.random_state,
        max_points=args.max_points,
        output_path=args.output,
        suptitle=args.suptitle,
    )


if __name__ == "__main__":
    main()
