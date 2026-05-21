"""Shared dual-panel t-SNE plots for pretrain / raw evaluation pipelines."""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_latent_tsne_dual(
    source_z: np.ndarray,
    target_z: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    mapping_int2str: Dict[int, str],
    save_path: str,
    suptitle: str = "Latent t-SNE (Test Split)",
    max_points: int = 3000,
    source_domain_label: str = "Source (CCLE)",
    target_domain_label: str = "Target (TCGA)",
    target_alpha: float = 0.85,
) -> None:
    """
    Panel A: color by domain (source vs target).
    Panel B: color by cancer type (shared palette; target uses triangle marker).
    """
    if len(source_z) == 0 or len(target_z) == 0:
        return

    source_z = np.asarray(source_z, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    source_labels = np.asarray(source_labels)
    target_labels = np.asarray(target_labels)

    source_z = np.nan_to_num(source_z, nan=0.0, posinf=0.0, neginf=0.0)
    target_z = np.nan_to_num(target_z, nan=0.0, posinf=0.0, neginf=0.0)
    all_feats = np.vstack([source_z, target_z])
    all_labels = np.concatenate([source_labels, target_labels])
    n_source = len(source_z)

    if all_feats.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(all_feats.shape[0], max_points, replace=False)
        all_feats = all_feats[idx]
        all_labels = all_labels[idx]
        domain_flags = np.array(
            ["source" if i < n_source else "target" for i in range(n_source + len(target_z))],
            dtype=object,
        )
        domain_flags = domain_flags[idx]
        n_source = int((domain_flags == "source").sum())
    else:
        domain_flags = np.array(
            ["source"] * n_source + ["target"] * len(target_z),
            dtype=object,
        )

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, max(2, len(all_feats) - 1)),
        init="random",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(all_feats)
    emb_source = emb[domain_flags == "source"]
    emb_target = emb[domain_flags == "target"]
    labels_source = all_labels[domain_flags == "source"]
    labels_target = all_labels[domain_flags == "target"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 7))

    ax_a.scatter(
        emb_source[:, 0], emb_source[:, 1],
        c="#1f77b4", s=14, alpha=0.85, marker="o", edgecolors="k", linewidths=0.3,
        label=source_domain_label,
    )
    ax_a.scatter(
        emb_target[:, 0], emb_target[:, 1],
        c="#ff7f0e", s=12, alpha=0.55, marker="^", edgecolors="k", linewidths=0.3,
        label=target_domain_label,
    )
    ax_a.set_title("A. t-SNE by Domain (Source / Target)")
    ax_a.set_xlabel("Dimension 1")
    ax_a.set_ylabel("Dimension 2")
    ax_a.legend(loc="best", fontsize=8)
    ax_a.grid(alpha=0.2)

    all_unique = np.unique(all_labels)
    cmap = plt.cm.get_cmap("tab20", max(20, len(all_unique)))
    colors = {lab: cmap(i % cmap.N) for i, lab in enumerate(all_unique)}

    for lab in np.unique(labels_source):
        idx = np.where(labels_source == lab)[0]
        ax_b.scatter(
            emb_source[idx, 0], emb_source[idx, 1],
            c=[colors[lab]], s=14, alpha=0.85, marker="o", edgecolors="k", linewidths=0.3,
        )
    for lab in np.unique(labels_target):
        idx = np.where(labels_target == lab)[0]
        ax_b.scatter(
            emb_target[idx, 0], emb_target[idx, 1],
            c=[colors[lab]], s=12, alpha=target_alpha, marker="^", edgecolors="k", linewidths=0.3,
        )

    ax_b.set_title("B. t-SNE by Cancer Type")
    ax_b.set_xlabel("Dimension 1")
    ax_b.set_ylabel("Dimension 2")
    handles = []
    for lab in all_unique:
        name = mapping_int2str.get(int(lab), str(lab))
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="w", label=name,
                markerfacecolor=colors[lab], markersize=6,
            )
        )
    ax_b.legend(handles=handles, fontsize=7, loc="best", ncol=2)
    ax_b.grid(alpha=0.2)

    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close()
