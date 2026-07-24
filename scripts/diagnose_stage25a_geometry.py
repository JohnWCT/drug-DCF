#!/usr/bin/env python3
"""Stage 25A geometry / context readiness diagnostics (Round 25).

Reads completed Stage2 exp dirs from round25_stage25a_metrics.csv and writes:
  reports/round25_stage25a_geometry.json
  reports/round25_stage25a_context_readiness.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _eff_rank(x: np.ndarray, eps: float = 1e-8) -> float:
    """Effective rank via singular-value entropy."""
    if x.size == 0:
        return 0.0
    x = x - x.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(x, compute_uv=False)
    except Exception:
        return 0.0
    s = np.clip(s, 0, None)
    total = float(s.sum())
    if total <= eps:
        return 0.0
    p = s / total
    p = p[p > eps]
    ent = float(-(p * np.log(p)).sum())
    return float(math.exp(ent))


def _load_latent_dict(path: Path):
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def _latent_matrix(obj) -> np.ndarray:
    if isinstance(obj, dict):
        # common patterns: {id: vec} or {"latent": array, "label": ...}
        if "latent" in obj:
            return np.asarray(obj["latent"], dtype=np.float64)
        vals = []
        for v in obj.values():
            if isinstance(v, dict) and "latent" in v:
                vals.append(v["latent"])
            else:
                vals.append(v)
        return np.asarray(vals, dtype=np.float64)
    return np.asarray(obj, dtype=np.float64)


def diagnose_exp(exp_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"exp_dir": str(exp_dir), "ok": False}
    ccle_p = exp_dir / "ccle_latent_dict.pkl"
    tcga_p = exp_dir / "tcga_latent_dict.pkl"
    if not ccle_p.exists() or not tcga_p.exists():
        out["error"] = "missing latent pickles"
        return out
    try:
        ccle = _latent_matrix(_load_latent_dict(ccle_p))
        tcga = _latent_matrix(_load_latent_dict(tcga_p))
        if ccle.ndim == 1:
            ccle = ccle.reshape(1, -1)
        if tcga.ndim == 1:
            tcga = tcga.reshape(1, -1)
        out.update(
            {
                "ok": True,
                "source_n": int(ccle.shape[0]),
                "target_n": int(tcga.shape[0]),
                "latent_dim": int(ccle.shape[1]),
                "source_variance": float(np.var(ccle)),
                "target_variance": float(np.var(tcga)),
                "source_effective_rank": _eff_rank(ccle),
                "target_effective_rank": _eff_rank(tcga),
                "source_mean_norm": float(np.linalg.norm(ccle, axis=1).mean()),
                "target_mean_norm": float(np.linalg.norm(tcga, axis=1).mean()),
            }
        )
        # provisional 5-block raw context readiness proxy using mean prototypes
        s = ccle.mean(axis=0)
        t = tcga.mean(axis=0)
        # sample a subset of target for context rank
        idx = np.linspace(0, len(tcga) - 1, num=min(256, len(tcga)), dtype=int)
        z = tcga[idx]
        raw = np.concatenate(
            [
                np.repeat(s[None, :], len(z), axis=0),
                np.repeat(t[None, :], len(z), axis=0),
                np.repeat((s - t)[None, :], len(z), axis=0),
                z - s,
                z - t,
            ],
            axis=1,
        )
        out["raw_context_dim"] = int(raw.shape[1])
        out["raw_context_variance"] = float(np.var(raw))
        out["raw_context_effective_rank"] = _eff_rank(raw)
        # PCA32 explained variance ratio (source-only fit proxy on raw of source)
        from sklearn.decomposition import PCA

        pca = PCA(n_components=min(32, raw.shape[1], raw.shape[0]))
        # fit on source-constructed raw analog
        zs = ccle[np.linspace(0, len(ccle) - 1, num=min(256, len(ccle)), dtype=int)]
        raw_s = np.concatenate(
            [
                np.repeat(s[None, :], len(zs), axis=0),
                np.repeat(t[None, :], len(zs), axis=0),
                np.repeat((s - t)[None, :], len(zs), axis=0),
                zs - s,
                zs - t,
            ],
            axis=1,
        )
        pca.fit(raw_s)
        c32 = pca.transform(raw)
        out["pca32_explained_variance_sum"] = float(pca.explained_variance_ratio_.sum())
        out["c32_effective_rank"] = _eff_rank(c32)
        out["c32_variance"] = float(np.var(c32))
    except Exception as exc:
        out["error"] = str(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="reports/round25_stage25a_metrics.csv")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    metrics_path = ROOT / args.metrics
    if not metrics_path.exists():
        raise SystemExit(f"missing metrics: {metrics_path}")

    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    geometry = {"created_at": datetime.now(timezone.utc).isoformat(), "variants": {}}
    context = {"created_at": datetime.now(timezone.utc).isoformat(), "variants": {}}

    for row in rows:
        if row.get("status") != "DONE":
            continue
        exp = Path(row["exp_dir"])
        vid = row["variant"]
        seed = row["seed"]
        diag = diagnose_exp(exp)
        geometry["variants"].setdefault(vid, {})[str(seed)] = diag
        context["variants"].setdefault(vid, {})[str(seed)] = {
            k: diag.get(k)
            for k in (
                "ok",
                "raw_context_dim",
                "raw_context_variance",
                "raw_context_effective_rank",
                "pca32_explained_variance_sum",
                "c32_effective_rank",
                "c32_variance",
                "error",
            )
        }

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "round25_stage25a_geometry.json").write_text(
        json.dumps(geometry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "round25_stage25a_context_readiness.json").write_text(
        json.dumps(context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"geometry": "reports/round25_stage25a_geometry.json", "context": "reports/round25_stage25a_context_readiness.json", "n_rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
