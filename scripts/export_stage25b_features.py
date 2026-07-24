#!/usr/bin/env python3
"""Export Round25 Stage25B features in Round19/20-compatible layout.

Writes feature_dir with:
  ccle_latent_proto.pkl  (ModelID -> float32[96] = Z64||C32)
  feature_metadata.json  (summary_dim=0, response_input_dim=96)

Raw context contract:
  r = [s, t, s-t, z-s, z-t]  (320-d)
  PCA/scaler fit on source only → C32
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def _as_latent_dict(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    out: Dict[str, np.ndarray] = {}
    for k, v in obj.items():
        out[str(k)] = np.asarray(v, dtype=np.float64).reshape(-1)
    return out


def _stack(d: Dict[str, np.ndarray]) -> Tuple[np.ndarray, list]:
    ids = sorted(d.keys())
    x = np.stack([d[i] for i in ids], axis=0)
    return x, ids


def raw_context(z: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    s2 = np.repeat(s.reshape(1, -1), len(z), axis=0)
    t2 = np.repeat(t.reshape(1, -1), len(z), axis=0)
    return np.concatenate([s2, t2, s2 - t2, z - s2, z - t2], axis=1)


def export_one(exp_dir: Path, out_dir: Path, tag: str) -> dict:
    ccle = _as_latent_dict(exp_dir / "ccle_latent_dict.pkl")
    tcga = _as_latent_dict(exp_dir / "tcga_latent_dict.pkl")
    zs, s_ids = _stack(ccle)
    zt, t_ids = _stack(tcga)
    if zs.shape[1] != 64:
        raise ValueError(f"expected Z64, got {zs.shape}")

    s_anchor = zs.mean(axis=0)
    t_proto = zt.mean(axis=0)
    raw_s = raw_context(zs, s_anchor, t_proto)
    raw_t = raw_context(zt, s_anchor, t_proto)

    scaler = StandardScaler()
    raw_s_sc = scaler.fit_transform(raw_s)
    raw_t_sc = scaler.transform(raw_t)
    pca = PCA(n_components=32, random_state=42)
    c32_s = pca.fit_transform(raw_s_sc)
    c32_t = pca.transform(raw_t_sc)

    feat96_s = np.concatenate([zs, c32_s], axis=1).astype(np.float32)
    feat96_t = np.concatenate([zt, c32_t], axis=1).astype(np.float32)
    # XA training uses CCLE ModelIDs; also keep TCGA ids for optional diagnostics.
    latent_proto = {sid: feat96_s[i] for i, sid in enumerate(s_ids)}
    for i, tid in enumerate(t_ids):
        latent_proto[tid] = feat96_t[i]

    z_only = {}
    zeros32 = np.zeros((32,), dtype=np.float32)
    for i, sid in enumerate(s_ids):
        # Keep 96-d wire format for collate; C32 zeros → model type biocda_xa_z_only ignores context.
        z_only[sid] = np.concatenate([zs[i].astype(np.float32), zeros32], axis=0)
    for i, tid in enumerate(t_ids):
        z_only[tid] = np.concatenate([zt[i].astype(np.float32), zeros32], axis=0)

    out_c32 = out_dir / "z_plus_context32"
    out_z = out_dir / "z_only"
    out_c32.mkdir(parents=True, exist_ok=True)
    out_z.mkdir(parents=True, exist_ok=True)

    with (out_c32 / "ccle_latent_proto.pkl").open("wb") as f:
        pickle.dump(latent_proto, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (out_z / "ccle_latent_proto.pkl").open("wb") as f:
        pickle.dump(z_only, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta_c32 = {
        "display_name": "z_plus_context32",
        "latent_dim": 64,
        "context_dim": 32,
        "summary_dim": 0,
        "response_input_dim": 96,
        "n_ids": len(latent_proto),
        "n_trainable_cancer_types": 18,
        "prototype_class_source": "checkpoint_metadata",
        "uses_legacy_28class_cache": False,
        "includes_own_plus_summary": False,
        "includes_projected_context": True,
        "mode": "round25_stage25b_pca32_no_summary",
        "scaler_fit_domain": "source_only",
        "pca_fit_domain": "source_only",
        "pca_explained_variance_sum": float(pca.explained_variance_ratio_.sum()),
        "round25_tag": tag,
        "source_exp_dir": str(exp_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_context_definition": {
            "vector": "concat(source, target, source-target, z-source, z-target)",
            "input_dim": 320,
            "projection_type": "pca",
            "fit_domain": "source_only",
            "random_state": 42,
        },
    }
    meta_z = dict(meta_c32)
    meta_z.update(
        {
            "display_name": "z_only",
            "context_dim": 32,
            "response_input_dim": 96,
            "includes_projected_context": False,
            "mode": "round25_stage25b_z_only_zero_c32",
            "n_ids": len(z_only),
            "note": "C32 channels explicitly zeroed; XA model_type=biocda_xa_z_only ignores context",
        }
    )
    (out_c32 / "feature_metadata.json").write_text(json.dumps(meta_c32, indent=2) + "\n", encoding="utf-8")
    (out_z / "feature_metadata.json").write_text(json.dumps(meta_z, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "pca_scaler.pkl").open("wb") as f:
        pickle.dump({"pca": pca, "scaler": scaler}, f)

    # hashes
    blob = pickle.dumps(latent_proto)
    meta_c32["feature_sha256"] = hashlib.sha256(blob).hexdigest()
    (out_c32 / "feature_metadata.json").write_text(json.dumps(meta_c32, indent=2) + "\n", encoding="utf-8")
    return meta_c32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", default="reports/round25_stage25a_decision.json")
    ap.add_argument("--metrics", default="reports/round25_stage25a_metrics.csv")
    ap.add_argument("--out-root", default="result/optimization_runs/round25_stage25b/features")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    decision = json.loads((ROOT / args.decision).read_text(encoding="utf-8"))
    selected = decision.get("selected_variant") or "S0"
    rows = list(csv.DictReader((ROOT / args.metrics).open(encoding="utf-8")))

    def pick(vid: str):
        cands = [r for r in rows if r["variant"] == vid and r["status"] == "DONE"]
        if not cands:
            return None
        return sorted(cands, key=lambda r: int(r["seed"]))[0]

    b0 = pick("S0")
    b1 = pick(selected)
    if b0 is None or b1 is None:
        raise SystemExit(f"missing DONE exps for B0/S0 or B1/{selected}")

    out_root = ROOT / args.out_root
    lineage = {"created_at": datetime.now(timezone.utc).isoformat(), "selected_variant": selected, "exports": {}}
    for tag, row in [("B0_S0", b0), (f"B1_{selected}", b1)]:
        meta = export_one(Path(row["exp_dir"]), out_root / tag, tag)
        lineage["exports"][tag] = {
            "exp_dir": row["exp_dir"],
            "seed": row["seed"],
            "feature_dir": str((out_root / tag / "z_plus_context32").relative_to(ROOT)),
            "z_only_dir": str((out_root / tag / "z_only").relative_to(ROOT)),
            "meta": meta,
        }
        print(json.dumps({"exported": tag, "n_ids": meta["n_ids"], "pca_var": meta["pca_explained_variance_sum"]}))

    (ROOT / "reports/round25_artifact_lineage.json").write_text(
        json.dumps(lineage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
