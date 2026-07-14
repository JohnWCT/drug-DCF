"""Round 19 omics composition feature builder (O0–O4)."""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Existing Round 17R artifact layout for O3:
#   [Z(64) | context16(16) | summary(11)] = 91
# Round 19 O2 is Z|context (80) derived without refitting PCA.


OMICS_ALIAS = {
    "O0": "z_only",
    "O1": "z_plus_summary",
    "O2": "z_plus_context16",
    "O3": "z_plus_summary_context16",
    "O4": "z_plus_source_proto_features",
}

LEGACY_MODE_DIRS = {
    "O0": "none",
    "O1": "own_plus_summary",
    "O3": "own_proto_context_projected_16",
}

SOURCE_SUMMARY_NAMES = [
    "proto_own_source_cosine_dist",
    "proto_own_source_l2_dist",
    "proto_source_anchor_initialized",
    "proto_source_min_dist",
    "proto_source_top1_margin",
    "proto_source_mean_dist",
    "proto_source_std_dist",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_pkl(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict pickle at {path}")
    return {str(k): np.asarray(v, dtype=np.float32) for k, v in obj.items()}


def _write_feature_dir(
    out_dir: Path,
    features: Dict[str, np.ndarray],
    *,
    metadata: Dict,
    feature_names: List[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ccle_latent_proto.pkl").open("wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")


def default_r17_feature_root() -> Path:
    return Path("result/optimization_runs/round17r_18class/features/r13_exp_008")


def build_o2_from_o3(
    feature_root: Path,
    out_root: Path,
) -> Dict:
    """O2 = Z + projected context16 using the same O3 artifact / PCA (no refit)."""
    o3_dir = Path(feature_root) / LEGACY_MODE_DIRS["O3"]
    o3 = _load_pkl(o3_dir / "ccle_latent_proto.pkl")
    names = json.loads((o3_dir / "feature_names.json").read_text(encoding="utf-8"))
    if len(names) != 91:
        raise ValueError(f"Expected O3 feature_names length 91, got {len(names)}")
    # Existing layout: z(64) + context(16) + summary(11)
    o2 = {k: v[:80].astype(np.float32) for k, v in o3.items()}
    for v in o2.values():
        if v.shape != (80,):
            raise ValueError(f"O2 vector must be 80-d, got {v.shape}")
    proj_src = o3_dir / "projection_model.pkl"
    out_dir = Path(out_root) / OMICS_ALIAS["O2"]
    if proj_src.is_file():
        shutil.copy2(proj_src, out_dir.parent / "_tmp_projection_model.pkl")
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out_dir.parent / "_tmp_projection_model.pkl"), str(out_dir / "projection_model.pkl"))
        proj_hash = _sha256_file(out_dir / "projection_model.pkl")
        o3_hash = _sha256_file(proj_src)
        if proj_hash != o3_hash:
            raise RuntimeError("O2/O3 projection_model hash mismatch after copy")
    else:
        proj_hash = None
        o3_hash = None

    feature_names = list(names[:80])
    meta = {
        "mode": "own_proto_context_projected_16_no_summary",
        "display_name": "z_plus_context16",
        "latent_dim": 64,
        "summary_dim": 0,
        "context_dim": 16,
        "response_input_dim": 80,
        "includes_own_plus_summary": False,
        "includes_projected_context": True,
        "source_artifact": str(o3_dir),
        "o3_projection_sha256": o3_hash,
        "o2_projection_sha256": proj_hash,
        "n_ids": len(o2),
    }
    _write_feature_dir(out_dir, o2, metadata=meta, feature_names=feature_names)
    return meta


def build_o4_source_only_stub(
    feature_root: Path,
    out_root: Path,
) -> Dict:
    """
    O4 smoke artifact: Z + source-only summary scalars (no target prototype fields).

    Full source-only projected context16 can be refined later; Stage 19A QC requires
    no target prototype columns and identical ModelID coverage.
    """
    root = Path(feature_root)
    z = _load_pkl(root / LEGACY_MODE_DIRS["O0"] / "ccle_latent_proto.pkl")
    o1 = _load_pkl(root / LEGACY_MODE_DIRS["O1"] / "ccle_latent_proto.pkl")
    o1_names = json.loads((root / LEGACY_MODE_DIRS["O1"] / "feature_names.json").read_text(encoding="utf-8"))
    # O1 layout: Z(64)+summary(11); summary names are the 11 proto_* fields
    summary_names = o1_names if len(o1_names) == 11 else o1_names[64:]
    name_to_idx = {n: i for i, n in enumerate(summary_names)}
    missing = [n for n in SOURCE_SUMMARY_NAMES if n not in name_to_idx]
    if missing:
        raise KeyError(f"Missing source summary names in O1: {missing}")

    ids = sorted(set(z) & set(o1))
    if len(ids) != len(z) or len(ids) != len(o1):
        raise ValueError("O0/O1 ModelID sets differ for O4 build")

    out: Dict[str, np.ndarray] = {}
    feat_names = [f"z_dim{i:03d}" for i in range(64)] + list(SOURCE_SUMMARY_NAMES)
    banned_substrings = ("target", "tgt")
    for n in SOURCE_SUMMARY_NAMES:
        low = n.lower()
        if any(b in low for b in banned_substrings):
            raise AssertionError(f"O4 source feature looks target-related: {n}")

    for mid in ids:
        z_vec = z[mid]
        if z_vec.shape[0] != 64:
            # none may already be 64
            z_vec = z_vec[:64]
        summary = o1[mid]
        if summary.shape[0] == 75:
            summary = summary[64:]
        elif summary.shape[0] != 11:
            raise ValueError(f"Unexpected O1 vector dim {summary.shape} for {mid}")
        src = np.asarray([summary[name_to_idx[n]] for n in SOURCE_SUMMARY_NAMES], dtype=np.float32)
        out[mid] = np.concatenate([z_vec.astype(np.float32), src], axis=0)

    out_dir = Path(out_root) / OMICS_ALIAS["O4"]
    meta = {
        "mode": "z_plus_source_proto_features",
        "display_name": "z_plus_source_proto_features",
        "latent_dim": 64,
        "source_summary_dim": len(SOURCE_SUMMARY_NAMES),
        "response_input_dim": 64 + len(SOURCE_SUMMARY_NAMES),
        "includes_own_plus_summary": False,
        "includes_projected_context": False,
        "includes_target_prototype_fields": False,
        "n_ids": len(out),
        "note": "19A smoke O4 = Z + source-only summary scalars; source-only PCA16 to be added later",
    }
    _write_feature_dir(out_dir, out, metadata=meta, feature_names=feat_names)
    return meta


def symlink_or_copy_legacy(
    feature_root: Path,
    out_root: Path,
    omics_id: str,
) -> Path:
    legacy = LEGACY_MODE_DIRS[omics_id]
    src = Path(feature_root) / legacy
    dst = Path(out_root) / OMICS_ALIAS[omics_id]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copytree(src, dst)
    return dst


def build_round19_feature_set(
    *,
    feature_root: Optional[str] = None,
    out_root: str = "result/optimization_runs/round19_factorial/features",
) -> Dict:
    fr = Path(feature_root) if feature_root else default_r17_feature_root()
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    for oid in ("O0", "O1", "O3"):
        paths[oid] = str(symlink_or_copy_legacy(fr, out, oid))
    o2_meta = build_o2_from_o3(fr, out)
    o4_meta = build_o4_source_only_stub(fr, out)
    paths["O2"] = str(out / OMICS_ALIAS["O2"])
    paths["O4"] = str(out / OMICS_ALIAS["O4"])

    # Coverage QC
    id_sets = {}
    for oid, p in paths.items():
        feats = _load_pkl(Path(p) / "ccle_latent_proto.pkl")
        id_sets[oid] = set(feats)
        arr0 = next(iter(feats.values()))
        expected = {
            "O0": 64,
            "O1": 75,
            "O2": 80,
            "O3": 91,
            "O4": 64 + len(SOURCE_SUMMARY_NAMES),
        }[oid]
        if arr0.shape[-1] != expected:
            raise ValueError(f"{oid} dim {arr0.shape[-1]} != expected {expected}")
        if not np.isfinite(arr0).all():
            raise ValueError(f"{oid} contains non-finite values")

    base = id_sets["O0"]
    for oid, s in id_sets.items():
        if s != base:
            raise ValueError(f"ModelID set mismatch for {oid}: {len(s)} vs {len(base)}")

    # O2/O3 context hash identity via shared projection file when present
    o2_proj = Path(paths["O2"]) / "projection_model.pkl"
    o3_proj = Path(paths["O3"]) / "projection_model.pkl"
    if o2_proj.is_file() and o3_proj.is_file():
        if _sha256_file(o2_proj) != _sha256_file(o3_proj):
            raise RuntimeError("O2/O3 projection_model hashes differ")

    report = {
        "feature_root": str(fr),
        "out_root": str(out),
        "paths": paths,
        "n_ids": len(base),
        "o2_meta": o2_meta,
        "o4_meta": o4_meta,
        "dims": {"O0": 64, "O1": 75, "O2": 80, "O3": 91, "O4": 64 + len(SOURCE_SUMMARY_NAMES)},
    }
    (out / "round19_feature_build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def resolve_omics_dim(omics_id: str) -> int:
    return {
        "O0": 64,
        "O1": 75,
        "O2": 80,
        "O3": 91,
        "O4": 64 + len(SOURCE_SUMMARY_NAMES),
        "z_only": 64,
        "z_plus_summary": 75,
        "z_plus_context16": 80,
        "z_plus_summary_context16": 91,
        "z_plus_source_proto_features": 64 + len(SOURCE_SUMMARY_NAMES),
    }[omics_id]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--out-root", default="result/optimization_runs/round19_factorial/features")
    args = parser.parse_args()
    rep = build_round19_feature_set(feature_root=args.feature_root, out_root=args.out_root)
    print(json.dumps(rep, indent=2))
