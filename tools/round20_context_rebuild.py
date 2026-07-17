#!/usr/bin/env python3
"""Rebuild Round 20 context32 from the same raw context population as C16.

Contract:
- same checkpoint / prototypes / latent rows
- same raw context definition
- same PCA algorithm + random_state=42
- only n_components: 16 -> 32
- do not overwrite Round 17/19 artifacts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from tools.extract_round13_proto_features import (
    _filter_latents_to_trainable,
    _load_or_extract_prototypes,
    _sample_cancer_id,
)
from tools.prototype_response_features import (
    build_raw_context_vector,
    fit_context_projection,
    get_own_source_target_vectors,
)
from tools.round9_diagnostics_common import (
    _load_cancer_maps,
    _load_latent_dict,
    find_latent_paths,
    resolve_path,
)
from tools.round20_context_adapter import audit_context_pair

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_C16_SOURCE = (
    PROJECT_ROOT
    / "result/optimization_runs/round17r_18class/features/r13_exp_008/own_proto_context_projected_16"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "result/optimization_runs/round12_proto_alignment/pretrain/exp_008"
)
DEFAULT_OUT_ROOT = (
    PROJECT_ROOT / "result/optimization_runs/round20_unseen_drug_closure"
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return _sha256_bytes(arr.tobytes() + str(arr.shape).encode())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct_raw_context_matrix(
    checkpoint_dir: Path,
    *,
    proto_cache_dir: Path,
) -> Tuple[np.ndarray, list, dict]:
    checkpoint_dir = Path(resolve_path(str(checkpoint_dir)))
    source_pkl, _target_pkl = find_latent_paths(str(checkpoint_dir))
    if not source_pkl:
        raise FileNotFoundError(f"Missing CCLE latent under {checkpoint_dir}")
    ccle_latent = _load_latent_dict(source_pkl)
    proto = _load_or_extract_prototypes(str(checkpoint_dir), str(proto_cache_dir), strict=False)
    mapping = proto["cancer_type_mapping"]
    name_to_id = mapping.get("name_to_id", {})
    ccle_map, tcga_map = _load_cancer_maps()
    ccle_latent, _ = _filter_latents_to_trainable(
        ccle_latent, {}, mapping, ccle_map=ccle_map, tcga_map=tcga_map
    )
    sample_ids = list(ccle_latent.keys())
    ccle_ids = [
        _sample_cancer_id(sid, "source", ccle_map, tcga_map, name_to_id) for sid in sample_ids
    ]
    ccle_z = np.stack([ccle_latent[sid] for sid in sample_ids], axis=0)
    latent_dim = int(ccle_z.shape[1])
    raw_rows = []
    for vec, cid in zip(ccle_z, ccle_ids):
        vecs = get_own_source_target_vectors(
            int(cid),
            proto["source_anchor_prototypes"],
            proto["target_prototypes"],
            source_initialized=proto["source_initialized"],
            target_initialized=proto["target_initialized"],
            strict=False,
            latent_dim=latent_dim,
        )
        raw_rows.append(build_raw_context_vector(vec, vecs["source_anchor"], vecs["target_proto"]))
    raw_mat = np.stack(raw_rows, axis=0)
    meta = {
        "n_rows": int(raw_mat.shape[0]),
        "input_dim": int(raw_mat.shape[1]),
        "latent_dim": latent_dim,
        "sample_ids_hash": _sha256_bytes("\n".join(map(str, sample_ids)).encode()),
        "raw_matrix_sha256": _sha256_array(raw_mat),
        "checkpoint_dir": str(checkpoint_dir),
        "proto_cache_dir": str(proto_cache_dir),
        "n_trainable_cancer_types": int(mapping.get("num_cancer_types", 0)),
    }
    return raw_mat, sample_ids, meta


def verify_c16_projection(
    raw_mat: np.ndarray,
    c16_projection_pkl: Path,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict:
    with c16_projection_pkl.open("rb") as f:
        legacy = pickle.load(f)
    refit = fit_context_projection(raw_mat, 16, random_state=42)
    legacy_t = legacy.transform(raw_mat)
    refit_t = refit.transform(raw_mat)
    max_abs = float(np.max(np.abs(legacy_t - refit_t)))
    ok = bool(np.allclose(legacy_t, refit_t, atol=atol, rtol=rtol))
    # Component sign flips are possible across sklearn versions; transform agreement is the contract.
    return {
        "ok": ok,
        "max_abs_transform_delta": max_abs,
        "legacy_explained_variance_ratio_sum": float(np.sum(legacy.explained_variance_ratio_)),
        "refit_explained_variance_ratio_sum": float(np.sum(refit.explained_variance_ratio_)),
        "legacy_n_components": int(legacy.n_components_),
        "refit_n_components": int(refit.n_components_),
        "random_state": 42,
    }


def _archive_projection(
    out_dir: Path,
    *,
    projection_model,
    metadata: dict,
    attestation: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "projection.pkl").open("wb") as f:
        pickle.dump(projection_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    _write_json(out_dir / "metadata.json", metadata)
    _write_json(out_dir / "comparability_attestation.json", attestation)
    sha = _sha256_file(out_dir / "projection.pkl")
    (out_dir / "sha256.txt").write_text(sha + "\n", encoding="utf-8")


def _build_o2_feature_dir(
    *,
    full_feature_pkl: Path,
    full_feature_names: list,
    context_dim: int,
    out_dir: Path,
    source_artifact: str,
    attestation: dict,
    projection_pkl: Path,
) -> dict:
    with full_feature_pkl.open("rb") as f:
        full = pickle.load(f)
    if not isinstance(full, dict):
        raise TypeError(full_feature_pkl)
    expected = 64 + int(context_dim)
    # Full projected_* layout: Z(64) + PCA(context_dim) + summary(11)
    o2 = {}
    for mid, vec in full.items():
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] < expected:
            raise ValueError(f"{mid} dim {arr.shape[0]} < {expected}")
        o2[str(mid)] = arr[:expected].astype(np.float32)
    names = list(full_feature_names[:expected])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ccle_latent_proto.pkl").open("wb") as f:
        pickle.dump(o2, f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copy2(projection_pkl, out_dir / "projection_model.pkl")
    proj_hash = _sha256_file(out_dir / "projection_model.pkl")
    meta = {
        "mode": f"own_proto_context_projected_{context_dim}_no_summary",
        "display_name": f"z_plus_context{context_dim}",
        "latent_dim": 64,
        "summary_dim": 0,
        "context_dim": int(context_dim),
        "response_input_dim": expected,
        "includes_own_plus_summary": False,
        "includes_projected_context": True,
        "source_artifact": source_artifact,
        "o2_projection_sha256": proj_hash,
        "n_ids": len(o2),
        "round20_rebuild": True,
        "comparability_attestation": attestation,
    }
    _write_json(out_dir / "feature_metadata.json", meta)
    _write_json(out_dir / "feature_names.json", names)
    _write_json(
        out_dir / "projection_metadata.json",
        {
            "projection_type": "pca",
            "fit_domain": "source_only",
            "input_dim": int(attestation["raw_context_input_dim"]),
            "requested_output_dim": int(context_dim),
            "output_dim": int(context_dim),
            "explained_variance_ratio_sum": attestation.get(
                f"c{context_dim}_explained_variance_ratio_sum"
            ),
            "random_state": 42,
        },
    )
    _write_json(out_dir / "comparability_attestation.json", attestation)
    return meta


def rebuild_context32(
    *,
    c16_source_dir: Path,
    checkpoint_dir: Path,
    out_root: Path,
    random_state: int = 42,
) -> dict:
    c16_source_dir = Path(c16_source_dir)
    checkpoint_dir = Path(checkpoint_dir)
    out_root = Path(out_root)
    proj_root = out_root / "projections"
    feat_root = out_root / "features"
    audit_root = out_root / "audit"
    for p in (proj_root, feat_root, audit_root):
        p.mkdir(parents=True, exist_ok=True)

    # Reuse C16 proto cache to keep prototype tensors identical.
    proto_cache = c16_source_dir / "_proto_cache"
    if not proto_cache.is_dir():
        raise FileNotFoundError(f"Missing C16 proto cache: {proto_cache}")

    raw_mat, sample_ids, raw_meta = reconstruct_raw_context_matrix(
        checkpoint_dir, proto_cache_dir=proto_cache
    )
    c16_proj = c16_source_dir / "projection_model.pkl"
    verify = verify_c16_projection(raw_mat, c16_proj)
    if not verify["ok"]:
        raise RuntimeError(
            "C16 projection verification failed against reconstructed raw matrix: "
            f"{verify}"
        )

    pca16 = fit_context_projection(raw_mat, 16, random_state=random_state)
    pca32 = fit_context_projection(raw_mat, 32, random_state=random_state)

    attestation = {
        "schema": "round20_context_comparability_attestation",
        "schema_version": 1,
        "raw_context_definition": {
            "vector": "concat(source, target, source-target, z-source, z-target)",
            "input_dim": 320,
            "projection_type": "pca",
            "fit_domain": "source_only",
            "random_state": int(random_state),
        },
        "raw_context_input_dim": int(raw_mat.shape[1]),
        "raw_matrix_sha256": raw_meta["raw_matrix_sha256"],
        "sample_ids_hash": raw_meta["sample_ids_hash"],
        "n_rows": raw_meta["n_rows"],
        "checkpoint_dir": raw_meta["checkpoint_dir"],
        "proto_cache_dir": raw_meta["proto_cache_dir"],
        "c16_source_dir": str(c16_source_dir),
        "c16_projection_verify": verify,
        "c16_explained_variance_ratio_sum": float(np.sum(pca16.explained_variance_ratio_)),
        "c32_explained_variance_ratio_sum": float(np.sum(pca32.explained_variance_ratio_)),
        "fit_population_hash": _sha256_bytes(
            json.dumps(
                {
                    "raw_matrix_sha256": raw_meta["raw_matrix_sha256"],
                    "sample_ids_hash": raw_meta["sample_ids_hash"],
                    "fit_domain": "source_only",
                    "n_rows": raw_meta["n_rows"],
                },
                sort_keys=True,
            ).encode()
        ),
        "raw_context_definition_hash": _sha256_bytes(
            json.dumps(
                {
                    "vector": "concat(source, target, source-target, z-source, z-target)",
                    "input_dim": 320,
                    "projection_type": "pca",
                    "fit_domain": "source_only",
                },
                sort_keys=True,
            ).encode()
        ),
        "normalization_hash": _sha256_bytes(
            json.dumps(
                {
                    "projection_type": "pca",
                    "fit_domain": "source_only",
                    "random_state": int(random_state),
                    "sklearn_pca_whiten": False,
                },
                sort_keys=True,
            ).encode()
        ),
        "source_encoder_checkpoint_hash": _sha256_bytes(raw_meta["checkpoint_dir"].encode()),
    }

    # Archive projections
    _archive_projection(
        proj_root / "context16",
        projection_model=pca16,
        metadata={
            "projection_type": "pca",
            "fit_domain": "source_only",
            "input_dim": 320,
            "requested_output_dim": 16,
            "output_dim": 16,
            "explained_variance_ratio_sum": attestation["c16_explained_variance_ratio_sum"],
            "random_state": random_state,
            "role": "reference_refit_from_same_raw_matrix",
        },
        attestation=attestation,
    )
    # Also keep a copy of the legacy C16 projection for provenance.
    shutil.copy2(c16_proj, proj_root / "context16" / "legacy_projection_model.pkl")

    _archive_projection(
        proj_root / "context32",
        projection_model=pca32,
        metadata={
            "projection_type": "pca",
            "fit_domain": "source_only",
            "input_dim": 320,
            "requested_output_dim": 32,
            "output_dim": 32,
            "explained_variance_ratio_sum": attestation["c32_explained_variance_ratio_sum"],
            "random_state": random_state,
            "role": "round20_rebuild",
        },
        attestation=attestation,
    )

    # Build O2-C32 directly: Z + StandardScaler(PCA32(raw)).
    # This mirrors Round 19 O2 (Z + scaled PCA context, no summary) with n_components=32.
    source_pkl, _ = find_latent_paths(str(Path(resolve_path(str(checkpoint_dir)))))
    ccle_latent = _load_latent_dict(source_pkl)
    o2_c32: Dict[str, np.ndarray] = {}
    ctx32 = pca32.transform(raw_mat).astype(np.float32)
    for i, sid in enumerate(sample_ids):
        z = np.asarray(ccle_latent[sid], dtype=np.float32).reshape(-1)[:64]
        if z.shape != (64,):
            raise ValueError(f"Z dim for {sid}: {z.shape}")
        o2_c32[str(sid)] = np.concatenate([z, ctx32[i]], axis=0)

    from sklearn.preprocessing import StandardScaler

    ctx_mat = np.stack([o2_c32[str(sid)][64:] for sid in sample_ids], axis=0)
    scaler = StandardScaler()
    ctx_scaled = scaler.fit_transform(ctx_mat).astype(np.float32)
    for i, sid in enumerate(sample_ids):
        o2_c32[str(sid)] = np.concatenate([o2_c32[str(sid)][:64], ctx_scaled[i]], axis=0)

    # For C16 reference feature dir under Round20, copy existing Round19 O2 (do not mutate).
    c16_feat = feat_root / "z_plus_context16"
    if c16_feat.exists():
        shutil.rmtree(c16_feat)
    shutil.copytree(
        PROJECT_ROOT / "result/optimization_runs/round19_factorial/features/z_plus_context16",
        c16_feat,
    )
    _write_json(c16_feat / "comparability_attestation.json", attestation)
    if not (c16_feat / "projection_metadata.json").is_file():
        shutil.copy2(
            c16_source_dir / "projection_metadata.json",
            c16_feat / "projection_metadata.json",
        )
    if not (c16_feat / "projection_model.pkl").is_file():
        shutil.copy2(c16_proj, c16_feat / "projection_model.pkl")
    c16_meta = _load_json(c16_feat / "feature_metadata.json")
    c16_meta["comparability_attestation"] = attestation
    c16_meta["round20_attestation_attached"] = True
    _write_json(c16_feat / "feature_metadata.json", c16_meta)

    # Write C32 O2 feature dir
    c32_feat = feat_root / "z_plus_context32"
    if c32_feat.exists():
        shutil.rmtree(c32_feat)
    names32 = [f"z_dim{i:03d}" for i in range(64)] + [
        f"proto_context_pca32_dim{i:03d}" for i in range(32)
    ]
    c32_feat.mkdir(parents=True, exist_ok=True)
    with (c32_feat / "ccle_latent_proto.pkl").open("wb") as f:
        pickle.dump(o2_c32, f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copy2(proj_root / "context32" / "projection.pkl", c32_feat / "projection_model.pkl")
    meta32 = {
        "mode": "own_proto_context_projected_32_no_summary",
        "display_name": "z_plus_context32",
        "latent_dim": 64,
        "summary_dim": 0,
        "context_dim": 32,
        "response_input_dim": 96,
        "includes_own_plus_summary": False,
        "includes_projected_context": True,
        "source_artifact": str(proj_root / "context32"),
        "o2_projection_sha256": _sha256_file(c32_feat / "projection_model.pkl"),
        "n_ids": len(o2_c32),
        "round20_rebuild": True,
        "context_scaler": {
            "type": "standard",
            "fit_on": "context32_only_source_population",
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "comparability_attestation": attestation,
        "note": (
            "C32 rebuilt from same raw context matrix as C16; PCA random_state=42; "
            "O2 uses Z + StandardScaler(context32) without summary."
        ),
    }
    _write_json(c32_feat / "feature_metadata.json", meta32)
    _write_json(c32_feat / "feature_names.json", names32)
    _write_json(
        c32_feat / "projection_metadata.json",
        {
            "projection_type": "pca",
            "fit_domain": "source_only",
            "input_dim": 320,
            "requested_output_dim": 32,
            "output_dim": 32,
            "explained_variance_ratio_sum": attestation["c32_explained_variance_ratio_sum"],
            "random_state": random_state,
        },
    )
    _write_json(c32_feat / "comparability_attestation.json", attestation)

    # Coverage check
    c16_ids = set(pickle.load(open(c16_feat / "ccle_latent_proto.pkl", "rb")).keys())
    c32_ids = set(o2_c32.keys())
    if c16_ids != c32_ids:
        raise RuntimeError(
            f"ModelID coverage mismatch: c16={len(c16_ids)} c32={len(c32_ids)} "
            f"only16={len(c16_ids-c32_ids)} only32={len(c32_ids-c16_ids)}"
        )

    report = {
        "schema": "round20_context32_rebuild_report",
        "ok": True,
        "attestation": attestation,
        "c16_feature_dir": str(c16_feat),
        "c32_feature_dir": str(c32_feat),
        "projections": {
            "context16": str(proj_root / "context16"),
            "context32": str(proj_root / "context32"),
        },
        "n_ids": len(c32_ids),
        "c16_verify": verify,
    }
    _write_json(audit_root / "context32_rebuild_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c16-source-dir", default=str(DEFAULT_C16_SOURCE))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--audit-out",
        default=str(DEFAULT_OUT_ROOT / "audit" / "context_audit.json"),
    )
    args = parser.parse_args()
    report = rebuild_context32(
        c16_source_dir=Path(args.c16_source_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        out_root=Path(args.out_root),
        random_state=int(args.random_state),
    )
    audit = audit_context_pair(
        c16_dir=report["c16_feature_dir"],
        c32_dir=report["c32_feature_dir"],
        out=args.audit_out,
        fail_closed=False,
    )
    print(
        json.dumps(
            {
                "rebuild_ok": report["ok"],
                "audit_comparable": audit["comparable"],
                "mismatches": audit["mismatches"],
                "c32_feature_dir": report["c32_feature_dir"],
                "audit_out": args.audit_out,
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["ok"] and audit["comparable"] else 4)


if __name__ == "__main__":
    main()
