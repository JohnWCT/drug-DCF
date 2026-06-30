"""Prototype-distance response features for Round 13 Step 2 predictor."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

SUPPORTED_MODES = frozenset(
    {
        "none",
        "own_cancer",
        "all_source_anchors",
        "all_source_and_target",
        "own_plus_summary",
        "own_plus_summary_no_l2",
        "own_plus_summary_no_gap",
        "own_plus_summary_no_initialized_flags",
        "own_plus_summary_zscore",
        "own_plus_summary_robust_scaler",
        "own_proto_delta",
        "own_proto_context",
        "own_proto_context_projected_16",
        "own_proto_context_projected_32",
        "own_proto_interaction",
        "own_proto_delta_only",
        "own_plus_summary_plus_delta",
        "own_plus_summary_no_delta_control",
        "own_proto_delta_projected_16",
        "own_proto_delta_projected_32",
        "own_proto_delta_projected_8",
        "own_proto_delta_projected_64",
        "own_proto_delta_normed",
        "own_plus_summary_plus_delta_projected_16",
        "source_proto_delta_projected_16",
        "target_available_context_projected_16",
        "minimal_source_only_min_margin",
    }
)
OWN_PROTO_CONTEXT_MODES = frozenset(
    {
        "own_proto_delta",
        "own_proto_context",
        "own_proto_context_projected_16",
        "own_proto_context_projected_32",
        "own_proto_interaction",
    }
)
OWN_PROTO_DELTA_REPLACEMENT_MODES = frozenset(
    {
        "own_proto_delta_only",
        "own_plus_summary_plus_delta",
        "own_plus_summary_no_delta_control",
        "own_proto_delta_projected_16",
        "own_proto_delta_projected_32",
        "own_proto_delta_projected_8",
        "own_proto_delta_projected_64",
        "own_proto_delta_normed",
        "own_plus_summary_plus_delta_projected_16",
        "source_proto_delta_projected_16",
    }
)
ROUND17_STANDALONE_MODES = frozenset(
    {
        "target_available_context_projected_16",
        "minimal_source_only_min_margin",
    }
)
SUPPORTED_METRICS = frozenset({"cosine", "euclidean"})
SENTINEL_DISTANCE = 1.0


def normalize_feature_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode in OWN_PROTO_CONTEXT_MODES or mode in OWN_PROTO_DELTA_REPLACEMENT_MODES:
        return mode
    if mode.startswith("own_plus_summary"):
        return "own_plus_summary"
    return mode


def is_own_proto_delta_replacement_mode(mode: str) -> bool:
    return str(mode).lower() in OWN_PROTO_DELTA_REPLACEMENT_MODES


def is_round17_standalone_mode(mode: str) -> bool:
    return str(mode).lower() in ROUND17_STANDALONE_MODES


def is_round17_proto_feature_mode(mode: str) -> bool:
    mode = str(mode).lower()
    return (
        is_own_proto_delta_replacement_mode(mode)
        or is_round17_standalone_mode(mode)
        or mode in OWN_PROTO_DELTA_REPLACEMENT_MODES
        or mode in ROUND17_STANDALONE_MODES
    )


def get_projected_delta_dim(mode: str) -> int:
    mode = str(mode).lower()
    if mode == "own_proto_delta_projected_8":
        return 8
    if mode == "own_proto_delta_projected_16":
        return 16
    if mode == "own_proto_delta_projected_32":
        return 32
    if mode == "own_proto_delta_projected_64":
        return 64
    if mode in ("own_plus_summary_plus_delta_projected_16", "source_proto_delta_projected_16"):
        return 16
    return 0


def is_own_proto_context_mode(mode: str) -> bool:
    return str(mode).lower() in OWN_PROTO_CONTEXT_MODES


def get_projected_context_dim(mode: str) -> int:
    mode = str(mode).lower()
    if mode == "own_proto_context_projected_16":
        return 16
    if mode == "own_proto_context_projected_32":
        return 32
    if mode == "target_available_context_projected_16":
        return 16
    return 0


def get_projection_raw_kind(mode: str) -> str:
    """How to build the matrix row before PCA for Round 17 projection modes."""
    mode = str(mode).lower()
    if mode == "source_proto_delta_projected_16":
        return "source_delta_only"
    if mode == "target_available_context_projected_16":
        return "target_available_context"
    return "full_delta"


def parse_feature_variant(mode: str) -> dict:
    mode = str(mode).lower()
    return {
        "base_mode": normalize_feature_mode(mode),
        "drop_l2": mode == "own_plus_summary_no_l2",
        "drop_gap": mode == "own_plus_summary_no_gap",
        "drop_initialized_flags": mode == "own_plus_summary_no_initialized_flags",
        "scaler": (
            "robust"
            if mode == "own_plus_summary_robust_scaler"
            else "standard"
            if mode == "own_plus_summary_zscore"
            else "default"
        ),
    }


def resolve_feature_mode_options(
    feature_mode: str,
    *,
    include_l2_distance: bool = True,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = True,
    proto_feature_scaler: str = "standard",
) -> dict:
    variant = parse_feature_variant(feature_mode)
    if (
        is_own_proto_context_mode(feature_mode)
        or is_own_proto_delta_replacement_mode(feature_mode)
        or is_round17_standalone_mode(feature_mode)
    ):
        return {
            "mode": str(feature_mode).lower(),
            "include_l2_distance": include_l2_distance,
            "include_same_cancer_gap": include_same_cancer_gap,
            "include_initialized_flag": include_initialized_flag,
            "proto_feature_scaler": proto_feature_scaler,
            "feature_mode_label": str(feature_mode).lower(),
        }
    scaler = variant["scaler"] if variant["scaler"] != "default" else proto_feature_scaler
    return {
        "mode": variant["base_mode"],
        "include_l2_distance": False if variant["drop_l2"] else include_l2_distance,
        "include_same_cancer_gap": False if variant["drop_gap"] else include_same_cancer_gap,
        "include_initialized_flag": False if variant["drop_initialized_flags"] else include_initialized_flag,
        "proto_feature_scaler": scaler,
        "feature_mode_label": str(feature_mode).lower(),
    }


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float(SENTINEL_DISTANCE)
    return float(1.0 - np.dot(a, b) / (na * nb))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(a - b))


def _distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    metric = str(metric).lower()
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric={metric!r}")
    dist = cosine_distance(a, b) if metric == "cosine" else euclidean_distance(a, b)
    if not np.isfinite(dist):
        return float(SENTINEL_DISTANCE)
    return float(dist)


def _safe_vector(vec: Optional[np.ndarray], dim: int) -> Tuple[np.ndarray, bool]:
    if vec is None:
        return np.zeros(dim, dtype=np.float32), False
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] != dim:
        raise ValueError(f"Prototype dim {arr.shape[0]} != latent dim {dim}")
    if not np.all(np.isfinite(arr)):
        return np.zeros(dim, dtype=np.float32), False
    return arr, True


def get_own_source_target_vectors(
    sample_cancer_id: int,
    source_anchor_prototypes: np.ndarray,
    target_prototypes: Optional[np.ndarray],
    source_initialized: Optional[np.ndarray] = None,
    target_initialized: Optional[np.ndarray] = None,
    strict: bool = True,
    latent_dim: Optional[int] = None,
) -> dict:
    """Return own-cancer source anchor and target prototype vectors."""
    cid = int(sample_cancer_id)
    dim = int(latent_dim or source_anchor_prototypes.shape[1])
    src_init = (
        np.asarray(source_initialized, dtype=bool)
        if source_initialized is not None
        else np.ones(len(source_anchor_prototypes), dtype=bool)
    )
    tgt_protos = target_prototypes if target_prototypes is not None else source_anchor_prototypes
    tgt_init = (
        np.asarray(target_initialized, dtype=bool)
        if target_initialized is not None
        else np.ones(len(tgt_protos), dtype=bool)
    )

    if cid < 0 or cid >= len(source_anchor_prototypes):
        if strict:
            raise ValueError(f"Unknown cancer id {cid}")
        return {
            "source_anchor": np.zeros(dim, dtype=np.float32),
            "target_proto": np.zeros(dim, dtype=np.float32),
            "source_initialized": False,
            "target_initialized": False,
        }

    src_vec, src_ok = _safe_vector(source_anchor_prototypes[cid], dim)
    tgt_vec, tgt_ok = _safe_vector(tgt_protos[cid], dim)
    src_ready = bool(src_init[cid]) and src_ok
    tgt_ready = bool(tgt_init[cid]) and tgt_ok

    if strict:
        if not src_ready:
            raise ValueError(f"Missing source anchor prototype for cancer id {cid}")
        if not tgt_ready:
            raise ValueError(f"Missing target prototype for cancer id {cid}")

    if not src_ready:
        src_vec = np.zeros(dim, dtype=np.float32)
    if not tgt_ready:
        tgt_vec = np.zeros(dim, dtype=np.float32)

    return {
        "source_anchor": src_vec.astype(np.float32),
        "target_proto": tgt_vec.astype(np.float32),
        "source_initialized": src_ready,
        "target_initialized": tgt_ready,
    }


def get_own_cancer_prototype_vectors(
    cancer_id: int,
    source_anchor_prototypes: np.ndarray,
    target_prototypes: Optional[np.ndarray] = None,
    source_initialized: Optional[np.ndarray] = None,
    target_initialized: Optional[np.ndarray] = None,
    strict: bool = True,
    latent_dim: Optional[int] = None,
) -> dict:
    """Alias for get_own_source_target_vectors (Round 16F API)."""
    return get_own_source_target_vectors(
        cancer_id,
        source_anchor_prototypes,
        target_prototypes,
        source_initialized=source_initialized,
        target_initialized=target_initialized,
        strict=strict,
        latent_dim=latent_dim,
    )


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def compute_own_proto_delta_vectors(
    z: np.ndarray,
    source_anchor_c: np.ndarray,
    target_proto_c: np.ndarray,
    normalize: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    z_vec = np.asarray(z, dtype=np.float32).reshape(-1)
    src = np.asarray(source_anchor_c, dtype=np.float32).reshape(-1)
    tgt = np.asarray(target_proto_c, dtype=np.float32).reshape(-1)
    latent_dim = int(z_vec.shape[0])
    z_minus_source = z_vec - src
    z_minus_target = z_vec - tgt
    source_minus_target = src - tgt
    if normalize:
        z_minus_source = _l2_normalize(z_minus_source)
        z_minus_target = _l2_normalize(z_minus_target)
        source_minus_target = _l2_normalize(source_minus_target)
    features = np.concatenate([z_minus_source, z_minus_target, source_minus_target]).astype(np.float32)
    names = (
        _dim_feature_names("proto_delta_z_minus_source", latent_dim)
        + _dim_feature_names("proto_delta_z_minus_target", latent_dim)
        + _dim_feature_names("proto_delta_source_minus_target", latent_dim)
    )
    return features, names


def build_raw_delta_vector(
    z: np.ndarray,
    source_anchor_c: np.ndarray,
    target_proto_c: np.ndarray,
) -> np.ndarray:
    z_vec = np.asarray(z, dtype=np.float32).reshape(-1)
    delta, _ = compute_own_proto_delta_vectors(z_vec, source_anchor_c, target_proto_c, normalize=False)
    return delta


def build_source_only_delta_vector(
    z: np.ndarray,
    source_anchor_c: np.ndarray,
) -> np.ndarray:
    z_vec = np.asarray(z, dtype=np.float32).reshape(-1)
    src = np.asarray(source_anchor_c, dtype=np.float32).reshape(-1)
    return (z_vec - src).astype(np.float32)


def build_target_available_context_vector(
    z: np.ndarray,
    source_anchor_c: np.ndarray,
    target_proto_c: np.ndarray,
    target_available: bool,
) -> np.ndarray:
    z_vec = np.asarray(z, dtype=np.float32).reshape(-1)
    src = np.asarray(source_anchor_c, dtype=np.float32).reshape(-1)
    tgt = np.asarray(target_proto_c, dtype=np.float32).reshape(-1)
    if not target_available:
        tgt = src.copy()
    return build_raw_context_vector(z_vec, src, tgt)


def build_projection_raw_row(
    z: np.ndarray,
    source_anchor_c: np.ndarray,
    target_proto_c: np.ndarray,
    mode: str,
    target_available: bool = True,
) -> np.ndarray:
    kind = get_projection_raw_kind(mode)
    if kind == "source_delta_only":
        return build_source_only_delta_vector(z, source_anchor_c)
    if kind == "target_available_context":
        return build_target_available_context_vector(
            z, source_anchor_c, target_proto_c, target_available=target_available
        )
    return build_raw_delta_vector(z, source_anchor_c, target_proto_c)


def _dim_feature_names(prefix: str, dim: int) -> List[str]:
    return [f"{prefix}_dim{i:03d}" for i in range(dim)]


def _own_plus_summary_feature_names(
    cancer_names: Sequence[str],
    metric: str,
    include_l2_distance: bool,
    include_same_cancer_gap: bool,
    include_initialized_flag: bool,
) -> List[str]:
    return _feature_names_for_mode(
        "own_plus_summary",
        cancer_names,
        metric,
        include_l2_distance,
        include_same_cancer_gap,
        include_initialized_flag,
    )


def _feature_names_for_own_proto_context_mode(
    mode: str,
    latent_dim: int,
    cancer_names: Sequence[str],
    metric: str = "cosine",
    include_l2_distance: bool = True,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = True,
    projection_dim: int = 0,
) -> List[str]:
    mode = str(mode).lower()
    summary_names = _own_plus_summary_feature_names(
        cancer_names, metric, include_l2_distance, include_same_cancer_gap, include_initialized_flag
    )
    if mode == "own_proto_delta":
        names = (
            _dim_feature_names("proto_delta_z_minus_source", latent_dim)
            + _dim_feature_names("proto_delta_z_minus_target", latent_dim)
            + _dim_feature_names("proto_delta_source_minus_target", latent_dim)
            + summary_names
        )
        return names
    if mode == "own_proto_context":
        names = (
            _dim_feature_names("proto_context_source", latent_dim)
            + _dim_feature_names("proto_context_target", latent_dim)
            + _dim_feature_names("proto_context_source_minus_target", latent_dim)
            + summary_names
        )
        return names
    if mode in ("own_proto_context_projected_16", "own_proto_context_projected_32"):
        pfx = f"proto_context_pca{projection_dim}"
        return _dim_feature_names(pfx, projection_dim) + summary_names
    if mode == "own_proto_interaction":
        names = (
            _dim_feature_names("proto_interact_z_times_source", latent_dim)
            + _dim_feature_names("proto_interact_z_times_target", latent_dim)
            + _dim_feature_names("proto_delta_z_minus_source", latent_dim)
            + _dim_feature_names("proto_delta_z_minus_target", latent_dim)
            + summary_names
        )
        return names
    raise ValueError(f"Unsupported own_proto_context mode={mode!r}")


def build_raw_context_vector(
    z: np.ndarray,
    source_anchor_c: np.ndarray,
    target_proto_c: np.ndarray,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    src = np.asarray(source_anchor_c, dtype=np.float32).reshape(-1)
    tgt = np.asarray(target_proto_c, dtype=np.float32).reshape(-1)
    return np.concatenate([src, tgt, src - tgt, z - src, z - tgt]).astype(np.float32)


def compute_own_plus_summary_vector(
    z: np.ndarray,
    cancer_id: int,
    source_anchor_prototypes: np.ndarray,
    target_prototypes: Optional[np.ndarray],
    cancer_type_mapping: Optional[Dict],
    metric: str = "cosine",
    include_l2_distance: bool = True,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = True,
    strict: bool = False,
    source_initialized: Optional[np.ndarray] = None,
    target_initialized: Optional[np.ndarray] = None,
) -> np.ndarray:
    pack = compute_proto_distance_features(
        z,
        cancer_id,
        source_anchor_prototypes,
        target_prototypes=target_prototypes,
        cancer_type_mapping=cancer_type_mapping,
        mode="own_plus_summary",
        metric=metric,
        include_l2_distance=include_l2_distance,
        include_same_cancer_gap=include_same_cancer_gap,
        include_initialized_flag=include_initialized_flag,
        strict=strict,
        source_initialized=source_initialized,
        target_initialized=target_initialized,
    )
    return np.asarray(pack["features"], dtype=np.float32).reshape(-1)


def compute_own_proto_context_features(
    z: np.ndarray,
    cancer_id: int,
    source_anchor_prototypes: np.ndarray,
    target_prototypes: Optional[np.ndarray] = None,
    mode: str = "own_proto_delta",
    own_summary_features: Optional[np.ndarray] = None,
    cancer_type_mapping: Optional[Dict] = None,
    metric: str = "cosine",
    include_l2_distance: bool = True,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = True,
    source_initialized: Optional[np.ndarray] = None,
    target_initialized: Optional[np.ndarray] = None,
    projection_model: Optional[object] = None,
    strict: bool = True,
) -> Tuple[np.ndarray, List[str], dict]:
    """Compute extra (non-z) own-prototype context features for one sample."""
    mode = str(mode).lower()
    if mode not in OWN_PROTO_CONTEXT_MODES:
        raise ValueError(f"Unsupported own_proto_context mode={mode!r}")

    z_vec = np.asarray(z, dtype=np.float32).reshape(-1)
    latent_dim = int(z_vec.shape[0])
    vecs = get_own_source_target_vectors(
        cancer_id,
        source_anchor_prototypes,
        target_prototypes,
        source_initialized=source_initialized,
        target_initialized=target_initialized,
        strict=strict,
        latent_dim=latent_dim,
    )
    src = vecs["source_anchor"]
    tgt = vecs["target_proto"]

    if own_summary_features is None:
        own_summary_features = compute_own_plus_summary_vector(
            z_vec,
            cancer_id,
            source_anchor_prototypes,
            target_prototypes,
            cancer_type_mapping,
            metric=metric,
            include_l2_distance=include_l2_distance,
            include_same_cancer_gap=include_same_cancer_gap,
            include_initialized_flag=include_initialized_flag,
            strict=strict,
            source_initialized=source_initialized,
            target_initialized=target_initialized,
        )
    own_summary_features = np.asarray(own_summary_features, dtype=np.float32).reshape(-1)

    mapping = cancer_type_mapping or {}
    id_to_name = {int(k): str(v) for k, v in mapping.get("id_to_name", {}).items()}
    if not id_to_name:
        id_to_name = {i: str(i) for i in range(len(source_anchor_prototypes))}
    cancer_names = [id_to_name[i] for i in sorted(id_to_name.keys())]
    proj_dim = get_projected_context_dim(mode)

    if mode == "own_proto_delta":
        features = np.concatenate([z_vec - src, z_vec - tgt, src - tgt, own_summary_features])
    elif mode == "own_proto_context":
        features = np.concatenate([src, tgt, src - tgt, own_summary_features])
    elif mode in ("own_proto_context_projected_16", "own_proto_context_projected_32"):
        if projection_model is None:
            raise ValueError(f"projection_model required for mode={mode}")
        raw_context = build_raw_context_vector(z_vec, src, tgt).reshape(1, -1)
        projected = projection_model.transform(raw_context).reshape(-1)
        features = np.concatenate([projected, own_summary_features])
        proj_dim = int(projected.shape[0])
    elif mode == "own_proto_interaction":
        features = np.concatenate([z_vec * src, z_vec * tgt, z_vec - src, z_vec - tgt, own_summary_features])
    else:
        raise ValueError(f"Unsupported mode={mode!r}")

    feature_names = _feature_names_for_own_proto_context_mode(
        mode,
        latent_dim,
        cancer_names,
        metric,
        include_l2_distance,
        include_same_cancer_gap,
        include_initialized_flag,
        projection_dim=proj_dim,
    )
    if len(feature_names) != len(features):
        raise ValueError(f"feature_names length {len(feature_names)} != features length {len(features)}")

    metadata = {
        "mode": mode,
        "latent_dim": latent_dim,
        "feature_dim": int(len(features)),
        "projection_dim": proj_dim,
        "includes_own_plus_summary": True,
    }
    return features.astype(np.float32), feature_names, metadata


def compute_own_proto_context_features_batch(
    z: np.ndarray,
    cancer_ids: Sequence[int],
    source_anchor_prototypes: np.ndarray,
    target_prototypes: Optional[np.ndarray] = None,
    mode: str = "own_proto_delta",
    cancer_type_mapping: Optional[Dict] = None,
    metric: str = "cosine",
    include_l2_distance: bool = True,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = True,
    source_initialized: Optional[np.ndarray] = None,
    target_initialized: Optional[np.ndarray] = None,
    projection_model: Optional[object] = None,
    strict: bool = False,
) -> Dict:
    z_arr = np.asarray(z, dtype=np.float32)
    if z_arr.ndim == 1:
        z_arr = z_arr.reshape(1, -1)
    cancer_arr = np.asarray(cancer_ids)
    rows = []
    names: Optional[List[str]] = None
    meta: dict = {}
    for vec, cid in zip(z_arr, cancer_arr):
        feat, feat_names, row_meta = compute_own_proto_context_features(
            vec,
            int(cid),
            source_anchor_prototypes,
            target_prototypes=target_prototypes,
            mode=mode,
            cancer_type_mapping=cancer_type_mapping,
            metric=metric,
            include_l2_distance=include_l2_distance,
            include_same_cancer_gap=include_same_cancer_gap,
            include_initialized_flag=include_initialized_flag,
            source_initialized=source_initialized,
            target_initialized=target_initialized,
            projection_model=projection_model,
            strict=strict,
        )
        rows.append(feat)
        names = feat_names
        meta = row_meta
    features = np.stack(rows, axis=0)
    return {"features": features, "feature_names": names or [], "metadata": meta}


def _delta_replacement_flags(mode: str) -> dict:
    mode = str(mode).lower()
    projected_delta_modes = (
        "own_proto_delta_projected_8",
        "own_proto_delta_projected_16",
        "own_proto_delta_projected_32",
        "own_proto_delta_projected_64",
        "own_plus_summary_plus_delta_projected_16",
        "source_proto_delta_projected_16",
    )
    return {
        "uses_own_plus_summary": mode
        in (
            "own_plus_summary",
            "own_plus_summary_no_delta_control",
            "own_plus_summary_plus_delta",
            "own_plus_summary_plus_delta_projected_16",
        ),
        "uses_delta": mode
        in (
            "own_proto_delta_only",
            "own_plus_summary_plus_delta",
            "own_proto_delta_normed",
            *projected_delta_modes,
        ),
        "uses_projection": mode in projected_delta_modes,
    }


def compute_own_proto_delta_replacement_features(
    z: np.ndarray,
    cancer_id: int,
    source_anchor_prototypes: np.ndarray,
    target_prototypes: Optional[np.ndarray] = None,
    mode: str = "own_proto_delta_only",
    cancer_type_mapping: Optional[Dict] = None,
    metric: str = "cosine",
    include_l2_distance: bool = True,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = True,
    source_initialized: Optional[np.ndarray] = None,
    target_initialized: Optional[np.ndarray] = None,
    projection_model: Optional[object] = None,
    strict: bool = True,
) -> Tuple[np.ndarray, List[str], dict]:
    """Compute extra (non-z) features for Round 16F delta replacement / ablation modes."""
    feature_mode_label = str(mode).lower()
    mode = feature_mode_label
    if mode == "own_plus_summary_no_delta_control":
        mode = "own_plus_summary"
    if mode not in OWN_PROTO_DELTA_REPLACEMENT_MODES and mode != "own_plus_summary":
        raise ValueError(f"Unsupported delta replacement mode={mode!r}")

    z_vec = np.asarray(z, dtype=np.float32).reshape(-1)
    latent_dim = int(z_vec.shape[0])
    vecs = get_own_source_target_vectors(
        cancer_id,
        source_anchor_prototypes,
        target_prototypes,
        source_initialized=source_initialized,
        target_initialized=target_initialized,
        strict=strict,
        latent_dim=latent_dim,
    )
    src = vecs["source_anchor"]
    tgt = vecs["target_proto"]

    mapping = cancer_type_mapping or {}
    id_to_name = {int(k): str(v) for k, v in mapping.get("id_to_name", {}).items()}
    if not id_to_name:
        id_to_name = {i: str(i) for i in range(len(source_anchor_prototypes))}
    cancer_names = [id_to_name[i] for i in sorted(id_to_name.keys())]
    summary_names = _own_plus_summary_feature_names(
        cancer_names, metric, include_l2_distance, include_same_cancer_gap, include_initialized_flag
    )

    if mode == "own_plus_summary":
        summary = compute_own_plus_summary_vector(
            z_vec,
            cancer_id,
            source_anchor_prototypes,
            target_prototypes,
            cancer_type_mapping,
            metric=metric,
            include_l2_distance=include_l2_distance,
            include_same_cancer_gap=include_same_cancer_gap,
            include_initialized_flag=include_initialized_flag,
            strict=strict,
            source_initialized=source_initialized,
            target_initialized=target_initialized,
        )
        features = summary
        feature_names = summary_names
        proj_dim = 0
    else:
        normalize_delta = mode == "own_proto_delta_normed"
        delta_features, delta_names = compute_own_proto_delta_vectors(z_vec, src, tgt, normalize=normalize_delta)
        proj_dim = get_projected_delta_dim(mode)

        if mode == "own_proto_delta_only":
            features = delta_features
            feature_names = delta_names
        elif mode == "own_plus_summary_plus_delta":
            summary = compute_own_plus_summary_vector(
                z_vec,
                cancer_id,
                source_anchor_prototypes,
                target_prototypes,
                cancer_type_mapping,
                metric=metric,
                include_l2_distance=include_l2_distance,
                include_same_cancer_gap=include_same_cancer_gap,
                include_initialized_flag=include_initialized_flag,
                strict=strict,
                source_initialized=source_initialized,
                target_initialized=target_initialized,
            )
            features = np.concatenate([summary, delta_features])
            feature_names = summary_names + delta_names
        elif mode in (
            "own_proto_delta_projected_8",
            "own_proto_delta_projected_16",
            "own_proto_delta_projected_32",
            "own_proto_delta_projected_64",
            "source_proto_delta_projected_16",
        ):
            if projection_model is None:
                raise ValueError(f"projection_model required for mode={mode}")
            raw_delta = build_projection_raw_row(z_vec, src, tgt, mode).reshape(1, -1)
            projected = projection_model.transform(raw_delta).reshape(-1)
            proj_dim = int(projected.shape[0])
            features = projected.astype(np.float32)
            feature_names = _dim_feature_names(f"proto_delta_pca{proj_dim}", proj_dim)
        elif mode == "own_plus_summary_plus_delta_projected_16":
            if projection_model is None:
                raise ValueError(f"projection_model required for mode={mode}")
            summary = compute_own_plus_summary_vector(
                z_vec,
                cancer_id,
                source_anchor_prototypes,
                target_prototypes,
                cancer_type_mapping,
                metric=metric,
                include_l2_distance=include_l2_distance,
                include_same_cancer_gap=include_same_cancer_gap,
                include_initialized_flag=include_initialized_flag,
                strict=strict,
                source_initialized=source_initialized,
                target_initialized=target_initialized,
            )
            raw_delta = build_raw_delta_vector(z_vec, src, tgt).reshape(1, -1)
            projected = projection_model.transform(raw_delta).reshape(-1)
            proj_dim = int(projected.shape[0])
            features = np.concatenate([summary, projected.astype(np.float32)])
            feature_names = summary_names + _dim_feature_names(f"proto_delta_pca{proj_dim}", proj_dim)
        elif mode == "own_proto_delta_normed":
            features = delta_features
            feature_names = delta_names
        else:
            raise ValueError(f"Unsupported mode={mode!r}")

    if len(feature_names) != len(features):
        raise ValueError(f"feature_names length {len(feature_names)} != features length {len(features)}")

    flags = _delta_replacement_flags(feature_mode_label)
    metadata = {
        "mode": feature_mode_label,
        "latent_dim": latent_dim,
        "feature_dim": int(len(features)),
        "projection_dim": proj_dim,
        **flags,
    }
    return features.astype(np.float32), feature_names, metadata


def compute_own_proto_delta_replacement_features_batch(
    z: np.ndarray,
    cancer_ids: Sequence[int],
    source_anchor_prototypes: np.ndarray,
    target_prototypes: Optional[np.ndarray] = None,
    mode: str = "own_proto_delta_only",
    cancer_type_mapping: Optional[Dict] = None,
    metric: str = "cosine",
    include_l2_distance: bool = True,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = True,
    source_initialized: Optional[np.ndarray] = None,
    target_initialized: Optional[np.ndarray] = None,
    projection_model: Optional[object] = None,
    strict: bool = False,
) -> Dict:
    z_arr = np.asarray(z, dtype=np.float32)
    if z_arr.ndim == 1:
        z_arr = z_arr.reshape(1, -1)
    cancer_arr = np.asarray(cancer_ids)
    rows = []
    names: Optional[List[str]] = None
    meta: dict = {}
    for vec, cid in zip(z_arr, cancer_arr):
        feat, feat_names, row_meta = compute_own_proto_delta_replacement_features(
            vec,
            int(cid),
            source_anchor_prototypes,
            target_prototypes=target_prototypes,
            mode=mode,
            cancer_type_mapping=cancer_type_mapping,
            metric=metric,
            include_l2_distance=include_l2_distance,
            include_same_cancer_gap=include_same_cancer_gap,
            include_initialized_flag=include_initialized_flag,
            source_initialized=source_initialized,
            target_initialized=target_initialized,
            projection_model=projection_model,
            strict=strict,
        )
        rows.append(feat)
        names = feat_names
        meta = row_meta
    features = np.stack(rows, axis=0)
    return {"features": features, "feature_names": names or [], "metadata": meta}


def compute_round17_standalone_features(
    z: np.ndarray,
    cancer_id: int,
    source_anchor_prototypes: np.ndarray,
    target_prototypes: Optional[np.ndarray] = None,
    mode: str = "minimal_source_only_min_margin",
    cancer_type_mapping: Optional[Dict] = None,
    metric: str = "cosine",
    include_l2_distance: bool = True,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = False,
    source_initialized: Optional[np.ndarray] = None,
    target_initialized: Optional[np.ndarray] = None,
    projection_model: Optional[object] = None,
    strict: bool = True,
) -> Tuple[np.ndarray, List[str], dict]:
    """Round 17 standalone proto feature modes (not delta-replacement family)."""
    feature_mode_label = str(mode).lower()
    if feature_mode_label not in ROUND17_STANDALONE_MODES:
        raise ValueError(f"Unsupported round17 standalone mode={feature_mode_label!r}")

    z_vec = np.asarray(z, dtype=np.float32).reshape(-1)
    latent_dim = int(z_vec.shape[0])
    vecs = get_own_source_target_vectors(
        cancer_id,
        source_anchor_prototypes,
        target_prototypes,
        source_initialized=source_initialized,
        target_initialized=target_initialized,
        strict=strict,
        latent_dim=latent_dim,
    )
    src = vecs["source_anchor"]
    tgt = vecs["target_proto"]
    tgt_available = bool(vecs.get("target_initialized", True))

    if feature_mode_label == "minimal_source_only_min_margin":
        summary_pack = compute_proto_distance_features(
            z_vec,
            cancer_id,
            source_anchor_prototypes,
            target_prototypes=target_prototypes,
            cancer_type_mapping=cancer_type_mapping,
            mode="own_plus_summary",
            metric=metric,
            include_l2_distance=include_l2_distance,
            include_same_cancer_gap=include_same_cancer_gap,
            include_initialized_flag=False,
            strict=strict,
            source_initialized=source_initialized,
            target_initialized=target_initialized,
        )
        full_names = list(summary_pack["feature_names"])
        summary_vals = np.asarray(summary_pack["features"], dtype=np.float32).reshape(-1)
        feature_names = [
            "proto_own_source_cosine_dist",
            "proto_source_min_dist",
            "proto_source_top1_margin",
        ]
        features = np.asarray(
            [summary_vals[full_names.index(name)] for name in feature_names],
            dtype=np.float32,
        )
        proj_dim = 0
    elif feature_mode_label == "target_available_context_projected_16":
        if projection_model is None:
            raise ValueError(f"projection_model required for mode={feature_mode_label}")
        raw_context = build_projection_raw_row(
            z_vec, src, tgt, feature_mode_label, target_available=tgt_available
        ).reshape(1, -1)
        projected = projection_model.transform(raw_context).reshape(-1)
        proj_dim = int(projected.shape[0])
        features = projected.astype(np.float32)
        feature_names = _dim_feature_names(f"proto_context_pca{proj_dim}", proj_dim)
    else:
        raise ValueError(f"Unsupported mode={feature_mode_label!r}")

    metadata = {
        "mode": feature_mode_label,
        "latent_dim": latent_dim,
        "feature_dim": int(len(features)),
        "projection_dim": proj_dim,
        "target_proto_available": tgt_available,
        "uses_own_plus_summary": feature_mode_label == "minimal_source_only_min_margin",
        "uses_projection": feature_mode_label == "target_available_context_projected_16",
    }
    return features.astype(np.float32), feature_names, metadata


def compute_round17_standalone_features_batch(
    z: np.ndarray,
    cancer_ids: Sequence[int],
    source_anchor_prototypes: np.ndarray,
    target_prototypes: Optional[np.ndarray] = None,
    mode: str = "minimal_source_only_min_margin",
    cancer_type_mapping: Optional[Dict] = None,
    metric: str = "cosine",
    include_l2_distance: bool = True,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = False,
    source_initialized: Optional[np.ndarray] = None,
    target_initialized: Optional[np.ndarray] = None,
    projection_model: Optional[object] = None,
    strict: bool = False,
) -> Dict:
    z_arr = np.asarray(z, dtype=np.float32)
    if z_arr.ndim == 1:
        z_arr = z_arr.reshape(1, -1)
    cancer_arr = np.asarray(cancer_ids)
    rows = []
    names: Optional[List[str]] = None
    meta: dict = {}
    for vec, cid in zip(z_arr, cancer_arr):
        feat, feat_names, row_meta = compute_round17_standalone_features(
            vec,
            int(cid),
            source_anchor_prototypes,
            target_prototypes=target_prototypes,
            mode=mode,
            cancer_type_mapping=cancer_type_mapping,
            metric=metric,
            include_l2_distance=include_l2_distance,
            include_same_cancer_gap=include_same_cancer_gap,
            include_initialized_flag=include_initialized_flag,
            source_initialized=source_initialized,
            target_initialized=target_initialized,
            projection_model=projection_model,
            strict=strict,
        )
        rows.append(feat)
        names = feat_names
        meta = row_meta
    features = np.stack(rows, axis=0)
    return {"features": features, "feature_names": names or [], "metadata": meta}


def fit_context_projection(
    raw_context_matrix: np.ndarray,
    n_components: int,
    random_state: int = 42,
):
    from sklearn.decomposition import PCA

    mat = np.asarray(raw_context_matrix, dtype=np.float64)
    n_components = min(int(n_components), mat.shape[0], mat.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(mat)
    return pca


def _feature_names_for_mode(
    mode: str,
    cancer_names: Sequence[str],
    metric: str,
    include_l2_distance: bool,
    include_same_cancer_gap: bool,
    include_initialized_flag: bool,
) -> List[str]:
    mode = str(mode).lower()
    if mode == "none":
        return []
    if mode == "own_cancer":
        names = [f"proto_own_source_{metric}_dist"]
        if include_l2_distance:
            names.append("proto_own_source_l2_dist")
        names.append(f"proto_own_target_{metric}_dist")
        if include_l2_distance:
            names.append("proto_own_target_l2_dist")
        if include_same_cancer_gap:
            names.append("proto_same_cancer_gap")
        if include_initialized_flag:
            names.extend(["proto_source_anchor_initialized", "proto_target_proto_initialized"])
        return names
    if mode == "all_source_anchors":
        return [f"proto_source_{metric}_dist_{name}" for name in cancer_names]
    if mode == "all_source_and_target":
        src = [f"proto_source_{metric}_dist_{name}" for name in cancer_names]
        tgt = [f"proto_target_{metric}_dist_{name}" for name in cancer_names]
        return src + tgt
    if mode == "own_plus_summary":
        base = _feature_names_for_mode(
            "own_cancer",
            cancer_names,
            metric,
            include_l2_distance,
            include_same_cancer_gap,
            include_initialized_flag,
        )
        base.extend(
            [
                "proto_source_min_dist",
                "proto_source_top1_margin",
                "proto_source_mean_dist",
                "proto_source_std_dist",
            ]
        )
        return base
    raise ValueError(f"Unsupported mode={mode!r}")


def compute_proto_distance_features(
    z: Union[np.ndarray, Sequence[float]],
    cancer_ids: Union[int, Sequence[int]],
    source_anchor_prototypes: np.ndarray,
    target_prototypes: Optional[np.ndarray] = None,
    cancer_type_mapping: Optional[Dict] = None,
    mode: str = "own_cancer",
    metric: str = "cosine",
    include_l2_distance: bool = False,
    include_same_cancer_gap: bool = True,
    include_initialized_flag: bool = True,
    normalize: bool = False,
    strict: bool = False,
    source_initialized: Optional[np.ndarray] = None,
    target_initialized: Optional[np.ndarray] = None,
) -> Dict:
    """Compute prototype-distance features for one or many samples."""
    del normalize  # scaler applied in extraction stage
    mode = str(mode).lower()
    base_mode = normalize_feature_mode(mode)
    if base_mode not in SUPPORTED_MODES and mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode={mode!r}")

    variant = parse_feature_variant(mode)
    if variant["drop_l2"]:
        include_l2_distance = False
    if variant["drop_gap"]:
        include_same_cancer_gap = False
    if variant["drop_initialized_flags"]:
        include_initialized_flag = False

    z_arr = np.asarray(z, dtype=np.float32)
    single = z_arr.ndim == 1
    if single:
        z_arr = z_arr.reshape(1, -1)

    cancer_arr = np.asarray(cancer_ids)
    if single and cancer_arr.ndim == 0:
        cancer_arr = cancer_arr.reshape(1)
    if len(cancer_arr) != len(z_arr):
        raise ValueError("cancer_ids length must match batch size")

    mapping = cancer_type_mapping or {}
    id_to_name = {int(k): str(v) for k, v in mapping.get("id_to_name", {}).items()}
    if not id_to_name:
        id_to_name = {i: str(i) for i in range(len(source_anchor_prototypes))}
    cancer_names = [id_to_name[i] for i in sorted(id_to_name.keys())]

    base_mode = normalize_feature_mode(mode)
    feature_names = _feature_names_for_mode(
        base_mode,
        cancer_names,
        metric,
        include_l2_distance,
        include_same_cancer_gap,
        include_initialized_flag,
    )
    n_feat = len(feature_names)
    features = np.zeros((len(z_arr), n_feat), dtype=np.float32)

    src_init = (
        np.asarray(source_initialized, dtype=bool)
        if source_initialized is not None
        else np.ones(len(source_anchor_prototypes), dtype=bool)
    )
    proto_for_len = target_prototypes if target_prototypes is not None else source_anchor_prototypes
    tgt_init = (
        np.asarray(target_initialized, dtype=bool)
        if target_initialized is not None
        else np.ones(len(proto_for_len), dtype=bool)
    )
    target_prototypes = proto_for_len

    metadata = {
        "mode": mode,
        "metric": metric,
        "feature_dim": n_feat,
        "latent_dim": int(z_arr.shape[1]),
        "num_cancer_types": len(cancer_names),
    }

    if base_mode == "none":
        features_out = features[0] if single else features
        return {"features": features_out, "feature_names": feature_names, "metadata": metadata}

    for row_idx, (vec, cid) in enumerate(zip(z_arr, cancer_arr)):
        cid_int = int(cid)
        if cid_int < 0 or cid_int >= len(source_anchor_prototypes):
            if strict:
                raise ValueError(f"Unknown cancer id {cid_int}")
            if base_mode in ("own_cancer", "own_plus_summary"):
                features[row_idx, :] = SENTINEL_DISTANCE
                if include_initialized_flag and "proto_source_anchor_initialized" in feature_names:
                    init_idx = feature_names.index("proto_source_anchor_initialized")
                    features[row_idx, init_idx] = 0.0
                    features[row_idx, init_idx + 1] = 0.0
            continue

        src_vec, src_ok = _safe_vector(source_anchor_prototypes[cid_int], z_arr.shape[1])
        tgt_vec, tgt_ok = _safe_vector(target_prototypes[cid_int], z_arr.shape[1])
        src_ready = bool(src_init[cid_int]) and src_ok
        tgt_ready = bool(tgt_init[cid_int]) and tgt_ok

        col = 0
        if base_mode in ("own_cancer", "own_plus_summary"):
            src_cos = _distance(vec, src_vec, metric) if src_ready else SENTINEL_DISTANCE
            features[row_idx, col] = src_cos
            col += 1
            if include_l2_distance:
                features[row_idx, col] = (
                    euclidean_distance(vec, src_vec) if src_ready else SENTINEL_DISTANCE
                )
                col += 1
            tgt_cos = _distance(vec, tgt_vec, metric) if tgt_ready else SENTINEL_DISTANCE
            features[row_idx, col] = tgt_cos
            col += 1
            if include_l2_distance:
                features[row_idx, col] = (
                    euclidean_distance(vec, tgt_vec) if tgt_ready else SENTINEL_DISTANCE
                )
                col += 1
            if include_same_cancer_gap:
                gap = _distance(src_vec, tgt_vec, metric) if src_ready and tgt_ready else SENTINEL_DISTANCE
                features[row_idx, col] = gap
                col += 1
            if include_initialized_flag:
                features[row_idx, col] = float(src_ready)
                col += 1
                features[row_idx, col] = float(tgt_ready)
                col += 1

        if base_mode in ("all_source_anchors", "all_source_and_target"):
            start_col = col
            for c_idx, _name in enumerate(cancer_names):
                s_vec, s_ok = _safe_vector(source_anchor_prototypes[c_idx], z_arr.shape[1])
                features[row_idx, start_col + c_idx] = (
                    _distance(vec, s_vec, metric)
                    if bool(src_init[c_idx]) and s_ok
                    else SENTINEL_DISTANCE
                )
            col = start_col + len(cancer_names)

        if base_mode == "all_source_and_target":
            start_col = col
            for c_idx, _name in enumerate(cancer_names):
                t_vec, t_ok = _safe_vector(target_prototypes[c_idx], z_arr.shape[1])
                features[row_idx, start_col + c_idx] = (
                    _distance(vec, t_vec, metric)
                    if bool(tgt_init[c_idx]) and t_ok
                    else SENTINEL_DISTANCE
                )
            col = start_col + len(cancer_names)

        if base_mode == "own_plus_summary":
            dists = []
            for c_idx in range(len(cancer_names)):
                s_vec, s_ok = _safe_vector(source_anchor_prototypes[c_idx], z_arr.shape[1])
                if bool(src_init[c_idx]) and s_ok:
                    dists.append(_distance(vec, s_vec, metric))
            if dists:
                dists_arr = np.asarray(dists, dtype=np.float64)
                sorted_d = np.sort(dists_arr)
                min_d = float(sorted_d[0])
                margin = float(sorted_d[1] - sorted_d[0]) if len(sorted_d) > 1 else 0.0
                mean_d = float(np.mean(dists_arr))
                std_d = float(np.std(dists_arr))
            else:
                min_d = margin = mean_d = std_d = SENTINEL_DISTANCE
            features[row_idx, col : col + 4] = [min_d, margin, mean_d, std_d]

    if single:
        features = features[0]
    return {"features": features, "feature_names": feature_names, "metadata": metadata}


def concat_latent_and_proto_features(
    latent_z: np.ndarray,
    proto_pack: Dict,
) -> np.ndarray:
    proto = np.asarray(proto_pack["features"], dtype=np.float32)
    single = proto.ndim == 1
    if single:
        proto = proto.reshape(1, -1)
    latent = np.asarray(latent_z, dtype=np.float32)
    latent_single = latent.ndim == 1
    if latent_single:
        latent = latent.reshape(1, -1)
    if proto.shape[0] != latent.shape[0]:
        raise ValueError("Latent and prototype feature batch sizes differ")
    if proto.shape[1] == 0:
        out = latent
    else:
        out = np.concatenate([latent, proto], axis=1)
    if single or latent_single:
        return out[0]
    return out
