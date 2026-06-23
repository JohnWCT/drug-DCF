"""Prototype-distance response features for Round 13 Step 2 predictor."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

SUPPORTED_MODES = frozenset(
    {"none", "own_cancer", "all_source_anchors", "all_source_and_target", "own_plus_summary"}
)
SUPPORTED_METRICS = frozenset({"cosine", "euclidean"})
SENTINEL_DISTANCE = 1.0


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
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode={mode!r}")

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

    feature_names = _feature_names_for_mode(
        mode,
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
    tgt_init = (
        np.asarray(target_initialized, dtype=bool)
        if target_initialized is not None
        else np.ones(len(target_prototypes or source_anchor_prototypes), dtype=bool)
    )
    target_prototypes = target_prototypes if target_prototypes is not None else source_anchor_prototypes

    metadata = {
        "mode": mode,
        "metric": metric,
        "feature_dim": n_feat,
        "latent_dim": int(z_arr.shape[1]),
        "num_cancer_types": len(cancer_names),
    }

    if mode == "none":
        features_out = features[0] if single else features
        return {"features": features_out, "feature_names": feature_names, "metadata": metadata}

    for row_idx, (vec, cid) in enumerate(zip(z_arr, cancer_arr)):
        cid_int = int(cid)
        if cid_int < 0 or cid_int >= len(source_anchor_prototypes):
            if strict:
                raise ValueError(f"Unknown cancer id {cid_int}")
            if mode in ("own_cancer", "own_plus_summary"):
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
        if mode in ("own_cancer", "own_plus_summary"):
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

        if mode in ("all_source_anchors", "all_source_and_target"):
            start_col = col
            for c_idx, _name in enumerate(cancer_names):
                s_vec, s_ok = _safe_vector(source_anchor_prototypes[c_idx], z_arr.shape[1])
                features[row_idx, start_col + c_idx] = (
                    _distance(vec, s_vec, metric)
                    if bool(src_init[c_idx]) and s_ok
                    else SENTINEL_DISTANCE
                )
            col = start_col + len(cancer_names)

        if mode == "all_source_and_target":
            start_col = col
            for c_idx, _name in enumerate(cancer_names):
                t_vec, t_ok = _safe_vector(target_prototypes[c_idx], z_arr.shape[1])
                features[row_idx, start_col + c_idx] = (
                    _distance(vec, t_vec, metric)
                    if bool(tgt_init[c_idx]) and t_ok
                    else SENTINEL_DISTANCE
                )
            col = start_col + len(cancer_names)

        if mode == "own_plus_summary":
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
