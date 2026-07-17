#!/usr/bin/env python3
"""Round 20 O2 context feature adapter and C16/C32 comparability audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FeatureShapeError(ValueError):
    """Raised when an O2 feature vector has the wrong shape."""


class ContextAuditError(RuntimeError):
    """Raised when C16/C32 artifacts are not comparable."""


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


def inspect_context_dir(feature_dir: Path | str, *, expected_context_dim: int) -> dict[str, Any]:
    directory = Path(feature_dir)
    report: dict[str, Any] = {
        "path": str(directory),
        "exists": directory.is_dir(),
        "expected_context_dim": int(expected_context_dim),
        "ok": False,
        "issues": [],
    }
    if not directory.is_dir():
        report["issues"].append("directory_missing")
        return report

    meta_path = directory / "feature_metadata.json"
    names_path = directory / "feature_names.json"
    proj_path = directory / "projection_model.pkl"
    proj_meta_path = directory / "projection_metadata.json"
    latent_path = directory / "ccle_latent_proto.pkl"

    if not meta_path.is_file():
        report["issues"].append("feature_metadata_missing")
        return report

    meta = _load_json(meta_path)
    report["feature_metadata"] = meta
    attestation = meta.get("comparability_attestation")
    att_path = directory / "comparability_attestation.json"
    if attestation is None and att_path.is_file():
        attestation = _load_json(att_path)
    report["comparability_attestation"] = attestation
    latent_dim = int(meta.get("latent_dim", -1))
    context_dim = int(meta.get("context_dim", -1))
    response_dim = int(meta.get("response_input_dim", latent_dim + context_dim))
    if latent_dim != 64:
        report["issues"].append(f"latent_dim={latent_dim}!=64")
    if context_dim != int(expected_context_dim):
        report["issues"].append(
            f"context_dim={context_dim}!=expected={expected_context_dim}"
        )
    if response_dim != 64 + int(expected_context_dim):
        report["issues"].append(
            f"response_input_dim={response_dim}!=expected={64 + expected_context_dim}"
        )

    feature_names = None
    if names_path.is_file():
        feature_names = _load_json(names_path)
        report["feature_order_hash"] = _stable_hash(feature_names)
        report["n_feature_names"] = (
            len(feature_names) if isinstance(feature_names, list) else None
        )
    else:
        report["issues"].append("feature_names_missing")
        report["feature_order_hash"] = None

    report["projection_model_sha256"] = _sha256_file(proj_path)
    if report["projection_model_sha256"] is None:
        # Round19 O2 copies may omit projection_model; fall back to source artifact.
        source = meta.get("source_artifact")
        if source:
            source_proj = PROJECT_ROOT / str(source) / "projection_model.pkl"
            report["projection_model_sha256"] = _sha256_file(source_proj)
            report["projection_model_resolved_from"] = str(source_proj)
        if report["projection_model_sha256"] is None:
            report["issues"].append("projection_model_missing")

    if proj_meta_path.is_file():
        proj_meta = _load_json(proj_meta_path)
    elif meta.get("source_artifact"):
        src_meta = PROJECT_ROOT / str(meta["source_artifact"]) / "projection_metadata.json"
        proj_meta = _load_json(src_meta) if src_meta.is_file() else {}
        report["projection_metadata_resolved_from"] = str(src_meta)
    else:
        proj_meta = {}
    report["projection_metadata"] = proj_meta
    report["projection_method"] = proj_meta.get("projection_type") or meta.get("mode")
    report["fit_domain"] = proj_meta.get("fit_domain")
    report["raw_context_input_dim"] = proj_meta.get("input_dim")
    if isinstance(attestation, dict) and attestation.get("fit_population_hash"):
        report["fit_population_hash"] = attestation["fit_population_hash"]
        report["raw_context_definition_hash"] = attestation["raw_context_definition_hash"]
        report["normalization_hash"] = attestation["normalization_hash"]
        report["source_encoder_checkpoint_hash"] = attestation["source_encoder_checkpoint_hash"]
        report["projection_method"] = attestation.get("raw_context_definition", {}).get(
            "projection_type", report["projection_method"]
        )
    else:
        report["fit_population_hash"] = _stable_hash(
            {
                "fit_domain": proj_meta.get("fit_domain"),
                "input_dim": proj_meta.get("input_dim"),
                "source_artifact": meta.get("source_artifact"),
                "o2_projection_sha256": meta.get("o2_projection_sha256"),
            }
        )
        report["raw_context_definition_hash"] = _stable_hash(
            {
                "mode": meta.get("mode"),
                "projection_type": proj_meta.get("projection_type"),
                "input_dim": proj_meta.get("input_dim"),
                "includes_own_plus_summary": meta.get("includes_own_plus_summary"),
                "includes_projected_context": meta.get("includes_projected_context"),
            }
        )
        report["normalization_hash"] = _stable_hash(
            {
                "projection_type": proj_meta.get("projection_type"),
                "fit_domain": proj_meta.get("fit_domain"),
                "random_state": proj_meta.get("random_state", 42),
                "sklearn_pca_whiten": False,
            }
        )
        report["source_encoder_checkpoint_hash"] = meta.get("o2_projection_sha256") or meta.get(
            "o3_projection_sha256"
        )

    model_ids: list[str] = []
    nan_count = 0
    inf_count = 0
    if latent_path.is_file():
        with latent_path.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, Mapping):
            # Common layouts: {model_id: vector} or {"features": {...}}
            if all(isinstance(v, (np.ndarray, list, tuple)) for v in payload.values()):
                mapping = payload
            elif "features" in payload and isinstance(payload["features"], Mapping):
                mapping = payload["features"]
            else:
                mapping = {}
                report["issues"].append("latent_payload_unrecognized")
            for mid, vec in mapping.items():
                arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                model_ids.append(str(mid))
                nan_count += int(np.isnan(arr).sum())
                inf_count += int(np.isinf(arr).sum())
                expected = 64 + int(expected_context_dim)
                # Some pickle stores only projected context or full O2; accept both metadata-backed sizes.
                if arr.shape[0] not in (expected, int(expected_context_dim), 64, response_dim):
                    report["issues"].append(
                        f"unexpected_vector_dim model_id={mid} dim={arr.shape[0]}"
                    )
                    break
        else:
            report["issues"].append("latent_payload_not_mapping")
    else:
        # Round19 feature dir may only carry metadata + names; coverage comes from source.
        source = meta.get("source_artifact")
        if source:
            src_latent = PROJECT_ROOT / str(source) / "ccle_latent_proto.pkl"
            report["latent_resolved_from"] = str(src_latent)
            if src_latent.is_file():
                with src_latent.open("rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, Mapping):
                    if all(isinstance(v, (np.ndarray, list, tuple)) for v in payload.values()):
                        mapping = payload
                    elif "features" in payload and isinstance(payload["features"], Mapping):
                        mapping = payload["features"]
                    else:
                        mapping = {}
                    model_ids = [str(k) for k in mapping.keys()]
            else:
                report["issues"].append("latent_pickle_missing")
        else:
            report["issues"].append("latent_pickle_missing")

    report["n_ids_metadata"] = meta.get("n_ids")
    report["n_ids_observed"] = len(model_ids)
    report["model_id_coverage_hash"] = _stable_hash(sorted(model_ids))
    report["nan_count"] = nan_count
    report["inf_count"] = inf_count
    if nan_count or inf_count:
        report["issues"].append(f"nan={nan_count} inf={inf_count}")
    if meta.get("n_ids") is not None and model_ids and int(meta["n_ids"]) != len(model_ids):
        report["issues"].append(
            f"n_ids metadata {meta['n_ids']} != observed {len(model_ids)}"
        )

    report["component_slices"] = {
        "feature_mode": "O2",
        "latent_slice": [0, 64],
        "context_slice": [64, 64 + int(expected_context_dim)],
        "context_dim": int(expected_context_dim),
        "output_dim": 64 + int(expected_context_dim),
    }
    report["ok"] = len(report["issues"]) == 0
    return report


def audit_context_pair(
    *,
    c16_dir: Path | str,
    c32_dir: Path | str | None,
    out: Path | str | None = None,
    fail_closed: bool = False,
) -> dict[str, Any]:
    c16 = inspect_context_dir(c16_dir, expected_context_dim=16)
    if c32_dir is None or str(c32_dir).strip() in {"", "null", "None"}:
        c32 = {
            "path": None,
            "exists": False,
            "expected_context_dim": 32,
            "ok": False,
            "issues": ["directory_missing"],
        }
    else:
        c32 = inspect_context_dir(c32_dir, expected_context_dim=32)

    comparable_keys = (
        "raw_context_definition_hash",
        "fit_population_hash",
        "feature_order_hash",
        "normalization_hash",
        "projection_method",
        "source_encoder_checkpoint_hash",
        "model_id_coverage_hash",
    )
    comparisons: dict[str, Any] = {}
    mismatches: list[str] = []
    if c16.get("ok") and c32.get("ok"):
        for key in comparable_keys:
            left = c16.get(key)
            right = c32.get(key)
            same = left == right and left is not None
            # feature_order_hash intentionally differs by context dim length; compare prefix policy separately.
            if key == "feature_order_hash":
                comparisons[key] = {
                    "c16": left,
                    "c32": right,
                    "note": "expected to differ by trailing context feature names",
                }
                continue
            comparisons[key] = {"c16": left, "c32": right, "match": same}
            if not same:
                mismatches.append(key)
    else:
        if not c16.get("ok"):
            mismatches.append("c16_not_ok")
        if not c32.get("ok"):
            mismatches.append("c32_not_ok")

    report = {
        "schema": "round20_context_audit",
        "schema_version": 1,
        "c16": c16,
        "c32": c32,
        "comparable": len(mismatches) == 0 and c16.get("ok") and c32.get("ok"),
        "mismatches": mismatches,
        "comparisons": comparisons,
        "rebuild_guidance": {
            "allowed_if_missing_c32": True,
            "must_reuse": [
                "same raw context matrix",
                "same rows",
                "same normalization",
                "same projection algorithm",
                "same random seed",
                "only n_components: 16 -> 32",
            ],
            "auto_rebuild_in_audit": False,
        },
    }
    if out is not None:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if fail_closed and not report["comparable"]:
        raise ContextAuditError(
            "C16/C32 are not comparable: " + ", ".join(report["mismatches"])
        )
    return report


class Round20O2FeatureStore:
    """Lazy O2 feature store with strict shape checks."""

    def __init__(self, feature_dir: str | Path, context_dim: int):
        self.feature_dir = Path(feature_dir)
        self.context_dim = int(context_dim)
        self.expected_dim = 64 + self.context_dim
        self.metadata = inspect_context_dir(self.feature_dir, expected_context_dim=self.context_dim)
        if not self.metadata.get("ok"):
            raise ContextAuditError(
                f"Invalid O2 feature dir {self.feature_dir}: {self.metadata.get('issues')}"
            )
        self._cache: dict[str, np.ndarray] | None = None

    def _load_mapping(self) -> dict[str, np.ndarray]:
        if self._cache is not None:
            return self._cache
        candidates = [
            self.feature_dir / "ccle_latent_proto.pkl",
        ]
        source = self.metadata.get("feature_metadata", {}).get("source_artifact")
        if source:
            candidates.append(PROJECT_ROOT / str(source) / "ccle_latent_proto.pkl")
        mapping: dict[str, np.ndarray] = {}
        for path in candidates:
            if not path.is_file():
                continue
            with path.open("rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, Mapping):
                if "features" in payload and isinstance(payload["features"], Mapping):
                    raw = payload["features"]
                else:
                    raw = payload
                for key, value in raw.items():
                    arr = np.asarray(value, dtype=np.float32).reshape(-1)
                    mapping[str(key)] = arr
            if mapping:
                break
        if not mapping:
            raise ContextAuditError(f"No feature vectors found under {self.feature_dir}")
        self._cache = mapping
        return mapping

    def get(self, model_id: str) -> np.ndarray:
        mapping = self._load_mapping()
        if model_id not in mapping:
            raise KeyError(model_id)
        vector = np.asarray(mapping[model_id], dtype=np.float32).reshape(-1)
        if vector.shape != (self.expected_dim,):
            # If store contains only Z or only context, refuse rather than silently pad.
            raise FeatureShapeError(
                f"model_id={model_id} shape={vector.shape} expected={(self.expected_dim,)}"
            )
        if not np.isfinite(vector).all():
            raise FeatureShapeError(f"model_id={model_id} contains NaN/Inf")
        return vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit_p = sub.add_parser("audit", help="Audit C16/C32 comparability")
    audit_p.add_argument("--c16-dir", required=True)
    audit_p.add_argument("--c32-dir", default=None)
    audit_p.add_argument("--out", required=True)
    audit_p.add_argument("--fail-closed", action="store_true")

    args = parser.parse_args()
    if args.command == "audit":
        report = audit_context_pair(
            c16_dir=args.c16_dir,
            c32_dir=args.c32_dir,
            out=args.out,
            fail_closed=args.fail_closed,
        )
        print(json.dumps({"ok": report["comparable"], "out": args.out, "mismatches": report["mismatches"]}, indent=2))
        raise SystemExit(0 if report["comparable"] else 3)


if __name__ == "__main__":
    main()
