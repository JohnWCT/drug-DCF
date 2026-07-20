"""Release integrity validation beyond basic hash audit."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from tools.round20.result_contracts import (
    PLACEHOLDER_PATTERNS,
    load_json,
    portable_path,
    sha256_file,
    write_json,
)


def _scan_placeholders(text: str) -> List[str]:
    return PLACEHOLDER_PATTERNS.findall(text)


VALID_RELEASE_STATUSES = frozenset({
    "DRAFT", "AUDIT_FAILED", "RELEASE_CANDIDATE", "LOCKED_RELEASE", "LOCKED",
})


def validate_release_manifest(manifest: dict) -> List[str]:
    errors: List[str] = []
    status = manifest.get("release_status")
    if status not in VALID_RELEASE_STATUSES:
        errors.append(f"invalid_release_status={status}")
    for art in manifest.get("artifacts", []):
        for field in ("relative_path", "path"):
            if field in art and isinstance(art[field], str):
                if _scan_placeholders(art[field]):
                    errors.append(f"placeholder_in_artifact_{field}")
    sel = manifest.get("selection", {})
    for key in ("context", "predictor", "drug_encoder"):
        val = str(sel.get(key, ""))
        if not val or _scan_placeholders(val):
            errors.append(f"placeholder_selection_{key}")
    return errors


def validate_release_directory(release_dir: Path, *, strict: bool = True) -> dict:
    release_dir = Path(release_dir)
    errors: List[str] = []
    manifest_path = release_dir / "RELEASE_MANIFEST.json"
    if not manifest_path.is_file():
        return {"status": "FAIL", "errors": ["missing_RELEASE_MANIFEST"]}
    manifest = load_json(manifest_path)
    errors.extend(validate_release_manifest(manifest))
    hashes_path = release_dir / "hashes/artifact_hashes.json"
    if hashes_path.is_file():
        hashes = load_json(hashes_path)
        for rel, expected in hashes.items():
            p = release_dir / rel
            if not p.is_file():
                errors.append(f"missing:{rel}")
            elif sha256_file(p) != expected:
                errors.append(f"hash_mismatch:{rel}")
    required = [
        "MODEL_CARD.md",
        "DATASET_CARD.md",
        "INFERENCE_GUIDE.md",
        "LIMITATIONS.md",
        "configs/final_model_lock.json",
    ]
    for rel in required:
        if not (release_dir / rel).is_file():
            errors.append(f"required_missing:{rel}")
    n_ckpt = len(list((release_dir / "checkpoints").glob("*.pt")))
    if n_ckpt != 15:
        errors.append(f"checkpoint_count={n_ckpt}")
    status = "PASS" if not errors else "FAIL"
    release_status = "LOCKED_RELEASE" if status == "PASS" else "AUDIT_FAILED"
    if status == "PASS" and manifest.get("release_status") in {"LOCKED", "LOCKED_RELEASE"}:
        release_status = "LOCKED_RELEASE"
    report = {
        "status": status,
        "release_status": release_status,
        "errors": errors,
        "n_checkpoints": n_ckpt,
        "n_hashed_artifacts": len(load_json(hashes_path)) if hashes_path.is_file() else 0,
    }
    write_json(release_dir / "hashes/release_integrity.json", report)
    if manifest.get("release_status") != report["release_status"]:
        manifest["release_status"] = report["release_status"]
        write_json(manifest_path, manifest)
    if strict and status != "PASS":
        raise SystemExit(f"ROUND20_RELEASE_AUDIT=FAIL errors={errors}")
    print(f"ROUND20_RELEASE_STATUS={report['release_status']}")
    print(f"ROUND20_RELEASE_AUDIT={status}")
    return report


def build_public_model_lock(lock_path: Path, output_path: Path) -> dict:
    lock = load_json(lock_path)
    public = json_deep_copy_portable(lock)
    write_json(output_path, public)
    return public


def json_deep_copy_portable(obj):
    import copy

    out = copy.deepcopy(obj)
    if isinstance(out, dict):
        if "feature_dir" in out.get("selected_context", {}):
            out["selected_context"]["feature_dir"] = portable_path(out["selected_context"]["feature_dir"])
        for k, v in list(out.items()):
            out[k] = json_deep_copy_portable(v) if isinstance(v, (dict, list)) else (
                portable_path(v) if isinstance(v, str) and "result/optimization_runs" in v else v
            )
    elif isinstance(out, list):
        return [json_deep_copy_portable(x) for x in out]
    return out
