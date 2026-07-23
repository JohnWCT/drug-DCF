"""Round 24 all-target gate and lock helpers."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def weighted_score(per_target_auc: Dict[str, float], weights: Dict[str, int]) -> float:
    return float(sum(weights[t] * per_target_auc[t] for t in weights))


def evaluate_all_target_gate(
    per_target_fold_mean_auc: Dict[str, float],
    gate_table: Dict[str, Dict[str, float]],
    *,
    target_priority: List[str],
    target_weights: Dict[str, int],
) -> Dict[str, Any]:
    """Hard gate: every target fold-mean AUROC must strictly exceed gate."""
    results = {}
    n_pass = 0
    deltas = []
    for t in target_priority:
        auc = float(per_target_fold_mean_auc[t])
        gate = float(gate_table[t]["gate_auroc"])
        passed = auc > gate
        n_pass += int(passed)
        delta = auc - gate
        deltas.append(delta)
        results[t] = {
            "fold_mean_DrugMacro_AUC": auc,
            "gate_auroc": gate,
            "delta": delta,
            "pass": passed,
        }
    all_pass = n_pass == len(target_priority)
    status = "PASS" if all_pass else "NO_LOCK"
    return {
        "status": status,
        "n_pass": n_pass,
        "n_targets": len(target_priority),
        "min_delta": float(min(deltas)) if deltas else None,
        "weighted_DrugMacro_AUC": weighted_score(per_target_fold_mean_auc, target_weights),
        "per_target": results,
        "note": (
            "Weighted score is ranking-only after all-target PASS; "
            "it cannot override NO_LOCK."
        ),
    }


def rank_passing_candidates(
    candidates: List[Dict[str, Any]],
    *,
    target_priority: List[str],
    target_weights: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Only candidates with status PASS are ranked."""
    passing = [c for c in candidates if c.get("gate", {}).get("status") == "PASS"]

    def sort_key(c: Dict[str, Any]) -> Tuple:
        aucs = {t: c["per_target_fold_mean_auc"][t] for t in target_priority}
        auprcs = c.get("per_target_fold_mean_auprc", {})
        globals_auc = c.get("per_target_fold_mean_global_auc", {})
        globals_auprc = c.get("per_target_fold_mean_global_auprc", {})
        return (
            weighted_score(aucs, target_weights),
            tuple(auprcs.get(t, 0.0) for t in target_priority),
            tuple(globals_auc.get(t, 0.0) for t in target_priority),
            tuple(globals_auprc.get(t, 0.0) for t in target_priority),
        )

    return sorted(passing, key=sort_key, reverse=True)


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def build_lock_manifest(
    *,
    cfg: Dict[str, Any],
    gate_result: Dict[str, Any],
    candidate: Optional[Dict[str, Any]],
    protocol_manifest: Dict[str, Any],
    candidate_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = gate_result["status"] if candidate else "NO_LOCK"
    if candidate is None:
        status = "NO_LOCK"
    supersedes = []
    for p in cfg.get("locks", {}).get("supersedes", []):
        path = PROJECT_ROOT / p
        supersedes.append(
            {
                "path": p,
                "sha256": sha256_file(path) if path.is_file() else "missing",
                "role": "historical_superseded",
            }
        )
    payload = {
        "status": status,
        "protocol_name": "eval3",
        "selection_role_gdsc": "none",
        "tcga_used_for_selection": True,
        "tcga_role": "selection_benchmark",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config_path": cfg.get("_config_path"),
        "config_sha256": cfg.get("_config_sha256"),
        "protocol_manifest_sha256": sha256_json(protocol_manifest),
        "candidate_manifest_sha256": sha256_json(candidate_manifest) if candidate_manifest else None,
        "gate": gate_result,
        "champion": candidate,
        "supersedes": supersedes,
        "forbidden": {
            "gdsc_test_for_selection": True,
            "per_target_champion": True,
            "weighted_override_of_failed_target": True,
            "tcga_in_early_stopping": True,
        },
    }
    return payload


def write_lock_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
