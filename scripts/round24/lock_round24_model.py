"""Stage 24E/F select and lock."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from biocda.validation.round24_gate import (
    build_lock_manifest,
    evaluate_all_target_gate,
    rank_passing_candidates,
    write_lock_manifest,
)
from biocda.validation.round24_protocol import gate_table

ROOT = Path(__file__).resolve().parents[2]


def run_select(cfg: Dict[str, Any], *, preregister_only: bool = False, strict: bool = True) -> Dict[str, Any]:
    out = ROOT / cfg["paths"]["reports_root"] / "stage24e"
    out.mkdir(parents=True, exist_ok=True)
    if preregister_only:
        # Seed preregister from Stage24C winners if present, else empty template
        manifest = {
            "status": "PREREGISTERED",
            "candidates": [],
            "note": "Fill after Stage24C/D; freeze hash before Stage24F.",
            "forbid_post_formal_mutation": True,
        }
        path = out / "candidate_manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        import hashlib

        h = hashlib.sha256(path.read_bytes()).hexdigest()
        (out / "candidate_manifest.sha256").write_text(h + "\n")
        return {"status": "PREREGISTERED", "sha256": h, "path": str(path)}

    # Formal select from Stage24B evaluate report if available
    eval_path = ROOT / cfg["paths"]["reports_root"] / "stage24b" / "evaluate_summary.json"
    if not eval_path.is_file():
        return {"status": "WAITING_STAGE24B", "path": str(eval_path)}
    summary = json.loads(eval_path.read_text())
    candidates = summary.get("candidates", [])
    ranked = rank_passing_candidates(
        candidates,
        target_priority=cfg["target_priority"],
        target_weights=cfg["target_weights"],
    )
    return {
        "status": "PASS" if ranked else "NO_LOCK",
        "n_passing": len(ranked),
        "champion": ranked[0] if ranked else None,
        "strict_all_targets": strict,
    }


def run_lock(cfg: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
    stage24a = ROOT / cfg["paths"]["reports_root"] / "stage24a"
    protocol = json.loads((stage24a / "eval3_manifest.json").read_text())
    select = run_select(cfg, preregister_only=False, strict=True)
    if select.get("status") == "WAITING_STAGE24B" and not force:
        return select
    champion = select.get("champion")
    if champion is None:
        gate = {"status": "NO_LOCK", "n_pass": 0, "n_targets": 5, "per_target": {}}
    else:
        gate = champion.get("gate") or evaluate_all_target_gate(
            champion["per_target_fold_mean_auc"],
            gate_table(cfg),
            target_priority=cfg["target_priority"],
            target_weights=cfg["target_weights"],
        )
    cand_manifest = None
    cm_path = ROOT / cfg["paths"]["reports_root"] / "stage24e" / "candidate_manifest.json"
    if cm_path.is_file():
        cand_manifest = json.loads(cm_path.read_text())
    payload = build_lock_manifest(
        cfg=cfg,
        gate_result=gate,
        candidate=champion,
        protocol_manifest=protocol,
        candidate_manifest=cand_manifest,
    )
    out = Path(cfg["locks"]["output"])
    if not out.is_absolute():
        out = ROOT / out
    write_lock_manifest(out, payload)
    return {"status": payload["status"], "path": str(out)}
