"""Evaluate Stage24B candidates against all-target gate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from biocda.validation.round24_gate import evaluate_all_target_gate
from biocda.validation.round24_protocol import gate_table

ROOT = Path(__file__).resolve().parents[2]


def run_evaluate_24b(cfg: Dict[str, Any], *, stage: str = "24b") -> Dict[str, Any]:
    out = ROOT / cfg["paths"]["reports_root"] / "stage24b"
    out.mkdir(parents=True, exist_ok=True)
    candidates: List[Dict[str, Any]] = []

    # B0 from Stage24A baseline
    b0_path = ROOT / cfg["paths"]["reports_root"] / "stage24a" / "baseline_summary.json"
    if b0_path.is_file():
        b0 = json.loads(b0_path.read_text())
        aucs = {k: v["fold_mean_DrugMacro_AUC"] for k, v in b0["targets"].items()}
        auprcs = {k: v["fold_mean_DrugMacro_AUPRC"] for k, v in b0["targets"].items()}
        gate = evaluate_all_target_gate(
            aucs,
            gate_table(cfg),
            target_priority=cfg["target_priority"],
            target_weights=cfg["target_weights"],
        )
        candidates.append(
            {
                "candidate_id": "B0",
                "architecture": "pooled_mlp",
                "feature": "own_plus_summary",
                "per_target_fold_mean_auc": aucs,
                "per_target_fold_mean_auprc": auprcs,
                "gate": gate,
            }
        )

    # B1/B2 placeholders until fold training completes
    for cid in ("B1", "B2"):
        status_path = out / cid / "candidate_summary.json"
        if status_path.is_file():
            payload = json.loads(status_path.read_text())
            if "per_target_fold_mean_auc" in payload:
                gate = evaluate_all_target_gate(
                    payload["per_target_fold_mean_auc"],
                    gate_table(cfg),
                    target_priority=cfg["target_priority"],
                    target_weights=cfg["target_weights"],
                )
                payload["gate"] = gate
                candidates.append(payload)

    decision = {
        "stage": stage,
        "candidates": candidates,
        "any_all_target_pass": any(c.get("gate", {}).get("status") == "PASS" for c in candidates),
        "next_stage": None,
    }
    if decision["any_all_target_pass"]:
        decision["next_stage"] = "24F"
    elif any(c["candidate_id"] == "B0" for c in candidates) and len(candidates) == 1:
        decision["next_stage"] = "WAIT_B1_B2_OR_24C"
        decision["note"] = (
            "B0 evaluated. If B0 fails all-target gate and B1/B2 not ready, "
            "complete B1/B2 training or proceed to Stage24C after formal B1/B2 evaluate."
        )
    else:
        decision["next_stage"] = "24C"

    (out / "evaluate_summary.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision
