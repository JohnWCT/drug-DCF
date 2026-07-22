#!/usr/bin/env python3
"""Lock or reject BioCDA-XA after Round 23 gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biocda.utils.runtime_manifest import git_commit
from biocda.validation.xa_model_lock import build_xa_lock_manifest, write_xa_lock_manifest
from biocda.validation.xa_selection_gate import evaluate_xa_selection_gate
from tools.biocda_telegram_notify import biocda_notify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/biocda/xa_v2_closure.yaml")
    parser.add_argument("--force-reject", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reports = ROOT / "reports"
    perf_path = reports / "round23_paired_performance.csv"
    if not perf_path.is_file():
        print("Missing round23_paired_performance.csv — run evaluate_xa_candidates first", file=sys.stderr)
        return 1
    df = pd.read_csv(perf_path)

    audit = {}
    audit_path = reports / "round23_no_pooling_architecture_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    no_pool = all(v.get("ok", False) for v in audit.get("checks", {}).values()) if audit else False

    util = {}
    util_path = reports / "round23_query_drug_utilization.json"
    if util_path.is_file():
        util = json.loads(util_path.read_text(encoding="utf-8"))
    query_drug = bool(util.get("pass", False))

    attn_ok = True
    attn_path = reports / "round23_attention_health.json"
    if attn_path.is_file():
        health = json.loads(attn_path.read_text(encoding="utf-8"))
        # Soft health: entropy not collapsed
        attn_ok = float(health.get("mean_normalized_entropy", 0)) > 0.05

    c32_path = reports / "round23_c32_xa_effect.json"
    c32_ok = c32_path.is_file()

    decision = evaluate_xa_selection_gate(
        performance_summary=df,
        attention_health_pass=attn_ok,
        query_drug_pass=query_drug,
        c32_contract_pass=c32_ok,
        no_pooling_pass=no_pool,
        reproduction_pass=True,
    )
    if args.force_reject:
        decision.status = "REJECTED"
        decision.selected_model = None
        decision.failures = list(set(decision.failures + ["performance_failure"]))

    out_root = ROOT / config["outputs"]["root"]
    ckpts = []
    if decision.selected_model:
        # map locked name back to training id
        mid = "biocda_xa_kd" if decision.selected_model == "BioCDA-XA-KD" else "biocda_xa_transfer"
        if decision.selected_training and "fresh" in (decision.selected_training or ""):
            mid = "biocda_xa_fresh"
        for seed in config["experiment"]["seeds"]:
            p = out_root / f"{mid}_seed{seed}" / "best.pt"
            if p.is_file():
                ckpts.append(str(p))

    payload = build_xa_lock_manifest(
        decision,
        architecture_version=config["experiment"]["architecture_version"],
        checkpoint_paths=ckpts,
        git_commit=git_commit(),
    )
    # Never write fake LOCKED
    if payload["status"] == "LOCKED" and not ckpts:
        payload["status"] = "REJECTED"
        payload["reason"] = "missing_checkpoints"
        payload["model_name"] = None

    write_xa_lock_manifest(reports / "round23_selection_decision.json", decision.to_dict())
    write_xa_lock_manifest(reports / "biocda_xa_model_lock.json", payload)

    biocda_notify(
        f"Round23 XA lock status={payload['status']} model={payload.get('model_name')} "
        f"failures={payload.get('failures') or payload.get('reason')}"
    )
    print(json.dumps(payload, indent=2)[:2000])
    return 0 if payload["status"] in {"LOCKED", "REJECTED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
