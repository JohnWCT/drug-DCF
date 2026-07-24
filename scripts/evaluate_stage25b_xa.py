#!/usr/bin/env python3
"""Evaluate Stage25B paired XA runs and apply promotion gates."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


def _load_metrics(run_dir: Path) -> Optional[dict]:
    for name in ("metrics_by_seed.json", "metrics.json", "summary.json", "run_manifest.json"):
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    for p in run_dir.rglob("metrics_by_seed.json"):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def _auc(m: dict) -> Optional[float]:
    for k in (
        "DrugMacro_AUC",
        "val_drug_macro_auc",
        "validation_drug_macro_auc",
        "drug_macro_auc",
        "mean_drug_macro_auc",
    ):
        if k in m and m[k] is not None:
            return float(m[k])
    for nest in ("validation", "metrics", "best", "summary", "test"):
        if isinstance(m.get(nest), dict):
            v = _auc(m[nest])
            if v is not None:
                return v
    return None


def _auprc(m: dict) -> Optional[float]:
    for k in (
        "DrugMacro_AUPRC",
        "val_drug_macro_auprc",
        "validation_drug_macro_auprc",
        "drug_macro_auprc",
        "mean_drug_macro_auprc",
    ):
        if k in m and m[k] is not None:
            return float(m[k])
    for nest in ("validation", "metrics", "best", "summary", "test"):
        if isinstance(m.get(nest), dict):
            v = _auprc(m[nest])
            if v is not None:
                return v
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="outputs/round25_stage25b")
    ap.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    ap.add_argument("--out", default="reports/round25_stage25b_paired_performance.csv")
    ap.add_argument("--decision-out", default="reports/round25_selection_decision.json")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    root = ROOT / args.runs_root
    rows: List[dict] = []
    for arm in ("B0", "B1", "B2"):
        for seed in args.seeds:
            # common layout: outputs/.../B0/biocda_xa_fresh_seed17
            cands = list((root / arm).glob(f"*seed{seed}"))
            if not cands:
                cands = list((root / arm).glob(f"*_{seed}"))
            run_dir = cands[0] if cands else root / arm / f"missing_seed{seed}"
            m = _load_metrics(run_dir) if run_dir.exists() else None
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "auc": None if not m else _auc(m),
                    "auprc": None if not m else _auprc(m),
                    "status": "DONE" if m and _auc(m) is not None else "MISSING",
                }
            )

    out_csv = ROOT / args.out
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def mean_arm(arm: str, key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r["arm"] == arm and r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    b0_auc, b1_auc = mean_arm("B0", "auc"), mean_arm("B1", "auc")
    b0_auprc, b1_auprc = mean_arm("B0", "auprc"), mean_arm("B1", "auprc")
    deltas = []
    for seed in args.seeds:
        a0 = next((r["auc"] for r in rows if r["arm"] == "B0" and r["seed"] == seed), None)
        a1 = next((r["auc"] for r in rows if r["arm"] == "B1" and r["seed"] == seed), None)
        if a0 is not None and a1 is not None:
            deltas.append(a1 - a0)
    noninf = sum(1 for d in deltas if d >= 0)
    worst = min(deltas) if deltas else None
    mean_auc_delta = (b1_auc - b0_auc) if (b0_auc is not None and b1_auc is not None) else None
    mean_auprc_delta = (b1_auprc - b0_auprc) if (b0_auprc is not None and b1_auprc is not None) else None

    promote = False
    reason = "KEEP_S0"
    blocking = []
    if mean_auc_delta is None:
        blocking.append("missing paired AUCs")
        status = "INCONCLUSIVE_TECHNICAL"
    else:
        if mean_auc_delta < 0:
            blocking.append(f"mean AUC delta {mean_auc_delta:.4f} < 0")
        if mean_auprc_delta is not None and mean_auprc_delta < -0.005:
            blocking.append(f"mean AUPRC delta {mean_auprc_delta:.4f} < -0.005")
        if noninf < 3:
            blocking.append(f"noninferior seeds {noninf}/3 < 3")
        if worst is not None and worst <= -0.010:
            blocking.append(f"worst-seed delta {worst:.4f} <= -0.010")
        if not blocking:
            promote = True
            reason = "PROMOTE_STAGE2"
            status = "PROMOTE_STAGE2"
        else:
            status = "KEEP_S0"
            reason = "B1 failed Stage25B paired gates vs B0"

    decision = {
        "round": 25,
        "stage": "25B",
        "status": status,
        "promote_stage2": promote,
        "b0_mean_auc": b0_auc,
        "b1_mean_auc": b1_auc,
        "mean_auc_delta": mean_auc_delta,
        "b0_mean_auprc": b0_auprc,
        "b1_mean_auprc": b1_auprc,
        "mean_auprc_delta": mean_auprc_delta,
        "noninferior_seed_count": noninf,
        "worst_seed_delta": worst,
        "blocking_failures": blocking,
        "reason": reason,
        "tcga_used_for_selection": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / args.decision_out).write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))
    if args.strict and status == "INCONCLUSIVE_TECHNICAL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
