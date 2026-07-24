#!/usr/bin/env python3
"""Materialize Stage 25A decision JSON from screen metrics + geometry (Round 25).

Allowed statuses:
  PROMOTE_S2 | PROMOTE_S1 | PROMOTE_S3 | KEEP_S0 | RUN_S3 | RUN_S2B | INCONCLUSIVE_TECHNICAL

Hard rule: never wrap poor performance as bare INCONCLUSIVE.
TCGA labels are never used for selection here.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

ALLOWED = {
    "PROMOTE_S2",
    "PROMOTE_S1",
    "PROMOTE_S3",
    "KEEP_S0",
    "RUN_S3",
    "RUN_S2B",
    "INCONCLUSIVE_TECHNICAL",
}


def _mean(vals: List[float]) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None and v != ""]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--metrics", default="reports/round25_stage25a_metrics.csv")
    ap.add_argument("--geometry", default="reports/round25_stage25a_geometry.json")
    ap.add_argument("--context", default="reports/round25_stage25a_context_readiness.json")
    ap.add_argument("--config", default="config/round25_stage2_margin_screen.yaml")
    ap.add_argument("--out", default="reports/round25_stage25a_decision.json")
    args = ap.parse_args()

    metrics_path = ROOT / args.metrics
    if not metrics_path.exists():
        decision = {
            "round": 25,
            "stage": "25A",
            "status": "INCONCLUSIVE_TECHNICAL",
            "baseline_variant": "S0",
            "evaluated_variants": ["S0", "S2", "S1"],
            "selected_variant": "",
            "run_s3": False,
            "run_s2b": False,
            "reason": "stage25a metrics missing; screen not completed",
            "blocking_failures": ["missing reports/round25_stage25a_metrics.csv"],
            "tcga_used_for_selection": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (ROOT / args.out).write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(decision, indent=2))
        return 1 if args.strict else 0

    cfg = _load_yaml(ROOT / args.config)
    sel = dict(cfg.get("selection") or {})
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    by_var: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r)

    geom = {}
    gpath = ROOT / args.geometry
    if gpath.exists():
        geom = json.loads(gpath.read_text(encoding="utf-8")).get("variants", {})
    ctx = {}
    cpath = ROOT / args.context
    if cpath.exists():
        ctx = json.loads(cpath.read_text(encoding="utf-8")).get("variants", {})

    def variant_ok(vid: str) -> Dict[str, Any]:
        jobs = by_var.get(vid, [])
        fails = [j for j in jobs if j.get("status") != "DONE"]
        info: Dict[str, Any] = {
            "variant": vid,
            "n_jobs": len(jobs),
            "n_fail": len(fails),
            "pass_gates": False,
            "blocking": [],
        }
        if fails:
            info["blocking"].append(f"{vid}: {len(fails)} failed seeds")
            return info
        if vid not in by_var or not jobs:
            info["blocking"].append(f"{vid}: missing")
            return info

        # Geometry gates vs S0
        if vid != "S0" and "S0" in geom:
            s0_ranks = [
                geom["S0"][s].get("target_effective_rank")
                for s in geom["S0"]
                if geom["S0"][s].get("ok")
            ]
            v_ranks = [
                geom[vid][s].get("target_effective_rank")
                for s in geom.get(vid, {})
                if geom[vid][s].get("ok")
            ]
            s0_r = _mean(s0_ranks)
            v_r = _mean(v_ranks)
            ratio_min = float(sel.get("target_effective_rank_ratio_min", 0.90))
            if s0_r and v_r is not None and v_r < ratio_min * s0_r:
                info["blocking"].append(
                    f"{vid}: target_effective_rank {v_r:.4f} < {ratio_min}*{s0_r:.4f}"
                )
            s0_c = _mean(
                [
                    ctx.get("S0", {}).get(s, {}).get("c32_effective_rank")
                    for s in ctx.get("S0", {})
                ]
            )
            v_c = _mean(
                [
                    ctx.get(vid, {}).get(s, {}).get("c32_effective_rank")
                    for s in ctx.get(vid, {})
                ]
            )
            cmin = float(sel.get("c32_effective_rank_ratio_min", 0.90))
            if s0_c and v_c is not None and v_c < cmin * s0_c:
                info["blocking"].append(
                    f"{vid}: c32_effective_rank {v_c:.4f} < {cmin}*{s0_c:.4f}"
                )

        if vid == "S2":
            fracs = []
            for j in jobs:
                v = j.get("prototype_hinge_active_fraction")
                if v not in (None, ""):
                    fracs.append(float(v))
            if fracs:
                m = sum(fracs) / len(fracs)
                lo = float(sel.get("prototype_hinge_active_fraction_min", 0.05))
                hi = float(sel.get("prototype_hinge_active_fraction_max", 0.95))
                if not (lo < m < hi):
                    info["blocking"].append(
                        f"S2: prototype_hinge_active_fraction={m:.4f} outside ({lo},{hi})"
                    )
            else:
                info["blocking"].append("S2: missing prototype_hinge_active_fraction")

        if vid == "S1":
            # source recon should be < target recon during AE discriminator updates
            s_err = _mean([j.get("source_reconstruction_error") for j in jobs])
            t_err = _mean([j.get("target_reconstruction_error") for j in jobs])
            if s_err is not None and t_err is not None and not (s_err < t_err + 1e-6):
                info["blocking"].append(
                    f"S1: source_recon={s_err:.4f} not < target_recon={t_err:.4f}"
                )

        info["pass_gates"] = len(info["blocking"]) == 0
        return info

    evaluated = [v for v in ("S0", "S2", "S1") if v in by_var]
    reports = {v: variant_ok(v) for v in evaluated}
    blocking = []
    for v, info in reports.items():
        blocking.extend(info["blocking"])

    # Selection priority: S2 > S1 > KEEP_S0 (among passers). S3/S2b are conditional.
    status = "KEEP_S0"
    selected = "S0"
    run_s3 = False
    run_s2b = False
    reason = "baseline retained"

    s0_ok = reports.get("S0", {}).get("pass_gates", False)
    s2_ok = reports.get("S2", {}).get("pass_gates", False)
    s1_ok = reports.get("S1", {}).get("pass_gates", False)

    if not s0_ok and "S0" in reports:
        status = "INCONCLUSIVE_TECHNICAL"
        selected = ""
        reason = "S0 baseline failed technical gates"
    elif s2_ok:
        status = "PROMOTE_S2"
        selected = "S2"
        reason = "S2 passed Stage25A gates vs S0"
        run_s3 = bool(s1_ok or s2_ok)
    elif s1_ok:
        status = "PROMOTE_S1"
        selected = "S1"
        reason = "S1 passed Stage25A gates vs S0; S2 did not"
        run_s3 = True
    else:
        status = "KEEP_S0"
        selected = "S0"
        reason = "neither S1 nor S2 passed Stage25A gates; keep S0"

    # Overlap evidence gate for S2b (prototype near-zero + rank collapse)
    overlap = False
    if "S0" in geom:
        # crude: if mean target-source distance from metrics near 0 and c32 rank drops
        dists = []
        for j in by_var.get("S0", []):
            d = j.get("mean_target_to_source_anchor_distance")
            if d not in (None, ""):
                dists.append(float(d))
        mean_d = _mean(dists)
        if mean_d is not None and mean_d < 0.02:
            overlap = True
            run_s2b = True
            if status.startswith("PROMOTE") or status == "KEEP_S0":
                # annotate but do not override promote
                reason = reason + "; prototype_overlap_gate=true → RUN_S2B"

    if status == "PROMOTE_S2" and s1_ok:
        run_s3 = True
        reason = reason + "; S1 also passed → RUN_S3 eligible"

    decision = {
        "round": 25,
        "stage": "25A",
        "status": status,
        "baseline_variant": "S0",
        "evaluated_variants": evaluated,
        "selected_variant": selected,
        "run_s3": run_s3,
        "run_s2b": run_s2b,
        "prototype_overlap_gate": overlap,
        "reason": reason,
        "blocking_failures": blocking,
        "per_variant": reports,
        "tcga_used_for_selection": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    assert decision["status"] in ALLOWED
    (ROOT / args.out).write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    if args.strict and status == "INCONCLUSIVE_TECHNICAL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
