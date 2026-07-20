"""Automatic model selection gates for Round 21."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class GateResult:
    name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionOutcome:
    status: str
    selected_model: Optional[str]
    gates: List[GateResult]
    failures: List[str] = field(default_factory=list)


def evaluate_selection_gates(
    *,
    functional_checks: Dict[str, bool],
    performance_summary: pd.DataFrame,
    context_checks: Dict[str, bool],
    attention_checks: Dict[str, bool],
) -> SelectionOutcome:
    gates: List[GateResult] = []
    failures: List[str] = []

    gate_a = all(functional_checks.values())
    gates.append(GateResult("functional_correctness", gate_a, functional_checks))
    if not gate_a:
        failures.extend([f"functional:{k}" for k, v in functional_checks.items() if not v])

    m2 = performance_summary[performance_summary["model"] == "biocda_xa_zc"]
    m0 = performance_summary[performance_summary["model"] == "pooled_baseline"]
    perf_ok = True
    perf_details: Dict[str, Any] = {}
    if not m2.empty and not m0.empty:
        auc_delta = float(m2["drug_macro_auc"].mean() - m0["drug_macro_auc"].mean())
        auprc_delta = float(m2["drug_macro_auprc"].mean() - m0["drug_macro_auprc"].mean())
        per_seed = []
        for seed in sorted(set(m2["seed"]) & set(m0["seed"])):
            d = float(
                m2.loc[m2["seed"] == seed, "drug_macro_auc"].iloc[0]
                - m0.loc[m0["seed"] == seed, "drug_macro_auc"].iloc[0]
            )
            per_seed.append({"seed": int(seed), "auc_delta": d})
        nonworse = sum(1 for x in per_seed if x["auc_delta"] >= -0.005)
        perf_ok = (
            auc_delta >= -0.005
            and auprc_delta >= -0.010
            and any(x["auc_delta"] > -0.020 for x in per_seed)
            and nonworse >= max(1, int(len(per_seed) * 2 / 3))
        )
        perf_details = {
            "mean_auc_delta": auc_delta,
            "mean_auprc_delta": auprc_delta,
            "per_seed_auc_delta": per_seed,
            "nonworse_seed_count": nonworse,
        }
    else:
        perf_ok = False
        perf_details = {"error": "missing M0 or M2 metrics"}
    gates.append(GateResult("performance_guardrails", perf_ok, perf_details))
    if not perf_ok:
        failures.append("performance_guardrails")

    ctx_ok = all(context_checks.values())
    gates.append(GateResult("context_utilization", ctx_ok, context_checks))
    if not ctx_ok:
        failures.extend([f"context:{k}" for k, v in context_checks.items() if not v])

    attn_ok = all(attention_checks.values())
    gates.append(GateResult("attention_health", attn_ok, attention_checks))
    if not attn_ok:
        failures.extend([f"attention:{k}" for k, v in attention_checks.items() if not v])

    all_pass = gate_a and perf_ok and ctx_ok and attn_ok
    if all_pass:
        return SelectionOutcome("LOCKED", "BioCDA-XA-ZC", gates, failures)
    if gate_a and perf_ok and not ctx_ok:
        return SelectionOutcome("NEEDS_REVISION", "biocda_xa_z", gates, failures)
    if gate_a and not perf_ok:
        return SelectionOutcome("REJECTED", "pooled_baseline", gates, failures)
    return SelectionOutcome("NEEDS_REVISION", None, gates, failures)
