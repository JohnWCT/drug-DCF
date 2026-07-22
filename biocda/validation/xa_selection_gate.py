"""Round 23 XA selection gate vs BioCDA-Predictive (P0)."""
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
class XASelectionDecision:
    status: str  # LOCKED | REJECTED | CANDIDATE
    selected_model: Optional[str]
    selected_training: Optional[str]
    gates: List[GateResult]
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "selected_model": self.selected_model,
            "selected_training": self.selected_training,
            "failures": self.failures,
            "gates": [{"name": g.name, "passed": g.passed, "details": g.details} for g in self.gates],
        }


def _perf_vs_p0(summary: pd.DataFrame, candidate: str, reference: str = "biocda_predictive") -> GateResult:
    cand = summary[summary["model"] == candidate]
    ref = summary[summary["model"] == reference]
    if cand.empty or ref.empty:
        return GateResult("performance_guardrails", False, {"error": "missing candidate or P0"})

    auc_delta = float(cand["drug_macro_auc"].mean() - ref["drug_macro_auc"].mean())
    auprc_delta = float(cand["drug_macro_auprc"].mean() - ref["drug_macro_auprc"].mean())
    per_seed = []
    for seed in sorted(set(cand["seed"]) & set(ref["seed"])):
        d = float(
            cand.loc[cand["seed"] == seed, "drug_macro_auc"].iloc[0]
            - ref.loc[ref["seed"] == seed, "drug_macro_auc"].iloc[0]
        )
        per_seed.append({"seed": int(seed), "auc_delta": d})
    nonworse = sum(1 for x in per_seed if x["auc_delta"] >= -0.005)
    need = max(1, int(len(per_seed) * 2 / 3))
    floor_ok = all(x["auc_delta"] >= -0.020 for x in per_seed)
    ok = (
        auc_delta >= -0.005
        and auprc_delta >= -0.010
        and nonworse >= need
        and floor_ok
        and len(per_seed) > 0
    )
    return GateResult(
        "performance_guardrails",
        ok,
        {
            "candidate": candidate,
            "mean_auc_delta": auc_delta,
            "mean_auprc_delta": auprc_delta,
            "per_seed_auc_delta": per_seed,
            "nonworse_seed_count": nonworse,
            "nonworse_required": need,
            "floor_ok": floor_ok,
        },
    )


def evaluate_xa_selection_gate(
    *,
    performance_summary: pd.DataFrame,
    candidate_order: Optional[List[str]] = None,
    attention_health_pass: bool = False,
    query_drug_pass: bool = False,
    c32_contract_pass: bool = False,
    no_pooling_pass: bool = False,
    reproduction_pass: bool = False,
) -> XASelectionDecision:
    """
    Prefer X1 (transfer) then X2 (KD) if both pass performance + health gates.
    """
    order = candidate_order or ["biocda_xa_transfer", "biocda_xa_kd", "biocda_xa_fresh"]
    gates: List[GateResult] = []
    failures: List[str] = []

    for name, passed in [
        ("no_pooling_contract", no_pooling_pass),
        ("attention_health", attention_health_pass),
        ("query_drug_sensitivity", query_drug_pass),
        ("c32_contract", c32_contract_pass),
        ("reproduction", reproduction_pass),
    ]:
        gates.append(GateResult(name, passed, {}))
        if not passed:
            failures.append(name)

    health_ok = no_pooling_pass and attention_health_pass and query_drug_pass and c32_contract_pass and reproduction_pass

    for cand in order:
        perf = _perf_vs_p0(performance_summary, cand)
        gates.append(GateResult(f"performance_{cand}", perf.passed, perf.details))
        if health_ok and perf.passed:
            training = {
                "biocda_xa_transfer": "transferred E3 GIN",
                "biocda_xa_kd": "transferred E3 GIN + pooled-teacher distillation",
                "biocda_xa_fresh": "fresh GIN",
            }.get(cand, cand)
            locked_name = "BioCDA-XA-KD" if cand == "biocda_xa_kd" else "BioCDA-XA"
            return XASelectionDecision(
                status="LOCKED",
                selected_model=locked_name,
                selected_training=training,
                gates=gates,
                failures=[],
            )

    failures.append("performance_failure")
    return XASelectionDecision(
        status="REJECTED",
        selected_model=None,
        selected_training=None,
        gates=gates,
        failures=failures,
    )
