#!/usr/bin/env python3
"""Emit and load the deterministic Round 19 deployment policy JSON artifact."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from tools.round19_deployment_policy import (
    CANCER_SHIFT_SPECIALIST,
    CHEMICAL_SHIFT_SPECIALIST,
    SOURCE_PERFORMANCE_CHAMPION,
    route,
    select_role,
)
from tools.round19_role_lock import load_role_lock, role_candidate_map

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = (
    PROJECT_ROOT
    / "result/optimization_runs/round19_factorial/reports/round19_deployment_policy.json"
)
REPO_REPORTS_POLICY = PROJECT_ROOT / "reports" / "round19_deployment_policy.json"
REPO_REPORTS_LOCK = PROJECT_ROOT / "reports" / "round19_final_role_lock.json"


@dataclass(frozen=True)
class RouteDecision:
    role: str
    candidate_id: str | None
    reason: str
    matched_rule_priority: int | None
    warnings: tuple[str, ...]
    rejected: bool
    novelty_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_deployment_policy(*, role_lock: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Serialize the executed deterministic novelty router as a policy document.

    Precedence mirrors ``tools.round19_deployment_policy.select_role`` and is
    intentionally pinned to the Stage 19F proposal-gate behavior already used in
    audits (source_like -> source_performance_champion).
    """
    roles = role_candidate_map(role_lock) if role_lock is not None else {}
    return {
        "schema_version": "1.0",
        "schema": "round19_deployment_policy",
        "policy_type": "deterministic_metadata_routing",
        "source_module": "tools.round19_deployment_policy",
        "single_champion": None,
        "rules": [
            {
                "priority": 10,
                "when": {"novelty_class": ["unseen_drug", "unseen_scaffold", "metadata_unknown"]},
                "route_to_role": CHEMICAL_SHIFT_SPECIALIST,
                "notes": "Chemical shift has precedence for drug/scaffold novelty and unknown metadata.",
            },
            {
                "priority": 20,
                "when": {"novelty_class": ["unseen_cancer_type"]},
                "route_to_role": CANCER_SHIFT_SPECIALIST,
            },
            {
                "priority": 30,
                "when": {"novelty_class": ["source_like"]},
                "route_to_role": SOURCE_PERFORMANCE_CHAMPION,
                "notes": (
                    "Executed local policy pins source_like to source_performance_champion. "
                    "Public-manual prose that prefers parsimonious_context_model is not applied."
                ),
            },
        ],
        "fallback_role": CHEMICAL_SHIFT_SPECIALIST,
        "reject_conditions": [],
        "required_metadata": [
            "canonical_drug_id",
            "canonical_scaffold_id",
            "cancer_type",
        ],
        "locked_role_candidates": roles,
        "selection_used_internal": False,
        "selection_used_tcga": False,
    }


class Round19DeploymentRouter:
    def __init__(self, role_lock: Mapping[str, Any], policy: Mapping[str, Any]):
        self.role_lock = role_lock
        self.policy = policy
        self.roles = role_candidate_map(role_lock)

    def route(
        self,
        metadata: Mapping[str, Any],
        *,
        novelty_class: str | None = None,
    ) -> RouteDecision:
        warnings: list[str] = []
        if novelty_class is None:
            raise ValueError(
                "Round19DeploymentRouter.route requires novelty_class; "
                "use tools.round19_novelty_classifier for classification"
            )
        role = select_role(novelty_class)
        priority = None
        for rule in self.policy.get("rules", []):
            when = rule.get("when", {})
            classes = when.get("novelty_class", [])
            if novelty_class in classes:
                priority = int(rule["priority"])
                expected = str(rule["route_to_role"])
                if expected != role:
                    raise AssertionError(
                        f"Policy/runtime drift for {novelty_class}: "
                        f"policy={expected} runtime={role}"
                    )
                break
        candidate = self.roles.get(role)
        if candidate is None:
            warnings.append(f"role {role} has no locked candidate")
        return RouteDecision(
            role=role,
            candidate_id=None if candidate is None else str(candidate),
            reason=f"fixed novelty routing: {novelty_class} -> {role}",
            matched_rule_priority=priority,
            warnings=tuple(warnings),
            rejected=False,
            novelty_class=novelty_class,
        )


def write_deployment_policy(
    path: Path,
    *,
    role_lock_path: Path | None = None,
    also_repo_reports: bool = True,
) -> dict[str, Any]:
    lock = load_role_lock(role_lock_path)
    policy = build_deployment_policy(role_lock=lock)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if also_repo_reports:
        REPO_REPORTS_POLICY.parent.mkdir(parents=True, exist_ok=True)
        REPO_REPORTS_POLICY.write_text(
            json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pointer = {
            "schema": "round19_final_role_lock_pointer",
            "schema_version": 1,
            "immutable_lock_path": str(
                Path(lock["_path"]).resolve().relative_to(PROJECT_ROOT)
            ),
            "roles": role_candidate_map(lock),
            "note": (
                "Canonical immutable lock remains under "
                "result/optimization_runs/round19_factorial/reports/; do not overwrite."
            ),
        }
        REPO_REPORTS_LOCK.write_text(
            json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--role-lock")
    parser.add_argument("--no-repo-reports", action="store_true")
    parser.add_argument("--smoke-route", action="store_true")
    args = parser.parse_args()
    policy = write_deployment_policy(
        Path(args.output),
        role_lock_path=Path(args.role_lock) if args.role_lock else None,
        also_repo_reports=not args.no_repo_reports,
    )
    result: dict[str, Any] = {"written": args.output, "rules": len(policy["rules"])}
    if args.smoke_route:
        lock = load_role_lock(Path(args.role_lock) if args.role_lock else None)
        router = Round19DeploymentRouter(lock, policy)
        result["smoke"] = {
            novelty: router.route({}, novelty_class=novelty).to_dict()
            for novelty in (
                "unseen_drug",
                "unseen_scaffold",
                "unseen_cancer_type",
                "source_like",
                "metadata_unknown",
            )
        }
        # Keep parity with the low-level router used by Stage 19G.
        for novelty, decision in result["smoke"].items():
            low_level = route(novelty)
            if low_level.selected_role != decision["role"]:
                raise AssertionError(
                    f"Router drift for {novelty}: "
                    f"{low_level.selected_role} vs {decision['role']}"
                )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
