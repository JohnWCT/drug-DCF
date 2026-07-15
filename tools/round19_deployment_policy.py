"""Fixed proposal-only routing policy for Round 19F."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional

from tools.round19_novelty_classifier import NoveltyResult, Round19NoveltyClassifier


CHEMICAL_SHIFT_SPECIALIST = "chemical_shift_specialist"
CANCER_SHIFT_SPECIALIST = "cancer_shift_specialist"
SOURCE_PERFORMANCE_CHAMPION = "source_performance_champion"


@dataclass(frozen=True)
class RoutingDecision:
    novelty_class: str
    selected_role: str
    confidence: str
    conservative_fallback: bool
    reason: str
    candidate_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)


def _novelty_name(novelty: object) -> str:
    if isinstance(novelty, NoveltyResult):
        return novelty.novelty_class
    if isinstance(novelty, Mapping):
        return str(novelty.get("novelty_class"))
    return str(novelty)


def select_role(novelty: object) -> str:
    """Select a role from novelty only; model confidence is never consulted."""
    novelty_class = _novelty_name(novelty)
    if novelty_class in {"unseen_drug", "unseen_scaffold", "metadata_unknown"}:
        return CHEMICAL_SHIFT_SPECIALIST
    if novelty_class == "unseen_cancer_type":
        return CANCER_SHIFT_SPECIALIST
    if novelty_class == "source_like":
        # Proposal gate is intentionally pinned, before any role lock exists.
        return SOURCE_PERFORMANCE_CHAMPION
    raise ValueError(f"Unknown novelty class: {novelty_class!r}")


def route(
    classifier_or_novelty: object,
    drug_id: object = None,
    canonical_smiles: object = None,
    cancer_type: object = None,
    *,
    proposal_roles: Optional[Mapping[str, str]] = None,
) -> RoutingDecision:
    """Classify and route, or route an already-computed novelty result.

    ``proposal_roles`` may map role names to proposal candidate ids.  It does
    not alter the role decision.
    """
    if isinstance(classifier_or_novelty, Round19NoveltyClassifier):
        novelty = classifier_or_novelty.classify(
            drug_id, canonical_smiles, cancer_type
        )
    else:
        novelty = classifier_or_novelty
    novelty_class = _novelty_name(novelty)
    role = select_role(novelty_class)
    fallback = novelty_class == "metadata_unknown"
    confidence = (
        "low"
        if fallback
        else (
            novelty.confidence
            if isinstance(novelty, NoveltyResult)
            else str(novelty.get("confidence", "high"))
            if isinstance(novelty, Mapping)
            else "high"
        )
    )
    return RoutingDecision(
        novelty_class=novelty_class,
        selected_role=role,
        confidence=confidence,
        conservative_fallback=fallback,
        reason=(
            "conservative chemical fallback for unknown metadata"
            if fallback
            else f"fixed novelty routing: {novelty_class} -> {role}"
        ),
        candidate_id=(
            str(proposal_roles[role])
            if proposal_roles is not None and role in proposal_roles
            else None
        ),
    )
