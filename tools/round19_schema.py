"""Canonical Round 19 IDs, job specs, and selection-input guards.

This module is the public-reconstruction compatibility surface.  It does not
retrain models or rewrite immutable locks; it validates contracts used by the
existing Round 19 factorial pipeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

OmicsID = Literal["O0", "O1", "O2", "O3", "O4"]
DrugID = Literal["D0", "D1", "D2", "D3", "D4"]
PredictorID = Literal["P0", "P1", "P2"]
ShiftID = Literal[
    "modelid_grouped",
    "cancer_type_held_out",
    "drug_held_out",
    "scaffold_held_out",
]

OMICS_IDS: tuple[str, ...] = ("O0", "O1", "O2", "O3", "O4")
DRUG_IDS: tuple[str, ...] = ("D0", "D1", "D2", "D3", "D4")
PREDICTOR_IDS: tuple[str, ...] = ("P0", "P1", "P2")
SHIFT_IDS: tuple[str, ...] = (
    "modelid_grouped",
    "cancer_type_held_out",
    "drug_held_out",
    "scaffold_held_out",
)

# Case-insensitive substring patterns that must never appear in selection inputs.
FORBIDDEN_SELECTION_SUBSTRINGS: tuple[str, ...] = (
    "internal",
    "internal_test",
    "tcga",
    "external",
    "integrated5",
    "posthoc",
)

# Explicit attestation / bookkeeping keys that contain forbidden substrings but
# are not selection metric columns.  Values must still be non-selecting.
ALLOWED_SELECTION_ATTESTATION_KEYS: frozenset[str] = frozenset(
    {
        "selection_used_internal",
        "selection_used_tcga",
        "internal_test_used",
        "tcga_used",
        "integrated5_used",
        "posthoc_classification",
        "posthoc_case",
        "is_posthoc_contrastive",
        "is_tcga_exploratory",
    }
)

REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "job_id",
    "stage",
    "omics_id",
    "drug_id",
    "predictor_id",
    "fold_id",
    "model_seed",
    "split_seed",
)

# Local factorial aliases used by existing manifests.
_MANIFEST_DRUG_ALIASES = ("drug_id", "drug_representation_id", "drug")
_MANIFEST_OMICS_ALIASES = ("omics_id", "omics")
_MANIFEST_PRED_ALIASES = ("predictor_id", "predictor")


@dataclass(frozen=True)
class Round19ModelSpec:
    omics_id: OmicsID
    drug_id: DrugID
    predictor_id: PredictorID
    residual_mode: str = "pure"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Round19JobSpec:
    stage: str
    model: Round19ModelSpec
    fold_id: int
    model_seed: int
    split_seed: int
    shift_id: ShiftID = "modelid_grouped"
    control_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"] = self.model.to_dict()
        return payload


def _require_member(value: str, allowed: Sequence[str], *, kind: str) -> str:
    text = str(value).strip()
    if text not in allowed:
        raise ValueError(f"Invalid {kind}: {text!r}; expected one of {list(allowed)}")
    return text


def validate_model_spec(spec: Round19ModelSpec) -> Round19ModelSpec:
    _require_member(spec.omics_id, OMICS_IDS, kind="omics_id")
    _require_member(spec.drug_id, DRUG_IDS, kind="drug_id")
    _require_member(spec.predictor_id, PREDICTOR_IDS, kind="predictor_id")
    if not str(spec.residual_mode).strip():
        raise ValueError("residual_mode must be non-empty")
    # Import lazily to avoid circular imports at module load.
    from tools.round19_registry import assert_compatible_cell

    assert_compatible_cell(spec.drug_id, spec.predictor_id)
    return spec


def validate_job_spec(job: Round19JobSpec) -> Round19JobSpec:
    if not str(job.stage).strip():
        raise ValueError("stage must be non-empty")
    validate_model_spec(job.model)
    _require_member(job.shift_id, SHIFT_IDS, kind="shift_id")
    if int(job.fold_id) < 0:
        raise ValueError(f"fold_id must be >= 0, got {job.fold_id}")
    if int(job.model_seed) < 0 or int(job.split_seed) < 0:
        raise ValueError("model_seed and split_seed must be non-negative")
    return job


def canonical_model_id(spec: Round19ModelSpec) -> str:
    validate_model_spec(spec)
    return f"{spec.omics_id}__{spec.drug_id}__{spec.predictor_id}__{spec.residual_mode}"


def canonical_job_id(job: Round19JobSpec) -> str:
    validate_job_spec(job)
    control = job.control_id or "none"
    return (
        f"{job.stage}__{job.model.drug_id}__{job.model.predictor_id}__"
        f"{job.model.omics_id}__split{job.split_seed}__seed{job.model_seed}__"
        f"fold{job.fold_id}__{job.shift_id}__{control}"
    )


def _column_hits_forbidden(name: str) -> list[str]:
    lowered = str(name).casefold()
    return [pattern for pattern in FORBIDDEN_SELECTION_SUBSTRINGS if pattern in lowered]


def validate_selection_input_columns(
    columns: Iterable[str],
    *,
    allow_attestation_keys: bool = True,
) -> None:
    """Reject selection frames that mention internal/TCGA/external/post-hoc fields."""
    violations: dict[str, list[str]] = {}
    for column in columns:
        name = str(column)
        if allow_attestation_keys and name in ALLOWED_SELECTION_ATTESTATION_KEYS:
            continue
        hits = _column_hits_forbidden(name)
        if hits:
            violations[name] = hits
    if violations:
        detail = ", ".join(
            f"{column} -> {hits}" for column, hits in sorted(violations.items())
        )
        raise AssertionError(
            "Selection input contains forbidden columns "
            f"(case-insensitive substring match): {detail}"
        )


def validate_manifest_columns(columns: Iterable[str]) -> None:
    present = {str(name) for name in columns}
    drug_ok = any(alias in present for alias in _MANIFEST_DRUG_ALIASES)
    omics_ok = any(alias in present for alias in _MANIFEST_OMICS_ALIASES)
    pred_ok = any(alias in present for alias in _MANIFEST_PRED_ALIASES)
    missing = []
    if "job_id" not in present:
        missing.append("job_id")
    if not drug_ok:
        missing.append("drug_id|drug_representation_id")
    if not omics_ok:
        missing.append("omics_id")
    if not pred_ok:
        missing.append("predictor_id")
    if "fold_id" not in present:
        missing.append("fold_id")
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")


def model_spec_from_mapping(value: Mapping[str, Any]) -> Round19ModelSpec:
    return validate_model_spec(
        Round19ModelSpec(
            omics_id=str(value["omics_id"]),  # type: ignore[arg-type]
            drug_id=str(value["drug_id"]),  # type: ignore[arg-type]
            predictor_id=str(value["predictor_id"]),  # type: ignore[arg-type]
            residual_mode=str(value.get("residual_mode", "pure")),
        )
    )
