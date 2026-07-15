"""Round 19F deployment novelty classification.

The classifier is fitted only from development-training metadata.  It does not
inspect model predictions or any internal/TCGA outcome.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import pandas as pd

from tools.round18_eligible_data import _normalize_drug_key
from tools.round19_scaffold_groups import graph_canonical_smiles, murcko_scaffold_id


NOVELTY_CLASSES = (
    "unseen_drug",
    "unseen_scaffold",
    "unseen_cancer_type",
    "source_like",
    "metadata_unknown",
)


def _present(value: object) -> bool:
    return value is not None and not pd.isna(value) and bool(str(value).strip())


@dataclass(frozen=True)
class NoveltyResult:
    novelty_class: str
    confidence: str
    normalized_drug_id: Optional[str]
    canonical_smiles: Optional[str]
    scaffold_id: Optional[str]
    cancer_type: Optional[str]
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)


class Round19NoveltyClassifier:
    """Classify deployment rows against development-training support."""

    def __init__(
        self,
        development_drug_ids: Iterable[object],
        development_canonical_smiles: Iterable[object],
        development_cancer_types: Iterable[object],
    ) -> None:
        drug_ids = list(development_drug_ids)
        smiles = list(development_canonical_smiles)
        cancer_types = list(development_cancer_types)
        self.development_drug_ids = {
            _normalize_drug_key(str(v)) for v in drug_ids if _present(v)
        }
        self.development_canonical_smiles = set()
        self.development_scaffolds = set()
        for value in smiles:
            if not _present(value):
                continue
            canonical = graph_canonical_smiles(str(value))
            if canonical is None:
                raise ValueError(f"Invalid development canonical SMILES: {value!r}")
            self.development_canonical_smiles.add(canonical)
            self.development_scaffolds.add(murcko_scaffold_id(canonical))
        self.development_cancer_types = {
            str(v).strip() for v in cancer_types if _present(v)
        }
        if not self.development_drug_ids:
            raise ValueError("development training drug-id set is empty")
        if not self.development_scaffolds:
            raise ValueError("development training scaffold set is empty")
        if not self.development_cancer_types:
            raise ValueError("development training cancer-type set is empty")

    @classmethod
    def from_development(
        cls,
        development: pd.DataFrame,
        *,
        drug_group_table: Optional[pd.DataFrame] = None,
        cancer_mapping: Optional[pd.DataFrame] = None,
        drug_column: str = "DRUG_NAME",
        model_id_column: str = "ModelID",
        cancer_type_column: str = "cancer_type",
    ) -> "Round19NoveltyClassifier":
        """Build support sets using the Round 19 drug and cancer mapping tables."""
        dev = development.copy()
        if drug_column not in dev:
            raise KeyError(drug_column)

        if drug_group_table is not None:
            required = {drug_column, "normalized_drug_id", "canonical_smiles"}
            missing = required - set(drug_group_table.columns)
            if missing:
                raise KeyError(f"drug_group_table missing columns: {sorted(missing)}")
            lookup = drug_group_table.drop_duplicates(drug_column).set_index(drug_column)
            drug_ids = dev[drug_column].map(lookup["normalized_drug_id"])
            smiles = dev[drug_column].map(lookup["canonical_smiles"])
        else:
            drug_ids = (
                dev["normalized_drug_id"]
                if "normalized_drug_id" in dev
                else dev[drug_column].map(lambda v: _normalize_drug_key(str(v)))
            )
            if "canonical_smiles" not in dev:
                raise KeyError(
                    "canonical_smiles (or drug_group_table) is required to fit novelty support"
                )
            smiles = dev["canonical_smiles"]

        if cancer_type_column in dev:
            cancer_types = dev[cancer_type_column]
        elif cancer_mapping is not None:
            if model_id_column not in dev or model_id_column not in cancer_mapping:
                raise KeyError(model_id_column)
            if cancer_type_column not in cancer_mapping:
                raise KeyError(cancer_type_column)
            mapping = cancer_mapping.drop_duplicates(model_id_column).copy()
            mapping[model_id_column] = mapping[model_id_column].astype(str)
            lookup = mapping.set_index(model_id_column)[cancer_type_column]
            cancer_types = dev[model_id_column].astype(str).map(lookup)
        else:
            raise KeyError(
                f"{cancer_type_column} (or cancer_mapping) is required to fit novelty support"
            )

        if pd.Series(drug_ids).isna().any() or pd.Series(smiles).isna().any():
            raise AssertionError("development drug metadata is incompletely mapped")
        if pd.Series(cancer_types).isna().any():
            raise AssertionError("development cancer metadata is incompletely mapped")
        return cls(drug_ids, smiles, cancer_types)

    def classify(
        self,
        drug_id: object,
        canonical_smiles: object,
        cancer_type: object,
    ) -> NoveltyResult:
        """Apply the fixed priority: drug, scaffold, cancer type, source-like."""
        missing = [
            name
            for name, value in (
                ("drug_id", drug_id),
                ("canonical_smiles", canonical_smiles),
                ("cancer_type", cancer_type),
            )
            if not _present(value)
        ]
        if missing:
            return NoveltyResult(
                "metadata_unknown",
                "low",
                None if not _present(drug_id) else _normalize_drug_key(str(drug_id)),
                None,
                None,
                None if not _present(cancer_type) else str(cancer_type).strip(),
                f"missing metadata: {', '.join(missing)}",
            )

        normalized_drug_id = _normalize_drug_key(str(drug_id))
        canonical = graph_canonical_smiles(str(canonical_smiles))
        if canonical is None:
            return NoveltyResult(
                "metadata_unknown",
                "low",
                normalized_drug_id,
                None,
                None,
                str(cancer_type).strip(),
                "invalid canonical_smiles",
            )
        scaffold = murcko_scaffold_id(canonical)
        cancer = str(cancer_type).strip()

        if normalized_drug_id not in self.development_drug_ids:
            novelty = "unseen_drug"
        elif scaffold not in self.development_scaffolds:
            novelty = "unseen_scaffold"
        elif cancer not in self.development_cancer_types:
            novelty = "unseen_cancer_type"
        else:
            novelty = "source_like"
        return NoveltyResult(
            novelty,
            "high",
            normalized_drug_id,
            canonical,
            scaffold,
            cancer,
            f"fixed-priority classification: {novelty}",
        )


def build_classifier(
    development: pd.DataFrame,
    *,
    drug_group_table: Optional[pd.DataFrame] = None,
    cancer_mapping: Optional[pd.DataFrame] = None,
) -> Round19NoveltyClassifier:
    """Convenience wrapper retained for manifest/inference callers."""
    return Round19NoveltyClassifier.from_development(
        development,
        drug_group_table=drug_group_table,
        cancer_mapping=cancer_mapping,
    )
