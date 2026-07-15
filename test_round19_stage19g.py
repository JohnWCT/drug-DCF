from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.round19_stage19f_final_lock import canonical_sha256
from tools.round19_stage19g_case_manifest import (
    build_task_manifests,
    validate_case_manifest,
)
from tools.round19_stage19g_case_selector import (
    CASE_COLUMNS,
    _contrastive_e1_e3,
    deterministic_take,
)


def _base_row(index: int, *, patient: bool = False, tcga: bool = False) -> dict:
    target = "aacdr_gdsc_intersect" if tcga else "drug_heldout"
    eval_row_id = (
        f"19f:{target}|T{index}|tcga_drug|{index % 2}|{index}"
        if tcga
        else f"19e:drug_heldout:{index}"
    )
    reason = (
        "patient_conditioned"
        if patient
        else "tcga_exploratory"
        if tcga
        else "representative_stratified_random"
    )
    drug = "patient_drug" if patient else "tcga_drug" if tcga else f"drug{index % 4}"
    return {
        "case_id": eval_row_id,
        "eval_row_id": eval_row_id,
        "candidate_role": "all_locked_roles",
        "candidate_id": "all_locked_sources",
        "source_stage": "19F" if tcga else "19E",
        "source_target": target,
        "source_row_id": str(index),
        "shift_strategy": "" if tcga else "drug_heldout",
        "ModelID": f"T{index}" if tcga else f"M{index}",
        "Patient_id": f"T{index}" if tcga else "",
        "DRUG_NAME": drug,
        "Label": index % 2,
        "cancer_type": f"C{index % 3}",
        "normalized_drug_id": drug,
        "drug_id": drug,
        "scaffold_id": f"S{index % 2}",
        "canonical_smiles": "CCO",
        "graph_smiles": "CCO",
        "graph_smiles_metadata_status": "legacy_graph_resolution_required",
        "selection_reason": reason,
        "selection_detail": "synthetic",
        "posthoc_case": tcga,
        "selection_eligible": not tcga,
        "is_posthoc_contrastive": False,
        "is_tcga_exploratory": tcga,
        "prediction_used_for_selection": False,
        "selection_seed": 19091,
        "available_drug_unique_modelids": 40 if patient else "",
        "available_drug_cancer_types": 8 if patient else "",
        "available_drug_labels": "0,1" if patient else "",
    }


def _valid_cases(n_representative: int = 113) -> pd.DataFrame:
    rows = [_base_row(index, patient=True) for index in range(20)]
    rows.extend(_base_row(1000 + index, tcga=True) for index in range(20))
    rows.extend(_base_row(100 + index) for index in range(n_representative))
    return pd.DataFrame(rows, columns=CASE_COLUMNS)


class Round19GSelectionTests(unittest.TestCase):
    def test_deterministic_take_ignores_prediction_values_and_input_order(self) -> None:
        frame = pd.DataFrame(
            {
                "eval_row_id": [f"19e:drug_heldout:{i}" for i in range(100)],
                "probability": [i / 100 for i in range(100)],
            }
        )
        first = deterministic_take(frame, 20, seed=19091, namespace="representative")
        changed = frame.sample(frac=1, random_state=9).copy()
        changed["probability"] = list(reversed(changed["probability"].tolist()))
        second = deterministic_take(changed, 20, seed=19091, namespace="representative")
        self.assertEqual(first["eval_row_id"].tolist(), second["eval_row_id"].tolist())

    def test_e1_e3_gain_and_loss_counts_are_posthoc(self) -> None:
        rows = []
        for shift in ("cancer_type_heldout", "drug_heldout", "scaffold_heldout"):
            for index in range(60):
                label = index % 2
                e1_correct = index % 4 < 2
                probability = 0.9 if (label == 1) == e1_correct else 0.1
                row = _base_row(index)
                row.update(
                    {
                        "eval_row_id": f"19e:{shift}:{index}",
                        "shift_strategy": shift,
                        "Label": label,
                        "probability": probability,
                    }
                )
                rows.append(row)
        e1 = pd.DataFrame(rows)
        e3 = e1.copy()
        e3["probability"] = 1.0 - e1["probability"]
        selected = pd.concat(_contrastive_e1_e3(e1, e3, 10, set()))
        counts = selected["selection_reason"].value_counts().to_dict()
        self.assertEqual(counts["contrastive_cancer_gain"], 10)
        self.assertEqual(counts["contrastive_cancer_loss"], 10)
        self.assertEqual(counts["contrastive_chemical_gain"], 10)
        self.assertEqual(counts["contrastive_chemical_loss"], 10)
        self.assertTrue(selected["is_posthoc_contrastive"].all())

    def test_tcga_must_be_exploratory(self) -> None:
        cases = _valid_cases()
        validate_case_manifest(cases)
        tcga_index = cases.index[cases["selection_reason"] == "tcga_exploratory"][0]
        cases.loc[tcga_index, "is_tcga_exploratory"] = False
        with self.assertRaisesRegex(AssertionError, "TCGA"):
            validate_case_manifest(cases)

    def test_patient_constraints_apply_per_selected_drug(self) -> None:
        cases = validate_case_manifest(_valid_cases())
        patient = cases[cases["selection_reason"] == "patient_conditioned"]
        for _, group in patient.groupby("drug_id"):
            self.assertGreaterEqual(group["ModelID"].nunique(), 20)
            self.assertGreaterEqual(group["cancer_type"].nunique(), 3)
            self.assertEqual(set(group["Label"].astype(int)), {0, 1})
            self.assertLessEqual(len(group), 30)
            self.assertGreaterEqual(
                int(group["available_drug_unique_modelids"].min()), 20
            )
        broken = _valid_cases()
        patient_indices = broken.index[
            broken["selection_reason"] == "patient_conditioned"
        ]
        broken.loc[patient_indices[-1], "ModelID"] = broken.loc[
            patient_indices[0], "ModelID"
        ]
        with self.assertRaisesRegex(AssertionError, "per-drug"):
            validate_case_manifest(broken)


class Round19GTaskManifestTests(unittest.TestCase):
    def test_15_member_and_task_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case_path = root / "cases.csv"
            _valid_cases().to_csv(case_path, index=False)
            inventory = []
            for source in ("P2_source", "P0_source"):
                for member in range(15):
                    checkpoint = root / f"{source}_{member}.pt"
                    checkpoint.write_bytes(f"{source}:{member}".encode())
                    inventory.append(
                        {
                            "source_candidate_id": source,
                            "member_id": f"seed{member // 5}_fold{member % 5}",
                            "checkpoint_path": str(checkpoint),
                            "checkpoint_sha256": hashlib.sha256(
                                checkpoint.read_bytes()
                            ).hexdigest(),
                            "checkpoint_size_bytes": checkpoint.stat().st_size,
                        }
                    )
            lock = {
                "lock_type": "round19_final_role_lock",
                "schema_version": 1,
                "immutable": True,
                "roles": {
                    "attention_role": {"source_candidate_id": "P2_source"},
                    "pooled_role": {"source_candidate_id": "P0_source"},
                    "duplicate_alias": {"source_candidate_id": "P2_source"},
                },
                "hashes": {"checkpoint_inventory": inventory},
            }
            lock["hashes"]["lock_payload_sha256"] = canonical_sha256(lock)
            lock_path = root / "final_lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            config = {
                "candidate_components": {
                    "P2_source": {"predictor_id": "P2"},
                    "P0_source": {"predictor_id": "P0"},
                },
                "methods": {
                    method: {"enabled": True}
                    for method in ("attention", "occlusion", "omics", "routing")
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            summary = build_task_manifests(
                case_manifest_path=case_path,
                config_path=config_path,
                final_lock_path=lock_path,
                project_root=root,
                output_dir=root / "tasks",
            )
            self.assertEqual(summary["case_count"], 153)
            self.assertEqual(summary["manifests"]["attention"]["tasks"], 90)
            self.assertEqual(summary["manifests"]["occlusion"]["tasks"], 180)
            self.assertEqual(summary["manifests"]["omics"]["tasks"], 180)
            self.assertEqual(summary["manifests"]["routing"]["tasks"], 180)
            self.assertEqual(
                summary["manifests"]["occlusion"]["members_per_candidate"], 15
            )
            attention = pd.read_csv(summary["manifests"]["attention"]["path"])
            self.assertEqual(
                set(attention["cohort_scope"]),
                {"primary_faithfulness", "tcga_exploratory"},
            )


if __name__ == "__main__":
    unittest.main()
