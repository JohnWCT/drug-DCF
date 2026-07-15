"""Synthetic analyzer checks for 19E."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tools.round19_stage19e_analyzer import _guardrail, analyze_stage19e


def test_guardrail_thresholds():
    assert _guardrail(0.003) == "PASS"
    assert _guardrail(0.0) == "NON_WORSE"
    assert _guardrail(-0.003) == "FAIL"


def test_analyzer_synthetic(tmp_path: Path):
    root = tmp_path / "round19"
    for strategy in ("drug_heldout", "scaffold_heldout", "cancer_type_heldout"):
        man_dir = root / "manifests"
        man_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for eid, base in [("E0", 0.50), ("E1", 0.55), ("E3", 0.52)]:
            for fold in range(5):
                job_id = f"{eid}__{strategy}__fold{fold}"
                rd = root / "stage19e" / strategy / job_id
                rd.mkdir(parents=True, exist_ok=True)
                (rd / "job_status.json").write_text(json.dumps({"status": "done"}))
                (rd / "val_metrics.json").write_text(
                    json.dumps(
                        {
                            "DrugMacro_AUC": base + 0.001 * fold,
                            "DrugMacro_AUPRC": 0.3,
                            "Global_AUC": 0.8,
                            "Global_AUPRC": 0.4,
                            "fallback_used": False,
                            "n_valid_drugs": 5,
                            "Brier": 0.2,
                            "ECE": 0.05,
                        }
                    )
                )
                rows.append(
                    {
                        "job_id": job_id,
                        "candidate_id": eid,
                        "source_candidate_id": eid,
                        "fold_id": fold,
                        "drug_id": "D0",
                        "predictor_id": "P0",
                        "omics_id": "O1",
                        "result_dir": str(rd),
                    }
                )
        pd.DataFrame(rows).to_csv(man_dir / f"stage19e_{strategy}_manifest.csv", index=False)
    summary = analyze_stage19e(str(root), require_complete=True)
    assert summary["n_done"] == 45
    assert (root / "reports" / "round19e_shift_guardrails.csv").is_file()
