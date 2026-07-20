"""Synthetic Round 20 fixtures for unit tests."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def synthetic_run_root(tmp_path: Path) -> Path:
    root = tmp_path / "round20_unseen_drug_closure"
    _build_minimal_run(root)
    return root


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _job_metrics(path: Path, auc: float, auprc: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {"status": "COMPLETE", "metrics": {"DrugMacro_AUC": auc, "DrugMacro_AUPRC": auprc}},
    )
    _write_json(path.parent / "status.json", {"status": "COMPLETE"})


def _build_minimal_run(root: Path) -> None:
    s0 = root / "stage20_0"
    _write_json(s0 / "stage20_0_go.json", {"status": "GO"})
    _write_json(s0 / "resolved_e3.json", {"contract": "e3"})

    s20a = root / "stage20a_dimension"
    manifest_a = []
    auc_map = {"C16": 0.70, "C32": 0.71}
    for seed in (52, 62, 72):
        for fold in range(5):
            for ctx in ("C16", "C32"):
                job_id = f"r20a__A_{ctx}_E3__ss{seed}__f{fold}__ms101"
                out = s20a / "jobs" / job_id
                _job_metrics(out / "metrics.json", auc_map[ctx])
                manifest_a.append(
                    {
                        "job_id": job_id,
                        "context_id": ctx,
                        "candidate_id": "A_E3",
                        "split_seed": seed,
                        "fold": fold,
                        "output_dir": str(out),
                        "split_assignment_sha256": f"sha_seed_{seed}",
                    }
                )
    (s20a / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest_a) + "\n", encoding="utf-8"
    )
    _write_json(
        s20a / "stage20a_dimension_decision.json",
        {
            "status": "LOCKED",
            "selected_context": "C32",
            "mean_auc_delta_c32_minus_c16": 0.01,
            "reason": "stable_improvement",
        },
    )
    (s20a / "reports").mkdir(parents=True, exist_ok=True)
    for name in (
        "stage20a_pairwise.csv",
        "stage20a_candidate_summary.csv",
        "stage20a_seed_summary.csv",
    ):
        (s20a / "reports" / name).write_text("x\n", encoding="utf-8")

    s20b = root / "stage20b_predictor"
    manifest_b = []
    for seed in (52, 62, 72):
        for fold in range(5):
            for cand, auc in (("B_E3", 0.72), ("B_GATED", 0.70)):
                skip = cand == "B_E3"
                job_id = f"r20b__{cand}__C32__ss{seed}__f{fold}__ms101"
                if skip:
                    out = s20a / "jobs" / f"r20a__A_C32_E3__ss{seed}__f{fold}__ms101"
                else:
                    out = s20b / "jobs" / job_id
                    _job_metrics(out / "metrics.json", auc)
                manifest_b.append(
                    {
                        "job_id": job_id,
                        "candidate_id": cand,
                        "split_seed": seed,
                        "fold": fold,
                        "output_dir": str(out),
                        "skip_train": skip,
                    }
                )
    (s20b / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest_b) + "\n", encoding="utf-8"
    )
    _write_json(
        s20b / "stage20b_guardrail_report.json",
        {"all_pass": False, "mean_auc_delta": -0.01, "guardrails": {"g1_mean_auc_nonworse": False}},
    )
    (s20b / "reports").mkdir(parents=True, exist_ok=True)
    for name in (
        "stage20b_pairwise.csv",
        "stage20b_candidate_summary.csv",
        "stage20b_seed_summary.csv",
    ):
        (s20b / "reports" / name).write_text("x\n", encoding="utf-8")

    s20c = root / "stage20c_lock"
    _write_json(
        s20c / "final_model_lock.json",
        {
            "stage": "20C",
            "status": "LOCKED",
            "created_at": "2026-07-17T10:00:00+00:00",
            "forbidden_metrics_used": False,
            "selection_reason": "gated_failed_guardrails",
            "selected_context": {
                "id": "C32",
                "dimension": 32,
                "omics_dimension": 96,
                "feature_dir": str(root / "features/C32"),
            },
            "selected_model": {
                "candidate_id": "B_E3",
                "predictor_type": "pooled_e3",
                "drug_encoder": "D0",
                "checkpoint_policy": "probability_mean_ensemble",
            },
            "development_metrics": {},
            "checkpoint_policy": {},
            "input_hashes": {},
            "source_git_sha": "abc",
        },
    )

    for seed in (52, 62, 72):
        for fold in range(5):
            job_id = f"r20a__A_C32_E3__ss{seed}__f{fold}__ms101"
            ckpt = s20a / "jobs" / job_id / "best_checkpoint.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_bytes(b"ckpt")

    s20d = root / "stage20d_tcga"
    _write_json(s20d / "stage20d_tcga_preflight.json", {"status": "PASS"})
    _write_json(s20d / "stage20d_tcga_summary.json", {"status": "COMPLETE"})
    rows = pd.DataFrame(
        {
            "row_id": ["r0", "r1"],
            "drug_id": ["d1", "d2"],
            "true_label": [0, 1],
            "prediction_probability": [0.3, 0.7],
            "checkpoint_count": [2, 2],
        }
    )
    ckpt = pd.DataFrame(
        {
            "_row_id": ["r0", "r1"],
            "prob_ckpt0": [0.2, 0.6],
            "prob_ckpt1": [0.4, 0.8],
        }
    )
    metrics_payload = {}
    for suffix in (
        "gdsc_intersect13",
        "tcga_only3",
        "dapl",
        "aacdr_tcga_only",
        "aacdr_gdsc_intersect",
    ):
        rows.to_csv(s20d / f"predictions_ensemble__{suffix}.csv", index=False)
        ckpt.to_csv(s20d / f"predictions_by_checkpoint__{suffix}.csv", index=False)
        metrics_payload[suffix] = {
            "DrugMacro_AUC": None,
            "DrugMacro_AUPRC": None,
            "Global_AUC": 1.0,
            "Global_AUPRC": 1.0,
            "per_drug_records": [],
        }
    _write_json(s20d / "tcga_metrics.json", metrics_payload)

    s20e = root / "stage20e_release"
    _write_json(
        s20e / "RELEASE_MANIFEST.json",
        {
            "release_status": "LOCKED",
            "artifacts": [],
            "selection": {"context": "C32", "predictor": "B_E3", "drug_encoder": "D0"},
        },
    )
    for doc in ("MODEL_CARD.md", "DATASET_CARD.md", "INFERENCE_GUIDE.md", "LIMITATIONS.md"):
        (s20e / doc).write_text("# doc\n", encoding="utf-8")
    _write_json(s20e / "hashes/release_audit.json", {"status": "PASS"})
    (s20e / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(s20c / "final_model_lock.json", s20e / "configs/final_model_lock.json")
    ckpt_dir = s20e / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for i in range(15):
        (ckpt_dir / f"fold{i}.pt").write_bytes(b"x")
