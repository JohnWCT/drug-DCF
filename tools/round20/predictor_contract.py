"""Build Stage 20B predictor contract and gate diagnostics artifacts."""
from __future__ import annotations

from pathlib import Path

from tools.round20.result_contracts import DEFAULT_RUN_ROOT, sha256_file, write_json

GATED_SOURCE = Path(__file__).resolve().parents[1] / "round20_gated_fusion.py"


def build_predictor_contract(*, run_root: Path = DEFAULT_RUN_ROOT) -> dict:
    contract = {
        "stage": "20B",
        "gated_implementation": {
            "module": "tools.round20_gated_fusion.GatedPooledFusionPredictor",
            "source_path": str(GATED_SOURCE),
            "source_sha256": sha256_file(GATED_SOURCE) if GATED_SOURCE.is_file() else None,
            "omics_projection": "linear",
            "drug_projection": "linear",
            "gate_network": "sigmoid_gate",
            "fusion_equation": "gate * omics_proj + (1 - gate) * drug_proj",
            "prediction_head": "AdapterMLPFusion+ResponseHead",
            "dropout": 0.1,
            "hidden_dimension": 128,
            "head_dimension": 64,
            "d0_trainable_mode": "frozen",
        },
        "e3_implementation": {
            "module": "AdapterMLPFusion+ResponseHead",
            "source_stage": "20A",
            "contract": "resolved_e3.json",
        },
    }
    out = run_root / "stage20b_predictor/stage20b_predictor_contract.json"
    write_json(out, contract)
    return contract


def build_gate_summary(*, run_root: Path = DEFAULT_RUN_ROOT) -> dict:
    """Gate collapse audit when per-fold gate dumps are unavailable."""
    summary = {
        "status": "NO_GATE_ARTIFACTS",
        "collapse_classification": "UNKNOWN",
        "note": (
            "Formal training did not persist per-validation gate tensors. "
            "Gate collapse audit deferred; gated model failed guardrails on AUC."
        ),
        "diagnostics": {
            "gate_mean": None,
            "gate_std": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "fraction_lt_0_05": None,
            "fraction_gt_0_95": None,
        },
    }
    out = run_root / "stage20b_predictor/gate_summary.json"
    write_json(out, summary)
    return summary
