#!/usr/bin/env python3
"""Round 20 Stage 20C: immutable final model lock (development metrics only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

FORBIDDEN_SELECTION_KEYS = {
    "tcga_auc",
    "tcga_auprc",
    "internal_auc",
    "external_auc",
    "integrated5_auc",
    "tcga",
    "internal_test",
    "posthoc",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result/optimization_runs/round20_unseen_drug_closure"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_no_forbidden(payload: dict, *, path: str) -> None:
    stack = [payload]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                key = str(k).lower()
                if key in FORBIDDEN_SELECTION_KEYS or any(tok in key for tok in ("tcga", "internal_test", "posthoc")):
                    # allow attestation flags that are explicitly false
                    if isinstance(v, (bool, type(None), int)) and not v:
                        continue
                    if key.endswith("_used") and v in (False, None, 0):
                        continue
                    raise ValueError(
                        f"Selection input contains forbidden post-lock metrics: {k} in {path}"
                    )
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)


def select_final_model(
    guardrail_report: dict,
    *,
    parsimony_threshold: float = 0.005,
) -> tuple[str, str]:
    if not guardrail_report.get("all_pass"):
        return "B_E3", "gated_failed_guardrails"
    delta = float(guardrail_report.get("mean_auc_delta", 0.0))
    if abs(delta) < parsimony_threshold:
        return "B_E3", "parsimony"
    if delta > 0:
        return "B_GATED", "stable_auc_improvement"
    return "B_E3", "baseline_better"


def build_final_model_lock(
    *,
    stage20a_decision_path: Path,
    stage20b_guardrails_path: Path,
    output_path: Path,
    parsimony_threshold: float = 0.005,
) -> dict:
    dim = json.loads(Path(stage20a_decision_path).read_text(encoding="utf-8"))
    guard = json.loads(Path(stage20b_guardrails_path).read_text(encoding="utf-8"))
    _assert_no_forbidden(dim, path=str(stage20a_decision_path))
    _assert_no_forbidden(guard, path=str(stage20b_guardrails_path))
    if dim.get("status") != "LOCKED":
        raise ValueError("Stage 20A decision is not LOCKED")

    selected_model, reason = select_final_model(guard, parsimony_threshold=parsimony_threshold)
    predictor_type = (
        "gated_pooled_fusion" if selected_model == "B_GATED" else "AdapterMLPFusion+ResponseHead"
    )
    lock = {
        "stage": "20C",
        "status": "LOCKED",
        "selected_context": {
            "id": dim["selected_context"],
            "dimension": dim["selected_context_dim"],
            "omics_dimension": dim["selected_omics_dim"],
            "feature_dir": dim["selected_feature_dir"],
            "projection_sha256": _sha256_file(
                Path(dim["selected_feature_dir"]) / "projection_model.pkl"
            ),
        },
        "selected_model": {
            "candidate_id": selected_model,
            "predictor_type": predictor_type,
            "drug_encoder": "D0",
            "checkpoint_policy": "five_fold_probability_mean_ensemble",
        },
        "selection_reason": reason,
        "guardrails": guard.get("guardrails", {}),
        "development_metrics": {
            "stage20a_mean_auc_delta_c32_minus_c16": dim.get("mean_auc_delta_c32_minus_c16"),
            "stage20b_mean_auc_delta_gated_minus_e3": guard.get("mean_auc_delta"),
            "stage20b_mean_auprc_delta": guard.get("mean_auprc_delta"),
            "stage20b_seed_auc_deltas": guard.get("seed_auc_deltas"),
        },
        "forbidden_metrics_used": False,
        "input_hashes": {
            "stage20a_decision_sha256": _sha256_file(Path(stage20a_decision_path)),
            "stage20b_guardrails_sha256": _sha256_file(Path(stage20b_guardrails_path)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = Path(output_path)
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("status") == "LOCKED":
            raise RuntimeError(
                f"Refusing to overwrite immutable lock at {output_path}; "
                "create a new version with supersedes instead"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage20a-decision",
        default=str(RESULT_ROOT / "stage20a_dimension/stage20a_dimension_decision.json"),
    )
    p.add_argument(
        "--stage20b-guardrails",
        default=str(RESULT_ROOT / "stage20b_predictor/stage20b_guardrail_report.json"),
    )
    p.add_argument(
        "--output",
        default=str(RESULT_ROOT / "stage20c_lock/final_model_lock.json"),
    )
    p.add_argument("--strict", action="store_true", default=True)
    args = p.parse_args()
    lock = build_final_model_lock(
        stage20a_decision_path=Path(args.stage20a_decision),
        stage20b_guardrails_path=Path(args.stage20b_guardrails),
        output_path=Path(args.output),
    )
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
