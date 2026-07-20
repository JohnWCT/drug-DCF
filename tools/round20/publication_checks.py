"""README and GitHub release note generation from locked artifacts."""
from __future__ import annotations

from pathlib import Path

from tools.round20.result_contracts import DEFAULT_RUN_ROOT, load_json


def build_readme_section(*, run_root: Path = DEFAULT_RUN_ROOT) -> str:
    lock = load_json(run_root / "stage20c_lock/final_model_lock.json")
    ctx = lock["selected_context"]
    model = lock["selected_model"]
    a = load_json(run_root / "stage20a_dimension/stage20a_dimension_decision.json")
    return f"""
## Current recommended drug-response model

The current locked model was selected under repeated drug-held-out validation in Round 20.

- Omics representation: **{ctx['id']}** O2 ({ctx['omics_dimension']}-d = Z64 + context{ctx['dimension']})
- Drug encoder: **{model['drug_encoder']}** GIN32 / graph32
- Predictor: **{model['predictor_type']}** (`{model['candidate_id']}`)
- Primary use case: unseen-drug prediction
- Model selection: development-only repeated drug-held-out (seeds 52/62/72 × 5 folds)
- Final TCGA evaluation: performed **after** model lock

Context selection (Stage 20A): {a.get('selected_context')} — ΔAUC(C32−C16) = {a.get('mean_auc_delta_c32_minus_c16', 0):.4f}

## Project reports

- [Round 20 final report](docs/round20_final_report.md)
- [Round 20 model card](docs/round20_model_card.md)
- [Round 20 inference guide](docs/round20_inference_guide.md)
- [Round 19 final report](docs/round19_final_report.md)

## Scope and limitations

The Round 20 model focuses on **unseen-drug transfer**. Unseen cancer-type transfer was not
optimized in this round. The omics encoder was frozen during formal model selection. The
repository retains an end-to-end-capable path, but encoder unfreezing was not validated as a
formal Round 20 experiment.
"""


def build_github_release_notes(*, run_root: Path = DEFAULT_RUN_ROOT) -> str:
    lock = load_json(run_root / "stage20c_lock/final_model_lock.json")
    a = load_json(run_root / "stage20a_dimension/stage20a_dimension_decision.json")
    b = load_json(run_root / "stage20b_predictor/stage20b_guardrail_report.json")
    return f"""# Round 20 unseen-drug closure

## Scope

Repeated drug-held-out evaluation of prototype-context dimension and pooled predictor architecture.

## Locked model

- Context: **{lock['selected_context']['id']}** ({lock['selected_context']['omics_dimension']}-d O2)
- Drug encoder: {lock['selected_model']['drug_encoder']}
- Predictor: {lock['selected_model']['predictor_type']} (`{lock['selected_model']['candidate_id']}`)
- Selection reason: {lock.get('selection_reason')}

## Validation (development only)

- Stage 20A: {a.get('selected_context')} selected (ΔAUC C32−C16 = {a.get('mean_auc_delta_c32_minus_c16', 0):.4f})
- Stage 20B: gated `all_pass={b.get('all_pass')}` (ΔAUC = {b.get('mean_auc_delta', 0):.4f})

## TCGA evaluation

Performed after model lock. See `docs/round20_stage20d_tcga_report.md`.

## Reproduction

See [docs/round20_inference_guide.md](docs/round20_inference_guide.md).

## Limitations

Unseen cancer-type optimization and formal encoder unfreezing were outside Round 20 scope.
"""
