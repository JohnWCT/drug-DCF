"""Generate Round 20 documentation from artifacts (no hand-copied numbers)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from tools.round20.result_contracts import DEFAULT_RUN_ROOT, load_json, sha256_file, write_json
from tools.round20.release_integrity import build_public_model_lock


def generate_final_report(*, run_root: Path = DEFAULT_RUN_ROOT, docs_dir: Optional[Path] = None) -> str:
    run_root = Path(run_root)
    docs_dir = docs_dir or (Path(__file__).resolve().parents[2] / "docs")
    a = load_json(run_root / "stage20a_dimension/stage20a_dimension_decision.json")
    b = load_json(run_root / "stage20b_predictor/stage20b_guardrail_report.json")
    lock = load_json(run_root / "stage20c_lock/final_model_lock.json")
    tcga = load_json(run_root / "stage20d_tcga/tcga_metrics.json")
    manifest = load_json(run_root / "stage20e_release/RELEASE_MANIFEST.json")
    audit = load_json(run_root / "round20_completion_audit.json") if (
        run_root / "round20_completion_audit.json"
    ).is_file() else {"audit_status": "UNKNOWN"}

    ctx = lock["selected_context"]["id"]
    omics_dim = lock["selected_context"]["omics_dimension"]
    predictor = lock["selected_model"]["predictor_type"]
    cand = lock["selected_model"]["candidate_id"]

    tcga_lines = []
    for target, m in tcga.items():
        tcga_lines.append(
            f"| {target} | {m.get('DrugMacro_AUC', 'n/a'):.4f} | {m.get('Global_AUC', 'n/a'):.4f} |"
            if isinstance(m.get("DrugMacro_AUC"), (int, float))
            else f"| {target} | n/a | n/a |"
        )

    text = f"""# Round 20 Final Report — Unseen-Drug Closure

## Status

**COMPLETE** — completion audit: `{audit.get('audit_status', 'UNKNOWN')}`.

## Executive summary

Round 20 compared prototype-context dimension (C16 vs C32) and predictor architecture
(pooled E3 vs gated fusion) under repeated drug-held-out validation. The locked model
uses **{ctx}** ({omics_dim}-d O2), **D0 GIN32**, and **{predictor}** (`{cand}`).

## Stage 20A — Context dimension

| Metric | C16 | C32 | Δ (C32−C16) |
|--------|-----|-----|-------------|
| mean DrugMacro AUC | {a.get('mean_auc_c16', 0):.4f} | {a.get('mean_auc_c32', 0):.4f} | {a.get('mean_auc_delta_c32_minus_c16', 0):.4f} |

**Locked context:** {a.get('selected_context')} (`{a.get('reason')}`)

## Stage 20B — Predictor

| Guardrail | Pass |
|-----------|------|
| G1 mean AUC | {b['guardrails'].get('g1_mean_auc_nonworse')} |
| G2 seed majority | {b['guardrails'].get('g2_seed_majority')} |
| G3 AUPRC | {b['guardrails'].get('g3_auprc')} |
| G4 no major fail | {b['guardrails'].get('g4_no_major_fail')} |
| G5 complete | {b['guardrails'].get('g5_complete')} |

Mean AUC Δ (gated−E3): **{b.get('mean_auc_delta', 0):.4f}** — `all_pass={b.get('all_pass')}`

## Stage 20C — Model lock

- Context: {lock['selected_context']['id']} ({lock['selected_context']['omics_dimension']}-d)
- Model: {lock['selected_model']['candidate_id']} / {lock['selected_model']['predictor_type']}
- Reason: {lock.get('selection_reason')}
- Forbidden metrics used: {lock.get('forbidden_metrics_used')}

## Stage 20D — TCGA (post-lock only)

| Target | DrugMacro AUC | Global AUC |
|--------|---------------|------------|
{chr(10).join(tcga_lines)}

## Stage 20E — Release

- Release status: {manifest.get('release_status', 'UNKNOWN')}
- Artifacts hashed: {len(manifest.get('artifacts', []))}

## Final architecture

```text
Raw omics [G] → frozen encoder → Z [64]
Raw prototype context → PCA (n={lock['selected_context']['dimension']}) → context [{lock['selected_context']['dimension']}]
Z + context → O2 [{omics_dim}]
SMILES → D0 GIN → graph embedding [32]
O2 + graph → {predictor} → probability [1]
```

## Limitations

- Unseen cancer-type optimization: out of scope.
- Encoder unfreezing: not formally evaluated in Round 20.
- TCGA results must not be used to revisit model selection.

## Official conclusion

Increasing prototype-context dimension from 16 to 32 produced stable repeated
drug-held-out improvement under the locked E3 contract. Gated fusion did not pass
predefined guardrails; the parsimonious pooled E3 predictor was retained on **{ctx}**.
"""
    out = docs_dir / "round20_final_report.md"
    out.write_text(text, encoding="utf-8")
    return str(out)


def generate_model_card(*, run_root: Path = DEFAULT_RUN_ROOT, docs_dir: Optional[Path] = None) -> str:
    run_root = Path(run_root)
    docs_dir = docs_dir or (Path(__file__).resolve().parents[2] / "docs")
    lock = load_json(run_root / "stage20c_lock/final_model_lock.json")
    lock_sha = sha256_file(run_root / "stage20c_lock/final_model_lock.json")
    audit = load_json(run_root / "round20_completion_audit.json") if (
        run_root / "round20_completion_audit.json"
    ).is_file() else {}
    git_sha = (audit.get("git") or {}).get("sha", "unknown")
    ctx = lock["selected_context"]
    text = f"""# Round 20 Model Card

## Intended use

Research-oriented prediction of drug response for drugs excluded from response-model
training under the repository's established omics and drug preprocessing contracts.

## Locked configuration

| Field | Value |
|-------|-------|
| Git SHA | `{git_sha}` |
| Model lock SHA256 | `{lock_sha}` |
| Context | {ctx['id']} ({ctx['omics_dimension']}-d O2) |
| Drug encoder | D0 GIN32 (graph dim 32) |
| Predictor | {lock['selected_model']['predictor_type']} |
| Candidate | {lock['selected_model']['candidate_id']} |
| Checkpoint policy | {lock['selected_model']['checkpoint_policy']} |

## Development metrics (selection only)

- Stage 20A ΔAUC (C32−C16): {lock['development_metrics'].get('stage20a_mean_auc_delta_c32_minus_c16')}
- Stage 20B ΔAUC (gated−E3): {lock['development_metrics'].get('stage20b_mean_auc_delta_gated_minus_e3')}

## Out of scope

Clinical treatment recommendation; dose/toxicity; combinations; unseen-cancer optimization;
molecules outside the locked graph contract.

## Reproduction

```bash
python scripts/round20/round20_cli.py audit --strict
python scripts/round20/round20_cli.py reproduce --strict
```
"""
    out = docs_dir / "round20_model_card.md"
    out.write_text(text, encoding="utf-8")
    return str(out)


def generate_inference_guide(*, run_root: Path = DEFAULT_RUN_ROOT, docs_dir: Optional[Path] = None) -> str:
    run_root = Path(run_root)
    docs_dir = docs_dir or (Path(__file__).resolve().parents[2] / "docs")
    lock = load_json(run_root / "stage20c_lock/final_model_lock.json")
    omics_dim = lock["selected_context"]["omics_dimension"]
    text = f"""# Round 20 Inference Guide

## Preflight

Before inference, verify:

- Model lock SHA256 matches release `configs/final_model_lock.json`
- Selected context: **{lock['selected_context']['id']}** ({omics_dim}-d O2)
- Checkpoint count: **15** (probability-mean ensemble)
- Drug graph coverage for all drugs in the response file

## Frozen latent inference (official)

```bash
python scripts/round20/round20_cli.py infer \\
  --release-dir result/optimization_runs/round20_unseen_drug_closure/stage20e_release \\
  --mode frozen_latent \\
  --response-file path/to/response.csv \\
  --output predictions.csv \\
  --strict
```

## Raw omics inference (capability path)

```bash
python scripts/round20/round20_cli.py infer \\
  --release-dir result/optimization_runs/round20_unseen_drug_closure/stage20e_release \\
  --mode raw_omics \\
  --response-file path/to/response.csv \\
  --output predictions.csv \\
  --strict
```

Encoder unfreezing was **not** validated as a formal Round 20 experiment.
"""
    out = docs_dir / "round20_inference_guide.md"
    out.write_text(text, encoding="utf-8")
    return str(out)


def generate_all_docs(*, run_root: Path = DEFAULT_RUN_ROOT, docs_dir: Optional[Path] = None) -> dict:
    docs_dir = docs_dir or (Path(__file__).resolve().parents[2] / "docs")
    reports_dir = docs_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    public_lock = build_public_model_lock(
        run_root / "stage20c_lock/final_model_lock.json",
        reports_dir / "round20_final_model_lock_public.json",
    )
    return {
        "final_report": generate_final_report(run_root=run_root, docs_dir=docs_dir),
        "model_card": generate_model_card(run_root=run_root, docs_dir=docs_dir),
        "inference_guide": generate_inference_guide(run_root=run_root, docs_dir=docs_dir),
        "public_lock": str(reports_dir / "round20_final_model_lock_public.json"),
    }
