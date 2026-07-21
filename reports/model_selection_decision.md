# BioCDA Round 21 Model Selection Decision

- Status: **REJECTED** (cross-attention candidates did not pass performance guardrails)
- Fallback / retained predictive baseline: **pooled_baseline** (M0)
- Protocol: repeated unseen-drug on GDSC development rows
- Primary metric: `drug_macro_auc` (DrugMacro AUROC)
- Secondary: `drug_macro_auprc`
- **TCGA not used for selection**

## Candidate models

| ID | Factory name | Query | Drug path |
|----|--------------|-------|-----------|
| M0 | `pooled_baseline` | `[Z;C]` sample repr | D0 GIN global max-pool + adapter fusion |
| M1 | `biocda_xa_z` | Z only | sample→atom cross-attention (no pool bypass) |
| M2 | `biocda_xa_zc` | `[Z;C]` | sample→atom cross-attention (no pool bypass) |

## Selection protocol

- Seeds: 17, 29, 43 (shared split manifest)
- Freeze Phase A: omics encoder, context, GIN frozen; train sample projection, cross-attention, response head
- Early stop: validation DrugMacro AUC, patience 20
- Gates: functional correctness, performance vs M0, context utilization, attention health

## Observed validation metrics (DrugMacro AUC)

| model | seed 17 | seed 29 | seed 43 | mean |
|-------|---------|---------|---------|------|
| pooled_baseline (M0) | 0.746 | 0.715 | 0.776 | **0.746** |
| biocda_xa_z (M1) | 0.701 | 0.688 | 0.752 | 0.714 |
| biocda_xa_zc (M2) | 0.687 | 0.694 | 0.745 | 0.709 |

Paired mean ΔAUC (M2 − M0) ≈ **−0.037** (fails ≥ −0.005 guardrail).  
Paired mean ΔAUPRC (M2 − M0) ≈ **−0.045** (fails ≥ −0.010 guardrail).  
No seed has M2 AUROC superior to M0.

## Gate results

| Gate | Result |
|------|--------|
| functional_correctness | **PASS** |
| performance_guardrails | **FAIL** |
| context_utilization | **PASS** |
| attention_health | **PASS** |

## Failures

- `performance_guardrails`: BioCDA-XA-ZC underperforms pooled baseline on DrugMacro AUC/AUPRC across all three split seeds.

## Selected / rejected

- **Rejected for LOCKED BioCDA-XA**: M1 and M2 (cross-attention)
- **Retained predictive baseline**: M0 `pooled_baseline`
- Outcome matches Round 21 manual **Outcome 3**: do **not** enter TCGA interpretability; next round should study gated residual / attention distillation / limited last-block GIN fine-tuning (Phase B)

## Attention interface preserved for next round

Per-head probabilities, pre-softmax logits, atom_mask, atom_ptr, model/original/rdkit atom indices are available on BioCDA checkpoints under `outputs/xa_validation/` (gitignored).
