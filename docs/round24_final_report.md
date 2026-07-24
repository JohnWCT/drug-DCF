# Round 24 — Final Report

**Status:** `LOCKED`  
**Container:** `DAPL` · `/workspace/DAPL`  
**Standard:** AACDR **stest0**（無 10% testset）— [`AACDR_drug_macro_auroc_auprc.md`](AACDR_drug_macro_auroc_auprc.md)  
**Hard gate:** `aacdr_gdsc_intersect` > **0.5279** ∧ `aacdr_tcga_only` > **0.4804**  
**Champion:** **`E-NH0`** — `pooled_mlp` × `own_plus_summary` × **NoHoldout** formal 5-fold

## Selection summary

| ID | Architecture × Feature | Train data | aacdr_gdsc | aacdr_tcga | Hard gate | Lock-eligible |
|----|------------------------|------------|-----------:|-----------:|:---------:|:-------------:|
| **E-NH0** | pooled_mlp × own_plus_summary | NoHoldout | **0.5648** | 0.4971 | **PASS** | Y ← **champion** |
| E-NH1 | biocda_predictive_e3 × C16 | NoHoldout | 0.5501 | 0.4992 | **PASS** | Y |
| E-NH2 | biocda_predictive_e3 × C32 | NoHoldout | 0.5210 | 0.5167 | NO_LOCK | Y |
| E-REF2 | predictive × C16（24C F2） | holdout ref | 0.5427 | 0.5398 | PASS | N |
| E-REF3 | predictive × C32（24C F3） | holdout ref | 0.5268 | 0.4730 | NO_LOCK | N |

Among NoHoldout PASS candidates, ranking used weights  
`aacdr_gdsc`(5) > `aacdr_tcga_only`(4) > `dapl`(3) > `gdsc_intersect13`(2) > `tcga_only3`(1).  
**E-NH0** wins primarily on `aacdr_gdsc_intersect`.

## Champion metrics (5-fold mean DrugMacro)

| Target | AUROC | AUPRC | Required for lock |
|--------|------:|------:|:-----------------:|
| `aacdr_gdsc_intersect` | 0.5648 | 0.6186 | Y |
| `aacdr_tcga_only` | 0.4971 | 0.6532 | Y |
| `dapl` | 0.4820 | 0.5416 | N |
| `gdsc_intersect13` | 0.5697 | 0.6121 | N |
| `tcga_only3` | 0.4845 | 0.6368 | N |

## Scientific notes

- Training protocol aligned with stest0: **no 10% GDSC internal holdout** (development ∪ internal_test → formal 5-fold).
- TCGA labels never entered loss / early stopping / checkpoint selection.
- Holdout 24C refs are diagnostic only and were excluded from lock ranking.
- Predictive C16/C32 under NoHoldout did **not** beat pooled own_plus_summary on the primary hard-gate axis (`aacdr_gdsc_intersect`).

## Artifacts

- Lock: [`reports/round24_final_model_lock.json`](../reports/round24_final_model_lock.json)
- Decision: [`reports/round24/stage24e/stage24e_decision.json`](../reports/round24/stage24e/stage24e_decision.json)
- Manifest: [`reports/round24/stage24e/candidate_manifest.json`](../reports/round24/stage24e/candidate_manifest.json)
- vs standard: [`reports/round24/vs_aacdr_standard.md`](../reports/round24/vs_aacdr_standard.md)
