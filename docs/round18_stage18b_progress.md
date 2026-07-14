# Round 18 Stage 18B Progress Report

**Date:** 2026-07-13  
**Status:** **ALL_DONE — 45/45 screening jobs complete**  
**Downstream:** Round 18C–18E complete；**18F pending** — see [`docs/round18_final_report.md`](round18_final_report.md).
**Root:** `result/optimization_runs/round18_architecture/`

---

## 1. Scope

Round 18B compares **pooled MLP** and **pooled Transformer** fusion on frozen Round 17R omics features, with ModelID-grouped 3-fold screening CV.

| Item | Value |
|------|-------|
| Jobs | **45/45 done** |
| Omics modes | `none`, `own_plus_summary`, `own_proto_context_projected_16` |
| Split | Eligible GDSC only (`round18_eligible_response.csv`, n=110279; internal test ≈10.03%) |
| Primary metric | Fold-mean **DrugMacro AUC** (no internal test / TCGA in selection) |
| Batching | target effective 1024; successful micro-batch typically **512** (accum=2) |
| Docker env | `scikit-learn==1.3.2` via `requirements-round18.txt`; `/dev/shm=64MB` → `ROUND18_NUM_WORKERS=0` |

---

## 2. Pilot (pre-screen)

Distinct `*_pilot100` job IDs / result dirs (do not collide with formal 18B):

| Job | best_epoch | DrugMacro AUC | n_epochs |
|-----|------------|---------------|----------|
| MLP × own_plus_summary fold0 | 73 | **0.583** | 100 |
| P0 historical Transformer (corrected mask) × own_plus_summary fold0 | 8 | **0.559** | 80 |

---

## 3. Full screening timeline

1. **First dispatch** (high packing, up to 40 jobs/GPU): finished 2026-07-12 ~16:59Z with **38 done / 7 SIGKILL (−9)** from mid-run restarts while tuning parallelism.
2. **Retry** of 7 killed jobs (`MAX_JOBS_PER_GPU=7`, workers=0): **7/7 done**, overall **45/45**.
3. **QC:** all 45 jobs have required artifacts; finite train loss; non-degenerate val probabilities; `n_valid_auc_drugs ≥ 3`.

Artifacts per job: `job_status.json`, `checkpoint.pt`, `train_history.csv`, `train_summary.json`, `val_predictions.csv`, `val_metrics.json`, `runtime_resource_summary.json`.

---

## 4. Final 18B ranking (3-fold mean DrugMacro AUC)

Source: `reports/round18_screening_architecture_ranking.csv` (via `tools/analyze_round18.py`).

| Rank | Architecture | Omics | mean DrugMacro AUC | mean AUPRC | mean Global AUC |
|------|--------------|-------|--------------------|------------|-----------------|
| 1 | P3_deeper128 | context16 | **0.6171** | 0.4172 | 0.8161 |
| 2 | P1_compact64 | context16 | 0.6128 | 0.4179 | 0.7994 |
| 3 | **pooled MLP** | own_plus_summary | 0.6127 | 0.4132 | 0.7943 |
| 4 | P2_standard128 | context16 | 0.6125 | 0.4112 | 0.7962 |
| 5 | pooled MLP | context16 | 0.6120 | 0.4057 | 0.7982 |
| 6 | P3_deeper128 | own_plus_summary | 0.6063 | 0.4044 | 0.8341 |
| 7 | P2_standard128 | none | 0.6038 | 0.4021 | 0.7965 |
| 8 | P2_standard128 | own_plus_summary | 0.6037 | 0.4046 | 0.8216 |
| 9 | P3_deeper128 | none | 0.6019 | 0.4040 | 0.8191 |
| 10 | P1_compact64 | own_plus_summary | 0.5980 | 0.3987 | 0.8069 |
| … | P0 historical (corrected mask) | all omics | 0.576–0.591 | — | — |

### Takeaways

- Best screening candidate: **`pooled_transformer__P3_deeper128__own_proto_context_projected_16`**.
- Strong MLP baseline: **`pooled_mlp__own_plus_summary`** ranks #3 and is competitive with larger Transformers.
- **P0 historical hparams (corrected mask)** underperforms modern P1–P3 configs under this protocol.
- Top ranks favor **`own_proto_context_projected_16`** (unlabeled TCGA prototype context — disclose in final report; not TCGA response-label leakage).

---

## 5. Ops / infrastructure notes

- Feature ModelID coverage preflight: **937 IDs identical** across O0/O1/O2 (dims 64/75/91).
- Stage **18D** refuses to build without `reports/round18_locked_selection.json` unless `--allow-placeholder-for-smoke`.
- Selection lock was written **after** 18C-A (48/48) + 18C-B none follow-up (6/6). Full results: `docs/round18_final_report.md`.

---

## 6. Scientific caveats

1. Omics encoder is **frozen**; CV is grouped response-prediction CV with pretrained/transductive omics representations.
2. `own_proto_context_projected_16` includes unlabeled target-domain prototype features.

---

## 7. Next steps

1. ~~Run Stage **18C** …~~ **DONE**
2. ~~`--write-lock` for formal candidates~~ **DONE**
3. ~~Stage **18D** formal 5CV~~ **DONE**
4. ~~**18E** internal + TCGA ensemble eval~~ **DONE**（external success = false；見 [`round18_stage18e_report.md`](round18_stage18e_report.md)）
5. **18F** attention export / masking — **尚未執行**（Round 18 全完成前的剩餘項）

---

## 8. Key paths

```text
manifests/stage18b_screening_manifest.csv
manifests/job_status.csv
manifests/job_status_retry7.csv
stage18b/**/fold_*/
reports/round18_screening_architecture_ranking.csv
reports/round18_job_completion_summary.csv
reports/round18_final_report.md
docs/round18_stage18b_progress.md
```
