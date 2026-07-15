# Round 19 Stage 19E Report — Domain-shift Validation

**Date:** 2026-07-15  
**Status:** **ALL_DONE — 90/90 jobs done，0 failed**  
**Stage gate:** Stage 19E = **GO complete**；**Formal selection lock = NO-GO**（需人工審查後寫 `round19_final_model_roles.json` / selection lock）  
**Root:** `result/optimization_runs/round19_factorial/`  
**Docker:** container `DAPL`，workdir `/workspace/DAPL`  
**19D baseline:** `f45c342`（90/90）

---

## 1. Scope

19E **不再搜尋架構**。在預註冊的三軸 shift 上驗證 E0–E5：

| Strategy | Seed | Jobs |
|----------|------|------|
| cancer_type_heldout | 19071 | 30 |
| drug_heldout | 19051 | 30 |
| scaffold_heldout | 19061 | 30 |
| **Total** | — | **90** |

Parallel pack：12 jobs / GPU（~90% util）+ Telegram。  
禁止：TCGA / internal-test 作選擇。

### Candidates

| ID | Source | Composition | Role |
|----|--------|-------------|------|
| E0 | F0 | D0×P0×O1 | historical MLP |
| E1 | F1 | D0×P2×O2 | primary O2 atom |
| E2 | F2 | D0×P2×O3 | O3 atom control |
| E3 | F3 | D0×P0×O2 | best pooled |
| E4 | F4 | D3×P2×O4 | source-only |
| E5 | F5 | D4×P1×O2 | MACCS efficient |

---

## 2. Completion

| Strategy | Done | Failed |
|----------|------|--------|
| cancer_type_heldout | **30/30** | 0 |
| drug_heldout | **30/30** | 0 |
| scaffold_heldout | **30/30** | 0 |

**Note:** E5 drug-heldout 初跑曾因 MACCS map 只載入 train drugs 而失敗；已修正 pipeline（preload train∪val）後重跑通過。這正是 drug-heldout 才會暴露的 bug（cancer-type 下藥物會同時出現在 train/val）。

---

## 3. Per-shift mean DrugMacro AUC

`reports/round19e_per_shift_summary.csv`

### 3.1 Cancer-type-heldout（最接近 18E 生物轉移）

| Rank | Cand | mean AUC | vs E0 |
|------|------|----------|-------|
| 1 | **E2** | **0.5824** | +0.029 |
| 2 | E1 | 0.5806 | +0.027 |
| 3 | E3 | 0.5774 | +0.024 |
| 4 | E4 | 0.5756 | +0.022 |
| 5 | E5 | 0.5726 | +0.019 |
| 6 | E0 | 0.5533 | — |

### 3.2 Drug-heldout

| Rank | Cand | mean AUC | vs E0 |
|------|------|----------|-------|
| 1 | **E3** | **0.7503** | +0.002 |
| 2 | E0 | 0.7482 | — |
| 3 | E5 | 0.7465 | −0.002 |
| 4 | E4 | 0.7413 | −0.007 |
| 5 | E2 | 0.7399 | −0.008 |
| 6 | E1 | 0.7328 | −0.015 |

### 3.3 Scaffold-heldout

| Rank | Cand | mean AUC | vs E0 |
|------|------|----------|-------|
| 1 | **E3** | **0.7456** | +0.001 |
| 2 | E0 | 0.7446 | — |
| 3 | E4 | 0.7428 | −0.002 |
| 4 | E5 | 0.7410 | −0.004 |
| 5 | E1 | 0.7376 | −0.007 |
| 6 | E2 | 0.7368 | −0.008 |

---

## 4. Paired fold deltas（每 shift 分開）

`reports/round19e_paired_fold_deltas.csv`

| Shift | Contrast | mean Δ | pos folds |
|-------|----------|--------|-----------|
| cancer | E1−E0 | **+0.0273** | **5/5** |
| cancer | E1−E3 | +0.0032 | 3/5 |
| cancer | E2−E1 | +0.0018 | 1/5 |
| cancer | E4−E1 | −0.0050 | 1/5 |
| drug | E1−E0 | **−0.0153** | **0/5** |
| drug | E1−E3 | −0.0175 | 1/5 |
| drug | E4−E1 | +0.0085 | 5/5 |
| scaffold | E1−E0 | −0.0071 | 1/5 |
| scaffold | E1−E3 | −0.0081 | 1/5 |
| scaffold | E4−E1 | +0.0052 | 5/5 |

---

## 5. Shift guardrails

`reports/round19e_shift_guardrails.csv`（±0.003；相對 E0 下降 >0.015 = MAJOR_FAIL）

| Cand | cancer vs E0 | drug vs E0 | scaffold vs E0 | MAJOR_FAIL |
|------|--------------|------------|----------------|------------|
| E1 | PASS | **FAIL** | FAIL | **yes（drug）** |
| E2 | PASS | FAIL | FAIL | no |
| E3 | PASS | NON_WORSE | NON_WORSE | no |
| E4 | PASS | FAIL | NON_WORSE | no |
| E5 | PASS | NON_WORSE | FAIL | no |

---

## 6. Interpretation（對應手冊情境）

### 情況混合：C（pooled 在 chemical shift 更穩）+ 部分 A（cancer 上 atom 仍優）

1. **Cancer-type shift：** E1/E2 相對 E0 **強且一致**（+0.027～0.029，5/5）；O2≈O3；E4 略低於 E1 但仍遠優於 E0。  
2. **Drug / scaffold shift：** **E3 pooled 與 E0 最佳**；atom E1/E2 反而落後 → source ModelID CV 優勢**無法**外推到 unseen drug/scaffold。  
3. **E1 MAJOR_FAIL** 於 drug-heldout（相對 E0 −0.015）。  
4. **E4 source-only** 在 drug/scaffold 上優於 E1（E4−E1 全 folds 為正），但 cancer 上略遜 E1。  
5. **E5 MACCS** 在 drug 上接近 E0，可作效率對照，不作 primary。

這直接呼應 Round 18E：source/internal 排名與 domain transfer 可不一致。

---

## 7. Role recommendations（非正式；formal lock 仍 NO-GO）

| Role | Recommendation | Rationale |
|------|----------------|-----------|
| Source-performance（19D） | F2 / F1 | 不變；19E 不改寫 19D |
| Recommended general | **E3（D0×P0×O2）** 優先考慮 | 三 shift 無 MAJOR_FAIL；drug/scaffold 最佳；cancer 仍 PASS vs E0 |
| Domain-generalization | **E4** 保留進 19F 討論 | chemical shift 優於 E1；cancer 未崩潰 |
| Atom primary | **勿單獨作為 general winner** | drug MAJOR_FAIL |
| Efficient | E5 optional | 接近 E0 on drug；資源低 |

若需「cancer 特化」敘事，可並列 E1/E2 為 biology-shift specialists，但不得覆蓋 E3 作為稳健 general 候選。

**Formal `round19_locked_selection.json` / `round19_final_model_roles.json`：仍 NO-GO，待人工審查。**

---

## 8. Artefacts

```text
metadata/round19_stage19d_baseline.json
reports/round19_stage19e_candidate_lock.json
reports/round19_stage19e_experiment_lock.json
reports/round19e_per_fold_metrics.csv
reports/round19e_per_shift_summary.csv
reports/round19e_paired_fold_deltas.csv
reports/round19e_shift_guardrails.csv
reports/round19e_*_generalization.csv
reports/round19e_analysis_summary.json
manifests/stage19e_*_manifest.csv
splits/round19e_*_heldout_5cv.csv
stage19e/{strategy}/{job_id}/
```

程式：`tools/round19_stage19e_*.py` · `tools/round19_drug_groups.py` ·  
`tools/round19_cancer_type_groups.py` · `tools/run_round19_stage19e_*.sh` ·  
`step1_finetune_latent_pipeline_round19.py`（MACCS union preload fix）

---

## 9. Go / No-Go

| Item | Verdict |
|------|---------|
| 90/90 complete | **GO** |
| Split overlap QC | **GO**（setup smoke） |
| Cancer atom≫historical | **GO** |
| Drug/scaffold atom generalizes | **NO**（E3/E0 更穩） |
| Formal selection lock | **NO-GO** |
| Next | **19F** final model roles（人工審查後） |

---

## 10. Related docs

- [`docs/round19_stage19e_ide_manual.md`](round19_stage19e_ide_manual.md)  
- [`docs/round19_stage19d_report.md`](round19_stage19d_report.md)
