# Round 19 Stage 19D Report — Repeated 5CV Confirmation

**Date:** 2026-07-15  
**Status:** **ALL_DONE — 90/90 jobs done，0 failed**  
**Stage gate:** Stage 19D = **GO complete**；**Formal selection lock = NO-GO**；**Round 19E：ALL_DONE（90/90）** — 見 [`docs/round19_stage19e_ide_manual.md`](round19_stage19e_ide_manual.md)（需 19E shift + 人工審查後才寫 `round19_locked_selection.json`）  
**Root:** `result/optimization_runs/round19_factorial/`  
**Docker:** container `DAPL`，workdir `/workspace/DAPL`  
**19C baseline:** `70cc0b4`（54/54）

---

## 1. Scope

19D **不再搜尋架構**。在新 ModelID 5CV seeds `52/62/72` 上確認 19B/19C 角色候選是否穩健。

| Item | Value |
|------|-------|
| Candidates | **6**（F0–F5；F6 未達門檻未納入） |
| Protocol | 3 seeds × 5 folds |
| Total jobs | 6 × 15 = **90** |
| Model seed | 101（固定） |
| Parallel pack | 12 jobs / GPU（~90% util） |
| Telegram | start / done |
| Primary metric | **mean-of-means DrugMacro AUC**（先每 seed 5-fold mean，再跨 seed 平均） |

禁止：TCGA、internal-test、Integrated5 作選擇。Internal test 僅凍結、不進 ranking。

---

## 2. Candidate set（experiment lock）

`reports/round19_stage19d_candidate_proposal.json`  
`reports/round19_stage19d_experiment_lock.json`

| ID | Composition | Role | Mandatory |
|----|-------------|------|-----------|
| F0 | D0×P0×O1 | historical anchor | yes |
| F1 | D0×P2×O2 | primary context (O2) | yes |
| F2 | D0×P2×O3 | full-omics control | yes |
| F3 | D0×P0×O2 | best pooled O2 | yes |
| F4 | D3×P2×O4 | source-only DG | yes |
| F5 | D4×P1×O2 | MACCS efficient | no（gap≤0.005 vs F3） |

Splits：`splits/round19d_seed{52,62,72}_5cv_assignments.csv`

---

## 3. Completion

| Metric | Value |
|--------|-------|
| `job_status=done` | **90/90** |
| Failed | **0** |
| Pack | `jobs_per_gpu=12`，micro-batch 256（無 OOM 降級） |
| Dispatch log | `logs/round19_stage19d_parallel_20260714T152440Z.log` |
| Status CSV | `manifests/stage19d_job_status.csv` |

---

## 4. Cross-seed ranking

`reports/round19d_cross_seed_summary.csv`

| Rank | Candidate | Cell | mean-of-means AUC | std(seeds) | mean-of-means AUPRC |
|------|-----------|------|-------------------|------------|---------------------|
| 1 | **F2** | D0×P2×O3 | **0.6201** | 0.0024 | **0.4508** |
| 2 | F4 | D3×P2×O4 | 0.6194 | 0.0031 | 0.4446 |
| 3 | F1 | D0×P2×O2 | 0.6194 | 0.0034 | 0.4447 |
| 4 | F5 | D4×P1×O2 | 0.6170 | 0.0041 | 0.4452 |
| 5 | F3 | D0×P0×O2 | 0.6079 | 0.0070 | 0.4365 |
| 6 | F0 | D0×P0×O1 | 0.5947 | 0.0026 | 0.4232 |

Per-seed 5CV means：`reports/round19d_per_seed_5cv_summary.csv`  
Per-seed rank-1：**F2 / F1 / F4**（各贏 1 個 seed）→ 頂層三候選彼此接近，**無單一絕對冠軍**。

---

## 5. Paired hypothesis checks（seed-level）

`reports/round19d_paired_seed_deltas.csv`  
`reports/round19d_paired_seed_delta_summary.csv`

| Hypothesis | Contrast | mean Δ AUC | pos seeds |
|------------|----------|------------|-----------|
| Primary ≫ historical | F1 − F0 | **+0.0247** | **3/3** |
| Atom ≫ pooled（同 O2） | F1 − F3 | **+0.0115** | **3/3** |
| O2 ≈ O3（atom） | F1 − F2 | −0.0007 | 1/3 |
| Target context ≈ source-only | F1 − F4 | ≈ 0.0000 | 1/3 |

Fold-level F1−F2：mean Δ ≈ −0.0007，**8/15** folds F1 較高（`round19d_paired_fold_deltas.csv`）。

**解讀：**

1. **F1 相對 F0 強且一致** → context + atom 相對歷史 baseline 在新 splits 上成立。  
2. **F1 相對 F3 一致為正** → atom cross-attn（P2）優於 pooled MLP（P0）在 O2 上可複現。  
3. **O2 vs O3 實質打平** → 預設仍可優先較簡的 **O2**；若以 AUPRC / mean-of-means 取單一代表，**F2 略勝**。  
4. **F4 幾乎貼齊 F1** → source-only O4 在 development 5CV 上意外強；**不得因此當最終 winner**，必須在 19E domain-shift 再驗。  
5. **F5** 落後 F1/F2 約 0.002–0.003，但仍遠高於 F0；可作資源／非 graph 對照保留。

---

## 6. Role-lock recommendations（非正式）

| Role | Recommended | Note |
|------|-------------|------|
| Primary atom | **F1 (D0×P2×O2)** 或 **F2 (…O3)** | O2/O3 打平；預設 O2，F2 作 full-omics 對照 |
| Historical | F0 | 必須保留敘事 baseline |
| Pooled | F3 | 明顯弱於 atom；不作 primary |
| Source-only | **F4** | 19E 必跑；development 強不代表 shift 強 |
| MACCS | F5 | optional efficiency lane |

**Formal `round19_locked_selection.json`：仍 NO-GO。**

---

## 7. Artefacts

```text
reports/round19_stage19d_candidate_proposal.json
reports/round19_stage19d_experiment_lock.json
reports/round19d_cross_seed_summary.csv
reports/round19d_per_seed_5cv_summary.csv
reports/round19d_paired_seed_deltas.csv
reports/round19d_paired_seed_delta_summary.csv
reports/round19d_paired_fold_deltas.csv
reports/round19d_o2_vs_o3.csv
reports/round19d_rank_counts.csv
reports/round19d_resource_summary.csv
reports/round19d_analysis_summary.json
manifests/stage19d_manifest.csv
manifests/stage19d_job_status.csv
splits/round19d_seed{52,62,72}_5cv_assignments.csv
metadata/round19_stage19c_baseline.json
stage19d/{job_id}/
```

程式：`tools/round19_stage19d_selector.py` · `tools/round19_cv_splits.py` ·  
`tools/round19_config_builder.py --stage 19d` · `tools/run_round19_stage19d_repeated_5cv.sh` ·  
`tools/analyze_round19.py --stage 19d` · `tools/write_round19_stage19c_baseline.py`

---

## 8. Go / No-Go

| Item | Verdict |
|------|---------|
| 90/90 complete | **GO** |
| F1 ≫ F0 on new seeds | **GO** |
| Atom ≫ pooled (F1≫F3) | **GO** |
| O2 vs O3 atom | **NEUTRAL**（預設 O2） |
| F4 source-only for DG | **KEEP for 19E** |
| Formal selection lock | **NO-GO** |
| Next | **19E DONE** — 見 [`docs/round19_stage19e_report.md`](round19_stage19e_report.md)；下一步 19F |

---

## 9. Related docs

- [`docs/round19_stage19d_ide_manual.md`](round19_stage19d_ide_manual.md)  
- [`docs/round19_stage19c_report.md`](round19_stage19c_report.md)  
- [`docs/round19_stage19b_report.md`](round19_stage19b_report.md)
