# Round 23 — No-Pooling XA v2 Performance Closure

**狀態：** COMPLETE · GDSC gate **REJECTED** · TCGA 選模 **BioCDA-XA v2 Fresh**

完整彙總見 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md#round-23--no-pooling-xa-v2)。TCGA 架構選擇詳見 [`biocda_final_architecture_selection.md`](biocda_final_architecture_selection.md)。

---

## GDSC 配對 gate（development unseen-drug，seeds 17/29/43）

| 模型 | mean AUC | ΔAUC vs P0 | mean AUPRC | ΔAUPRC vs P0 |
|------|----------|------------|------------|--------------|
| P0 BioCDA-Predictive | **0.744** | — | **0.512** | — |
| X0 fresh XA | 0.740 | −0.0043 | 0.506 | −0.0059 |
| X2 transfer+KD | 0.720 | −0.0247 | 0.490 | −0.0214 |
| X1 transfer | 0.699 | −0.0455 | 0.477 | −0.0342 |

**X0 fresh（最接近）：** mean ΔAUC 達 −0.005 門檻，但 seed non-worse **1/3**（需 2/3）→ **REJECTED**（performance_failure only）。

---

## TCGA 外部驗證（選模依據，不含 GDSC test）

### 選模協議

| 項目 | 設定 |
|------|------|
| 評估域 | 僅 TCGA 五個 external target |
| Target 優先順序 | `gdsc_intersect13` > `tcga_only3` > `dapl` > `aacdr_gdsc_intersect` > `aacdr_tcga_only` |
| 主指標 | DrugMacro AUC |
| 加權 | 5 : 4 : 3 : 2 : 1 |
| 平手規則 | DrugMacro AUPRC → Global AUC → Global AUPRC |

資料：`reports/biocda_tcga_comparison/biocda_tcga_comparison_long.csv`（8 模型 × 5 target）

### 加權 DrugMacro AUC 排名（R23 候選 + 歷史 baseline）

| 模型 | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only | weighted |
|------|---:|---:|---:|---:|---:|---:|
| **X0 fresh XA (R23)** | 0.481 | **0.605** | **0.529** | **0.563** | **0.564** | **8.100** |
| M0 pooled_baseline (R21) | 0.476 | 0.549 | 0.502 | 0.541 | 0.604 | 7.769 |
| P0 BioCDA-Predictive (R23) | **0.513** | 0.520 | 0.465 | 0.575 | 0.501 | 7.691 |
| X1 transfer (R23) | 0.455 | 0.543 | 0.444 | 0.571 | 0.547 | 7.467 |
| X2 transfer+KD (R23) | 0.456 | 0.495 | 0.444 | 0.562 | 0.513 | 7.233 |
| BioCDA-Predictive R20 locked | 0.471 | 0.459 | 0.428 | 0.617 | 0.539 | 7.253 |

### 架構家族最佳（TCGA 加權）

| 家族 | 最佳模型 | weighted DrugMacro AUC |
|------|----------|------------------------|
| **xa_v2** | X0 fresh (R23) | **8.100** |
| predictive_pooled_e3 | M0 (R21) | 7.769 |
| xa_v1 | M1 xa_z (R21) | 7.425 |

---

## 最終架構決策

| 項目 | 決策 |
|------|------|
| **正式名稱** | **BioCDA-XA** |
| **選中模型** | `biocda_xa_fresh / X0 (R23)` |
| **架構版本** | `biocda-xa-v2`（no-pooling sample→atom cross-attention） |
| **訓練配方** | Fresh GIN，無 KD，無 E3 transfer |
| **選模方法** | TCGA 加權 DrugMacro AUC（**不以 GDSC test 為依據**） |
| **加權 DrugMacro AUC** | **8.100**（全候選最高） |

### 敏感性：gdsc_intersect13 絕對優先

若改以最高優先 target 單獨決勝，則選 **P0 BioCDA-Predictive**（gdsc_intersect13 = 0.513，X0 = 0.481）。

---

## 結論

### GDSC gate（Round 23 原始 closure）

- **BioCDA-Predictive** 維持 LOCKED_REFERENCE（GDSC unseen-drug 基準）。
- **BioCDA-XA-Candidate v2** GDSC gate **REJECTED**（performance_failure only）。
- Fresh no-pooling 幾乎追平均值（ΔAUC ≈ −0.004），transfer/KD 未縮小差距。
- X3（Z64-only C32 ablation）因性能已 reject 而 deferred。

### TCGA 選模（正式產品架構）

- **BioCDA-XA v2 Fresh（X0）** 在 TCGA 加權 DrugMacro AUC 下為**全候選最優**。
- X0 在 `tcga_only3` 及後三個 target 均領先；transfer / KD 配方均不如 fresh。
- XA v2 已超越 XA v1（R21）與 R20 locked Predictive 的 TCGA 表現。
- **正式選模不再以 GDSC development / validation / test 為依據**；TCGA 五 target 加權排序為準。

### 雙軌結論的關係

GDSC gate 與 TCGA 選模結論**並不矛盾**，但適用場景不同：

| 軸 | 結論 | 用途 |
|----|------|------|
| GDSC unseen-drug gate | XA **REJECTED** | 開發期配對性能門檻、架構健康審計 |
| TCGA external 選模 | XA v2 Fresh **SELECTED** | 正式部署 / 論文外部驗證指標 |

若採 TCGA 選模協議，需更新 `biocda_xa_model_lock.json` 與方法學描述；**不得**在未更新 lock manifest 的情況下同時宣稱 GDSC-REJECTED 與 TCGA-SELECTED。

### 後續

- 若再開 round，應從 **fresh XA** 出發；不可用 rejected XA attention 解釋 Predictive。
- 重現 TCGA 選模：`python3 scripts/select_biocda_architecture_tcga.py`
