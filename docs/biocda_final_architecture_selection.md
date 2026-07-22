# BioCDA 最終架構選擇（TCGA 優先順序）

## 選模原則

本決策**不以 GDSC development / unseen-drug validation / test 為依據**。

唯一評估域：**TCGA 五個 external target**，優先順序（高→低）：

```text
gdsc_intersect13 > tcga_only3 > dapl > aacdr_gdsc_intersect > aacdr_tcga_only
```

| 項目 | 設定 |
|------|------|
| 主指標 | DrugMacro AUC |
| 加權 | 5 : 4 : 3 : 2 : 1（對應上述順序） |
| 平手規則 | DrugMacro AUPRC → Global AUC → Global AUPRC |
| 次排序 | 字典序（lexicographic）同 target 順序 |

資料來源：`reports/biocda_tcga_comparison/biocda_tcga_comparison_long.csv`

---

## 加權 DrugMacro AUC 排名（決策主排序）

| Model | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only | weighted |
|---|---:|---:|---:|---:|---:|---:|
| biocda_xa_fresh / X0 (R23) | 0.481 | 0.605 | 0.529 | 0.563 | 0.564 | **8.100** |
| pooled_baseline / M0 (R21) | 0.476 | 0.549 | 0.502 | 0.541 | 0.604 | **7.769** |
| biocda_predictive / P0 (R23) | 0.513 | 0.520 | 0.465 | 0.575 | 0.501 | **7.691** |
| biocda_xa_transfer / X1 (R23) | 0.455 | 0.543 | 0.444 | 0.571 | 0.547 | **7.467** |
| biocda_xa_z / M1 (R21) | 0.501 | 0.482 | 0.444 | 0.570 | 0.521 | **7.425** |
| biocda_xa_zc / M2 (R21) | 0.484 | 0.407 | 0.491 | 0.611 | 0.553 | **7.298** |
| BioCDA-Predictive (R20 locked, 15-fold) | 0.471 | 0.459 | 0.428 | 0.617 | 0.539 | **7.253** |
| biocda_xa_kd / X2 (R23) | 0.456 | 0.495 | 0.444 | 0.562 | 0.513 | **7.233** |

## 加權 DrugMacro AUPRC

| Model | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only | weighted |
|---|---:|---:|---:|---:|---:|---:|
| biocda_xa_fresh / X0 (R23) | 0.581 | 0.714 | 0.554 | 0.626 | 0.696 | **9.372** |
| pooled_baseline / M0 (R21) | 0.606 | 0.703 | 0.552 | 0.603 | 0.686 | **9.389** |
| biocda_predictive / P0 (R23) | 0.578 | 0.672 | 0.516 | 0.627 | 0.648 | **9.027** |
| biocda_xa_transfer / X1 (R23) | 0.575 | 0.695 | 0.494 | 0.649 | 0.688 | **9.121** |
| biocda_xa_z / M1 (R21) | 0.594 | 0.684 | 0.526 | 0.625 | 0.656 | **9.189** |
| biocda_xa_zc / M2 (R21) | 0.586 | 0.635 | 0.587 | 0.637 | 0.669 | **9.177** |
| BioCDA-Predictive (R20 locked, 15-fold) | 0.559 | 0.663 | 0.480 | 0.647 | 0.680 | **8.864** |
| biocda_xa_kd / X2 (R23) | 0.573 | 0.675 | 0.520 | 0.635 | 0.687 | **9.081** |

---

## 最終建議（單一架構）

| 項目 | 決策 |
|------|------|
| **正式名稱** | **BioCDA-XA** |
| **選中模型** | **biocda_xa_fresh / X0 (R23)** |
| **架構版本** | `biocda-xa-v2` |
| **選模方法** | TCGA 加權 DrugMacro AUC（target 權重 5:4:3:2:1） |
| **加權 DrugMacro AUC** | **8.0996** |
| **加權 DrugMacro AUPRC** | 9.3724 |

### 架構定義（若選 XA）

```text
Z64 + C32 → sample query Q0 [B,1,128]
GIN 5×32 → atom nodes（no pooling）
2-layer sample→atom cross-attention（d=128, H=4）
response head(Qfinal[:,0,:]) → logit
```

訓練配方：`biocda_xa_fresh`（fresh GIN，無 KD，無 E3 transfer）

### 敏感性：字典序（gdsc_intersect13 絕對優先）

若改以**最高優先 target 單獨決勝**，則選 **biocda_predictive / P0 (R23)**（gdsc_intersect13 DrugMacro AUC = 0.513）。

若嚴格以最高優先 target gdsc_intersect13 的 DrugMacro AUC 為唯一標準，則選 biocda_predictive / P0 (R23)（0.513）。

### 加權排名對照

| 排名 | 模型 | weighted DrugMacro AUC |
|------|------|------------------------|
| 1 | biocda_xa_fresh / X0 (R23) | 8.0996 |
| 2 | pooled_baseline / M0 (R21) | 7.7695 |
| 字典序 #1 | biocda_predictive / P0 (R23) | 7.6913 |

### 架構家族最佳

| 家族 | 最佳模型 | weighted DrugMacro AUC |
|------|----------|------------------------|
| predictive_pooled_e3 | pooled_baseline / M0 (R21) | 7.7695 |
| xa_v1 | biocda_xa_z / M1 (R21) | 7.4252 |
| xa_v2 | biocda_xa_fresh / X0 (R23) | 8.0996 |

---

## 與 Round 23 GDSC 結論的關係

Round 23 以 GDSC unseen-drug 配對 gate 判定 XA **REJECTED**（performance_failure）。

若正式產品決策改以 **TCGA 優先順序** 為準，則需將本文件視為 **新的選模協議**，
並相應更新 `biocda_xa_model_lock.json` 與論文方法學描述（明確聲明不再以 GDSC test 選模）。

**不得**在未更新 lock manifest 的情況下，同時宣稱 GDSC-REJECTED 與 TCGA-SELECTED。

---

## 重現

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && python3 scripts/select_biocda_architecture_tcga.py'
```
