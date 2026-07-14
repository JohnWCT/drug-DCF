# Round 18 Final Report — Architecture Screening

**Date:** 2026-07-14  
**Status:** **ALL_DONE through Stage 18D**  
**Root:** `result/optimization_runs/round18_architecture/`  
**Pipeline:** 18A → 18B → 18C-A → 18C-B → lock → 18D（含 Telegram stage 通知）  
**Docker:** `docker exec -w /workspace/DAPL DAPL ...`（`ROUND18_NUM_WORKERS=0`）

---

## 1. Objective

在 **凍結的 Round 17R omics features** 與 **ModelID-grouped CV** 下，比較：

1. pooled MLP  
2. pooled Transformer（early pooling）  
3. atom-level **cross-attention**（drug atoms × omics CLS）

科學問題不只是「誰 AUC 最高」，而是：

- atom-level attention 是否優於 early pooling？  
- 若優於，是來自 **atom–omics interaction**，還是 **pooled GIN residual shortcut**？  
- 提升是否依賴 **prototype engineered omics（context16）**？

選擇規則：只用 screening / formal CV 的 **DrugMacro AUC**；**不使用** internal test 與 TCGA response labels 做選模。

---

## 2. Completion status

| Stage | Jobs | Result |
|-------|------|--------|
| 18A setup / eligible data / splits | — | DONE |
| 18B pooled screen | **45/45** | DONE |
| 18C-A cross-attn screen | **48/48** | DONE |
| 18C-B none follow-up | **6/6** | DONE |
| Selection lock | `round18_locked_selection.json` | DONE（需 45+48+6） |
| 18D formal 5CV | **25/25** | DONE |

Lock 條件已記錄：`18B 45/45`、`18C-A 48/48`、`18C-B 6/6`；`internal_test_used=false`，`tcga_used=false`；split seed **42**、model seed **101**。

---

## 3. Experimental design

### Shared protocol

| Item | Value |
|------|-------|
| Eligible response | `round18_eligible_response.csv`（n≈110279） |
| Feature ModelIDs | 937 identical across `none` / `own_plus_summary` / `context16`（dims 64 / 75 / 91） |
| Screening CV | 3-fold，ModelID-grouped |
| Formal CV | 5-fold，ModelID-grouped |
| Primary metric | mean **DrugMacro AUC** |
| Ops | process-level OOM retry；`/dev/shm=64MB` → workers=0 |

### Stage contents

| Stage | Grid |
|-------|------|
| **18B** | MLP + P0–P3 Transformer × 3 omics × 3 folds = 45 |
| **18C-A** | X0–X3 × {pure, pooled_residual} × {own_plus_summary, context16} × 3 folds = **48** |
| **18C-B** | best pure + best residual × **none** × 3 folds = **6** |
| **18D** | 5 locked candidates × 5 folds = **25** |

18C-B **不**取 cross-attn 總榜前兩名（可能兩個都是 residual），固定選：

1. best **pure** cross-attention  
2. best **pooled_residual** cross-attention  

實際選出：`X3` pure / residual × context16。

---

## 4. Stage 18B summary（screening）

詳見 `docs/round18_stage18b_progress.md`。摘要：

| Rank | Architecture | Omics | mean AUC |
|------|--------------|-------|----------|
| 1 | P3_deeper128 | context16 | **0.6171** |
| 2 | P1_compact64 | context16 | 0.6128 |
| 3 | pooled MLP | own_plus_summary | 0.6127 |

重點：pooled Transformer 與 **context16** 有架構–表現交互；只測 `own_plus_summary` 會低估 Transformer。

---

## 5. Stage 18C-A screening ranking（top）

Source: `reports/round18_screening_architecture_ranking.csv`

| Rank | Architecture | mean AUC | mean AUPRC | std AUC |
|------|--------------|----------|------------|---------|
| 1 | X3 × pooled_residual × context16 | **0.6230** | 0.4201 | 0.0053 |
| 2 | X2 × pooled_residual × context16 | 0.6227 | 0.4213 | 0.0077 |
| 3 | X3 × pure × context16 | 0.6210 | 0.4218 | 0.0032 |
| 4 | X0 × pooled_residual × context16 | 0.6207 | 0.4150 | 0.0046 |
| 5 | X2 × pure × context16 | 0.6188 | 0.4194 | 0.0050 |
| 6 | P3 × context16（18B） | 0.6171 | 0.4172 | 0.0134 |
| 10 | MLP × own_plus_summary | 0.6127 | 0.4132 | 0.0169 |

Cross-attention 在 screening 全面壓過 P3／MLP；且 top 幾乎都是 **context16**。

### Paired comparisons（X3 residual × context16，screening 3-fold）

| Comparison | mean Δ | folds positive | std Δ |
|------------|--------|----------------|-------|
| vs MLP × own_plus_summary | +0.0103 | 1/3 | 0.0220 |
| vs P3 × context16 | +0.0059 | **2/3** | 0.0185 |
| residual − pure（context16） | +0.0020 | 2/3 | 0.0042 |
| context16 − own_plus（pure） | +0.0179 | **3/3** | 0.0065 |
| context16 − own_plus（residual） | +0.0174 | **3/3** | 0.0065 |

判讀：

- 相對 **P3** 達較強門檻（mean Δ>0 且 ≥2/3 folds 改善）。  
- 相對 MLP 的 mean Δ>0，但 fold 間變異大（僅 1/3 明顯改善）。  
- **Omics 效應遠大於 residual 效應**：context16 相對 own_plus 約 +0.017～0.018。  
- Residual 在 screening 略有幫助，但幅度很小（~0.002）。

---

## 6. Stage 18C-B none follow-up

| Architecture | residual | mean AUC |
|--------------|----------|----------|
| X3 × pure × **none** | pure | 0.6085 |
| X3 × residual × **none** | pooled_residual | 0.6033 |
| X3 × pure × **context16** | pure | 0.6210 |
| X3 × residual × **context16** | pooled_residual | 0.6230 |

**結論：** none 明顯低於 context16（約 −0.013～−0.020）。cross-attention 的 screening 優勢 **高度依賴 prototype engineered omics**，不是「無 omics 也能單靠 atom attention」 alone。  
另：在 **none** 上 residual 甚至略差於 pure（mean Δ ≈ −0.005），不支持「主要靠 GIN shortcut 就能解釋整體提升」。

---

## 7. Formal lock（5 candidates）

Policy: `round18_explicit_5candidate`

| Role | Candidate | Screening mean AUC |
|------|-----------|--------------------|
| Anchor | MLP × own_plus_summary | 0.6127 |
| Best pooled Transformer | P3 × context16 | 0.6171 |
| Efficient Transformer | P1 × context16 | 0.6128 |
| Atom interaction | X3 **pure** × context16 | 0.6210 |
| Atom + shortcut | X3 **pooled_residual** × context16 | 0.6230 |

Artifacts:

- `reports/round18_locked_selection.json`  
- `reports/round18_18c_top_for_none.json`  
- Paired reports: `round18_cross_attention_paired_deltas.csv`、`round18_residual_effect_summary.csv`、`round18_omics_architecture_interaction.csv`

---

## 8. Stage 18D formal 5CV results

Source: `reports/round18_formal_5cv_summary.csv`（5/5 folds with AUC）

| Rank | Architecture | mean AUC | mean AUPRC | std AUC |
|------|--------------|----------|------------|---------|
| 1 | **X3 pure × context16** | **0.6181** | 0.4505 | 0.0174 |
| 2 | X3 pooled_residual × context16 | 0.6176 | 0.4511 | 0.0161 |
| 3 | P1 compact64 × context16 | 0.6169 | 0.4538 | 0.0099 |
| 4 | P3 deeper128 × context16 | 0.6105 | 0.4457 | 0.0134 |
| 5 | MLP × own_plus_summary | 0.6078 | 0.4399 | 0.0233 |

### Screening → formal 反轉／確認

| Finding | Detail |
|---------|--------|
| Best formal | **pure** cross-attn（screening 時 residual 略優） |
| Cross-attn vs pooled | X3 兩者皆勝過 P3／MLP |
| P1 vs P3 | formal 中 **P1 > P3**（與 screening 相反） |
| Residual shortcut | formal 中 residual **不優於** pure（差距極小） |

---

## 9. Scientific conclusions

1. **Atom-level cross-attention + context16** 在 3-fold screening 與 5-fold formal 皆優於 MLP baseline 與 P3 pooled Transformer。  
2. 優勢 **不是** 單純來自 pooled GIN residual：formal 上 pure ≈ residual；none 上 residual 甚至更差。  
3. 優勢 **與 omics representation 強交互**：context16 ≫ own_plus ≫ none（對 X3）。  
4. 因此 Round 18 的核心發現是 **architecture × representation interaction**，而非「任何 Transformer 都贏 MLP」。  
5. P1（較小）在 formal 5CV 可能比 P3 更穩，值得保留為 efficient pooled 對照。

---

## 10. Ops notes

- 18B 曾因過度平行出現 SIGKILL；workers=0 後可提高 packing。  
- 18C-A 最終以 `MAX_JOBS_PER_GPU=8` 完成；18C-B 用 6；18D 用 8。  
- Telegram：`tools/round18_telegram_notify.py` + stage scripts 的 `r18_notify`。  
- `--write-lock` 硬性要求 18B+18C-A+18C-B 全部完成。

---

## 11. Key paths

```
result/optimization_runs/round18_architecture/
  manifests/
    stage18b_screening_manifest.csv
    stage18c_cross_attention_manifest.csv
    stage18c_none_followup_manifest.csv
    stage18d_formal_5cv_manifest.csv
  reports/
    round18_locked_selection.json
    round18_18c_top_for_none.json
    round18_screening_architecture_ranking.csv
    round18_formal_5cv_summary.csv
    round18_cross_attention_paired_deltas.csv
    round18_residual_effect_summary.csv
    round18_omics_architecture_interaction.csv
    round18_final_report.md
```

---

## 12. Suggested next steps（未執行）

- 18E / 18F：internal test、TCGA ensemble、attention export（需另開，且不得回頭污染選模）。  
- 若要以 single champion 進下游：目前 formal 最佳為 **`cross_attn__X3__pure__own_proto_context_projected_16`**。  
- 文件披露：`own_proto_context_projected_16` 使用 unlabeled TCGA prototype context（非 TCGA response labels）。
