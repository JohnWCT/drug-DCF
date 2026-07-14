# Round 18 Final Report — Architecture Screening

**Date:** 2026-07-14  
**Status:** **18A–18E DONE；18F NOT DONE → Round 18 整體尚未全部完成**  
**Root:** `result/optimization_runs/round18_architecture/`  
**Pipeline:** 18A → 18B → 18C-A → 18C-B → lock → 18D → **18E**（18F 待做）  
**Docker:** `docker exec -w /workspace/DAPL DAPL ...`（`ROUND18_NUM_WORKERS=0`）

---

## 0. Round 18 completion verdict

| 問題 | 答案 |
|------|------|
| 選模／formal CV 是否完成？ | **是**（18A–18D） |
| 鎖定 5 candidates 後的 external eval（internal + TCGA ensemble）是否完成？ | **是**（18E） |
| 可解釋性／attention export（18F）是否完成？ | **否**（`export_attention` 仍為 stub） |
| Round 18 是否可宣告全部完成？ | **否** — 差 Stage **18F** |

**18E 外部成功判定（固定規則，未回頭改選模）：**  
`cross_attention_external_success = false`  
（internal X3_pure 勝過 MLP，但 5 個 TCGA DrugMacro 目標中僅 **2/5** non-worse；門檻為 ≥3/5）

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
- formal CV 優勢能否落到 **held-out internal test** 與 **TCGA response**（選模後才評估）？

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
| 18E internal ensemble infer | **25/25** | DONE |
| 18E TCGA ensemble infer | **125/125** | DONE |
| 18E analyze（metrics + paired bootstrap） | reports + verdict | DONE |
| 18F attention export / masking | — | **NOT DONE** |

Lock 條件：`18B 45/45`、`18C-A 48/48`、`18C-B 6/6`；`internal_test_used=false`，`tcga_used=false`；split seed **42**、model seed **101**。

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
| 18E ensemble | 5-fold probability **mean**（禁止 best-fold） |
| Ops | process-level OOM retry；`/dev/shm=64MB` → workers=0 |

### Stage contents

| Stage | Grid |
|-------|------|
| **18B** | MLP + P0–P3 Transformer × 3 omics × 3 folds = 45 |
| **18C-A** | X0–X3 × {pure, pooled_residual} × {own_plus_summary, context16} × 3 folds = **48** |
| **18C-B** | best pure + best residual × **none** × 3 folds = **6** |
| **18D** | 5 locked candidates × 5 folds = **25** |
| **18E** | 5 candidates ×（internal + 5 TCGA targets）× 5 folds = **25 + 125** |

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

## 9. Stage 18E external evaluation（選模後）

**規則：** 只用 18D checkpoints；ensemble = 5-fold 機率平均；**不**用 internal／TCGA 回頭改 architecture selection。

Artifacts:

- `reports/round18_internal_test_summary.csv`  
- `reports/round18_five_target_tcga_summary.csv`  
- `reports/round18_integrated5_summary.csv`  
- `reports/round18e_paired_bootstrap_deltas.csv`  
- `reports/round18e_success_verdict.json`

### 9.1 Internal held-out test（ensemble）

| Rank | Architecture | DrugMacro AUC | Global AUC |
|------|--------------|---------------|------------|
| 1 | P3 × context16 | **0.6131** | 0.8586 |
| 2 | X3 residual × context16 | 0.6110 | 0.8579 |
| 3 | X3 pure × context16 | 0.6056 | **0.8632** |
| 4 | P1 × context16 | 0.5905 | 0.8244 |
| 5 | MLP × own_plus_summary | 0.5358 | 0.7705 |

**判讀：**

- X3 pure **明顯勝過** MLP（DrugMacro ≈ +0.070；paired bootstrap P(Δ>0)≈0.9995）。  
- 相對 P3／residual，DrugMacro 略差或持平（bootstrap CI 含 0）。  
- Internal 支持「cross-attn + context16 ≫ MLP」，但 **champion 未必仍是 X3 pure**（P3 在此切面最高）。

### 9.2 TCGA five-target DrugMacro AUC

| Target | MLP | P3 | P1 | X3 pure | X3 res | X3 vs MLP |
|--------|-----|----|----|---------|--------|-----------|
| gdsc_intersect13 | **0.5415** | 0.4902 | 0.4624 | 0.4593 | 0.4951 | worse |
| tcga_only3 | **0.5508** | 0.4023 | 0.4321 | 0.3584 | 0.4059 | worse |
| dapl | **0.5154** | 0.4628 | 0.4181 | 0.4620 | 0.4095 | worse |
| aacdr_tcga_only | 0.5183 | 0.4450 | 0.5018 | **0.5340** | 0.4492 | non-worse |
| aacdr_gdsc_intersect | 0.5178 | 0.5667 | 0.5267 | **0.5601** | 0.5559 | non-worse |

X3 pure vs MLP：**2/5** non-worse（未達 ≥3/5 成功門檻）。

### 9.3 Integrated5（5 TCGA 目標平均 DrugMacro AUC）

| Rank | Architecture | Integrated5 DrugMacro |
|------|--------------|------------------------|
| 1 | **MLP × own_plus_summary** | **0.5288** |
| 2 | X3 pure × context16 | 0.4748 |
| 3 | P3 × context16 | 0.4734 |
| 4 | P1 × context16 | 0.4682 |
| 5 | X3 residual × context16 | 0.4631 |

TCGA 整體平均上 **MLP 反而最佳**；formal／internal 的 Transformer／cross-attn 優勢 **未外推**到這五個 TCGA response sets。

### 9.4 Paired bootstrap（X3 pure − comparator，DrugMacro AUC）

重點（mean Δ；95% CI；P(Δ>0)）：

| Contrast | internal_test | 解讀 |
|----------|---------------|------|
| X3 pure − MLP | +0.065 [+0.024, +0.108]；P≈0.9995 | 穩健勝過 MLP |
| X3 pure − P3 | −0.006 [−0.031, +0.019]；P≈0.31 | 無穩定勝 P3 |
| X3 pure − residual | −0.005 [−0.032, +0.021]；P≈0.35 | pure≈residual |

TCGA 上相對 MLP：多數目標 mean Δ **為負**（尤其 `tcga_only3`、`gdsc_intersect13`）；僅 AACDR 相關子集傾向非負。

### 9.5 18E success verdict

```json
{
  "cross_attention_external_success": false,
  "notes": [
    "internal X3_pure=0.6056 vs MLP=0.5358; TCGA non-worse=2/5",
    "internal residual=0.6110 (delta vs pure=0.0054)"
  ],
  "prefer_pure_for_18F": true
}
```

若仍做 18F，鎖定對象維持 formal champion：**X3 pure × context16**（residual 無外部加分）。

---

## 10. Scientific conclusions（更新至 18E）

1. **Screening／formal CV：** atom-level cross-attention + **context16** 優於 MLP 與多數 pooled Transformer；優勢來自 **architecture × omics representation**，不是 residual shortcut alone。  
2. **Omics 依賴：** context16 ≫ own_plus ≫ none；none follow-up 否定「純 atom attention 即可」。  
3. **Residual：** formal 與 internal 上 pure ≈ residual；不支持「主要靠 GIN pooled residual」。  
4. **Internal held-out：** X3／P3 皆遠勝 MLP，確認 in-domain generalization 對 cross-attn／Transformer+context16 有利。  
5. **TCGA external：** 固定成功規則下 **未通過**（2/5 non-worse；Integrated5 MLP 最高）。formal／internal 增益 **未能穩定外推**到 TCGA response labels。  
6. **Round 18 核心結論應分成兩層：**  
   - **方法層（CV／internal）：** cross-attn + context16 是合理的 in-domain 架構贏家；  
   - **外部臨床／TCGA 層：** 目前 **不能**宣告 cross-attention external success。  
7. Round 18 **尚未全部完成**：缺 **18F**（attention 可解釋性／masking）。

---

## 11. Ops notes

- 18B 曾因過度平行出現 SIGKILL；workers=0 後可提高 packing。  
- 18C-A／18D／18E 多用 `MAX_JOBS_PER_GPU=8`；18C-B 用 6。  
- Telegram：`tools/round18_telegram_notify.py` + stage scripts 的 `r18_notify`。  
- `--write-lock` 硬性要求 18B+18C-A+18C-B 全部完成。  
- 18E analyze：paired bootstrap 為 96 jobs（4 pairs × 6 targets × 4 metrics）；改 `ProcessPoolExecutor`（`--n-jobs`，預設約 16）後約 **4 分鐘**完成（先前單執行緒過慢）。

---

## 12. Key paths

```
result/optimization_runs/round18_architecture/
  manifests/
    stage18b_screening_manifest.csv
    stage18c_cross_attention_manifest.csv
    stage18c_none_followup_manifest.csv
    stage18d_formal_5cv_manifest.csv
    stage18e_internal_test_manifest.csv
    stage18e_tcga_manifest.csv
  reports/
    round18_locked_selection.json
    round18_18c_top_for_none.json
    round18_screening_architecture_ranking.csv
    round18_formal_5cv_summary.csv
    round18_internal_test_summary.csv
    round18_five_target_tcga_summary.csv
    round18_integrated5_summary.csv
    round18e_paired_bootstrap_deltas.csv
    round18e_success_verdict.json
  stage18e_internal/ ...
  stage18e_tcga/ ...
```

Code（18E）：

- `tools/run_round18_stage18e_locked_eval.sh`  
- `tools/analyze_round18_external_eval.py`  
- `tools/round18_tcga_dataset.py`  
- `tools/round18_prediction_ensemble.py`  
- `step1_finetune_latent_pipeline_round18_cv.py`（`infer_internal_test` / `infer_tcga`）

---

## 13. Remaining / next（未全部完成項）

1. **Stage 18F（必要，才算 Round 18 全完成）：**  
   - 對 `cross_attn__X3__pure__own_proto_context_projected_16` 做 attention export、一致性、masking ablation。  
   - 不得用 18E／18F 結果回頭改 18D 選模。  
2. 若探討 TCGA 失敗原因：domain shift、藥物覆蓋、標籤雜訊、omics alignment（屬後續分析，不屬本 round 選模）。  
3. 文件披露：`own_proto_context_projected_16` 使用 unlabeled TCGA prototype context（非 TCGA response labels）。
