# DAPL 實驗結果彙總

**用途：** 各 Round 重要分數與科學結論；不含實作細節、測試紀錄或失敗清單。  
**主指標：** Pretrain 主線 → `Average_TCGA_AUC_mean`（gdsc_intersect13）；Round 17+ → 同上或 `DrugMacro AUC`；Round 18+ → `DrugMacro AUC`（unseen-drug / CV）。

---

## 命名對照（Round 20+）

| 角色 | 名稱 | 狀態 |
|------|------|------|
| Round 20 鎖定預測器 | **BioCDA-Predictive**（pooled E3 + C32） | LOCKED |
| Round 21/23 交叉注意力 | **BioCDA-XA-Candidate** | REJECTED |

BioCDA-Predictive ≠ BioCDA-XA：前者無 atom cross-attention；後者為可解釋性候選，尚未通過性能門檻。

---

## Pretrain 主線（Round 1–16）

下游主指標：`Average_TCGA_AUC_mean`（TCGA gdsc_intersect13，4 finetune combo 平均）。

| Round | 主題 | 最佳模型 | Avg TCGA | 結論 |
|-------|------|----------|----------|------|
| R1–R3 | VAEwC + Prototype InfoNCE | exp_018（control） | **0.5695** | InfoNCE 無明顯下游增益；K-means 與下游部分相關 |
| R4.1 | t2s cross-domain InfoNCE | exp_035 | 0.5339 | t2s 緩解結構崩潰，但未超越 R3 |
| R5 | Control + class-gap | exp_001 | **0.5403** | 超越 R4.1；實質為 pure control |
| R6 | Tumor topology / VICReg | exp_010 | **0.5569** | 超越 R5；最佳仍 λ=0 |
| R7 | exp_010 鄰域 + VICReg ablation | exp_048 | **0.5918** | **全專案 pretrain 主線峰值** |
| R8 | 廣泛架構確認 | exp_188 | 0.5777 | 未超越 R7 exp_048 |
| R9 | Deconfounding QC | exp_048 repro 最佳 | 0.5671 | global 對齊佳，conditional leakage 仍高 |
| R10 | Conditional ADV | exp_111 | 0.5749 | 略優 R9 repro，低於 R7 原始 |
| R11 | 10C 穩定化 + SmoothL1 | exp_035 | **0.5828** | conditional leakage 下降 |
| R12 | Source-anchor prototype align | exp_037 | **0.5972** | 超越 R11 與 R7 |
| R13 | Prototype response features | r13_exp_008_own_plus_summary | **0.6112** | proto feature 有效；stretch 0.62 未達 |
| R14 | VICReg 再整合 | r14_exp_078 | 0.5909 | 未超越 R13 |
| R15 | 可重現性 + exp_008 rescue | r15c_exp_005 | 0.6083 | 接近 R13（−0.0029） |
| R16 | Bruteforce | — | ~0.6068 | 未超越 R13 |

**歷史基準：** exp_746 Avg TCGA **0.5462**（嚴格 filter 通過的 control）。

**Production checkpoint（pretrain 路線）：** R7 **exp_048**（0.5918）→ 後續 R12 **exp_037**（0.5972）→ R13 **exp_008**（0.6112）為下游 finetune 峰值。

---

## Round 17 — Direct prototype features

| 階段 | 最佳 | Avg TCGA | vs R13 (0.6112) |
|------|------|----------|-----------------|
| 17C 10-seed | r13_exp_008 / context_16 | 0.5892 ± 0.034 | −0.022 |
| 17A 單點峰值 | r13_exp_008_control / own_plus_summary | 0.5998 | −0.011 |

**結論：** direct prototype 未全面超越 `own_plus_summary`；以 headline 指標看仍低於 R13。

---

## Round 17R — 18-class-clean 重跑

| 階段 | 最佳 | Avg TCGA | vs R13 |
|------|------|----------|--------|
| 17R-D 10-seed | r13_exp_008 / own_plus_summary | **0.5915 ± 0.036** | −0.020 |
| 17R-B 單點峰值 | r15c_exp_024 / own_plus_summary | 0.6074 | −0.004 |

**結論：** 18-class 修正後 primary strategy 仍為 `own_plus_summary`；10-seed 確認未達 R13。

---

## Round 18 — Architecture screening

**Formal 5CV DrugMacro AUC（選模依據）：**

| 排名 | 架構 | mean AUC |
|------|------|----------|
| 1 | X3 pure × context16 | **0.6181** |
| 2 | X3 pooled_residual × context16 | 0.6176 |
| 3 | P1 compact64 × context16 | 0.6169 |
| 4 | P3 deeper128 × context16 | 0.6105 |
| 5 | MLP × own_plus_summary | 0.6078 |

**Screening 峰值：** X3 × pooled_residual × context16 **0.6230**。

**Internal held-out（選模後）：** X3 pure **0.6056** vs MLP **0.5358**（穩健勝過 MLP）。

**TCGA external：** X3 pure vs MLP **2/5** non-worse → `cross_attention_external_success = false`；Integrated5 最高為 MLP **0.5288**。

**結論：**
- Cross-attention + context16 在 CV / internal 優於 MLP；優勢高度依賴 **context16**（none 約 −0.015～−0.020）。
- Residual shortcut 貢獻極小（formal 中 pure ≈ residual）。
- TCGA 外推未通過預定成功門檻；18F 可解釋性未完成。

---

## Round 19 — Factorial + domain shift + role lock

### 19D/19E 開發集 DrugMacro AUC

**19D（5-fold CV mean-of-means）：** F2（D0×P2×O3）最高 **~0.620**。

**19E per-shift 摘要：**

| Shift | 最佳 | mean AUC | 備註 |
|-------|------|----------|------|
| cancer_type_heldout | E2 | 0.5824 | |
| drug_heldout | E3 | **0.7503** | 後續 Round 20 E3 來源 |
| scaffold_heldout | E1 | 0.5806 | |

### 19F 角色鎖定（無 single champion）

| 角色 | 候選 | 依據 |
|------|------|------|
| Historical anchor | E0 | 固定 |
| Source-performance champion | F2 | 19D 最高 |
| Parsimonious context | F1 | F2−F1 ≤ 0.003 |
| Cancer-shift specialist | E1 | 與 E2 差距 ≤ 0.003 |
| Chemical-shift specialist | E3 | drug/scaffold maximin |
| General recommended | **E3** | 三 shift 皆 pass |

### 19G 可解釋性

**Verdict：** `PARTIALLY_SUPPORTED` — 模型使用 drug 與 omics/context；高排名 perturbation 通常大於 matched random；attention 跨 member 穩定度不足以支持因果解釋。

---

## Round 20 — Unseen-drug closure（BioCDA-Predictive）

**狀態：** COMPLETE，LOCKED_RELEASE。

### Stage 20A — C16 vs C32

| | C16 | C32 | Δ |
|--|-----|-----|---|
| mean DrugMacro AUC | 0.7434 | **0.7509** | **+0.0074** |

**鎖定：** C32（stable_improvement）。

### Stage 20B — Pooled E3 vs Gated fusion

| | B_E3 | B_GATED | Δ |
|--|------|---------|---|
| mean DrugMacro AUC | baseline | −0.0020 | gated 未過 guardrails |

**鎖定：** B_E3 / AdapterMLPFusion + ResponseHead（gated_failed_guardrails）。

### Stage 20D — TCGA（選模後描述性）

| Target | DrugMacro AUC | Global AUC |
|--------|---------------|------------|
| aacdr_gdsc_intersect | **0.6173** | 0.6020 |
| aacdr_tcga_only | 0.5391 | 0.4182 |
| gdsc_intersect13 | 0.4714 | 0.5506 |
| tcga_only3 | 0.4591 | 0.3826 |
| dapl | 0.4284 | 0.4632 |

**結論：** C32 在固定 E3 下穩定提升 unseen-drug AUC；gated fusion 未過預定門檻；TCGA 不得用於選模。

**鎖定架構：** Z64 + C32 → O2[96] + D0 GIN32 → AdapterMLPFusion → ResponseHead（無 cross-attention）。

---

## Round 21 — Cross-Attention v1（BioCDA-XA-Candidate）

**狀態：** COMPLETE；**REJECTED**（未 LOCKED）。

| 模型 | mean DrugMacro AUC | Δ vs M0 |
|------|-------------------|---------|
| M0 pooled_baseline | **0.746** | — |
| M1 biocda_xa_z | 0.714 | −0.032 |
| M2 biocda_xa_zc | 0.709 | **−0.037** |

**根因：** performance_failure（非 attention collapse、非 domain shift）。

**結論：** XA v1 顯著落後 pooled baseline；C32 改變 attention 但未改善 XA 預測；保留 M0 / BioCDA-Predictive 作為唯一正式預測模型。TCGA 可解釋性延後。

---

## Round 23 — No-Pooling XA v2

**狀態：** COMPLETE；**REJECTED**。

| 模型 | mean AUC | ΔAUC vs P0 | mean AUPRC | ΔAUPRC vs P0 |
|------|----------|------------|------------|--------------|
| P0 BioCDA-Predictive | **0.744** | — | **0.512** | — |
| X0 fresh XA | 0.740 | −0.0043 | 0.506 | −0.0059 |
| X2 transfer+KD | 0.720 | −0.0247 | 0.490 | −0.0214 |
| X1 transfer | 0.699 | −0.0455 | 0.477 | −0.0342 |

**最接近候選（X0 fresh）：** mean ΔAUC −0.0043 達 mean 門檻，但 **1/3** seed non-worse（需 2/3）→ 整體 REJECTED。

**結論：** No-pooling XA 幾乎追平 P0 均值，但 seed 穩定性不足；transfer/KD 未縮小差距。BioCDA-Predictive 維持 LOCKED_REFERENCE；XA attention 不可用於解釋 Predictive 預測。

---

## TCGA 事後比較（5 targets，mean DrugMacro AUC）

僅描述性，**不用於選模**。

| 模型 | mean_5targets |
|------|---------------|
| biocda_xa_fresh / X0 (R23) | **0.5484** |
| pooled_baseline / M0 (R21) | 0.5344 |
| biocda_xa_zc / M2 (R21) | 0.5093 |
| BioCDA-Predictive R20 15-fold | 0.5031 |
| biocda_predictive / P0 (R23) | 0.5150 |

**注意：** TCGA 上 XA 候選可能高於 locked Predictive，但 development unseen-drug 選模已 REJECTED XA；不可混用名稱或 claim。

---

## 決策時間線（精簡）

```text
R13 exp_008 (0.6112) ──► R17/R17R 未重現 ──► R18 XA+context16 CV 贏、TCGA 未過
    ──► R19 E3/F3 pooled 鎖角色 ──► R20 C32+E3 = BioCDA-Predictive LOCKED
    ──► R21 XA v1 REJECTED ──► R23 XA v2 REJECTED（fresh 最接近）
```

**當前正式預測模型：** **BioCDA-Predictive**（Round 20 locked）。  
**可解釋性路線：** BioCDA-XA-Candidate 待未來 round 先達 paired performance parity。

---

## 相關文件

| 文件 | 內容 |
|------|------|
| `round{N}_final_report.md` | 各 Round 精簡版 |
| `model_cards/round19_locked_models.md` | Round 19 鎖定模型卡片 |
| `proposal.md` / `design.md` | 原始提案與設計 |
