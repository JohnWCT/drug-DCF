# VAEwC / AEwC Pretrain 管線 — 結果摘要

> **完整跨 Round 彙總：** 請讀 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md)  
> 本文件僅保留 Pretrain 主線（Round 1–16）分數與結論。

**主指標：** `Average_TCGA_AUC_mean`（gdsc_intersect13，4 finetune combo 平均）  
**歷史基準：** exp_746 = **0.5462**

---

## Round 1–3（Prototype InfoNCE）

| Round | 最佳 | Avg TCGA | 結論 |
|-------|------|----------|------|
| R1 | exp_031 | 0.544 | InfoNCE 略優 control 組平均 |
| R3 | exp_018（control） | **0.5695** | control 超越 exp_746；InfoNCE 無增益 |
| R3 發現 | — | — | `score_kmeans` 與下游 r≈0.52；`score_total` 與下游 r≈0.03 |

**三方對照（R3）：** exp_746 0.5462 · exp_018 0.5695（+4.3% TCGA，CCLE Test 退步）· exp_100 InfoNCE 0.5153

---

## Round 4–8

| Round | 最佳 | Avg TCGA | 結論 |
|-------|------|----------|------|
| R4.1 | exp_035 | 0.5339 | t2s InfoNCE 緩解 K-means 崩潰，下游未超 R3 |
| R5 | exp_001 | 0.5403 | 超越 R4.1；pure control |
| R6 | exp_010 | 0.5569 | 超越 R5 |
| R7 | exp_048 | **0.5918** | **pretrain 主線峰值** |
| R8 | exp_188 | 0.5777 | 未超越 R7 |

---

## Round 9–16（Conditional ADV → Proto features）

| Round | 最佳 | Avg TCGA | vs 前輪最佳 | 結論 |
|-------|------|----------|-------------|------|
| R9 repro | exp_048 seed303 | 0.5671 | < R7 | global 對齊佳，conditional leakage 高 |
| R10 | exp_111 | 0.5749 | > R9 repro | 低於 R7 原始 |
| R11 | exp_035 | 0.5828 | +0.008 vs R10 | leakage 下降 |
| R12 | exp_037 | **0.5972** | +0.014 vs R11 | prototype align 有效 |
| R13 | r13_exp_008_own_plus_summary | **0.6112** | +0.014 vs R12 | proto response feature 峰值 |
| R14 | r14_exp_078 | 0.5909 | −0.020 vs R13 | VICReg 未超越 |
| R15 | r15c_exp_005 | 0.6083 | −0.003 vs R13 | 接近 R13 |
| R16 | — | ~0.6068 | — | 未超越 R13 |

---

## 主線 checkpoint 演進

```text
exp_746 (0.546) → R3 exp_018 (0.570) → R7 exp_048 (0.592)
  → R12 exp_037 (0.597) → R13 exp_008 (0.611) ← downstream finetune 峰值
```

Round 17 起進入 feature / architecture screening，見 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md)。
