# Round 24 — 目標更新與問題定義 Plan

**狀態：** PROBLEM_DEFINITION · **不含解法**

**前置文件：**
- [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) — 全專案分數與 round 結論
- [`round23_final_report.md`](round23_final_report.md) — R23 GDSC gate 與 TCGA 雙軌結論
- [`biocda_final_architecture_selection.md`](biocda_final_architecture_selection.md) — TCGA 加權選模（R23 事後）

**本文件範圍：** 定義 Round 24 的**更新目標**、**待解問題**、**成功標準**與**已知約束**。具體實驗設計、架構改動、訓練配方留待後續 plan 補充。

---

## 1. 目標演進（為何需要 Round 24）

### 1.1 歷史目標（Round 1–16，pretrain / finetune 主線）

| 項目 | 設定 |
|------|------|
| 特徵 | 11D `proto_summary` + 75D omics latent → **`own_plus_summary`** |
| 選模依據 | TCGA **`gdsc_intersect13`** 上的 `Average_TCGA_AUC_mean`（4 finetune combo 平均） |
| 峰值 | R7 exp_048 **0.5918**；R13 own_plus_summary **0.6112** |
| 當時 TCGA 表現（Macro Average AUROC，使用者紀錄） | |

| 評估子集 | DrugMacro AUROC | DrugMacro AUPRC |
|----------|----------------:|----------------:|
| 13 seen（gdsc_intersect13） | **0.5918** | 0.6522 |
| 3 unseen（tcga_only3） | 0.5675 | 0.7284 |
| 5 single drug（DAPL） | 0.4739 | 0.5230 |

此階段的成功定義：**在 TCGA 外部驗證上最大化 gdsc_intersect13**，並以 prototype response 特徵為核心。

### 1.2 中期目標（Round 18–23，BioCDA 主線）

| 項目 | 設定 |
|------|------|
| 特徵 | Z64 + C32 → 96-d O2（**不含** `own_plus_summary`） |
| 選模依據 | **GDSC development unseen-drug** DrugMacro AUC（Round 20 E3 → ~0.75） |
| TCGA | 明確標示為**選模後描述性評估**（Round 20D）或**事後比較**（R23 TCGA benchmark） |
| 架構 | BioCDA-Predictive（pooled E3）；BioCDA-XA v1/v2（交叉注意力候選） |

此階段的成功定義：**GDSC unseen-drug 性能 closure**；TCGA 不作為 lock gate。

### 1.3 **Round 24 更新目標（本 plan 定義）**

**正式產品 / 論文的外部驗證 north star 改回 TCGA 五 target，且必須超越下列 5-fold 基準。**

| 評估集 | 資料來源 | 藥物數 | 觀測 pair 數 | DrugMacro AUROC | DrugMacro AUPRC |
|--------|----------|--------|--------------|-----------------|-----------------|
| `gdsc_intersect13` | DAPL / eval3 | 12 | 906 | **0.5184 ± 0.0437** | 0.6011 ± 0.0252 |
| `tcga_only3` | DAPL / eval3 | 3 | 129 | **0.5586 ± 0.0442** | 0.7130 ± 0.0272 |
| `TCGA_drug_response_from_DAPL`（dapl） | DAPL / eval3（auxiliary） | 5 | 178 | **0.5356 ± 0.0570** | 0.5591 ± 0.0318 |
| `aacdr_gdsc_intersect` | AACDR / target_infer | 11 | 425 | **0.5582 ± 0.0618** | 0.6017 ± 0.0487 |
| `aacdr_tcga_only` | AACDR / target_infer | 8 | 97 | **0.4394 ± 0.0372** | 0.5942 ± 0.0206 |

**選模協議（延續 R23 TCGA 結論，更新 gate 定義）：**

| 項目 | 設定 |
|------|------|
| 評估域 | 僅上表五個 TCGA external target |
| **不得**作為選模依據 | GDSC development / validation / **test** |
| Target 優先順序（高→低） | `gdsc_intersect13` > `tcga_only3` > `dapl` > `aacdr_gdsc_intersect` > `aacdr_tcga_only` |
| 主指標 | DrugMacro AUROC |
| 加權（整體排序） | 5 : 4 : 3 : 2 : 1 |
| 平手規則 | DrugMacro AUPRC → Global AUROC → Global AUPRC |
| 成功定義 | **五 target 的 point estimate 均超越上表 mean**；std 不得顯著劣化（具體統計 gate 待 Round 24 實驗協議鎖定） |

**歷史 stretch goal（非 Round 24 硬性 gate，但作為長期參照）：**  
R13 own_plus_summary gdsc_intersect13 **0.6112** / 使用者紀錄 **0.5918** — 代表在一致 protocol 下曾達到的上限。

---

## 2. 現況快照（BioCDA，R23 TCGA benchmark）

資料來源：`reports/biocda_tcga_comparison/biocda_tcga_comparison_long.csv`  
協議：**3-seed probability ensemble、單次 TCGA inference**（與 eval3 5-fold 基準**尚未對齊**，見 §3.4）。

### 2.1 與 Round 24 基準的差距（DrugMacro AUROC）

| Target | 基準 mean | X0 fresh (R23) | P0 Predictive (R23) | X0 Δ vs 基準 | P0 Δ vs 基準 |
|--------|----------:|---------------:|--------------------:|-------------:|-------------:|
| gdsc_intersect13 | 0.5184 | 0.481 | **0.513** | **−0.037** | −0.005 |
| tcga_only3 | 0.5586 | **0.605** | 0.520 | **+0.046** | −0.039 |
| dapl | 0.5356 | 0.529 | 0.465 | −0.007 | −0.071 |
| aacdr_gdsc_intersect | 0.5582 | **0.563** | **0.575** | +0.005 | +0.017 |
| aacdr_tcga_only | 0.4394 | **0.564** | 0.501 | **+0.125** | +0.062 |

### 2.2 現況摘要

- **未達標 target：** 以 X0（TCGA 加權最優）計，**gdsc_intersect13**、**dapl** 仍低於基準；其餘三 target 已超越。
- **最大結構性缺口：** `gdsc_intersect13`（−0.037）；P0 在此 target 僅差 −0.005，接近基準 noise band（std 0.044）。
- **模型間分歧：** 無單一現行 checkpoint 在五 target 上同時超越基準；X0 與 P0 優勢 target 互補。
- **Global vs DrugMacro 不一致（X0 / gdsc_intersect13）：** Global AUROC **0.544** vs DrugMacro **0.481** — 整體排序能力與 per-drug macro 聚合結果脫節。

### 2.3 歷史 vs 現行（gdsc_intersect13 單點）

| 時代 | 指標 | 數值 | 備註 |
|------|------|-----:|------|
| R7 / 使用者紀錄 | Macro AUROC（13 seen） | 0.5918 | own_plus_summary 時代 |
| R13 peak | Average_TCGA_AUC | 0.6112 | pretrain finetune 主線峰值 |
| Round 24 基準 | DrugMacro AUROC 5-fold mean | 0.5184 | eval3 |
| BioCDA X0 (R23) | DrugMacro AUROC | 0.481 | 現行 TCGA benchmark |
| BioCDA P0 (R23) | DrugMacro AUROC | 0.513 | 現行 TCGA benchmark |

---

## 3. 問題定義

以下為 Round 24 需**先釐清或解決**的問題陳述，**不含**解法。

---

### P1 — 目標函數漂移（Objective Misalignment）

**陳述：**  
Round 18 起，正式選模主軸由「TCGA gdsc_intersect13 外推」轉為「GDSC development unseen-drug DrugMacro AUC」。Round 20 鎖定 BioCDA-Predictive 時，GDSC unseen-drug 顯著提升（~0.75），但 TCGA `gdsc_intersect13` 由歷史 ~0.59 量級跌至 ~0.47–0.51 量級。

**待回答：**
- GDSC unseen-drug 優化與 TCGA 五 target 表現之間，是否存在可量化的 **Pareto trade-off**？
- Round 24 是否正式宣告：**外部驗證 north star 優先於 GDSC development gate**？
- 若兩者衝突，哪個 axis 作為 lock / reject 依據？

**影響：** 決定 Round 24 的成功定義是否覆寫 Round 20 LOCKED_RELEASE 的 implicit objective。

---

### P2 — 特徵配方斷裂（Feature Regime Break）

**陳述：**  
歷史峰值（R13，0.6112）依賴 **`own_plus_summary`**（11D proto_summary + 75D omics latent 路線）。Round 20 C32 重建時，O2 特徵為 Z64 + PCA32(raw context)，且 **`includes_own_plus_summary: false`**。現行 BioCDA 全線使用此 96-d O2。

**待回答：**
- `gdsc_intersect13` 的回落，有多少比例可歸因於 **移除 own_plus_summary**，而非架構或訓練目標？
- C16 vs C32 vs own_plus_summary 三者在**同一 BioCDA 架構、同一 TCGA eval protocol** 下的貢獻分解是否已量測？
- 歷史 0.5918 / 0.6112 與現行 0.48–0.51 的差距中，多少是**特徵**、多少是**模型**、多少是**評估協議**？

**影響：** 若不分解此問題，Round 24 可能在錯誤的 levers（架構 / XA / KD）上投入資源。

---

### P3 — 內部 CV 增益無法外推（Internal–External Gap）

**陳述：**  
Round 18 已記錄：cross-attention + context16 在 formal 5CV DrugMacro AUC 上達 **0.618**，但 TCGA external Integrated5 最高為 MLP own_plus_summary **0.529**；`cross_attention_external_success = false`。Round 19–23 延續 GDSC 內部選模，未解決此外推失敗。

**待回答：**
- GDSC CV / development 上的 architecture ranking，對 TCGA 五 target  ranking 的 **Spearman 相關**是否為負或近零？
- 「在 development 上 non-worse」是否為 TCGA 成功的必要條件？（R23 X0：GDSC ΔAUC ≈ −0.004 但 TCGA 加權最優 — 已出現反例）
- Round 24 是否需要**獨立的 TCGA validation fold**（非 test、非 GDSC）作為 early stopping / model selection？

**影響：** 若 internal gate 與 external gate 脫鉤，Round 24 選模流程必須重新設計，不能沿用 R20/R23 GDSC paired gate。

---

### P4 — 單 target 瓶頸：`gdsc_intersect13`（Primary Bottleneck）

**陳述：**  
五 target 中，僅 `gdsc_intersect13` 對所有現行 BioCDA 候選呈現**系統性 underperform**（X0 −0.037，P0 −0.005 vs 基準）。此 target 與 pretrain/GDSC 藥物交集最大（12 drugs，906 pairs），且為 target 優先順序最高項（權重 5）。

**待回答：**
- 哪些 drug 在 `gdsc_intersect13` 上 per-drug AUC 最低？是否集中於特定 scaffold / MOA？
- X0 Global AUROC（0.544）高於 DrugMacro（0.481）的原因：support threshold、class imbalance、還是 calibration？
- 歷史 own_plus_summary 在**同一 906 pairs** 上的 per-drug 分布，與 BioCDA 是否同一組 weakness pattern？

**影響：** Round 24 整體是否達標，幾乎取決於此單 target 能否 +0.037（X0）或 +0.005（P0）。

---

### P5 — 架構角色衝突（Architecture Role Conflict）

**陳述：**  
Round 23 存在**雙軌結論**：

| 軸 | 結論 | 依據 |
|----|------|------|
| GDSC unseen-drug gate | BioCDA-XA v2 **REJECTED**；Predictive **LOCKED_REFERENCE** | 配對 ΔAUC、seed non-worse |
| TCGA 加權選模 | BioCDA-XA v2 Fresh **SELECTED** | 五 target DrugMacro 加權 |

兩軌對「正式部署模型」給出不同答案；lock manifest（`biocda_xa_model_lock.json`）仍為 GDSC-REJECTED，尚未更新為 TCGA-SELECTED。

**待回答：**
- Round 24 的單一 champion 是否必須**同時**通過 GDSC gate 與 TCGA gate，還是允許 **TCGA-only champion**？
- BioCDA-Predictive 與 BioCDA-XA 是否維持**角色分離**（predictive vs interpretable），還是合併為單一產品模型？
- 若 XA 在 TCGA 加權勝出但在 gdsc_intersect13 輸給 P0，最終 lock 以哪個 target 優先順序為準？

**影響：** 未定義前，Round 24 可能產出第三套互相矛盾的 lock 狀態。

---

### P6 — 評估協議不對齊（Evaluation Protocol Mismatch）

**陳述：**  
Round 24 基準來自 **eval3 + 5-fold CV + mean ± std**。現行 BioCDA TCGA 分數來自 **3-seed checkpoint ensemble + 單次 inference**，且歷史 pretrain 峰值使用 **`Average_TCGA_AUC`**（finetune 4 combo 平均），與 **`DrugMacro AUROC`**（Round 18+ robust per-drug macro，min support 10/2/2）定義不同。

**待回答：**
- 在**同一模型、同一 checkpoint** 上，eval3 5-fold DrugMacro 與現行 TCGA benchmark pipeline 的差值有多大？
- `Average_TCGA_AUC` 與 `DrugMacro_AUC` 在 gdsc_intersect13 上對 historical best 的換算是否可行？
- Round 24 正式 gate 是否**鎖定 eval3 5-fold** 為唯一可比 protocol？

**影響：** 若 protocol 未對齊，P4 的 −0.037 gap 可能部分為度量 artifact，而非模型能力問題。

---

### P7 — 跨 target 表現分裂（Cross-Target Performance Split）

**陳述：**  
現行候選模型在不同 target 上優劣互斥：

| 模型 | 強項 target | 弱項 target |
|------|-------------|-------------|
| X0 fresh XA | tcga_only3, dapl, aacdr_* | gdsc_intersect13 |
| P0 Predictive | gdsc_intersect13, aacdr_gdsc_intersect | tcga_only3, dapl |
| R20 15-fold locked | aacdr_gdsc_intersect | gdsc_intersect13, tcga_only3, dapl |

**待回答：**
- 此分裂是 **architecture effect**、**training recipe effect**，還是 **ensemble seed variance**？
- Round 24 是否接受 **per-target 不同模型** 作為過渡方案，還是要求 **單一 unified model** 五 target 全勝？
- target 優先順序若改為 `aacdr_gdsc_intersect` 優先（敏感性分析已做），加權 winner 仍為 X0 — 但 gdsc_intersect13 缺口依舊；**加權排序是否足以代表產品目標**？

**影響：** 定義 Round 24 需要的是「加權最優」還是「五 target 全過 gate」。

---

### P8 — GDSC 高分的科學含義未驗證（GDSC–TCGA Construct Validity）

**陳述：**  
BioCDA 在 GDSC development unseen-drug 上達 ~0.74–0.75 DrugMacro AUC，但 TCGA `gdsc_intersect13` 未達 0.5184 基準。這引发：**GDSC unseen-drug 高分是否預測 TCGA 外部成功**？

**待回答：**
- GDSC drug-held-out split 與 TCGA gdsc_intersect13 的 drug / sample 重疊結構為何？
- 高 GDSC AUC 是否主要反映 **cell line 域内插值**，而非 **跨域外推**？
- Round 24 是否應降低 GDSC development 在方法學敘事中的权重？

**影響：** 論文 / 產品對「模型泛化能力」的 claim 需要與此問題的 answer 一致。

---

### P9 — 文檔與 lock 狀態不一致（Documentation / Lock Drift）

**陳述：**  
`RESULTS_SUMMARY.md` 記載 BioCDA-XA 為 **REJECTED**；`round23_final_report.md` 記載 TCGA 選模 **BioCDA-XA v2 Fresh**；`biocda_xa_model_lock.json` 仍為 GDSC performance_failure REJECTED。Round 24 基準（eval3 5-fold）尚未寫入任何 lock manifest。

**待回答：**
- Round 24 結束時，lock 文件應反映哪一套 protocol（GDSC / TCGA / 雙 gate）？
- `biocda_final_architecture_selection.md` 中的 target 優先順序是否為 Round 24 正式 protocol？

**影響：** 實驗進行前需先定義「何謂 LOCKED / SELECTED / REJECTED」的判準，避免 Round 24 結束後再次出現雙軌結論。

---

## 4. Round 24 成功標準（問題層級，非解法）

在 §3 問題获得量測或決策前，Round 24 **暫定**以下 exit criteria：

| ID | Criterion | 類型 |
|----|-----------|------|
| G1 | 五 TCGA target DrugMacro AUROC **point estimate** 均 > §1.3 基準 mean | 硬性（待 protocol 對齊後生效） |
| G2 | `gdsc_intersect13` 為首要診斷 target；gap 量化为 per-drug 可解釋 | 硬性 |
| G3 | 完成 eval3 5-fold 與現行 benchmark 的 **protocol 對照表** | 前置（P6） |
| G4 | 完成 own_plus_summary vs C32 vs 現行 O2 的 **特徵 attribution 表** | 前置（P2） |
| G5 | 單一 lock manifest 更新，消除 GDSC-REJECTED / TCGA-SELECTED 矛盾 | 前置（P5, P9） |
| G6 | **不以 GDSC test 選模** | 約束（延續 R23） |

**明確排除（本 plan 階段）：** 具體 hyperparameter、架構改動、loss 設計、訓練 schedule — 留待「Round 24 解法 plan」。

---

## 5. 問題優先級（建議討論順序）

```text
P6 評估協議對齊     →  否則所有 gap 數字不可信
P2 特徵配方斷裂     →  最大結構變因
P4 gdsc_intersect13 瓶頸  →  離 G1 最近 / 最遠的 target
P1 目標函數漂移     →  決定 Round 24 選模 axis
P5 / P9 架構與 lock 衝突  →  避免重複 R23 雙軌
P3 內外推 gap       →  方法學
P7 跨 target 分裂   →  定義 unified vs ensemble 產品形態
P8 GDSC 構念效度    →  敘事與 claim
```

---

## 6. 待使用者確認的決策點

在進入解法 plan 前，需確認：

1. **Round 24 基準表（§1.3）是否為唯一 north star？** 歷史 0.5918 / 0.6112 是否降為參考而非 gate？
2. **五 target 全過 vs 加權最優：** 是否要求 G1 五 target 全部超越，還是加權分數最高即可？
3. **單模型 vs 多模型：** 是否允許過渡期的 per-target champion？
4. **GDSC gate 是否保留？** 若保留，與 TCGA gate 衝突時的優先級？
5. **eval3 5-fold 是否為 Round 24 唯一正式 TCGA protocol？** 基準模型 checkpoint / 特徵配方為何（用于 P6 對照）？

---

## 7. 下一步（不在本文件）

- 使用者確認 §6 決策點
- 撰寫 **Round 24 解法 plan**（對應 P1–P9 的量測實驗與 gate 設計）
- 更新 `RESULTS_SUMMARY.md` 與 lock manifest（Round 24 結束後）

---

## 附錄 A — 名詞對照

| 名詞 | 含義 |
|------|------|
| `own_plus_summary` | R13 peak 使用的 prototype response 特徵模式 |
| O2 | Sample omics 輸入向量（現行 Z64+C32=96-d） |
| DrugMacro AUROC | 符合 support 門檻的 per-drug AUROC 之 macro mean（Round 18+） |
| `Average_TCGA_AUC` | Finetune pipeline 的 TCGA average metrics（pretrain 主線） |
| eval3 | Round 24 基準表所標資料來源 / 評估配置 |
| GDSC unseen-drug | Round 20+ development drug-held-out DrugMacro 評估 |

## 附錄 B — 相關 artifact 路徑

| 用途 | 路徑 |
|------|------|
| R23 TCGA 長表 | `reports/biocda_tcga_comparison/biocda_tcga_comparison_long.csv` |
| TCGA 架構選擇 | `reports/biocda_tcga_architecture_selection.json` |
| R20 lock | `reports/round20_final_model_lock_public.json` |
| XA lock（GDSC） | `reports/biocda_xa_model_lock.json` |
| C32 特徵（無 summary） | `result/.../round20_unseen_drug_closure/features/z_plus_context32` |
