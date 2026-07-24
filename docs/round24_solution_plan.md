# Round 24 — TCGA Recovery 解題計畫

**狀態：** **COMPLETE · LOCKED** · champion `E-NH0`（pooled × own_plus_summary × NoHoldout）· 硬閘 = AACDR stest0 · 見 [`round24_status_report.md`](round24_status_report.md) / [`round24_final_report.md`](round24_final_report.md)  
**問題定義：** [`round24_problem_definition_plan.md`](round24_problem_definition_plan.md)  
**操作手冊：** [`round24_ide_manual.md`](round24_ide_manual.md)

---

## 1. 任務與不可變契約

Round 24 的任務是選出一個**單一 unified model**，在同一套 `eval3` 5-fold 協議下，使 **AACDR 兩組** TCGA target 的 5-fold mean DrugMacro AUROC 超越既定標準（其餘三組必報，但不擋 lock）。

### 1.1 硬性成功門檻（Stage 24E 起修訂）

**PASS / LOCK 條件（僅此兩組；標準 = AACDR `stest0` / 無 10% testset）：**

| Target | AUROC gate | 基準 std | AUPRC reference（追蹤） |
|--------|-----------:|---------:|------------------------:|
| `aacdr_gdsc_intersect` | > **0.5279** | 0.0312 | 0.5710 ± 0.0122 |
| `aacdr_tcga_only` | > **0.4804** | 0.0414 | 0.6300 ± 0.0419 |

標準來源：[`AACDR_drug_macro_auroc_auprc.md`](AACDR_drug_macro_auroc_auprc.md)（`target_infer_stest0`）。  
兩組皆嚴格大於標準 mean → `PASS`；缺任一 → `NO_LOCK`。

**必報但不擋 lock（diagnostic / soft ranking；`eval3_stest0`）：**

| Target | AUROC 參照 | AUPRC 參照 |
|--------|-----------:|-----------:|
| `dapl` | 0.5304 | 0.5570 |
| `gdsc_intersect13` | 0.5197 | 0.5981 |
| `tcga_only3` | 0.5536 | 0.6960 |

### 1.1a 最終模型排序優先序（多個 PASS 時）

若有多個 `PASS` 候選，依下表 **5:4:3:2:1** 加權 DrugMacro AUROC 排序（高→低）：

| 權重 | Target | 角色 |
|-----:|--------|------|
| **5** | `aacdr_gdsc_intersect` | 硬閘 + 首要選模軸 |
| **4** | `aacdr_tcga_only` | 硬閘 + 次要 |
| **3** | `dapl` | 僅排序／診斷 |
| **2** | `gdsc_intersect13` | 僅排序／診斷 |
| **1** | `tcga_only3` | 僅排序／診斷 |

平手：DrugMacro AUPRC → Global AUROC → Global AUPRC。  
**不作排名：** GDSC 訓練／內部 CV。  
設定：`configs/round24/eval3.yaml` → `gate_required_targets` / `target_weights`。

### 1.1b TCGA 彙整欄位（強制）

每個 candidate / stage 的 TCGA 結果彙整必須對下列五檔同時給出 **DrugMacro AUROC** 與 **DrugMacro AUPRC**：

1. `aacdr_gdsc_intersect`（**硬閘**）
2. `aacdr_tcga_only`（**硬閘**）
3. `dapl`（診斷）
4. `gdsc_intersect13`（診斷）
5. `tcga_only3`（診斷）

缺 target 或缺 AUPRC 視為彙整失敗。

### 1.2 防洩漏與科學敘事

- TCGA 標籤不得進入 loss、early stopping、checkpoint selection 或超參數搜尋。
- 候選矩陣必須在正式 TCGA gate 前寫入 manifest 並鎖 hash。
- 正式 gate 完成後不得依結果追加候選；若需新候選，必須開新 round。
- GDSC development / validation / test 只作 diagnostic，不影響選模。
- 因五個 TCGA target 已用於 Round 24 selection，最終報告稱其為 **selection benchmark**，不可再稱 untouched external test。

---

## 2. 正式 eval3 協議

`eval3` 不是新資料集名稱，而是 Round 24 對以下既有流程的具名鎖定：

1. Round 18 source 5-fold 訓練與 validation。
2. 每 fold 使用 source validation 選 checkpoint。
3. 每個 fold checkpoint 分別推論五個 TCGA target。
4. 報告 fold-level DrugMacro AUROC/AUPRC 的 mean ± std。
5. 另報告五 fold probability mean ensemble，但 ensemble point estimate 不取代硬 gate。

### 2.1 重用契約

| 契約 | 來源 |
|------|------|
| 五 target 路徑 | `tools/finetune_tcga_eval.py` / `tools/round20_tcga.py` |
| source folds | `config/round18_architecture_settings.json` |
| fold inference | `step1_finetune_latent_pipeline_round18_cv.py` |
| Stage 18E orchestration | `tools/run_round18_stage18e_locked_eval.sh` |
| DrugMacro | `tools/round18_cv_metrics.py` |
| ensemble 唯一性 | `tools/round18_prediction_ensemble.py` |
| external analysis | `tools/analyze_round18_external_eval.py` |

### 2.2 Cohort 契約

Stage 24A 必須建立 `reports/round24/eval3_manifest.json`，至少包含：

- 五個 TCGA CSV 的 SHA256、raw rows、unique patients、drugs、class counts。
- omics、SMILES、drug-name normalization 後的 eligible rows。
- 每筆被排除資料的 `row_id` 與唯一 drop reason。
- source fold assignment、feature artifact、checkpoint、config hashes。
- DrugMacro support：`min_samples=10`、`min_positive=2`、`min_negative=2`。

`gdsc_intersect13` 906 raw pairs 與現有 Round 18 886 eligible rows 必須逐列解釋。若無法重現 cohort 或基準，不得進入 Stage 24B。

目前可核對的 Round 18 `pooled_mlp × own_plus_summary` artifact 為
`result/optimization_runs/round18_architecture/reports/round18_external_eval_summary.csv`：
`gdsc_intersect13` DrugMacro AUROC = **0.5415**、n_rows = **886**。它不是 §1.1 的
0.5184 / 906 基準本身，只是建立 eval3 provenance 的起點；Stage 24A 必須解釋兩者差異。

---

## 3. Stage 24A — 協議與基準鎖定

**解決：** P6、P9；交付 G3 前置。

### 工作

1. 建立 eval3 preflight，稽核資料、fold、feature、SMILES 與 checkpoint。
2. 從 Round 18 5-fold 資產重建 `pooled_mlp × own_plus_summary` 基準。
3. 在相同 prediction 上同時計算：
   - DrugMacro AUROC/AUPRC；
   - Global AUROC/AUPRC；
   - 歷史 `Average_TCGA_AUC`（僅對照）。
4. 對照使用者基準表與重算結果，明確區分 raw cohort、eligible cohort、fold mean、ensemble。

### 輸出

```text
reports/round24/stage24a/
├── eval3_manifest.json
├── cohort_coverage.csv
├── dropped_rows.csv
├── baseline_fold_metrics.csv
├── baseline_summary.json
└── protocol_alignment_report.md
```

### Gate 24A

- `PASS`：五個資料 hash、5 folds、所有 drop reasons、五 target metrics 均完整，且差異可追溯。
- `BLOCKED`：906/886 或其他 row count 差異無法解釋、fold/checkpoint 不完整、指標無法從 predictions 重算。

---

## 4. Stage 24B — 現況同協議重建

**解決：** P3、P6、P7。

### 候選

| ID | 模型 | 特徵 | 用途 |
|----|------|------|------|
| B0 | Round 18 pooled MLP | `own_plus_summary` | 可重現基準 |
| B1 | P0 pooled E3 | C32 | gdsc_intersect13 接近基準的 predictive anchor |
| B2 | X0 XA fresh | C32 | R23 TCGA 加權 winner |

P0/X0 必須按同一 source folds 重新訓練；R23 三 seed checkpoint 只能作 protocol 對照，不能充當五 fold。

### 輸出

- 每 candidate × fold × target predictions。
- fold mean ± std、ensemble、per-drug metrics、coverage。
- eval3 vs R23 3-seed vs R20 15-fold protocol delta。
- paired candidate-minus-B0 fold delta。

### Gate 24B

- 若 B1 或 B2 單一模型五 target 全過，跳至 Stage 24F。
- 若皆未全過，進入 Stage 24C。
- 不完整 fold 或 cohort/hash 不一致的候選判 `INVALID`，不得排名。

---

## 5. Stage 24C — 特徵 attribution

**解決：** P2；交付 G4。

固定 pooled E3 架構、source folds、optimizer、training budget 與 early-stopping 規則，只改 feature recipe。

| ID | Feature recipe | 預期維度 | 既有來源 |
|----|----------------|---------:|----------|
| F0 | `own_plus_summary` | 86 | R13/R17R artifact |
| F1 | O1 `z_plus_summary` | 75 | Round 19 feature builder |
| F2 | C16 `z_plus_context16` | 80 | Round 19/20 |
| F3 | C32 `z_plus_context32` | 96 | Round 20 current O2 |
| F4 | O3 `z_plus_summary_context16` | 91 | Round 19 |

維度不同時各自訓練，不共享 checkpoint。每個 artifact 必須保存 feature names、dimension、source artifact 與 projection hashes。

### 排序與收斂

1. 第一順位：超越基準的 target 數。
2. 第二順位：所有 target 中最小 AUROC delta（maximin）。
3. 第三順位：§1.1a 優先序加權 AUROC（`aacdr_gdsc` 5 → `tcga_only3` 1）。
4. 平手：DrugMacro AUPRC → Global AUROC → Global AUPRC。

最多保留兩個 feature recipe。若任一 recipe 已五 target 全過，停止擴展並進 Stage 24F。  
**24C 結果：** top2 = **F2（C16）**、**F3（C32）** → 進入 Stage 24E。

---

## 6. Stage 24D — gdsc_intersect13 根因診斷

**解決：** P4；交付 G2。

### 分析層

- **Coverage：** 906 raw、eligible、各 drug 被排除 row 與原因。
- **Support：** 每 drug 的 n、positive、negative、是否通過 10/2/2。
- **Ranking：** per-drug AUROC/AUPRC 與 fold variance。
- **Aggregation：** Global AUROC 與 DrugMacro 差異由哪些 drugs 造成。
- **Calibration：** 各 drug probability distribution、Brier score、reliability bins（只診斷，不以 threshold 調 AUROC）。
- **Weakness overlap：** B0/B1/B2/最佳 feature 的 bottom drugs、scaffold/MOA（資料可得時）。

### 輸出

```text
reports/round24/stage24d/
├── gdsc_intersect13_per_drug.csv
├── coverage_and_support.csv
├── weakness_overlap.csv
├── calibration_summary.csv
└── gdsc_intersect13_diagnostic.md
```

診斷只能用於預登記 Stage 24E 候選，不得在正式 gate 後迭代。

---

## 7. Stage 24E — NoHoldout 確認 × 優選架構／特徵

**解決：** P1、P3、P7、P8。  
**狀態：** NEXT（預登記 → 訓練 → 24F formal gate）。

### 7.0 為何必須重測（資料協議 ≠ 架構結論）

| 已完成實驗 | 訓練資料 | 能回答什麼 | **不能**直接當硬閘結論 |
|------------|----------|------------|------------------------|
| 24B / 24C（B0–B2, F0–F4） | Round18 **含 ~10% holdout** | 同資料下架構／特徵相對排序 | NoHoldout 下誰過 AACDR 硬閘 |
| train-source ablation | NoHoldout **僅** `pooled_mlp × own_plus_summary` | 資料用量可抬 `aacdr_gdsc` | 其他架構／特徵在 NoHoldout 是否更好 |

**結論：** NoHoldout 只改「用多少 GDSC 訓練列」，與模型架構正交。硬閘已改為 AACDR 兩組後，**不可**用 holdout 上的 F2/F3 分數宣稱能否 PASS，也**不可**只用 NoHoldout×pooled 代表全部架構。  
**24E 主軸：** 挑 holdout 上硬閘相關表現較佳的少數候選，在 **同一套 NoHoldout 5-fold** 上重訓＋eval3，再比對硬閘。

### 7.1 硬閘與挑選依據

標準：[`AACDR_drug_macro_auroc_auprc.md`](AACDR_drug_macro_auroc_auprc.md)。  
**PASS：** `aacdr_gdsc_intersect` >0.5279 **且** `aacdr_tcga_only` >0.4804（stest0）。

**Holdout 上依硬閘相關指標的優選（重測清單來源；對照 stest0 後）：**

| 優先 | 候選 | aacdr_gdsc | aacdr_tcga | 硬閘(stest0) | 理由 |
|-----:|------|-----------:|-----------:|:------------:|------|
| 1 | **F2** pred×C16 | 0.5427 | 0.5398 | **PASS** | 正式優選；仍須 NoHoldout 確認（標準對齊無 10% test） |
| 2 | **F3** pred×C32 | 0.5268 | 0.4730 | NO_LOCK | top2；差 `aacdr_gdsc` −0.001 / `aacdr_tcga` −0.007 |
| 3 | **NoHoldout pooled** | 0.5648 | 0.4971 | **PASS** | 資料基準 |
| 4 | B0/Ctrl pooled | 0.5285 | 0.4861 | **PASS** | holdout 下已過硬閘；作對照 |
| 5 | B2 XA×C32（可選→C16） | 0.5263 | 0.4858 | NO_LOCK | 架構對照 |

### 7.2 預登記候選矩陣（NoHoldout 確認為主）

**訓練資料協議（鎖死）：** `development ∪ internal_test` 全量重建 formal 5-fold（與 ablation NoHoldout 相同）；early-stop 僅 source val DrugMacro。  
**Holdout 對照：** 可重用 24C F2/F3 checkpoint 作「舊協議參考」，**不**計入 NoHoldout 硬閘排名。

| ID | Architecture | Feature | 資料 | 角色 |
|----|--------------|---------|------|------|
| **E-NH0** | `pooled_mlp` | own_plus_summary | **NoHoldout** | 資料基準；優先重用 ablation 產物並寫入 manifest（或同等協議重跑） |
| **E-NH1** | `biocda_predictive_e3` | **C16（F2）** | **NoHoldout** | **主確認**：優選架構×優選特徵×新資料 |
| **E-NH2** | `biocda_predictive_e3` | **C32（F3）** | **NoHoldout** | 特徵對照確認 |
| **E-NH3** | `biocda_xa_fresh` | **C16** | **NoHoldout** | 架構對照（單一 feature；可選，預算緊可砍） |
| E-REF2 | `biocda_predictive_e3` | C16 | holdout（24C） | 僅參考錨；不參與 NoHoldout 排名 |
| E-REF3 | `biocda_predictive_e3` | C32 | holdout（24C） | 僅參考錨 |

**禁止：** 把 holdout 結果與 NoHoldout 結果混排爭 lock；無界超參 sweep；依 TCGA 調參；F0/F1/F4 全量 NoHoldout 重跑。

### 7.3 執行順序

1. **Preregister：** `reports/round24/stage24e/candidate_manifest.json` + `.sha256`（明示每臂 `train_source=no_holdout|holdout_ref`）。
2. **Smoke：** 每新訓臂 1 fold；確認無 TCGA 洩漏、`num_workers=0`。
3. **Formal：** 先跑 **E-NH1 / E-NH2**（必要）；E-NH0 重用或補跑；E-NH3 視 GPU 並行。
4. **Eval3：** 五 target AUROC+AUPRC；硬閘只看 AACDR 兩組；更新 `vs_aacdr_standard`。
5. **解讀：**  
   - 若 E-NH1/NH2 過硬閘 → 在 PASS 候選間用 §1.1a 加權選 champion → 24F。  
   - 若僅 E-NH0 過 → 記錄「資料效應主導」；仍可 lock pooled NoHoldout（若已預登記），並在報告標註架構未再增益。  
   - 若皆不過 → `NO_LOCK`，報告 NoHoldout 下 gap。
6. Telegram 僅完整 round 結束時發送。

### 7.4 Early stopping 與選模

- 只使用 source-fold validation DrugMacro。
- TCGA／歷史結果不能選 epoch。
- Soft 監控可看 AACDR 兩組，**不得**據此改候選矩陣。

### Gate 24E

- 預登記候選（含重用路徑）完成後封存 manifest；事後改矩陣則作廢、需新 round。

---

## 8. Stage 24F — 一次性正式 gate 與 lock

**解決：** P5、P7、P9；交付 G1、G5、G6、G7。

### Pass / fail

```text
PASS = 單一 candidate 在 gate_required_targets
       （aacdr_gdsc_intersect ∧ aacdr_tcga_only）
       的 5-fold mean DrugMacro AUROC 全部嚴格大於 AACDR 標準。
```

- `PASS` 候選超過一個時，才套用 §1.1a 加權與平手規則。
- 無候選 `PASS`：寫 `NO_LOCK`，保留最佳（依硬閘 min_delta / 加權）與各 target gap。
- 不允許 per-target ensemble、人工選 fold、遺漏 fold 或事後改權重。
- 其餘三 target 失敗**不**構成 `NO_LOCK`（仍須完整報告）。

### 統一 manifest

`reports/round24_final_model_lock.json` 必須包含：

- `status`: `LOCKED` 或 `NO_LOCK`。
- model/feature/architecture/checkpoint paths 與 hashes。
- eval3 protocol、cohort、fold、config 與 candidate manifest hashes。
- 五 target baseline、candidate mean/std、delta、pass/fail。
- ensemble、AUPRC、Global metrics（supporting only）。
- GDSC diagnostic（`selection_role: none`）。
- `supersedes`：舊 R20/R23 lock 路徑；舊檔不修改。
- git commit 與生成時間。

---

## 9. Stage 24G — 科學結論與文件一致性

**解決：** P1、P3、P8、P9。

### 分析

- GDSC unseen-drug vs TCGA target / weighted score 的 Spearman。
- 候選的 Pareto frontier 與內外推 ranking inversion。
- feature attribution、architecture effect、protocol effect 的分解。
- 回答 P1–P9，不能把相關性描述為因果。

### 最終文件

- `docs/round24_final_report.md`
- `docs/RESULTS_SUMMARY.md`
- `docs/biocda_final_architecture_selection.md`
- `reports/round24_final_model_lock.json`

---

## 10. P1–P9 對應矩陣

| 問題 | 解題 Stage | 核心 artifact | 完成判準 |
|------|------------|---------------|----------|
| P1 目標漂移 | 24E/24G | objective alignment report | TCGA 為唯一 gate；trade-off 已量化 |
| P2 特徵斷裂 | 24C | feature attribution | 同架構五 feature 完整比較 |
| P3 內外推 gap | 24B/24G | protocol delta、Spearman | ranking inversion 可重現 |
| P4 gdsc_intersect13 | 24D | per-drug diagnostic | 906/886 與 macro gap 可解釋 |
| P5 架構衝突 | 24F | unified lock | 只保留單一正式狀態 |
| P6 協議不一致 | 24A/24B | eval3 manifest | cohort/fold/metric hashes 鎖定 |
| P7 target 分裂 | 24B/24E/24F | all-target gate | 不接受 per-target champion |
| P8 GDSC 構念效度 | 24G | Pareto/Spearman | 僅作 diagnostic claim |
| P9 lock 漂移 | 24A/24F/24G | final lock | 舊 lock historical superseded |

---

## 11. 預計實作契約

下列入口已實作並可在 Docker `DAPL` 執行（正式訓練進度見狀態報告）：

```text
configs/round24/eval3.yaml
scripts/round24/run_round24.py
biocda/validation/round24_protocol.py
biocda/validation/round24_gate.py
scripts/round24/analyze_features.py
scripts/round24/diagnose_gdsc_intersect13.py
scripts/round24/analyze_objective_alignment.py
scripts/round24/lock_round24_model.py
tests/test_round24_protocol.py
tests/test_round24_gate.py
```

統一 CLI 預計提供：

```text
preflight → protocol → baseline → features → diagnose
          → train → evaluate → select → lock → all
```

---

## 12. 終止條件

以下任一條件成立即停止當前 stage：

- cohort/hash/fold 不一致。
- TCGA 被用於 training、early stopping 或 checkpoint selection。
- candidate manifest 未預登記或 formal 後被修改。
- 任一 formal candidate 缺 fold、缺 target、缺 predictions。
- 無單一模型五 target 全過：Round 24 可正常結束，但狀態必須是 `NO_LOCK`。

