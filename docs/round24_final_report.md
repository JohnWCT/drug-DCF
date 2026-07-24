# Round 24 — TCGA Recovery（Final）

**狀態：** `LOCKED` · champion **`E-NH0`**  
**容器：** `DAPL` · `/workspace/DAPL`  
**Lock：** [`reports/round24_final_model_lock.json`](../reports/round24_final_model_lock.json)

---

## 1. 實驗方法

### 1.1 目標

選出**單一 unified model**（禁止 per-target champion），在固定 `eval3` 協議下通過 AACDR 硬閘並鎖定。

### 1.2 評估協議（eval3）

| 項目 | 設定 |
|------|------|
| 訓練 folds | Round 18 source **5-fold**；每 fold 以 source validation 選 checkpoint |
| TCGA 推論 | 每 fold checkpoint 分別推論五個 external target |
| 主指標 | 5-fold mean **DrugMacro AUROC / AUPRC** |
| Ensemble | 五 fold 機率平均僅作補充，**不取代** fold-mean gate |
| 防洩漏 | TCGA 標籤**不進入** loss / early stopping / checkpoint / 超參搜尋 |
| GDSC 角色 | development / validation / test **僅診斷**，不參與選模 |

### 1.3 五組 TCGA target（皆須報告）

| Key | 角色 | 資料 |
|-----|------|------|
| `aacdr_gdsc_intersect` | **硬閘** | AACDR ∩ GDSC |
| `aacdr_tcga_only` | **硬閘** | AACDR TCGA-only |
| `dapl` | 診斷／排序 | DAPL 標註 |
| `gdsc_intersect13` | 診斷／排序 | DAPL ∩ GDSC 13 drugs |
| `tcga_only3` | 診斷／排序 | DAPL TCGA-only 3 drugs |

### 1.4 超越標準與選模規則（stest0）

標準來源：[`AACDR_drug_macro_auroc_auprc.md`](AACDR_drug_macro_auroc_auprc.md)（**無 10% internal testset**）。

| Target | AUROC gate | AUPRC 參照 |
|--------|-----------:|-----------:|
| `aacdr_gdsc_intersect` | > **0.5279** | 0.5710 |
| `aacdr_tcga_only` | > **0.4804** | 0.6300 |
| `dapl` | 0.5304（不擋 lock） | 0.5570 |
| `gdsc_intersect13` | 0.5197（不擋 lock） | 0.5981 |
| `tcga_only3` | 0.5536（不擋 lock） | 0.6960 |

| 規則 | 內容 |
|------|------|
| Hard PASS | 僅兩組 AACDR 硬閘皆嚴格大於標準 |
| NO_LOCK | 任一硬閘未過 |
| PASS 後排序 | 加權 5:4:3:2:1（`aacdr_gdsc` → `tcga_only3`） |
| 平手 | DrugMacro AUPRC → Global AUROC → Global AUPRC |
| Lock 池 | **僅 NoHoldout** 合格臂；holdout 參考不混排 |

### 1.5 實驗矩陣（摘要）

| Stage | 方法 | 目的 |
|-------|------|------|
| 24A | 鎖定 eval3 manifest／cohort（906→886 miss_latent） | 協議與基準可追溯 |
| 24B | B0 pooled×own_plus_summary；B1 predictive×C32；B2 XA×C32 | 同協議重建現況 |
| 24C | 固定 predictive_e3，掃 F0–F4 feature | 特徵 attribution；top2=F2(C16)、F3(C32) |
| Ablation | B0 × {Ctrl, **NoHoldout**, AACDR} | 診斷訓練資料用量（NoHoldout 可過硬閘） |
| 24E/F | NoHoldout 確認 E-NH0/1/2 + holdout refs | 正式 gate／lock |

**NoHoldout：** development ∪ internal_test 併入 formal 5-fold（對齊 stest0，無另留 10% test）。

---

## 2. 結果

### 2.1 Stage 24B（holdout 協議）

| ID | 模型 × 特徵 | aacdr_gdsc | aacdr_tcga | 硬閘 |
|----|-------------|-----------:|-----------:|:----:|
| B0 | pooled_mlp × own_plus_summary | 0.5285 | 0.4861 | PASS |
| B1 | predictive_e3 × C32 | 0.5268 | 0.4730 | NO_LOCK |
| B2 | xa_fresh × C32 | 0.5263 | 0.4858 | NO_LOCK |

### 2.2 Stage 24C（feature sweep，predictive_e3）

| ID | Feature | aacdr_gdsc | aacdr_tcga | 硬閘 |
|----|---------|-----------:|-----------:|:----:|
| **F2** | C16 | **0.5427** | **0.5398** | PASS |
| F3 | C32 | 0.5268 | 0.4730 | NO_LOCK |
| F0 | own_plus_summary | 0.5299 | 0.4351 | NO_LOCK |
| F1 | z_only | 0.5221 | 0.4537 | NO_LOCK |
| F4 | C64 | 0.5080 | 0.4361 | NO_LOCK |

### 2.3 Train-source ablation（診斷，B0）

| Arm | gdsc13 | tcga3 | dapl | aacdr_gdsc | aacdr_tcga | 硬閘 |
|-----|-------:|------:|-----:|-----------:|-----------:|:----:|
| Ctrl | 0.530 | 0.544 | 0.508 | 0.528 | 0.486 | PASS |
| **NoHoldout** | **0.570** | 0.485 | 0.482 | **0.565** | **0.497** | **PASS** |
| AACDR | 0.474 | 0.448 | **0.537** | 0.506 | 0.494 | NO_LOCK |

NoHoldout 顯著抬高 `aacdr_gdsc_intersect`／`gdsc_intersect13`，成為 24E 正式確認起點。

### 2.4 Stage 24E — NoHoldout lock 池

| ID | Architecture × Feature | aacdr_gdsc | aacdr_tcga | 硬閘 | Lock |
|----|------------------------|-----------:|-----------:|:----:|:----:|
| **E-NH0** | pooled_mlp × own_plus_summary | **0.5648** | 0.4971 | **PASS** | **champion** |
| E-NH1 | predictive_e3 × C16 | 0.5501 | 0.4992 | **PASS** | — |
| E-NH2 | predictive_e3 × C32 | 0.5210 | 0.5167 | NO_LOCK | — |
| E-REF2 | predictive × C16（holdout） | 0.5427 | 0.5398 | PASS | 不進 lock |
| E-REF3 | predictive × C32（holdout） | 0.5268 | 0.4730 | NO_LOCK | 不進 lock |

兩臂 PASS 時依加權（`aacdr_gdsc` 權重最高）→ **E-NH0**。

### 2.5 Champion（E-NH0）五組 DrugMacro（5-fold mean）

| Target | AUROC | AUPRC | 硬閘 |
|--------|------:|------:|:----:|
| `aacdr_gdsc_intersect` | **0.5648** | 0.6186 | Y（+0.037） |
| `aacdr_tcga_only` | **0.4971** | 0.6532 | Y（+0.017） |
| `dapl` | 0.4820 | 0.5416 | N |
| `gdsc_intersect13` | 0.5697 | 0.6121 | N（報） |
| `tcga_only3` | 0.4845 | 0.6368 | N |

完整候選 vs stest0：[`reports/round24/vs_aacdr_standard.md`](../reports/round24/vs_aacdr_standard.md)  
分 stage 表：[`reports/round24/tcga_metric_tables.md`](../reports/round24/tcga_metric_tables.md)

---

## 3. 結論

1. **鎖定模型：** `E-NH0` = **pooled MLP × `own_plus_summary` × NoHoldout** formal 5-fold。
2. **資料協議對齊 stest0（無 10% holdout）** 是通過 AACDR 硬閘的關鍵；同架構 holdout Ctrl 僅勉強過閘。
3. **Predictive E3 × C16/C32** 在 NoHoldout 下未勝過 pooled×own_plus_summary 的硬閘主軸（`aacdr_gdsc_intersect`）。
4. **XA（B2）** 在本 round eval3 協議下未通過硬閘。
5. TCGA 五組為 Round 24 **selection benchmark**（不可再稱 untouched external test）。
6. `tcga_only3` AUROC 仍無候選超越 stest0（0.5536）；champion 為 0.4845。

---

## 4. 產物

```text
configs/round24/eval3.yaml
reports/round24_final_model_lock.json
reports/round24/stage24e/stage24e_decision.json
reports/round24/stage24e/candidate_manifest.json
reports/round24/vs_aacdr_standard.md
reports/round24/tcga_metric_tables.md
docs/AACDR_drug_macro_auroc_auprc.md
```
