# Round 25 — Stage2 Margin / AADA → No-Pooling XA（Final）

**狀態：** `LOCKED_KEEP_S0` · Stage2 維持 **S0**  
**容器：** `DAPL` · `/workspace/DAPL`  
**Lock：** [`reports/biocda_xa_stage2_lock.json`](../reports/biocda_xa_stage2_lock.json)

---

## 1. 實驗方法

### 1.1 目標

只搜尋 **Stage2 alignment / prototype strategy**；固定下游 **fresh no-pooling BioCDA-XA v2**，不搜尋 XA 拓撲，也不用 TCGA 選模。

固定下游：

```text
Z64 + C32 → single query → fresh GIN atoms → no pooling → sample-to-atom XA → response head
```

### 1.2 Stage2 變體矩陣

| ID | 內容 | 時機 |
|----|------|------|
| **S0** | dual WGAN + always-on prototype（基準） | 必跑 |
| **S2** | dual WGAN + **margin-gated** prototype | 第一優先 |
| **S1** | **AADA AE** 取代 global WGAN + always-on prototype | 第二優先 |
| **S3** | AADA + margin-gated | 僅 S1 或 S2 通過 25A |
| **S2b** | prototype distance band | 僅過度重疊證據成立（本輪未觸發） |

### 1.3 Margin 欄位（不得混用）

| 欄位 | 用途 |
|------|------|
| `prototype_upper_margin` | S2 hinge：距離 > δ 才拉近 |
| `prototype_lower_margin` | S2b band：過近輕推 |
| `reconstruction_margin` | S1 AADA：target 重建誤差 hinge |

`δ`（`delta_c`）由 source minibatch→EMA anchor 距離 P90 估計，warm-up 後 freeze + SHA256。

### 1.4 階段設計

| Stage | 方法 | 選模依據 |
|-------|------|----------|
| **25A** | 共用 AE 一次 → S0/S2/S1（條件 S3）平行 screen | geometry／對齊／C32 readiness（相對 S0 不崩壞）；**不用 TCGA** |
| **25B** | 晉升 Stage2 重產特徵；固定 XA 配對 **B0(S0)／B1(晉升)／B2(Z-only)** | mean AUC／AUPRC／seed 非劣與 worst-seed floor |
| **25C** | B1 vs B2（同 S1 checkpoint，C32=0 pad） | C32 是否具可重現預測／attention 貢獻 |
| **Lock** | 寫入 `biocda_xa_stage2_lock.json` | **不覆寫** R23 `biocda_xa_model_lock.json`（維持 `REJECTED`） |

### 1.5 25B 晉升門檻（相對 B0）

- mean AUC Δ ≥ 0
- noninferior seeds ≥ 3/3
- worst-seed Δ > −0.010  
任一失敗 → `KEEP_S0`。

---

## 2. 結果

### 2.1 Stage 25A

| Variant | 結果 | 說明 |
|---------|------|------|
| S0 | PASS | 基準 |
| S2 | FAIL | `prototype_hinge_active_fraction=0`（margin 過寬，loss 失效） |
| **S1** | **PASS → `PROMOTE_S1`** | AADA 過對齊／幾何閘 |
| S3 | FAILED_MARGIN_INACTIVE | 因 S1 PASS 而執行；同樣 hinge=0，不晉升 |
| S2b | 未觸發 | 無過度重疊證據 |

### 2.2 Stage 25B（validation DrugMacro）

| Arm | Stage2 特徵 | mean AUC | mean AUPRC |
|-----|-------------|----------:|-----------:|
| **B0** | S0 | **0.6303** | 0.4151 |
| B1 | S1（25A 晉升） | 0.6241 | 0.4258 |
| B2 | S1 Z64-only（C32=0） | 0.6327 | — |

| Gate (B1−B0) | 結果 |
|--------------|------|
| mean AUC Δ | **−0.0063** &lt; 0 |
| noninferior seeds | **2/3** &lt; 3 |
| worst-seed Δ | **−0.0599** ≤ −0.010 |

**決策：`KEEP_S0`（不晉升 Stage2）**

### 2.3 Stage 25C（C32 ablation）

| 項目 | 結果 |
|------|------|
| B1−B2 AUC Δ | −0.0086 |
| predictive / attention effect | weak |
| **claim** | **`do_not_emphasize_C32`** |

C32 干預會改變 query／attention（zero/shuffle/wrong-cancer），但預測增益不可重現；不應強調「生物 context 導引」。

### 2.4 Lock

| 項目 | 值 |
|------|-----|
| Status | `LOCKED_KEEP_S0` |
| `promoted_stage2_variant` | **S0** |
| R23 GDSC XA lock | 維持 **REJECTED** |
| TCGA used for selection | **false** |

---

## 3. 結論

1. **25A** 幾何／對齊 screen 可晉升 **S1（AADA）**；margin-gated S2/S3 因 hinge 未啟動而失敗。
2. **25B** 固定下游 XA 後，S1 **未勝過** S0 → **不晉升 Stage2**。
3. **25C** C32 無穩定預測增益 → **`do_not_emphasize_C32`**。
4. Round 25 只鎖定 Stage2 配方（S0）；不改變 Round 23 XA `REJECTED` 敘事。

---

## 4. 產物

```text
reports/biocda_xa_stage2_lock.json
reports/round25_stage25a_decision.json
reports/round25_selection_decision.json
reports/round25_c32_xa_effect.json
reports/round25_stage25b_paired_performance.csv
config/round25_stage2_margin_screen.yaml
```
