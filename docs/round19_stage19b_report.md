# Round 19 Stage 19B Report — Drug × Predictor × Omics Screen

**Date:** 2026-07-14  
**Status:** **ALL_DONE — 117/117 jobs done，0 failed**  
**Stage gate:** Stage 19B = **GO complete**；Stage 19C selection 前需另做交互分析與 candidate list（本報告已含主效果對比，**尚未寫 selection lock**）  
**Root:** `result/optimization_runs/round19_factorial/`  
**Docker:** container `DAPL`，workdir `/workspace/DAPL`

---

## 1. Scope

Stage 19B 在 **ModelID-grouped screening 3-fold** 上，對所有相容 Drug × Predictor cells 同時測三個 omics anchors：

| Omics | 內容 | Dim |
|-------|------|-----|
| O1 | Z + own summary | 75 |
| **O2** | **Z + context16（不含 own summary）** | **80** |
| O3 | Z + summary + context16 | 91 |

設計刻意把 **O2 納入主 manifest**（非事後補測），避免先用含 summary 的模式淘汰架構再測 context-only 的 selection bias。

| Item | Value |
|------|-------|
| Compatible cells | 13 |
| Jobs | 13 × 3 omics × 3 folds = **117** |
| Split seed | 42 |
| Model seed | 101 |
| Early-stop | Robust DrugMacro AUC（`n_valid_auc_drugs ≥ 3`） |
| Selection uses TCGA / internal-test | **No** |
| Forbidden | D1×P2、D4×P2、MACCS+GIN/GINE hybrid |

---

## 2. Completion & execution

| Metric | Value |
|--------|-------|
| Manifest jobs | 117 |
| `job_status=done` | **117** |
| Failed / OOM-exhausted | **0** |
| Successful micro-batch | **256**（全部；未需降到 128） |
| Parallel pack | 1 GPU × **10** concurrent processes |
| Mean wall time / job | ~1423 s |
| Dispatch summary | `manifests/stage19b_job_status.summary.json` |
| Baseline commit（dispatch 前） | `4c66773`（見 `metadata/round19_baseline_git.json`） |

**Pilot（正式前）:** 6-job full `train_fold`（fold 0）全部 artifacts 通過；寫入 `pilot_job_status.json`，未污染正式 `job_status.json`。

| Pilot cell | best DrugMacro | best epoch |
|------------|----------------|------------|
| D0×P0×O1 | 0.5667 | 3 |
| D1×P1×O3 | 0.5968 | 62 |
| D2×P1×O3 | 0.5940 | 32 |
| D2×P2×O2 | 0.6189 | 20 |
| D3×P2×O3 | 0.6233 | 31 |
| D4×P1×O2 | 0.6084 | 59 |

修復（pilot 啟動時）：

1. 無邊分子 `edge_index` 強制 `(2, E)`（含 `(2,0)`）
2. `val_metrics.json` 使用 `metrics_to_jsonable`（避免 DataFrame 序列化失敗）

---

## 3. Primary ranking（cell mean over 3 folds）

Primary metric：**mean DrugMacro AUC**（3 folds）。完整表：`reports/stage19b_cell_ranking.csv`。

| Rank | Drug | Predictor | Omics | mean DrugMacro AUC | std |
|------|------|-----------|-------|--------------------|-----|
| 1 | D0 | P2 | O3 | **0.6275** | 0.0052 |
| 2 | D3 | P2 | O3 | 0.6256 | 0.0045 |
| 3 | D0 | P0 | O2 | 0.6248 | 0.0098 |
| 4 | D2 | P0 | O3 | 0.6245 | 0.0165 |
| 5 | D0 | P2 | O2 | 0.6241 | 0.0052 |
| 6 | D2 | P2 | O3 | 0.6219 | 0.0056 |
| 7 | D2 | P1 | O2 | 0.6216 | 0.0161 |
| 8 | D3 | P2 | O2 | 0.6209 | 0.0104 |
| 9 | D4 | P1 | O2 | 0.6204 | 0.0025 |
| 10 | D2 | P2 | O2 | 0.6182 | 0.0037 |

Top-10 中 **O2 佔 6 / O3 佔 4 / O1 佔 0**。若當時只用 O1／O3 篩架構會嚴重偏誤。

所有 job：`n_valid_auc_drugs ≥ 164`（未依賴 Global-AUC fallback）。

---

## 4. Planned contrast effects（paired over compatible cells）

跨相同 Drug×Predictor cell、對比不同 omics／表示（mean Δ DrugMacro AUC）：

### 4.1 Omics

| Contrast | n cells | mean Δ | median Δ | frac(Δ>0) |
|----------|---------|--------|----------|-----------|
| **O2 − O1**（context vs summary） | 13 | **+0.0123** | +0.0086 | **13/13** |
| **O3 − O2**（summary added to context） | 13 | ≈0（+4e-5） | +0.0016 | 8/13 |
| **O3 − O1**（context added to summary） | 13 | **+0.0123** | +0.0106 | **13/13** |

**解讀：** 在這個 population 上，**prototype context16 相對於 own summary 是主導增益**；在已有 context 後再加 summary 幾乎中性。這正是把 O2 納入主屏的科學理由。

Per-omics cell-mean averages：O1 **0.6041** · O2 **0.6163** · O3 **0.6164**。

### 4.2 Drug representation

| Contrast | Scope | mean Δ | n |
|----------|-------|--------|---|
| D1 − D0（node 64 vs 32） | P0/P1 × 3 omics | −0.0012 | 6 |
| D2 − D1（graph 64 vs 32 bottleneck） | P0/P1 × 3 omics | **+0.0065** | 6 |
| D3 − D2（GINE bond-aware） | 全部共有 predictor×omics | −0.0034 | 9 |

**解讀：** 單純把 node 從 32→64（D1）未帶來一致增益；**放寬 graph pooling bottleneck（D2）**较明顯。GINE bond（D3）相對 D2 平均略負，但 **D3×P2×O3 仍排第 2**——bond 與 atom cross-attn 的交互需在 19C 分開看，不能只看平均主效應。

### 4.3 Predictor integration

| Contrast | mean Δ | n paired cells |
|----------|--------|----------------|
| P1 − P0（pooled Transformer vs MLP） | −0.0022 | 15 |
| **P2 − P1**（atom cross-attn vs pooled） | **+0.0101** | 9 |

Predictor family means：P2 **0.6197** · P0 **0.6112** · P1 **0.6089**。

**解讀：** P1 相對 P0 未主導；**P2 atom cross-attention 是較強的 integration 主效應**（與 Round 18 X3 方向一致）。P1 與 P2 不應合併解讀。

---

## 5. Artefacts

```text
result/optimization_runs/round19_factorial/
  manifests/stage19b_drug_predictor_manifest.csv
  manifests/stage19b_job_status.csv
  manifests/stage19b_job_status.summary.json
  metadata/round19_baseline_git.json
  stage19b/{job_id}/   # checkpoint, val_*, train_*, job_status.json
  stage19b_pilot/      # 6 pilot jobs
  reports/
    stage19b_job_metrics.csv
    stage19b_cell_means.csv
    stage19b_cell_ranking.csv
    stage19b_analysis_summary.json
```

平行調度：`tools/round19_oom_runner.py` · `tools/run_round19_stage19b_parallel.sh`

---

## 6. Go / No-Go

| Item | Verdict |
|------|---------|
| 117/117 complete | **GO** |
| 3 folds / cell；DrugMacro 有效 | **GO** |
| 無 TCGA／internal selection | **GO** |
| O2 全 cell 比較完成 | **GO** |
| Stage 19C candidate list / selection lock | **NO-GO（尚未做）** |
| 19D repeated 5CV | **NO-GO** |
| 依本輪結果加 GIN128 / JK-cat | **NO-GO（無充分瓶頸證據）** |

### Suggested 19C directions（非正式 lock）

1. Omics 主線優先 **O2／O3**；O1 可作對照，不宜作唯一篩選軸。  
2. Predictor 主線優先保留 **P2**；P0 作為強 MLP baseline（尤其 O2）。  
3. Drug 不要只留單一赢家：至少保留 **D0**、**D2**、**D3**，並保留 **D4×P1×O2** 作為 MACCS 對照。  
4. 正式 selection 仍須遵守：不看 TCGA、不看 internal-test 排名。

---

## 7. Related docs

- [`docs/round19_factorial_ide_manual.md`](round19_factorial_ide_manual.md)  
- Round 18 背景：[`docs/round18_final_report.md`](round18_final_report.md) · [`docs/round18_stage18e_report.md`](round18_stage18e_report.md)
