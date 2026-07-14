# Round 19 Stage 19C Report — Omics Completion & Context Faithfulness

**Date:** 2026-07-14  
**Status:** **ALL_DONE — 54/54 jobs done，0 failed**  
**Stage gate:** Stage 19C = **GO complete**；**19D candidate proposal = READY（待人工審查）**；**Formal selection lock = NO-GO**（尚未寫 `round19_locked_selection.json`）  
**Root:** `result/optimization_runs/round19_factorial/`  
**Docker:** container `DAPL`，workdir `/workspace/DAPL`  
**19B baseline:** `c3c41f9`（117/117）

---

## 1. Scope

19C **不再搜尋架構**。在 19B 角色鎖定的 7 個 unique Drug×Predictor cells 上補齊：

| Mode | 內容 | 來源 |
|------|------|------|
| O0 | Z-only | 19C 新跑 |
| O1 | Z + summary | 19B 既有 |
| O2 | Z + context16 | 19B 既有 |
| O3 | Z + summary + context16 | 19B 既有 |
| O4 | Z + source-only prototype | 19C 新跑 |
| O2/O3_shuffled | context ModelID derangement | 19C 負對照 |

| Item | Value |
|------|-------|
| Unique selected cells | **7** |
| Core jobs | 7 × {O0,O4} × 3 folds = **42** |
| Context-shuffle controls | **12** |
| Total | **54** |
| Parallel pack | 12 jobs / GPU（~90% util） |
| Telegram | start / done 通知 |

禁止：TCGA、internal-test、Integrated5 作選擇。

---

## 2. Candidate lock（非正式）

`reports/round19_stage19c_candidate_lock.json`

| Role | Cell | selection_score = mean(O2,O3) |
|------|------|-------------------------------|
| R0 | D0×P0 | 0.6209 |
| R1 | D0×P1 | 0.6096 |
| R2 | D0×P2 | 0.6258 |
| R3 | D2×P0 | 0.6192 |
| R4 | D2×P1 | 0.6187 |
| R5 | D3×P2 | 0.6232 |
| R6 | D4×P1 | 0.6141 |

Shuffle controls：atom = **D0×P2**；pooled = **D0×P0**（最高 O2 among P0/P1）。

---

## 3. Completion

| Metric | Value |
|--------|-------|
| `job_status=done` | **54/54** |
| Failed | **0** |
| Core done | 42/42 |
| Controls done | 12/12 |
| Dispatch summary | `manifests/stage19c_job_status.summary.json` |

---

## 4. Full composition（selected cells）

Primary：3-fold mean DrugMacro AUC。完整表：`reports/round19c_full_composition_ranking.csv`。

### 4.1 Per-omics averages（across 7 cells）

| Omics | mean DrugMacro AUC |
|-------|--------------------|
| O0 | 0.5994 |
| O1 | 0.6040 |
| **O2** | **0.6192** |
| O3 | 0.6184 |
| O4 | 0.6057 |

### 4.2 Top compositions

| Rank | Cell | Omics | mean AUC |
|------|------|-------|----------|
| 1 | D0×P2 | O3 | **0.6275** |
| 2 | D3×P2 | O3 | 0.6256 |
| 3 | D0×P0 | O2 | 0.6248 |
| 4 | D2×P0 | O3 | 0.6245 |
| 5 | D0×P2 | O2 | 0.6241 |

每個 cell 的最佳 omics：

| Role | Cell | best omics |
|------|------|------------|
| R0 | D0×P0 | O2 |
| R1 | D0×P1 | O3 |
| R2 | D0×P2 | O3 |
| R3 | D2×P0 | O3 |
| R4 | D2×P1 | O2 |
| R5 | D3×P2 | O3 |
| R6 | D4×P1 | O2 |

---

## 5. Omics effects（paired folds）

`reports/round19c_omics_effects.csv`

| Comparison | mean Δ AUC | positive folds |
|------------|------------|----------------|
| **O2 − O0**（context effect） | **+0.0198** | **21/21** |
| O3 − O1（context added to summary） | +0.0144 | 17/21 |
| O2 − O4（target-informed vs source-only） | +0.0135 | 17/21 |
| O4 − O0（source-only effect） | +0.0063 | 14/21 |
| O1 − O0（summary effect） | +0.0045 | 17/21 |
| **O3 − O2**（summary added to context） | **−0.0009** | 10/21 |

**解讀：**

1. Context 相對 Z-only 是 **強且一致** 的增益（全 folds 為正）。  
2. Summary 在已有 context 後 **平均中性／略負** → 預設優先 **O2**，除非單 cell 符合 O3 門檻。  
3. Source-only O4 優於 O0，但明顯低於 O2（representation comparison，非純 ablation）。

### O3 vs O2 預設規則（≥+0.003 且 ≥2/3 folds）

| Cell | mean(O3−O2) | pos folds | 建議 |
|------|-------------|-----------|------|
| D0×P0 | −0.0079 | 1/3 | **O2** |
| D0×P1 | +0.0016 | 2/3 | **O2** |
| D0×P2 | +0.0034 | 2/3 | O3（剛過門檻） |
| D2×P0 | +0.0106 | 2/3 | O3 |
| D2×P1 | −0.0058 | 1/3 | **O2** |
| D3×P2 | +0.0047 | 2/3 | O3 |
| D4×P1 | −0.0125 | 0/3 | **O2** |

19B formal winner **D0×P2×O3** 仍保留為歷史最佳紀錄；方法敘述上多數 cell 可優先 O2。

---

## 6. Context faithfulness（shuffle）

`reports/round19c_context_shuffle_control.csv`

| Cell | Omics | mean(true − shuffled) | pos folds |
|------|-------|----------------------|-----------|
| D0×P0 | O2 | **+0.0343** | 3/3 |
| D0×P0 | O3 | **+0.0220** | 3/3 |
| D0×P2 | O2 | **+0.0178** | 2/3 |
| D0×P2 | O3 | **+0.0185** | 3/3 |

Overall：mean Δ = **+0.0231**，**11/12** folds 為正。

**判定：** Atom（P2）與 pooled（P0）皆明顯下降 →  
context16 含有 **sample-specific（ModelID）訊號**，不是單純增加維度／參數容量。  
對應手冊情境 **A（強證據）**。

---

## 7. O4 source-only retention

保留條件：距同 cell 最佳 O2/O3 ≤ 0.010，或至少不低於 O1。

| Cell | O4 | best O2/O3 | gap | within 0.010 | ≥ O1 |
|------|-----|------------|-----|--------------|------|
| D0×P0 | 0.5880 | 0.6248 | 0.0369 | no | no |
| D0×P1 | 0.5936 | 0.6104 | 0.0168 | no | no |
| D0×P2 | 0.6155 | 0.6275 | 0.0120 | no | yes |
| D2×P0 | 0.5980 | 0.6245 | 0.0265 | no | no |
| D2×P1 | 0.6129 | 0.6216 | 0.0087 | **yes** | yes |
| D3×P2 | 0.6188 | 0.6256 | 0.0068 | **yes** | yes |
| D4×P1 | 0.6130 | 0.6204 | 0.0073 | **yes** | yes |

**建議保留進 19D／shift validation 的 O4 候選：**  
至少 **D3×P2×O4**（最佳 source-only，gap 最小）與 **D2×P1×O4 / D4×P1×O4**。  
D0×P2×O4 雖 gap≈0.012，但優於 O1，可作 atom 線的 source-only 對照。

---

## 8. Suggested 19D candidate proposal（非正式 lock）

| ID | Role | Suggested composition | 理由 |
|----|------|----------------------|------|
| F0 | historical baseline | D0×P0×O1 | Round 18／19 legacy |
| F1 | best atom | D0×P2×O3（或 O2） | 19B/19C 冠軍；faithfulness 成立 |
| F2 | best pooled | D0×P0×O2 | pooled 線最強 O2 |
| F3 | source-only | D3×P2×O4 | 最佳 O4、gap≤0.010 |
| F4 | MACCS | D4×P1×O2 | R6；資源／非 graph 對照 |
| F5 | nonbaseline graph | D2×P0×O3 或 D3×P2×O3 | 相對 D0 有競爭力 |

去重後預期 **4–6** candidates。正式 seeds：`52,62,72`；5-fold；model seed 101。  
**不得因「較新」強制淘汰 D0。**

---

## 9. Artefacts

```text
reports/round19_stage19c_candidate_lock.json
reports/round19c_full_composition_ranking.csv
reports/round19c_omics_effects.csv
reports/round19c_context_redundancy.csv
reports/round19c_source_only_control.csv
reports/round19c_context_shuffle_control.csv
reports/round19c_role_candidate_summary.csv
reports/round19c_analysis_summary.json
manifests/stage19c_manifest.csv
manifests/stage19c_job_status.summary.json
stage19c/{job_id}/
metadata/round19_stage19b_baseline.json
```

程式：`tools/round19_stage19c_selector.py` · `tools/round19_context_controls.py` ·  
`tools/run_round19_stage19c_omics_interaction.sh` · `tools/round19_telegram_notify.py`

---

## 10. Go / No-Go

| Item | Verdict |
|------|---------|
| 54/54 complete | **GO** |
| O0–O4 每 selected cell × 3 folds | **GO** |
| Context shuffle faithfulness | **GO（強）** |
| O4 至少保留 1 候選 | **GO** |
| Formal `round19_locked_selection.json` | **NO-GO（需人工審查 19D list）** |
| 19D repeated 5CV | **READY to start after lock review** |

---

## 11. Related docs

- [`docs/round19_stage19c_ide_manual.md`](round19_stage19c_ide_manual.md)  
- [`docs/round19_stage19b_report.md`](round19_stage19b_report.md)  
- [`docs/round19_factorial_ide_manual.md`](round19_factorial_ide_manual.md)
