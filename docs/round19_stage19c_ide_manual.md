# Round 19C IDE 操作手冊

## Omics Composition Completion、Source-only Control 與 Context Faithfulness

**前置狀態：** Round 19B 117-job factorial screening 完成  
**19B commit：** `c3c41f9`  
**執行環境：** Docker container `DAPL`（`/workspace/DAPL`）  
**主要選擇指標：** 3-fold mean DrugMacro AUC  
**Tie-breaker：** DrugMacro AUPRC  
**禁止使用：** internal test、TCGA、Integrated5、18E external ranking  

---

## 1. 設計目標

19B 已證明 `context > summary`、`P2 > P1`。19C **不再搜尋架構**，只回答：

1. O2/O3 相對 Z-only（O0）的完整增益  
2. context 增益是否依賴正確 ModelID↔context 配對（shuffled 負對照）  
3. source-only O4 是否仍可保留進後續 domain-generalization  
4. O3 相對 O2 是否仍值得多 11 維 summary  

---

## 2. 角色式 candidate 選擇（非 Top-N）

| Role | 固定／規則 | 目的 |
|------|------------|------|
| R0 | D0×P0 | legacy MLP anchor |
| R1 | D0×P1 | pooled Transformer anchor |
| R2 | D0×P2 | atom cross-attn champion |
| R3 | best D1–D4×P0 by mean(O2,O3) | 非 baseline MLP drug |
| R4 | best D1–D4×P1 by mean(O2,O3) | 非 baseline pooled drug |
| R5 | best D2/D3×P2 by mean(O2,O3) | 非 baseline atom drug |
| R6 | best D4×P0/P1 | MACCS-only |

選擇分數：`(mean_auc_O2 + mean_auc_O3) / 2`，避免 max(O2,O3) 偶然偏差。  
重複 cell 只保留一次 → `N_selected ∈ [5,7]`。

---

## 3. Job 組成

```text
core   = N_selected × {O0,O4} × 3 folds
control= 2 cells × {O2_shuffled,O3_shuffled} × 3 folds = 12
total  = core + 12
```

本輪實測：`N_selected=7` → **54 jobs**。

Shuffled context：

- 以 **ModelID** 為單位 derangement  
- train / val **各自** within-partition shuffle  
- seed：`19031 + fold*100 + {1,2}`  
- 不預寫全域 shuffled feature 檔，Dataset 記憶體替換 context16  

---

## 4. 程式入口

| 元件 | 路徑 |
|------|------|
| Selector | `tools/round19_stage19c_selector.py` |
| Context controls | `tools/round19_context_controls.py` |
| Manifest | `tools/round19_config_builder.py --stage 19c` |
| Runner | `tools/run_round19_stage19c_omics_interaction.sh` |
| Telegram | `tools/round19_telegram_notify.py` |
| Analyzer | `tools/analyze_round19.py --stage 19c` |

Candidate lock：`reports/round19_stage19c_candidate_lock.json`（**非正式** selection lock）  
Formal lock：`round19_locked_selection.json` 在 19C 完成前 **NO-GO**。

---

## 5. 執行（Docker）

```bash
# tests
docker exec -w /workspace/DAPL -e PYTHONPATH=/workspace/DAPL DAPL \
  pytest tests/test_round19_context_shuffle.py \
         tests/test_round19_stage19c_*.py -q

# smoke
docker exec -w /workspace/DAPL -e PYTHONPATH=/workspace/DAPL DAPL \
  bash tools/run_round19_stage19c_omics_interaction.sh --smoke-only

# formal（~90% GPU packing）
docker exec -w /workspace/DAPL -e PYTHONPATH=/workspace/DAPL \
  -e ROUND19_JOBS_PER_GPU=12 \
  DAPL bash tools/run_round19_stage19c_omics_interaction.sh
```

---

## 6. Gate

- 19B = 117/117  
- 19C core = N×2×3 全部完成  
- context controls = 12/12  
- 每個 selected cell：O0–O4 皆有 3-fold metrics（O1–O3 來自 19B）  
- 無 TCGA / internal selection 欄位  

完成後才建立 19D 4–6 candidate proposal。
