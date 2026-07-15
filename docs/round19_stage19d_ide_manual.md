# Round 19D IDE 操作手冊

## Repeated 5CV Confirmation（非架構搜尋）

**前置狀態：** Round 19C 54-job omics／faithfulness 完成  
**19C commit：** `70cc0b4`  
**19D 狀態：** **ALL_DONE（90/90）** — 見 [`docs/round19_stage19d_report.md`](round19_stage19d_report.md)  
**執行環境：** Docker container `DAPL`（`/workspace/DAPL`）  
**主要選擇指標：** mean-of-means DrugMacro AUC（3 seeds × 5 folds）  
**禁止使用：** internal test、TCGA、Integrated5 作選擇  

---

## 1. 設計目標

在 **新 ModelID splits**（seeds 52/62/72）上確認：

1. Primary atom+O2（F1）是否穩定優於歷史 F0  
2. Atom（P2）是否穩定優於 pooled（P0）  
3. O2 vs O3 是否仍接近中性  
4. Source-only F4 是否仍值得進 19E shift  

---

## 2. Candidates

| ID | Composition | Role |
|----|-------------|------|
| F0 | D0×P0×O1 | historical |
| F1 | D0×P2×O2 | primary |
| F2 | D0×P2×O3 | O3 control |
| F3 | D0×P0×O2 | best pooled O2 |
| F4 | D3×P2×O4 | source-only |
| F5 | D4×P1×O2 | MACCS（optional） |

Jobs：`6 × 3 × 5 = 90`。

---

## 3. 程式入口

| 元件 | 路徑 |
|------|------|
| Selector | `tools/round19_stage19d_selector.py` |
| Splits | `tools/round19_cv_splits.py`（`build_round19d_splits`） |
| Manifest / lock | `tools/round19_config_builder.py --stage 19d` |
| Runner | `tools/run_round19_stage19d_repeated_5cv.sh` |
| Telegram | `tools/round19_telegram_notify.py` |
| Analyzer | `tools/analyze_round19.py --stage 19d` |
| 19C baseline | `tools/write_round19_stage19c_baseline.py` |

---

## 4. 執行（Docker）

```bash
# tests
docker exec -w /workspace/DAPL -e PYTHONPATH=/workspace/DAPL DAPL \
  pytest tests/test_round19_stage19d_*.py -q

# smoke
docker exec -w /workspace/DAPL -e PYTHONPATH=/workspace/DAPL DAPL \
  bash tools/run_round19_stage19d_repeated_5cv.sh --smoke-only

# formal（~90% GPU packing + Telegram）
docker exec -w /workspace/DAPL -e PYTHONPATH=/workspace/DAPL \
  -e ROUND19_JOBS_PER_GPU=12 \
  DAPL bash tools/run_round19_stage19d_repeated_5cv.sh

# analyze
docker exec -w /workspace/DAPL -e PYTHONPATH=/workspace/DAPL DAPL \
  python3 tools/analyze_round19.py --stage 19d --require-complete \
  --outdir result/optimization_runs/round19_factorial
```

---

## 5. Gate

- 19C = 54/54  
- 19D = 90/90、0 failed  
- 每候選 × 3 seeds × 5 folds 皆有 DrugMacro metrics  
- 無 TCGA / internal selection 欄位  
- Formal lock 仍 **NO-GO**，下一步 **19E**
