# Round 19E IDE操作手冊

## Drug、Scaffold與Cancer-type Domain-shift Validation

**前置狀態：** Round 19D ALL_DONE（90/90）  
**19D commit：** `f45c342`  
**執行環境：** Docker container `DAPL`（`/workspace/DAPL`）  
**本階段性質：** Locked secondary validation（**禁止**依 19E 重搜架構／超參）  
**Formal selection：** NO-GO until 19E complete  

---

## 1. 設計思路（摘要）

19D 證明 F1≈F2≈F4 ≫ F0/F3（ModelID CV）。19E 只回答：

1. unseen drug / scaffold / cancer-type 是否仍成立  
2. O2 vs O3 在 shift 下誰較穩  
3. source-only O4 是否更適合 biological domain transfer  
4. pooled F3 是否反而更穩（因此 **必須保留 E3**）

工作量：E0–E5（含 MACCS E5）× 3 shifts × 5 folds ≈ **90 jobs**（cancer-type 若 QC 降為 3-fold 另記於 lock）。

---

## 2. 候選（自 19D lock 複製，不可改定義）

| ID | Source | Composition | Role |
|----|--------|-------------|------|
| E0 | F0 | D0×P0×O1 | historical |
| E1 | F1 | D0×P2×O2 | primary O2 atom |
| E2 | F2 | D0×P2×O3 | O3 control |
| E3 | F3 | D0×P0×O2 | pooled comparator |
| E4 | F4 | D3×P2×O4 | source-only |
| E5 | F5 | D4×P1×O2 | MACCS（gap≤0.015 + efficiency） |

---

## 3. Split seeds（預註冊）

| Strategy | seed | group |
|----------|------|-------|
| drug_heldout | 19051 | `normalized_drug_id` |
| scaffold_heldout | 19061 | `MURCKO:` / `ACYCLIC:sha256` |
| cancer_type_heldout | 19071 | `cancer_type`（GDSC code；ACH-000708→COREAD 由 CCLE 補） |

Internal test 永不進入 assignments。

---

## 4. Docker 指令

```bash
# setup smoke（lock + splits + manifests + tests + data smoke + Telegram）
docker exec -w /workspace/DAPL -e PYTHONPATH=/workspace/DAPL DAPL \
  bash tools/run_round19_stage19e_setup_smoke.sh

# 正式：先 cancer-type（~90% GPU pack + Telegram）
docker exec -w /workspace/DAPL -e PYTHONPATH=/workspace/DAPL \
  -e ROUND19_JOBS_PER_GPU=12 \
  DAPL bash tools/run_round19_stage19e_shift_validation.sh \
    --strategy cancer_type_heldout

# 再 drug → scaffold
docker exec -w /workspace/DAPL -e PYTHONPATH=/workspace/DAPL \
  -e ROUND19_JOBS_PER_GPU=12 \
  DAPL bash tools/run_round19_stage19e_shift_validation.sh --strategy drug_heldout

docker exec -w /workspace/DAPL -e PYTHONPATH=/workspace/DAPL \
  -e ROUND19_JOBS_PER_GPU=12 \
  DAPL bash tools/run_round19_stage19e_shift_validation.sh --strategy scaffold_heldout

# 分析（三份 manifest 皆完成後）
docker exec -w /workspace/DAPL -e PYTHONPATH=/workspace/DAPL DAPL \
  python3 tools/analyze_round19.py --stage 19e --require-complete \
  --outdir result/optimization_runs/round19_factorial
```

---

## 5. 關鍵產物

```text
metadata/round19_stage19d_baseline.json
reports/round19_stage19e_candidate_lock.json
reports/round19_stage19e_experiment_lock.json
splits/round19e_*_heldout_5cv.csv
manifests/stage19e_*_manifest.csv
reports/round19e_*.csv
docs/round19_stage19e_report.md   # 全完成後撰寫
```

---

## 6. Guardrails（每 shift 分開，不混 15 folds）

相對 E0 / E3：Δ≥+0.003 PASS；|Δ|<0.003 NON_WORSE；Δ≤−0.003 FAIL；相對 E0 下降 >0.015 = MAJOR_FAIL。

Recommended general：至少兩種 shift 對 E0 與 E3 皆 PASS/NON_WORSE 且無 MAJOR_FAIL；E1/E2 皆過則優先 E1。
