# Round 19F IDE 操作手冊

## Final Role Proposal 與 Scenario-aware Policy

**前置狀態：** Round 19E ALL_DONE（90/90，0 failed）  
**19E commit：** `7c7fb93`  
**執行環境：** Docker container `DAPL`（`/workspace/DAPL`）  
**目前 Gate：** `ALL_DONE`（540/540 post-hoc jobs；0 failed）  
**Formal role selection：** `LOCKED`（immutable；90 checkpoint hashes pinned）  
**Single champion：** NONE  
**TCGA / internal：** 不得參與 role proposal

正式結果見 [`docs/round19_stage19f_report.md`](round19_stage19f_report.md)。

---

## 1. 本階段範圍

第一個 19F gate 只允許：

1. 由 19D repeated source CV 與 19E 三種 shift 結果產生 role proposal。
2. 建立 deterministic scenario-aware routing policy。
3. 驗證每個候選的 15-member checkpoint inventory。
4. 執行 selector、guardrail、routing、ensemble completeness smoke tests。

此 gate **不建立 final role lock**，也不執行 internal／TCGA post-hoc inference；須先人工審查 proposal。

---

## 2. 預註冊角色規則

| Role | 規則 |
|------|------|
| historical_anchor | 固定 E0/F0 |
| source_performance_champion | 19D mean-of-three-5CV-means 最高 |
| parsimonious_context_model | F1 與 F2 差距 ≤0.003，且 cancer shift 不劣於 E2 超過 0.003 |
| cancer_shift_specialist | E1/E2；AUC 差距 ≤0.003 時優先 E1/O2 |
| chemical_shift_specialist | drug/scaffold 對 E0 無 major fail；最大化 worst-shift delta |
| source_only_domain_candidate | E4 通過 cancer、chemical 與 collapse guards |
| efficient_model | E5 至少兩種 shift 不劣於 E0，且效率門檻成立 |
| general_recommended_model | 三種 shift 對 E0 均 PASS/NON_WORSE，且無 MAJOR_FAIL；可為 null |

禁止建立任意跨 shift 加權總分。

---

## 3. Scenario-aware routing

```text
unseen drug       -> chemical_shift_specialist
unseen scaffold   -> chemical_shift_specialist
unseen cancer     -> cancer_shift_specialist
source-like       -> source_performance_champion
metadata unknown  -> chemical_shift_specialist（low confidence）
```

這是 metadata-based deterministic policy，不是 learned mixture-of-experts。

---

## 4. Docker smoke / proposal

```bash
docker exec \
  -w /workspace/DAPL \
  -e PYTHONPATH=/workspace/DAPL \
  DAPL \
  bash tools/run_round19_stage19f_role_lock.sh --smoke-only

docker exec \
  -w /workspace/DAPL \
  -e PYTHONPATH=/workspace/DAPL \
  DAPL \
  python3 tools/round19_stage19f_role_selector.py \
    --root result/optimization_runs/round19_factorial \
    --policy config/round19_stage19f_role_policy.json \
    --output result/optimization_runs/round19_factorial/reports/round19_final_role_proposal.json \
    --require-complete
```

Smoke 通過後產生：

```text
reports/round19_final_role_proposal.json
reports/round19_stage19f_checkpoint_inventory.csv
reports/round19_stage19f_checkpoint_inventory_summary.json
```

---

## 5. 人工審查 Gate

必須確認：

- E1 不因 cancer 表現好而成為 universal model。
- E3 由 maximin 規則選為 chemical specialist。
- E1/E2 cancer 近似時依規則優先 E1/O2。
- F2 保留 source champion。
- General model 由三 shift full-precision guardrail 決定，可為 null。
- Proposal 的輸入 hash、candidate definitions 與 15 checkpoints 完整。
- `selection_used_internal=false`、`selection_used_tcga=false`。

審查前不得建立 `round19_final_role_lock.json`，也不得開始 post-hoc inference。
