# Round 15 Repro + exp_008 Route Rescue Manual

Round 15 定位：**Round 13 best 可重現性** + **exp_008 route 強制 downstream** + **compact proto-response**（`none` / `own_plus_summary`），不做 importance-aware weighting。

## 快速執行

```bash
# 測試
pytest tests/test_round15_*.py -q

# Smoke
python tools/round15_config_builder.py \
  --settings config/round15_repro_rescue_settings.json \
  --outdir result/optimization_runs/round15_repro_rescue_smoke \
  --force

# 正式
FINETUNE_PARALLEL=12 PRETRAIN_PARALLEL=12 bash tools/run_round15_repro_rescue_pipeline.sh
```

## 分支

| 分支 | 目的 |
|------|------|
| 15A | exp_008 × 5 seeds × (none, own_plus_summary) |
| 15B | 6-model forced pool（含 Round 14 14B exp_008 candidates） |
| 15C | λ ∈ {0, 3e-6, 1e-5, 3e-5} × schedules 60→120 / 90→150 × 3 seeds |
| 15D | vs Round 13/14 best 最終比較 |

## 成功標準

- exp_008 `own_plus_summary` mean Avg TCGA ≥ **0.6000**
- `own_plus_summary` > z-only
- 可重現或接近 Round 13 **0.6112**

## 輸出

- `result/optimization_runs/round15_repro_rescue/manifests/`
- `result/optimization_runs/round15_repro_rescue/final_report/round15_final_report.md`

詳見 IDE 手冊 Round 15 章節。
