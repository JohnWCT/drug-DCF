# Round 15 Final Report (placeholder)

Pipeline 尚未執行。執行完成後由 `tools/analyze_round15_repro_rescue.py` 產生正式報告於：

`result/optimization_runs/round15_repro_rescue/final_report/round15_final_report.md`

## 參考基準

| 模型 | Avg TCGA |
|------|----------|
| Round 13 best r13_exp_008_own_plus_summary | 0.6112 |
| Round 14 best r14_exp_078_own_plus_summary | 0.5909 |
| Round 12 exp_037 | 0.5972 |

## Round 15 待回答

1. Round 13 best 是否 5-seed 可重現？
2. exp_008 route 是否穩定受益於 own_plus_summary？
3. Round 14 漏測 exp_008 是否影響結論？
4. ultra-low / late VICReg 是否有幫助？
5. 是否進 Round 16 importance-aware weighting？
