# Round 12 Final Report

**Run:** `result/optimization_runs/round12_proto_alignment`  
**Completed:** 2026-06-23  
**Pipeline status:** DONE (pretrain 66/66, finetune 120/120)

## References

| Benchmark | Average_TCGA_AUC_mean |
|-----------|------------------------|
| Round 11 best exp_035 | 0.5828 |
| Round 10 best exp_111 | 0.5749 |
| R7 original exp_048 | 0.5918 |
| **Round 12 best exp_037** | **0.5972** |

## Pipeline summary

| Stage | Result |
|-------|--------|
| Round 12A baseline QC | completed |
| Pretrain | 66/66 success |
| Selection | top-30 |
| Finetune | 120/120 success |

## Downstream

- **Best model:** `exp_037`
- **Average_TCGA_AUC_mean:** 0.5971789386885913
- **Global_TCGA_AUC_mean:** 0.605292624852765
- **vs Round 11 exp_035 (0.5828):** +0.0144
- **vs R7 exp_048 (0.5918):** +0.0054

## Prototype alignment

- exp_035 baseline prototype distance: 0.0604103207588195
- Active proto configs reduced target-to-source anchor distance: **True**

## Round 13 decision

**Recommendation:** `go_response_features`

Prototype gap improved and downstream exceeded Round 11 baseline and R7 reference.

## Artifacts

- Runtime final report: `result/optimization_runs/round12_proto_alignment/final_report/round12_final_report.md`
- Baseline gap diagnostics: `result/optimization_runs/round12_proto_alignment/round12a_baseline_qc/`
- Aggregate: `result/optimization_runs/round12_proto_alignment/aggregate/aggregate_scores.csv`
- Selection: `result/optimization_runs/round12_proto_alignment/selection/pretrain_top10.csv`
- Logs: `result/optimization_runs/round12_proto_alignment/logs/pipeline.log`
