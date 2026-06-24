# Round 13 Final Report

**Run:** `result/optimization_runs/round13_proto_response`  
**Completed:** 2026-06-24（finetune retry 補跑完成）  
**Pipeline status:** ALL_DONE (feature extraction 30/30, finetune **120/120** success)

## Timeline

| Phase | When | Notes |
|-------|------|-------|
| Initial pipeline | 2026-06-23 09:58–13:00Z | 81/120 finetune success |
| Finetune retry | 2026-06-24 04:18–05:45Z | 39 OOM failures → `parallel=12` retry → **120/120** |

Initial failures were **CUDA OOM** from `FINETUNE_PARALLEL=26` GPU contention (shared node). Retry script: `tools/run_round13_finetune_retry.sh`.

## References

| Benchmark | Average_TCGA_AUC_mean |
|-----------|----------------------|
| R7 original exp_048 | 0.5918 |
| Round 11 best exp_035 | 0.5828 |
| Round 12 best exp_037 | 0.5972 |
| **Round 13 best r13_exp_008_own_plus_summary** | **0.6112** |

## Pipeline summary

| Stage | Planned | Completed | Notes |
|-------|---------|-----------|-------|
| Pretrain | 0 | — | Skipped; frozen Round 12 model pool |
| Prototype feature extraction | 30 | **30/30** | 6 models × 5 feature modes |
| Finetune | 120 | **120/120** | 6 models × 5 modes × 4 combos |
| Aggregate + analyze | — | Done | |

**Parallelism:** initial `FINETUNE_PARALLEL=26` → retry `FINETUNE_RETRY_PARALLEL=12`; `batch_size=12288`, `mini_batch_size=3072`, `epochs=1000`

## Downstream

- **Best model:** `r13_exp_008_own_plus_summary`（source `exp_008`, feature mode `own_plus_summary`）
- **Average_TCGA_AUC_mean:** 0.6112395039184843
- **Global_TCGA_AUC_mean:** 0.6204003275821104
- **vs Round 12 exp_037 (0.5972):** +0.0141
- **vs Round 11 exp_035 (0.5828):** +0.0284
- **vs R7 exp_048 (0.5918):** +0.0194
- **Stretch goal 0.6000:** **met**

### Top-5 downstream

| Rank | Model_ID | source | feature_mode | Average_TCGA_AUC_mean | Global_TCGA_AUC_mean |
|------|----------|--------|--------------|----------------------|---------------------|
| 1 | r13_exp_008_own_plus_summary | exp_008 | own_plus_summary | **0.6112** | 0.6204 |
| 2 | r13_exp_035_none | exp_035 | none (z-only) | 0.6059 | 0.6166 |
| 3 | r13_exp_035_own_cancer | exp_035 | own_cancer | 0.5954 | 0.6101 |
| 4 | r13_exp_057_own_cancer | exp_057 | own_cancer | 0.5900 | 0.6323 |
| 5 | r13_exp_037_none | exp_037 | none (z-only) | 0.5864 | 0.6084 |

Note: partial-run snapshot had `r13_exp_035_own_plus_summary` at 0.6127 (1 combo only); full 4-combo aggregate is **0.5845** — use 120/120 numbers below.

## Feature mode ablation

| Mode | Best model | Best Avg TCGA | Mean Avg TCGA (n models) |
|------|------------|---------------|--------------------------|
| **own_plus_summary** | r13_exp_008_own_plus_summary | **0.6112** | 0.5619 (6) |
| none (z-only) | r13_exp_035_none | 0.6059 | 0.5648 (6) |
| own_cancer | r13_exp_035_own_cancer | 0.5954 | 0.5631 (6) |
| all_source_and_target | r13_exp_035_all_source_and_target | 0.5857 | 0.5491 (6) |
| all_source_anchors | r13_exp_035_all_source_anchors | 0.5783 | 0.5463 (6) |

**Conclusion:** `own_plus_summary` wins at peak; z-only baseline remains strong for `exp_035`. Full anchor vectors still underperform compact features.

## z-only vs prototype features (per source model)

| source | z-only | best proto | delta |
|--------|--------|------------|-------|
| exp_008 | 0.5664 | 0.6112 (`own_plus_summary`) | **+0.0449** |
| exp_051 | 0.5024 | 0.5202 (`all_source_anchors`) | **+0.0178** |
| exp_057 | 0.5816 | 0.5900 (`own_cancer`) | **+0.0084** |
| exp_018 | 0.5459 | 0.5510 (`all_source_and_target`) | **+0.0051** |
| exp_037 | 0.5864 | 0.5834 (`own_plus_summary`) | −0.0030 |
| exp_035 | 0.6059 | 0.5954 (`own_cancer`) | −0.0105 |

Models with proto > z-only: **4/6** (exp_008, exp_051, exp_057, exp_018). Round 11/12 lineage `exp_035` and Round 12 champion `exp_037` did not benefit from proto features in this ablation.

## Round 14 decision

**Recommendation:** `go_vicreg_stabilizer`

Rationale:
1. Full 120-job aggregate best **0.6112** exceeds Round 12 (**0.5972**) and stretch **0.6000**.
2. Largest proto gain on `exp_008` (`own_plus_summary`); `exp_035` z-only still competitive — Round 14 should build on validated stacks, not partial-run peaks.
3. Next step: low-weight VICReg / latent stabilizer on top of Round 11–13 best candidates (`exp_008` proto-response, `exp_035` z-only / 10C stack).

## Artifacts

- Runtime final report: `result/optimization_runs/round13_proto_response/final_report/round13_final_report.md`
- Feature mode summary: `final_report/round13_feature_mode_summary.csv`
- z vs proto delta: `final_report/round13_z_vs_proto_delta.csv`
- Aggregate: `aggregate/aggregate_scores.csv`
- Manifests: `manifests/finetune_dispatch_manifest.csv`
- Logs: `logs/pipeline.log`, `logs/round13_finetune_retry.log`
