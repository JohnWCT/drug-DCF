# Round 13 Final Report

**Run:** `result/optimization_runs/round13_proto_response`  
**Completed:** 2026-06-23  
**Pipeline status:** DONE (feature extraction 30/30, finetune **81/120** success)  
**Duration:** 09:58:13Z → 13:00:32Z (~3h)

## References

| Benchmark | Average_TCGA_AUC_mean |
|-----------|----------------------|
| R7 original exp_048 | 0.5918 |
| Round 11 best exp_035 | 0.5828 |
| Round 12 best exp_037 | 0.5972 |
| **Round 13 best r13_exp_035_own_plus_summary** | **0.6127** |

## Pipeline summary

| Stage | Planned | Completed | Notes |
|-------|---------|-----------|-------|
| Pretrain | 0 | — | Skipped; frozen Round 12 model pool |
| Prototype feature extraction | 30 | **30/30** | 6 models × 5 feature modes |
| Finetune | 120 | **81 success / 39 failed** | 6 models × 5 modes × 4 combos |
| Aggregate + analyze | — | Done | |

**Parallelism:** `FINETUNE_PARALLEL=26`, `batch_size=12288`, `mini_batch_size=3072`, `epochs=1000`

Failures concentrated on `exp_035` (12), `exp_051` (14), `exp_018` (8); `exp_037` / `exp_057` completed all jobs.

## Downstream

- **Best model:** `r13_exp_035_own_plus_summary`（source `exp_035`, feature mode `own_plus_summary`）
- **Average_TCGA_AUC_mean:** 0.6127405840100312
- **Global_TCGA_AUC_mean:** 0.6357418369874579
- **vs Round 12 exp_037 (0.5972):** +0.0156
- **vs Round 11 exp_035 (0.5828):** +0.0299
- **vs R7 exp_048 (0.5918):** +0.0209
- **Stretch goal 0.6000:** **met**

### Top-5 downstream

| Rank | Model_ID | source | feature_mode | Average_TCGA_AUC_mean | Global_TCGA_AUC_mean |
|------|----------|--------|--------------|----------------------|---------------------|
| 1 | r13_exp_035_own_plus_summary | exp_035 | own_plus_summary | **0.6127** | 0.6357 |
| 2 | r13_exp_008_own_plus_summary | exp_008 | own_plus_summary | 0.6112 | 0.6204 |
| 3 | r13_exp_035_none | exp_035 | none (z-only) | 0.6059 | 0.6166 |
| 4 | r13_exp_035_own_cancer | exp_035 | own_cancer | 0.6046 | 0.6048 |
| 5 | r13_exp_057_own_cancer | exp_057 | own_cancer | 0.5900 | 0.6323 |

## Feature mode ablation

| Mode | Best model | Best Avg TCGA | Mean Avg TCGA (n models) |
|------|------------|---------------|--------------------------|
| **own_plus_summary** | r13_exp_035_own_plus_summary | **0.6127** | 0.5763 (5) |
| none (z-only) | r13_exp_035_none | 0.6059 | 0.5628 (5) |
| own_cancer | r13_exp_035_own_cancer | 0.6046 | 0.5651 (5) |
| all_source_and_target | r13_exp_037_all_source_and_target | 0.5830 | 0.5537 (4) |
| all_source_anchors | r13_exp_037_all_source_anchors | 0.5777 | 0.5460 (4) |

**Conclusion:** Compact prototype-distance features (`own_cancer`, `own_plus_summary`) outperform full anchor vectors. `own_plus_summary` is the winning branch.

## z-only vs prototype features (per source model)

| source | z-only | best proto | delta |
|--------|--------|------------|-------|
| exp_035 | 0.6059 | 0.6127 (`own_plus_summary`) | **+0.0068** |
| exp_008 | — | 0.6112 (`own_plus_summary`) | — |
| exp_057 | 0.5816 | 0.5900 (`own_cancer`) | **+0.0084** |
| exp_018 | 0.5374 | 0.5510 (`all_source_and_target`) | **+0.0136** |
| exp_037 | 0.5864 | 0.5834 (`own_plus_summary`) | −0.0030 |
| exp_051 | 0.5024 | 0.4947 (`own_cancer`) | −0.0077 |

Models with proto > z-only: **3/5** (exp_035, exp_057, exp_018). Round 12 champion `exp_037` did not benefit from proto features in this ablation.

## Round 14 decision

**Recommendation:** `go_vicreg_stabilizer`

Rationale:
1. Prototype-distance response features raised best downstream to **0.6127**, exceeding Round 12 (**0.5972**) and stretch goal **0.6000**.
2. `own_plus_summary` / `own_cancer` modes are sufficient; high-dimensional anchor vectors add little or hurt.
3. Next step: low-weight VICReg / latent stabilizer re-integration on top of the current best stack (Round 11–13 lineage via `exp_035`).

## Artifacts

- Runtime final report: `result/optimization_runs/round13_proto_response/final_report/round13_final_report.md`
- Feature mode summary: `final_report/round13_feature_mode_summary.csv`
- z vs proto delta: `final_report/round13_z_vs_proto_delta.csv`
- Aggregate: `aggregate/aggregate_scores.csv`
- Manifests: `manifests/finetune_dispatch_manifest.csv`, `manifests/proto_feature_manifest.csv`
- Logs: `logs/pipeline.log`
