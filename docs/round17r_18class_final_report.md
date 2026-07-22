# Round 17R — 18-class-clean Focused Rerun

**狀態：** ALL_DONE

見 [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md#round-17r--18-class-clean-重跑)。

## 基準

| 基準 | Avg TCGA |
|------|----------|
| R13 best | **0.6112** |
| Pre-18class 17C best | 0.5892 |

## 17R-D 10-seed 確認

| Rank | Model | Strategy | Avg TCGA |
|------|-------|----------|----------|
| 1 | r13_exp_008 | own_plus_summary | **0.5915 ± 0.036** |
| 2 | r13_exp_008_control | own_plus_summary | 0.5899 ± 0.031 |
| 5 | r13_exp_008 | own_proto_context_projected_16 | 0.5813 ± 0.019 |

vs R13：**−0.020**

## 各 dataset 冠軍（17R-D）

| Dataset | #1 | Strategy |
|---------|-----|----------|
| gdsc_intersect13 | r13_exp_008 | own_plus_summary |
| tcga_only3 | r15c_exp_024 | own_plus_summary |
| dapl | r13_exp_008 | context-16 |
| aacdr_tcga_only | r13_exp_008 | own_plus_summary |
| aacdr_gdsc_intersect | r13_exp_008 | own_plus_summary |

## 結論

18-class 修正後 primary strategy 仍為 **own_plus_summary**；10-seed 優於 pre-18class 17C 但未達 R13。單點峰值 17R-B 0.6074 接近 R13，確認後回落。
