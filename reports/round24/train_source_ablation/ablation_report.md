# Round24 train-source ablation (diagnostic)

Architecture: **pooled_mlp × own_plus_summary** (B0). Not a formal lock candidate.

## Fold-mean DrugMacro AUROC

| Arm | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only | n_pass |
|-----|------|------|------|------|------|-------|
| Ctrl | 0.5298 | 0.5437 | 0.5084 | 0.5285 | 0.4861 | 2/5 |
| NoHoldout | 0.5697 | 0.4845 | 0.4820 | 0.5648 | 0.4971 | 3/5 |
| AACDR | 0.4735 | 0.4480 | 0.5371 | 0.5063 | 0.4939 | 2/5 |

## Δ vs Ctrl

- **NoHoldout**: gdsc_intersect13 +0.0399, tcga_only3 -0.0592, dapl -0.0263, aacdr_gdsc_intersect +0.0364, aacdr_tcga_only +0.0110
- **AACDR**: gdsc_intersect13 -0.0563, tcga_only3 -0.0957, dapl +0.0288, aacdr_gdsc_intersect -0.0222, aacdr_tcga_only +0.0079

any_all_target_pass=False

