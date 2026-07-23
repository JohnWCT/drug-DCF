# Round 24 Stage 24A — Protocol Alignment Report

## Cohort

| Target | headline pairs | raw | eligible | dropped | miss_latent | miss_smiles |
|--------|---------------:|----:|---------:|--------:|------------:|------------:|
| gdsc_intersect13 | 906 | 906 | 886 | 20 | 20 | 0 |
| tcga_only3 | 129 | 129 | 129 | 0 | 0 | 0 |
| dapl | 178 | 178 | 177 | 1 | 1 | 0 |
| aacdr_gdsc_intersect | 425 | 425 | 417 | 8 | 8 | 0 |
| aacdr_tcga_only | 97 | 97 | 97 | 0 | 0 | 0 |

## gdsc_intersect13 906 → 886

Drop reason is exclusively `miss_latent` (patients absent from `tcga_latent_proto.pkl`).
Round 24 formal gate uses the **eligible cohort** (same as Round 18 Stage 18E).
Headline threshold table remains the product gate; paired comparisons must use eligible rows.

## Baseline vs gate (Round18 pooled_mlp × own_plus_summary, 5-fold mean)

| Target | fold-mean AUROC | gate | Δ | pass | eligible n | headline n |
|--------|----------------:|-----:|--:|:----|----------:|-----------:|
| gdsc_intersect13 | 0.5298 | 0.5184 | +0.0114 | True | 886 | 906 |
| tcga_only3 | 0.5437 | 0.5586 | -0.0149 | False | 129 | 129 |
| dapl | 0.5084 | 0.5356 | -0.0272 | False | 177 | 178 |
| aacdr_gdsc_intersect | 0.5285 | 0.5582 | -0.0297 | False | 417 | 425 |
| aacdr_tcga_only | 0.4861 | 0.4394 | +0.0467 | True | 97 | 97 |

## Metric notes

- Hard gate metric: 5-fold mean DrugMacro AUROC (support 10/2/2).
- Ensemble AUROC is supporting only.
- `Average_TCGA_AUC_proxy` = unfiltered per-drug AUC mean (historical style).
