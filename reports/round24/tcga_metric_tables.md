# Round 24 — TCGA DrugMacro 彙整表

**契約：** 以下五組 TCGA 必須同時報告 **DrugMacro AUROC** 與 **DrugMacro AUPRC**（5-fold mean）。  
**排名：** 以這五組為準；GDSC 訓練／內部 CV **不作**選模主軸。

| Key | Path |
|-----|------|
| `gdsc_intersect13` | `data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv` |
| `tcga_only3` | `data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv` |
| `dapl` | `data/TCGA/TCGA_drug_response_from_DAPL.csv` |
| `aacdr_gdsc_intersect` | `data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv` |
| `aacdr_tcga_only` | `data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv` |

**Gate（AUROC；硬閘僅 AACDR 兩組；標準 = stest0 / 無 10% testset）：**  
**必過：** aacdr_gdsc_intersect >**0.5279** · aacdr_tcga_only >**0.4804**  
**必報不擋 lock：** dapl 0.5304 · gdsc_intersect13 0.5197 · tcga_only3 0.5536

**PASS 後排序權重：** 5 / 4 / 3 / 2 / 1（`aacdr_gdsc` → `tcga_only3`）。

**超越標準：** [`docs/AACDR_drug_macro_auroc_auprc.md`](../docs/AACDR_drug_macro_auroc_auprc.md)（現行 = **stest0**）  
完整 Y/N + Δ：[`vs_aacdr_standard.md`](vs_aacdr_standard.md)  
**24E 計畫：** [`docs/round24_solution_plan.md`](../docs/round24_solution_plan.md) §7

---

## Stage 24A / Ctrl baseline（pooled_mlp × own_plus_summary）

| Metric | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only |
|--------|-----------------:|-----------:|-----:|---------------------:|---------------:|
| AUROC | 0.5298 | 0.5437 | 0.5084 | 0.5285 | 0.4861 |
| AUPRC | 0.5963 | 0.6962 | 0.5605 | 0.5765 | 0.6559 |

來源：`reports/round24/stage24a/baseline_summary.json`

---

## Stage 24B

### B1 — biocda_predictive_e3 × C32

| Metric | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only |
|--------|-----------------:|-----------:|-----:|---------------------:|---------------:|
| AUROC | 0.4999 | 0.4544 | 0.4661 | 0.5268 | 0.4730 |
| AUPRC | 0.5718 | 0.6377 | 0.5266 | 0.5890 | 0.6558 |

### B2 — biocda_xa_fresh × C32

| Metric | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only |
|--------|-----------------:|-----------:|-----:|---------------------:|---------------:|
| AUROC | 0.4840 | 0.4879 | 0.4945 | 0.5263 | 0.4858 |
| AUPRC | 0.5616 | 0.6530 | 0.5423 | 0.6028 | 0.6380 |

---

## Stage 24C（biocda_predictive_e3 × feature sweep）

來源：`reports/round24/stage24c/feature_attribution_summary.json`

### AUROC

| ID | Feature | n_pass | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only |
|----|---------|-------:|-----------------:|-----------:|-----:|---------------------:|---------------:|
| F2 | C16 | 2 | 0.5250 | 0.4534 | 0.4859 | 0.5427 | 0.5398 |
| F3 | C32 | 1 | 0.4999 | 0.4544 | 0.4661 | 0.5268 | 0.4730 |
| F0 | own_plus_summary | 1 | 0.5329 | 0.4357 | 0.4851 | 0.5299 | 0.4351 |
| F1 | z_only | 1 | 0.5046 | 0.4297 | 0.4739 | 0.5221 | 0.4537 |
| F4 | C64 | 0 | 0.5103 | 0.4264 | 0.4735 | 0.5080 | 0.4361 |

### AUPRC

| ID | Feature | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only |
|----|---------|-----------------:|-----------:|-----:|---------------------:|---------------:|
| F2 | C16 | 0.5773 | 0.6473 | 0.5466 | 0.5859 | 0.6824 |
| F3 | C32 | 0.5718 | 0.6377 | 0.5266 | 0.5890 | 0.6558 |
| F0 | own_plus_summary | 0.5826 | 0.6183 | 0.5447 | 0.5824 | 0.6359 |
| F1 | z_only | 0.5594 | 0.6112 | 0.5350 | 0.5784 | 0.6610 |
| F4 | C64 | 0.5800 | 0.6180 | 0.5285 | 0.5716 | 0.6431 |

---

## Train-source ablation（診斷）

### AUROC

| Arm | n_pass | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only |
|-----|-------:|-----------------:|-----------:|-----:|---------------------:|---------------:|
| Ctrl | 2 | 0.5298 | 0.5437 | 0.5084 | 0.5285 | 0.4861 |
| NoHoldout | 3 | 0.5697 | 0.4845 | 0.4820 | 0.5648 | 0.4971 |
| AACDR | 2 | 0.4735 | 0.4480 | 0.5371 | 0.5063 | 0.4939 |

### AUPRC

| Arm | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only |
|-----|-----------------:|-----------:|-----:|---------------------:|---------------:|
| Ctrl | 0.5963 | 0.6962 | 0.5605 | 0.5765 | 0.6559 |
| NoHoldout | 0.6121 | 0.6368 | 0.5416 | 0.6186 | 0.6532 |
| AACDR | 0.5487 | 0.6363 | 0.5648 | 0.5634 | 0.6667 |
