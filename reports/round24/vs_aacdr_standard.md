# Round 24 vs AACDR DrugMacro 標準（stest0 / 無 10% testset）

**標準來源：** [`docs/AACDR_drug_macro_auroc_auprc.md`](../docs/AACDR_drug_macro_auroc_auprc.md)（`eval3_stest0` / `target_infer_stest0`）

**硬閘 PASS：** `aacdr_gdsc_intersect` > 0.5279 ∧ `aacdr_tcga_only` > 0.4804。
其餘三組必報、不擋 lock。`Y` = 嚴格超越；括號 = Δ。

## 標準（stest0）

| 評估集 | DrugMacro AUROC | DrugMacro AUPRC |
|--------|----------------:|----------------:|
| `gdsc_intersect13` | 0.5197 | 0.5981 |
| `tcga_only3` | 0.5536 | 0.6960 |
| `dapl` | 0.5304 | 0.5570 |
| `aacdr_gdsc_intersect` | 0.5279 | 0.5710 |
| `aacdr_tcga_only` | 0.4804 | 0.6300 |

## 一眼看：哪些指標「曾被任一候選超越」

| 評估集 | AUROC 標準 | 曾超越？ | 最佳 AUROC（候選） | AUPRC 標準 | 曾超越？ | 最佳 AUPRC（候選） |
|--------|-----------:|:--------:|--------------------|-----------:|:--------:|--------------------|
| `gdsc_intersect13` | 0.5197 | **是** | 0.5697（Ablation NoHoldout） | 0.5981 | **是** | 0.6121（Ablation NoHoldout） |
| `tcga_only3` | 0.5536 | 否 | 0.5437（B0/Ctrl pooled_mlp x own_plus_summary） | 0.6960 | **是** | 0.6962（B0/Ctrl pooled_mlp x own_plus_summary） |
| `dapl` | 0.5304 | **是** | 0.5371（Ablation AACDR） | 0.5570 | **是** | 0.5648（Ablation AACDR） |
| `aacdr_gdsc_intersect` | 0.5279 | **是** | 0.5648（Ablation NoHoldout） | 0.5710 | **是** | 0.6186（Ablation NoHoldout） |
| `aacdr_tcga_only` | 0.4804 | **是** | 0.5398（F2 pred x C16） | 0.6300 | **是** | 0.6824（F2 pred x C16） |

### 結論摘要

- **AUROC 已有候選可超越：** `gdsc_intersect13`, `dapl`, `aacdr_gdsc_intersect`, `aacdr_tcga_only`
- **AUROC 目前無人超越：** `tcga_only3`
- **AUPRC 已有候選可超越：** `gdsc_intersect13`, `tcga_only3`, `dapl`, `aacdr_gdsc_intersect`, `aacdr_tcga_only`
- **AUPRC 目前無人超越：** （無）
- **硬閘 PASS 候選：** `B0/Ctrl pooled_mlp x own_plus_summary`, `F2 pred x C16`, `Ablation NoHoldout`, `E-NH0 pooled_mlp__own_plus_summary x own_plus_summary (NoHoldout)`, `E-NH1 biocda_predictive_e3 x z_plus_context16 (NoHoldout)`

## 各候選 × AUROC（vs stest0）

| Candidate | 硬閘 | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only | AUROC 超越數 |
|-----------|:----:|-----------------:|-----------:|-----:|---------------------:|---------------:|-------------:|
| B0/Ctrl pooled_mlp x own_plus_summary | PASS | Y 0.5298 (+0.0101) | N 0.5437 (-0.0099) | N 0.5084 (-0.0220) | Y 0.5285 (+0.0006) | Y 0.4861 (+0.0057) | 3/5 |
| B1 predictive x C32 | NO_LOCK | N 0.4999 (-0.0198) | N 0.4544 (-0.0992) | N 0.4661 (-0.0643) | N 0.5268 (-0.0011) | N 0.4730 (-0.0074) | 0/5 |
| B2 XA x C32 | NO_LOCK | N 0.4840 (-0.0357) | N 0.4879 (-0.0657) | N 0.4945 (-0.0359) | N 0.5263 (-0.0016) | Y 0.4858 (+0.0054) | 1/5 |
| F2 pred x C16 | PASS | Y 0.5250 (+0.0053) | N 0.4534 (-0.1002) | N 0.4859 (-0.0445) | Y 0.5427 (+0.0148) | Y 0.5398 (+0.0594) | 3/5 |
| F3 pred x C32 | NO_LOCK | N 0.4999 (-0.0198) | N 0.4544 (-0.0992) | N 0.4661 (-0.0643) | N 0.5268 (-0.0011) | N 0.4730 (-0.0074) | 0/5 |
| F0 pred x own+sum | NO_LOCK | Y 0.5329 (+0.0132) | N 0.4357 (-0.1179) | N 0.4851 (-0.0453) | Y 0.5299 (+0.0020) | N 0.4351 (-0.0453) | 2/5 |
| F1 pred x z_only | NO_LOCK | N 0.5046 (-0.0151) | N 0.4297 (-0.1239) | N 0.4739 (-0.0565) | N 0.5221 (-0.0058) | N 0.4537 (-0.0267) | 0/5 |
| F4 pred x C64 | NO_LOCK | N 0.5103 (-0.0094) | N 0.4264 (-0.1272) | N 0.4735 (-0.0569) | N 0.5080 (-0.0199) | N 0.4361 (-0.0443) | 0/5 |
| Ablation NoHoldout | PASS | Y 0.5697 (+0.0500) | N 0.4845 (-0.0691) | N 0.4820 (-0.0484) | Y 0.5648 (+0.0369) | Y 0.4971 (+0.0167) | 3/5 |
| Ablation AACDR | NO_LOCK | N 0.4735 (-0.0462) | N 0.4480 (-0.1056) | Y 0.5371 (+0.0067) | N 0.5063 (-0.0216) | Y 0.4939 (+0.0135) | 2/5 |
| E-NH0 pooled_mlp__own_plus_summary x own_plus_summary (NoHoldout) | PASS | Y 0.5697 (+0.0500) | N 0.4845 (-0.0691) | N 0.4820 (-0.0484) | Y 0.5648 (+0.0369) | Y 0.4971 (+0.0167) | 3/5 |
| E-NH1 biocda_predictive_e3 x z_plus_context16 (NoHoldout) | PASS | Y 0.5556 (+0.0359) | N 0.4492 (-0.1044) | N 0.4690 (-0.0614) | Y 0.5501 (+0.0222) | Y 0.4992 (+0.0188) | 3/5 |
| E-NH2 biocda_predictive_e3 x z_plus_context32 (NoHoldout) | NO_LOCK | Y 0.5276 (+0.0079) | N 0.4925 (-0.0611) | N 0.4628 (-0.0676) | N 0.5210 (-0.0069) | Y 0.5167 (+0.0363) | 2/5 |

## 各候選 × AUPRC（vs stest0）

| Candidate | gdsc_intersect13 | tcga_only3 | dapl | aacdr_gdsc_intersect | aacdr_tcga_only | AUPRC 超越數 |
|-----------|-----------------:|-----------:|-----:|---------------------:|---------------:|-------------:|
| B0/Ctrl pooled_mlp x own_plus_summary | N 0.5963 (-0.0018) | Y 0.6962 (+0.0002) | Y 0.5605 (+0.0035) | Y 0.5765 (+0.0055) | Y 0.6559 (+0.0259) | 4/5 |
| B1 predictive x C32 | N 0.5718 (-0.0263) | N 0.6377 (-0.0583) | N 0.5266 (-0.0304) | Y 0.5890 (+0.0180) | Y 0.6558 (+0.0258) | 2/5 |
| B2 XA x C32 | N 0.5616 (-0.0365) | N 0.6530 (-0.0430) | N 0.5423 (-0.0147) | Y 0.6028 (+0.0318) | Y 0.6380 (+0.0080) | 2/5 |
| F2 pred x C16 | N 0.5773 (-0.0208) | N 0.6473 (-0.0487) | N 0.5466 (-0.0104) | Y 0.5859 (+0.0149) | Y 0.6824 (+0.0524) | 2/5 |
| F3 pred x C32 | N 0.5718 (-0.0263) | N 0.6377 (-0.0583) | N 0.5266 (-0.0304) | Y 0.5890 (+0.0180) | Y 0.6558 (+0.0258) | 2/5 |
| F0 pred x own+sum | N 0.5826 (-0.0155) | N 0.6183 (-0.0777) | N 0.5447 (-0.0123) | Y 0.5824 (+0.0114) | Y 0.6359 (+0.0059) | 2/5 |
| F1 pred x z_only | N 0.5594 (-0.0387) | N 0.6112 (-0.0848) | N 0.5350 (-0.0220) | Y 0.5784 (+0.0074) | Y 0.6610 (+0.0310) | 2/5 |
| F4 pred x C64 | N 0.5800 (-0.0181) | N 0.6180 (-0.0780) | N 0.5285 (-0.0285) | Y 0.5716 (+0.0006) | Y 0.6431 (+0.0131) | 2/5 |
| Ablation NoHoldout | Y 0.6121 (+0.0140) | N 0.6368 (-0.0592) | N 0.5416 (-0.0154) | Y 0.6186 (+0.0476) | Y 0.6532 (+0.0232) | 3/5 |
| Ablation AACDR | N 0.5487 (-0.0494) | N 0.6363 (-0.0597) | Y 0.5648 (+0.0078) | N 0.5634 (-0.0076) | Y 0.6667 (+0.0367) | 2/5 |
| E-NH0 pooled_mlp__own_plus_summary x own_plus_summary (NoHoldout) | Y 0.6121 (+0.0140) | N 0.6368 (-0.0592) | N 0.5416 (-0.0154) | Y 0.6186 (+0.0476) | Y 0.6532 (+0.0232) | 3/5 |
| E-NH1 biocda_predictive_e3 x z_plus_context16 (NoHoldout) | N 0.5935 (-0.0046) | N 0.6339 (-0.0621) | N 0.5399 (-0.0171) | Y 0.5927 (+0.0217) | Y 0.6663 (+0.0363) | 2/5 |
| E-NH2 biocda_predictive_e3 x z_plus_context32 (NoHoldout) | N 0.5838 (-0.0143) | N 0.6723 (-0.0237) | N 0.5377 (-0.0193) | Y 0.5943 (+0.0233) | Y 0.6817 (+0.0517) | 2/5 |

## 圖例

- 標準 = **無 10% testset（stest0）** AACDR 基準。
- **硬閘：** 僅 `aacdr_gdsc_intersect` ∧ `aacdr_tcga_only`。
- Ablation 僅診斷，非正式 lock（除非寫入 24E manifest）。
- PASS 後選模序：`aacdr_gdsc`(5) > `aacdr_tcga_only`(4) > `dapl`(3) > `gdsc13`(2) > `tcga_only3`(1)。
