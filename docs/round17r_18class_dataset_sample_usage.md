# Round 17R Dataset Sample Usage (18-class-clean)

## Scope

本檔記錄 Round 17R 在 18-class-clean 設定下，實際「保留 / 剔除」的樣本數。

- Feature 參考：`result/optimization_runs/round17r_18class/features/r13_exp_008/own_plus_summary/`
- `feature_metadata.json` 顯示：
  - `n_trainable_cancer_types = 18`
  - `prototype_class_source = checkpoint_metadata`
  - `uses_legacy_28class_cache = false`

註：Round 17R-A 產出的 20 個 feature 組合，此樣本覆蓋數一致。

---

## 1) Metadata 全量分布（18 類 vs 非 18 類）

| Source | Total | In 18 classes | Not in 18 classes |
|---|---:|---:|---:|
| `data/ccle_sample_info_df.csv` | 1304 | 937 | 367 |
| `data/TCGA/xena_sample_info_df.csv` | 9808 | 8394 | 1414 |

---

## 2) GDSC finetune（CCLE latent）實際使用樣本

Response 檔：`data/GDSC2_fitted_dose_response_MaxScreen_raw.csv`

| Metric | Count |
|---|---:|
| Total rows | 180992 |
| Kept rows (ModelID 有 latent) | 110279 |
| Dropped rows (ModelID 無 latent) | 70713 |
| Unique ModelID total | 870 |
| Unique ModelID kept | 526 |
| Unique ModelID dropped | 344 |

Dropped `ModelID` 分解：

- 在 `ccle_sample_info_df.csv` 找得到：55（全部屬於 18 類外）
- 在 `ccle_sample_info_df.csv` 找不到：289（UNMAPPED）
- 18 類內被剔除：0

Top dropped cancer types（僅統計 `ccle_sample_info_df.csv` 可對應者）：

- Myeloma: 15
- Bone Cancer: 15
- Neuroblastoma: 14
- Prostate Cancer: 6
- Rhabdoid: 3
- Fibroblast: 1
- Gallbladder Cancer: 1

---

## 3) TCGA evaluation（patient-level）實際使用樣本

下表以各 eval dataset 的 `Patient_id` 與 `tcga_latent_proto.pkl` patient key 交集計算：

| Eval target | Patients total | Patients with latent | Patients missing latent | Missing patients in 18 classes | Missing patients not in 18 classes |
|---|---:|---:|---:|---:|---:|
| `gdsc_intersect13` | 702 | 684 | 18 | 0 | 18 |
| `tcga_only3` | 127 | 127 | 0 | 0 | 0 |
| `dapl` | 178 | 177 | 1 | 0 | 1 |
| `aacdr_tcga_only` | 95 | 95 | 0 | 0 | 0 |
| `aacdr_gdsc_intersect` | 406 | 398 | 8 | 0 | 8 |

缺失 patient 的癌別（Top）：

- `gdsc_intersect13`: Prostate Cancer 18
- `dapl`: Prostate Cancer 1
- `aacdr_gdsc_intersect`: Prostate Cancer 8

---

## 4) 現行處理規則（程式行為）

- **GDSC finetune**：若 `ModelID` 不在 `expression_latent_dict`，該樣本列直接 skip。
  - 實作位置：`step1_finetune_latent_pipeline_All_split.py`
- **TCGA eval**：若 `Patient_id` 對不到 `tcga_latent_dict`，該 patient（對應 drug）不納入評估。
  - 實作位置：`tools/inference_utils.py`
- **18-class-clean 的來源**：latent/prototype 先在 feature extraction 階段依 checkpoint 18 類清單過濾。
  - 實作位置：`tools/extract_round13_proto_features.py`

結論：目前對「無 cancer type 或不在 18 類」的樣本，行為是**不納入 latent，後續 finetune/eval 階段以 missing-latent 跳過**，不是補值或映射到其他類別。

---

## 5) GDSC 與 5 個 TCGA 資料集：藥物樣本分布（含 Pos/Neg）

### 5.1 GDSC（finetune source）

資料檔：`data/GDSC2_fitted_dose_response_MaxScreen_raw.csv`

| Metric | Count |
|---|---:|
| Total rows | 180992 |
| Unique cell lines (`ModelID`) | 870 |
| Unique drugs (`mapped_name`) | 230 |
| Label=1 (Pos) rows | 49398 |
| Label=0 (Neg) rows | 131594 |

### 5.2 TCGA `gdsc_intersect13`

資料檔：`data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv`  
總計：906 rows，702 patients，12 drugs

| drug_name | total | pos | neg |
|---|---:|---:|---:|
| Cisplatin | 221 | 170 | 51 |
| Paclitaxel | 141 | 95 | 46 |
| Gemcitabine | 139 | 69 | 70 |
| 5-Fluorouracil | 122 | 76 | 46 |
| Temozolomide | 91 | 10 | 81 |
| Docetaxel | 88 | 56 | 32 |
| Vinorelbine | 28 | 23 | 5 |
| Tamoxifen | 18 | 11 | 7 |
| Bicalutamide | 17 | 14 | 3 |
| Methotrexate | 14 | 7 | 7 |
| Sorafenib | 14 | 2 | 12 |
| Vinblastine | 13 | 6 | 7 |

### 5.3 TCGA `tcga_only3`

資料檔：`data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv`  
總計：129 rows，127 patients，3 drugs

| drug_name | total | pos | neg |
|---|---:|---:|---:|
| Doxorubicin | 81 | 53 | 28 |
| Etoposide | 31 | 23 | 8 |
| Pemetrexed | 17 | 9 | 8 |

### 5.4 TCGA `dapl`

資料檔：`data/TCGA/TCGA_drug_response_from_DAPL.csv`  
總計：178 rows，178 patients，5 drugs

| drug_name | total | pos | neg |
|---|---:|---:|---:|
| gemcitabine | 46 | 23 | 23 |
| temozolomide | 46 | 23 | 23 |
| cisplatin | 40 | 20 | 20 |
| sorafenib | 25 | 13 | 12 |
| 5-fluorouracil | 21 | 10 | 11 |

### 5.5 TCGA `aacdr_tcga_only`

資料檔：`data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv`  
總計：97 rows，95 patients，8 drugs

| drug_name | total | pos | neg |
|---|---:|---:|---:|
| doxorubicin | 23 | 7 | 16 |
| capecitabine | 20 | 16 | 4 |
| carboplatin | 19 | 12 | 7 |
| anastrozole | 13 | 10 | 3 |
| sunitinib | 7 | 0 | 7 |
| etoposide | 5 | 3 | 2 |
| ifosfamide | 5 | 2 | 3 |
| pazopanib | 5 | 1 | 4 |

### 5.6 TCGA `aacdr_gdsc_intersect`

資料檔：`data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv`  
總計：425 rows，406 patients，11 drugs

| drug_name | total | pos | neg |
|---|---:|---:|---:|
| cisplatin | 105 | 89 | 16 |
| temozolomide | 89 | 10 | 79 |
| gemcitabine | 66 | 25 | 41 |
| paclitaxel | 52 | 36 | 16 |
| 5-fluorouracil | 44 | 24 | 20 |
| docetaxel | 21 | 9 | 12 |
| tamoxifen | 15 | 12 | 3 |
| sorafenib | 14 | 1 | 13 |
| bicalutamide | 8 | 7 | 1 |
| oxaliplatin | 6 | 1 | 5 |
| erlotinib | 5 | 2 | 3 |

---

## 6) Filtered（18-class-clean, with-latent）per-drug total/pos/neg

以下統計使用 18-class-clean latent 後實際會進入模型的樣本：

- CCLE 來源：`result/optimization_runs/round17r_18class/features/r13_exp_008/own_plus_summary/ccle_latent_proto.pkl`
- TCGA 來源：`result/optimization_runs/round17r_18class/features/r13_exp_008/own_plus_summary/tcga_latent_proto.pkl`

### 6.1 Filtered GDSC

總計：110279 rows，526 unique cell lines，230 drugs

| mapped_name | total | pos | neg |
|---|---:|---:|---:|
| ulixertinib | 947 | 214 | 733 |
| oxaliplatin | 934 | 71 | 863 |
| fulvestrant | 931 | 2 | 929 |
| selumetinib | 927 | 186 | 741 |
| dactinomycin | 924 | 513 | 411 |
| uprosertib | 914 | 292 | 622 |
| gsk343 | 910 | 13 | 897 |
| docetaxel | 908 | 523 | 385 |
| acetalax | 820 | 234 | 586 |
| mg-132 | 526 | 522 | 4 |

註：完整 per-drug 表（230 drugs）過長，若需完整版可輸出成 csv 附檔。

### 6.2 Filtered TCGA `gdsc_intersect13`

總計：886 rows，684 patients，11 drugs

| drug_name | total | pos | neg |
|---|---:|---:|---:|
| Cisplatin | 221 | 170 | 51 |
| Paclitaxel | 141 | 95 | 46 |
| Gemcitabine | 139 | 69 | 70 |
| 5-Fluorouracil | 122 | 76 | 46 |
| Temozolomide | 91 | 10 | 81 |
| Docetaxel | 85 | 55 | 30 |
| Vinorelbine | 28 | 23 | 5 |
| Tamoxifen | 18 | 11 | 7 |
| Methotrexate | 14 | 7 | 7 |
| Sorafenib | 14 | 2 | 12 |
| Vinblastine | 13 | 6 | 7 |

### 6.3 Filtered TCGA `tcga_only3`

總計：129 rows，127 patients，3 drugs

| drug_name | total | pos | neg |
|---|---:|---:|---:|
| Doxorubicin | 81 | 53 | 28 |
| Etoposide | 31 | 23 | 8 |
| Pemetrexed | 17 | 9 | 8 |

### 6.4 Filtered TCGA `dapl`

總計：177 rows，177 patients，5 drugs

| drug_name | total | pos | neg |
|---|---:|---:|---:|
| temozolomide | 46 | 23 | 23 |
| gemcitabine | 45 | 22 | 23 |
| cisplatin | 40 | 20 | 20 |
| sorafenib | 25 | 13 | 12 |
| 5-fluorouracil | 21 | 10 | 11 |

### 6.5 Filtered TCGA `aacdr_tcga_only`

總計：97 rows，95 patients，8 drugs

| drug_name | total | pos | neg |
|---|---:|---:|---:|
| doxorubicin | 23 | 7 | 16 |
| capecitabine | 20 | 16 | 4 |
| carboplatin | 19 | 12 | 7 |
| anastrozole | 13 | 10 | 3 |
| sunitinib | 7 | 0 | 7 |
| etoposide | 5 | 3 | 2 |
| ifosfamide | 5 | 2 | 3 |
| pazopanib | 5 | 1 | 4 |

### 6.6 Filtered TCGA `aacdr_gdsc_intersect`

總計：417 rows，398 patients，10 drugs

| drug_name | total | pos | neg |
|---|---:|---:|---:|
| cisplatin | 105 | 89 | 16 |
| temozolomide | 89 | 10 | 79 |
| gemcitabine | 66 | 25 | 41 |
| paclitaxel | 52 | 36 | 16 |
| 5-fluorouracil | 44 | 24 | 20 |
| docetaxel | 21 | 9 | 12 |
| tamoxifen | 15 | 12 | 3 |
| sorafenib | 14 | 1 | 13 |
| oxaliplatin | 6 | 1 | 5 |
| erlotinib | 5 | 2 | 3 |
