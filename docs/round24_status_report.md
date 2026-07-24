# Round 24 — 執行狀態報告

**更新時間：** 2026-07-24  
**容器：** `DAPL` · `/workspace/DAPL`  
**最終狀態：** **`LOCKED`** · champion **`E-NH0`**（pooled_mlp × own_plus_summary × NoHoldout）

## TCGA 彙整契約（強制）

凡 Round 24 TCGA 結果彙整，**必須**同時報告下列五組資料的 **DrugMacro AUROC** 與 **DrugMacro AUPRC**（per-drug macro avg；support 10/2/2）。缺任一 target 或缺 AUPRC 即視為彙整不完整。

| Key | 資料檔 |
|-----|--------|
| `gdsc_intersect13` | `data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv` |
| `tcga_only3` | `data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv` |
| `dapl` | `data/TCGA/TCGA_drug_response_from_DAPL.csv` |
| `aacdr_gdsc_intersect` | `data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv` |
| `aacdr_tcga_only` | `data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv` |

**排名依據：**  
- **硬閘 PASS（stest0 / 無 10% testset）：** `aacdr_gdsc_intersect`（>**0.5279**）**且** `aacdr_tcga_only`（>**0.4804**）。  
- **PASS 後排序（5:4:3:2:1）：** `aacdr_gdsc_intersect` > `aacdr_tcga_only` > `dapl` > `gdsc_intersect13` > `tcga_only3`。  
- `dapl` / `gdsc_intersect13` / `tcga_only3`：**必報，不擋 lock**。  
- Lock 排名僅 NoHoldout 合格臂；holdout 參考不混排。

**超越標準：** [`docs/AACDR_drug_macro_auroc_auprc.md`](AACDR_drug_macro_auroc_auprc.md)（**現行 = stest0**）。  
**最終報告：** [`round24_final_report.md`](round24_final_report.md) · **Lock：** [`reports/round24_final_model_lock.json`](../reports/round24_final_model_lock.json)  
比對：[`vs_aacdr_standard.md`](../reports/round24/vs_aacdr_standard.md)

## 總覽

| Stage / 實驗 | 狀態 | 說明 |
|--------------|------|------|
| 24A 協議/基準 | **PASS** | eval3 manifest；906→886 miss_latent |
| 24B 同協議重建 | **COMPLETE** | B0/B1/B2 |
| 24C 特徵 attribution | **COMPLETE** | top2=F2/F3 |
| Train-source ablation | **COMPLETE（診斷）** | NoHoldout 可過硬閘 |
| 24D gdsc 診斷 | **DONE** | `reports/round24/stage24d/` |
| **24E NoHoldout 確認** | **COMPLETE** | E-NH0/NH1 PASS；E-NH2 NO_LOCK |
| **24F gate / lock** | **`LOCKED`** | champion **E-NH0** |
| 24G 最終報告 | **COMPLETE** | `docs/round24_final_report.md` |

## Stage 24E / 24F 正式結果（NoHoldout lock pool）

硬閘：`aacdr_gdsc_intersect` >0.5279 ∧ `aacdr_tcga_only` >0.4804（stest0）。

| ID | Architecture × Feature | aacdr_gdsc | aacdr_tcga | Hard gate |
|----|------------------------|-----------:|-----------:|:---------:|
| **E-NH0** | pooled_mlp × own_plus_summary | **0.5648** | 0.4971 | **PASS** ← champion |
| E-NH1 | predictive_e3 × C16 | 0.5501 | 0.4992 | **PASS** |
| E-NH2 | predictive_e3 × C32 | 0.5210 | 0.5167 | NO_LOCK |

**選模：** 兩臂 PASS 時依加權（`aacdr_gdsc` 權重 5 最高）→ **E-NH0**。  
**結論：** 在與 stest0 對齊的 NoHoldout 資料協議下，pooled MLP + own_plus_summary 勝過 predictive×C16/C32；架構升級未再抬高硬閘主軸 `aacdr_gdsc_intersect`。

### Champion（E-NH0）五組 DrugMacro（5-fold mean）

| Target | AUROC | AUPRC | 硬閘 |
|--------|------:|------:|:----:|
| `aacdr_gdsc_intersect` | 0.5648 | 0.6186 | Y |
| `aacdr_tcga_only` | 0.4971 | 0.6532 | Y |
| `dapl` | 0.4820 | 0.5416 | N |
| `gdsc_intersect13` | 0.5697 | 0.6121 | N |
| `tcga_only3` | 0.4845 | 0.6368 | N |

### Holdout 參考（不參與 lock）

| ID | aacdr_gdsc | aacdr_tcga | gate |
|----|-----------:|-----------:|:----:|
| E-REF2（F2 holdout） | 0.5427 | 0.5398 | PASS |
| E-REF3（F3 holdout） | 0.5268 | 0.4730 | NO_LOCK |

## 產物路徑

```text
configs/round24/eval3.yaml
scripts/round24/run_stage24e.py
reports/round24/stage24e/candidate_manifest.json
reports/round24/stage24e/stage24e_decision.json
reports/round24_final_model_lock.json
docs/round24_final_report.md
```

## 科學敘事約束

- TCGA 未進入 loss / early stopping / checkpoint selection。
- 正式 lock 僅比較 NoHoldout 臂；GDSC 內部分數不作選模主軸。
- Telegram 僅完整 round 結束時發送。
