# drug-DCF Round 17 IDE 操作手冊

## Direct Prototype Representation Optimization + 5-target TCGA Inference + Prototype tSNE

本文件為 Round 17 的 IDE 實作與執行手冊。  
**Phase 0（17D–17E 基礎設施）已完成**；**Phase 1+（17A–C、17F pipeline）已補齊工具與設定**，可啟動 Stage 17A。

---

## 0. Round 17 定版定位

Round 17 分成三條支線：

```text
Round 17A-C:
  Direct Prototype Representation Optimization

Round 17D-E:
  5-target TCGA inference expansion          ← Phase 0 已完成

Round 17F:
  Prototype-aware tSNE visualization         ← Phase 1+ 待實作
```

Round 17 的主方法目標是繼續優化 direct prototype representation：

```text
own_proto_delta
own_proto_context
projected delta/context
prototype-aware response head
```

同時保留 11 維 `own_plus_summary` 與 minimal features 作為 control。

---

## 1. Phase 0 實作狀態（已完成）

### 1.1 程式變更摘要

| 元件 | 變更 |
|------|------|
| `tools/finetune_tcga_eval.py` | 5 個 `DEFAULT_TCGA_EVAL_TARGETS`、AACDR prefix、`Integrated5_*` 指標、歷史 `Integrated_*` 仍僅 3 target |
| `step1_finetune_latent_pipeline_All_split.py` | `--drug-smiles-path` CLI；預設 AACDR extended SMILES；`run_config_snapshot` 記錄路徑與 eval targets |
| `tools/optimization_runner.py` | `--drug-smiles-path` 透傳至 finetune subprocess |
| `config/round17_direct_proto_settings.json` | Round 17 設定檔 |
| `tests/test_round17_*.py` | Phase 0 單元測試（3 檔） |

### 1.2 TCGA inference targets（5 個）

| # | eval key | 資料路徑 | 角色 |
|---|----------|----------|------|
| 1 | `gdsc_intersect13` | `data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv` | **歷史主指標來源** |
| 2 | `tcga_only3` | `data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv` | 歷史次要 |
| 3 | `dapl` | `data/TCGA/TCGA_drug_response_from_DAPL.csv` | 歷史次要 |
| 4 | `aacdr_tcga_only` | `data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv` | Round 17 新增 |
| 5 | `aacdr_gdsc_intersect` | `data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv` | Round 17 新增 |

每個 target 會輸出至：

```text
<outfolder>/fold_<k>/target_eval_<eval_key>/
```

### 1.3 Drug SMILES（Extended）

Round 17 預設改用：

```text
data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv
```

**重要：** AACDR extended 新增的 19 個藥物僅供 TCGA inference，**不**進入 GDSC response training labels、pretrain 或 prototype construction。

### 1.4 指標定義

#### 歷史指標（Round 13–16 可比）

| 欄位 | 來源 |
|------|------|
| `Global_TCGA_AUC` / `Average_TCGA_AUC` | `gdsc_intersect13`（headline alias） |
| `Integrated_*` | 僅 pool **3 個歷史 target**（gdsc_intersect13 + tcga_only3 + dapl） |

#### Round 17 新增 Integrated5（不取代歷史 Integrated）

| 欄位 | 定義 |
|------|------|
| `Integrated5_TargetMacro_TCGA_AUC` | 5 個 eval target 的 `Average_Metrics.AUC` macro mean |
| `Integrated5_TargetMacro_TCGA_AUPRC` | 同上（AUPRC） |
| `Integrated5_DrugMacro_TCGA_AUC` | 5 target 全部 valid per-drug AUC macro mean |
| `Integrated5_DrugMacro_TCGA_AUPRC` | 同上（AUPRC） |
| `Integrated5_n_tcga_eval_targets` | 應為 5 |

不使用 sample-count weighted mean 作為主 integrated 指標。

---

## 2. CLI 用法

### 2.1 單次 finetune（step1）

```bash
docker exec -w /workspace/DAPL DAPL python3 step1_finetune_latent_pipeline_All_split.py \
  --config config/params_finetune_mini.json \
  --model_select_path <path/to/model_select.csv> \
  --outfolder result/smoke_round17_phase0 \
  --drug-smiles-path data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv \
  --batch_size 2048 \
  --mini_batch_size 512 \
  --epochs 1
```

### 2.2 批次 finetune（optimization_runner）

```bash
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py finetune \
  --manifest <manifest.csv> \
  --run-dir result/optimization_runs/round17_direct_proto/stage17a \
  --finetune-config config/params_finetune_round17_direct_proto.json \
  --drug-smiles-path data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv \
  --batch-size 12288 \
  --mini-batch-size 3072 \
  --epochs 1500 \
  --max-parallel 8 \
  --round13-mode
```

環境變數建議（Phase 1+ 腳本使用）：

```bash
export DRUG_SMILES_PATH="${DRUG_SMILES_PATH:-data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv}"
```

---

## 3. Phase 0 驗證

### 3.1 單元測試

```bash
docker exec -w /workspace/DAPL DAPL pytest \
  tests/test_round17_tcga_eval_targets.py \
  tests/test_round17_drug_smiles_extended.py \
  tests/test_round17_integrated5_metrics.py \
  -q
```

### 3.2 資料 QC（手動）

```bash
docker exec -w /workspace/DAPL DAPL python3 - <<'PY'
import pandas as pd

paths = [
    "data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv",
    "data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv",
]
for p in paths:
    df = pd.read_csv(p)
    print(p, df.shape, "drugs:", df["drug_name"].nunique())
    assert {"Patient_id", "drug_name", "Label"}.issubset(df.columns)

smiles = pd.read_csv("data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv", index_col=0)
assert smiles["SMILES"].notna().all()
print("QC OK")
PY
```

### 3.3 Five-target smoke 檢查清單

finetune 完成後確認：

```text
target_eval_gdsc_intersect13/
target_eval_tcga_only3/
target_eval_dapl/
target_eval_aacdr_tcga_only/
target_eval_aacdr_gdsc_intersect/
eval_metrics_integrated_summary.csv   # 含 Integrated5_* 欄位
data_alignment_report.csv             # 5 個 TCGA dataset 列
config.json                           # drug_smiles_data_path + tcga_eval_targets
```

log 中不應出現大量 `Skipping <drug>: No SMILES found`（AACDR target 藥物）。

---

## 4. Phase 1+ 工具（已補齊）

| 支線 | 內容 | 產物 |
|------|------|------|
| **17A** | Direct prototype feature optimization | `tools/round17_direct_proto_config_builder.py`、`tools/run_round17_direct_proto_stage17a.sh` |
| **17B** | Prototype-aware response head search | `tools/run_round17_proto_head_stage17b.sh` |
| **17C** | 10-seed confirmation | `tools/run_round17_confirmation_stage17c.sh` |
| **17F** | Prototype-aware tSNE | `tools/visualize_round17_prototype_tsne.py`、`tools/run_round17_prototype_tsne_stage17f.sh` |
| Analyzer | 跨 feature/head 比較 | `tools/analyze_round17_direct_proto.py` |
| Finetune config | Round 17 專用超參 | `config/params_finetune_round17_direct_proto.json` |

> **注意：** 17B 的 `two_tower_proto` / `proto_film` 列在 `heads_pending_step1`；目前 Stage 17B 僅啟用 `concat_mlp`，待 `step1` classifier 接線後再開 head sweep。

### 本地啟動前檢查（P0 驗證）

```bash
docker exec -w /workspace/DAPL DAPL python3 -m py_compile \
  tools/prototype_response_features.py \
  tools/extract_round13_proto_features.py \
  tools/round17_direct_proto_config_builder.py

docker exec -w /workspace/DAPL DAPL pytest tests/test_round17_required_modes.py -q
```

若 GitHub `main` 仍缺 `9b317ce` 之後的 commit，請先 `git push` 再於遠端跑上述檢查。

### Stage 17A 最小 smoke

使用 `config/round17_direct_proto_smoke_settings.json`（6 個已驗證 modes、1 model、1 seed、5 epochs）：

```bash
ROUND17_ROOT=result/optimization_runs/round17_direct_proto_smoke \
FINETUNE_PARALLEL=1 \
python3 tools/round17_direct_proto_config_builder.py \
  --settings config/round17_direct_proto_smoke_settings.json \
  --outdir result/optimization_runs/round17_direct_proto_smoke \
  --stage 17a --force
```

### 4.1 17A 方法候選（規格保留）

```text
Feature modes:
  none, own_plus_summary
  own_proto_delta, own_proto_context
  own_proto_delta_projected_16/32/64
  own_proto_context_projected_16/32/64
  own_proto_delta_normed, own_proto_context_normed
```

### 4.1 17B Head 候選（規格保留）

```text
concat_mlp (baseline)
two_tower_proto_context
proto_film_mlp
minimal_source_only
```

### 4.2 17F tSNE 輸出（規格保留）

```text
prototype_tsne_samples_and_prototypes.png
prototype_tsne_samples_and_prototypes.pdf
prototype_tsne_coordinates.csv
prototype_tsne_metadata.json
```

source / target prototype 以不同色星號標註；missing target prototype 不畫 target star。

---

## 5. 成功標準

### Phase 0（基礎設施）

```text
1. 5-target eval 常數與 flatten prefix 正確
2. Integrated5_* 與歷史 Integrated_* 並存且定義不同
3. --drug-smiles-path 從 step1 與 optimization_runner 可透傳
4. AACDR TCGA 藥物在 extended SMILES 表皆有 SMILES
5. Phase 0 pytest 全數通過
```

### Round 17 整體（Phase 1+ 完成後）

**Basic success**

```text
1. 5-target inference 全部跑通
2. AACDR targets 無 missing SMILES / latent
3. tSNE 圖成功輸出且 prototype 標註正確
4. direct prototype candidates 至少與 own_plus_summary 落在 std 內
```

**Method success**

```text
1. direct prototype 在 Integrated5_TargetMacro_TCGA_AUC 優於 own_plus_summary
2. 或 Integrated5_DrugMacro_TCGA_AUC 優於 own_plus_summary
3. 或 prototype-aware head 使 direct prototype 10-seed std 更低
```

**Strong success**

```text
1. 10-seed mean >= 0.6112（Round 13 最佳）
2. 或 best >= 0.6200 且 10-seed std 可控
```

---

## 6. 風險與注意事項

| 風險 | 說明 | 建議 |
|------|------|------|
| 歷史主指標不可混用 | `Average_TCGA_AUC` 仍來自 gdsc_intersect13 | Round 17 新方法另看 `Integrated5_*` |
| AACDR vs 舊 tcga_only3 | 藥物集合不同（3 vs 8 drugs） | 用 `aacdr_tcga_only` key，勿與舊 tcga_only3 直接比 |
| 訓練不受影響 | 新 19 藥不在 GDSC train | 預期行為；只擴展 inference |
| Integrated 語意 | `Integrated_*` = 3 target；`Integrated5_*` = 5 target | 報告中分開呈現 |

---

## 7. 相關檔案

| 用途 | 路徑 |
|------|------|
| TCGA eval 核心 | `tools/finetune_tcga_eval.py` |
| Finetune 主腳本 | `step1_finetune_latent_pipeline_All_split.py` |
| 批次 runner | `tools/optimization_runner.py` |
| Round 17 設定 | `config/round17_direct_proto_settings.json` |
| Phase 0 測試 | `tests/test_round17_tcga_eval_targets.py` 等 |
| AACDR TCGA | `data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_*.csv` |
| AACDR SMILES | `data/GDSC_drug_merge_pubchem_dropNA_MACCS_AACDR_extended.csv` |

---

## 8. 建議執行順序（Phase 1+ 起）

```text
Step 0: pytest Phase 0 tests                          ← 現在可做
Step 1: five-target inference smoke（單 checkpoint）
Step 2: Stage 17A direct prototype feature optimization
Step 3: Stage 17F prototype tSNE
Step 4: Stage 17B prototype head search
Step 5: Stage 17C 10-seed confirmation
```

---

## 9. 進度更新（2026-07-07）

### 9.1 Stage 17A 完成摘要

- `stage17a_finetune_dispatch_manifest.csv` 已達 `1440/1440 success`（`failed=0`）。
- 本輪正式最佳候選之一：`finetune_r13_exp_008_own_plus_summary_plus_delta_projected_16`（後續進入 17B head search）。
- 目前 Round 17 狀態：**17A 完成，17B/17C 尚未執行**。

### 9.2 監控與通知

- `tools/run_round17_direct_proto_stage17a.sh`、`tools/run_round17_proto_head_stage17b.sh`、`tools/run_round17_confirmation_stage17c.sh`
  已內建 Telegram `stage-start` / `stage-done` 通知。
- `tools/run_round17_pipeline.sh` 已內建 pipeline start/done 與 stage fail 通知。

### 9.3 17B 接續前檢查

```text
必要前置：
1. 17A finetune manifest 全 success
2. 產出 reports_stage17a/round17_top_candidates.csv

建議流程：
1. 先做 stage17a aggregate + analyze（補齊 reports_stage17a）
2. 再啟動 run_round17_proto_head_stage17b.sh
```

---

*文件版本：Phase 0 + Stage17A 完成更新（2026-07）*
