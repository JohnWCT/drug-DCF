# drug-DCF Round 5+ 優化操作手冊

Control-centered Latent Optimization + Class-wise Prototype Gap Alignment

> 完整規劃章節（§0–§20）見本輪設計討論；執行摘要與指令亦收錄於 `docs/pipeline_summary.md` §13。

## 快速索引

| 章節 | 內容 |
|------|------|
| Branch A | `config/pretrain_sweeps/vaewc_round5_control_centered.json`（48 jobs） |
| Branch B | `config/pretrain_sweeps/vaewc_round5_class_gap_branch.json`（30 jobs） |
| Branch C | `config/pretrain_sweeps/vaewc_round5_t2s_infonce_appendix.json`（24 jobs） |
| Selection | `--selection-mode round5_structure_first` |
| 診斷 | `tools/analyze_round5_pretrain.py` |
| 一鍵 pretrain | `tools/run_round5_pretrain.sh` |

## 最高原則（摘要）

**不要：** 提高 cls 硬拉 K-means、symmetric InfoNCE 主線、FID 單獨選模、大 grid 交叉。

**必須：** `lambda_proto=0` baseline 不變、structure-first selection、finetune 固定納入 exp_018 / exp_746、主指標 `Average_TCGA_AUC_mean`。

## 新增 API

### Class-wise prototype gap

```python
from tools.classwise_alignment import compute_classwise_prototype_gap
loss, metrics = compute_classwise_prototype_gap(
    z_source, y_source, z_target, y_target,
    num_classes, metric="cosine", detach_source=True, detach_target=False,
)
```

### Config keys（pretrain）

```json
{
  "lambda_class_gap": 0.0003,
  "class_gap_metric": "cosine",
  "class_gap_start_epoch": 5,
  "class_gap_full_epoch": 30,
  "class_gap_min_samples_per_domain": 2,
  "class_gap_detach_source": true,
  "class_gap_detach_target": false
}
```

### paired_params（config generator）

```json
"paired_params": [
  {"latent_size": 32, "encoder_dims": [256, 128]},
  {"latent_size": 64, "encoder_dims": [512, 256, 128]},
  {"latent_size": 128, "encoder_dims": [1024, 512, 256]}
]
```

## 執行步驟

見 `docs/pipeline_summary.md` §13.5 或執行：

```bash
bash tools/run_round5_pretrain.sh
```

## 測試

```bash
pytest tests/test_classwise_prototype_gap.py \
       tests/test_classwise_alignment.py \
       tests/test_optimization_selection_round5.py \
       tests/test_analyze_round5_pretrain.py \
       tests/test_round5_config_generation.py -q
```

- `test_classwise_prototype_gap.py`：Round 5 主線 class-wise prototype gap
- `test_classwise_alignment.py`：cMMD（`compute_classwise_mmd`）

## 成功標準

1. `Average_TCGA_AUC_mean` > 0.5339（R4.1 exp_035）
2. `kmeans_ari` ≥ 0.65 或 ≥ 90% control mean
3. class-gap 分支有 non-collapse 且下游接近 control
