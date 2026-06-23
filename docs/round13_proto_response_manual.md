# drug-DCF Round 13 IDE 操作手冊

## Prototype-distance Response Features (Step 2 only)

Round 13 不再訓練 encoder，也不改 Step 1 loss。它把 Round 12 的 prototype geometry 轉成 Step 2 response predictor 的輔助 features：

```text
response_input = concat(z, prototype_distance_features)
```

## 快速開始

```bash
python -m compileall .

pytest tests/test_prototype_response_features.py \
  tests/test_round13_config_builder.py \
  tests/test_round13_proto_response_training_flags.py \
  tests/test_round13_selection.py \
  tests/test_analyze_round13_proto_response.py -q

python tools/round13_config_builder.py \
  --settings config/round13_proto_response_settings.json \
  --outdir result/optimization_runs/round13_proto_response_smoke \
  --force

bash tools/run_round13_proto_response_pipeline.sh
```

## Baselines

| Reference | Avg TCGA |
|-----------|----------|
| Round 12 exp_037 | 0.5972 |
| Round 11 exp_035 | 0.5828 |
| R7 exp_048 | 0.5918 |

## Feature modes

- `none` — z-only baseline
- `own_cancer` — 主分支（低維、可解釋）
- `all_source_anchors` — 全 cancer source-anchor distance vector
- `all_source_and_target` — 小型 source+target vector 測試
- `own_plus_summary` — 可選 global summary branch

## 主要檔案

- `config/round13_proto_response_settings.json`
- `tools/prototype_response_features.py`
- `tools/extract_round12_prototypes.py`
- `tools/extract_round13_proto_features.py`
- `tools/round13_config_builder.py`
- `tools/round13_selection.py`
- `tools/analyze_round13_proto_response.py`
- `tools/run_round13_proto_response_pipeline.sh`

## 成功標準

- 基本：feature extraction + finetune + aggregate 完成
- 方法：同模型 z+proto > z-only；Best Round 13 > 0.5972
- 強成功：Avg TCGA >= 0.6000

## Round 14

- **Go:** `go_vicreg_stabilizer`（低權重 latent stabilizer）
- **No-go:** Round 13.1 simplified features
