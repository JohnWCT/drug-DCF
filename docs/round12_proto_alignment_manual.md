# drug-DCF Round 12 IDE 操作手冊

## Conditional ADV + Source-anchor EMA Prototype Alignment

## 0. Round 12 核心定位

Round 12 的任務是：

```text
在 Round 11 best exp_035 的穩定 10C weak global guard 基礎上，
加入 Source-anchor EMA Prototype Alignment，
降低 same-cancer source/target prototype gap，
但不取代 Conditional ADV。
```

Round 12 主線：

```text
exp_035
+ conditional_plus_weak_global
+ source-anchor EMA prototype alignment
```

其中：

```text
exp_035:
  Round 11 best downstream model
  Average_TCGA_AUC_mean = 0.5828
  reconstruction_loss_type = mse
  λ_cond_adv = 0.0001
  global_adv_mode = conditional_plus_weak_global
  λ_global_mult = 0.25
```

## 1. Round 12 主要目標

Round 12 要回答：

```text
Q1. Source-anchor prototype alignment 是否降低 same-cancer source/target prototype distance？
Q2. 是否保留 inter-cancer margin？
Q3. 是否不惡化 conditional leakage？
Q4. 是否讓 downstream Average TCGA 超越 0.5828？
Q5. 是否接近或超越 R7 exp_048 = 0.5918？
Q6. hybrid / smooth_l1 reconstruction 在 prototype alignment 下是否有幫助？
```

## 2. Round 12 不做什麼

Round 12 不做：

```text
1. 不新增 SupCon。
2. 不新增 prototype compactness。
3. 不新增 importance-aware weighting。
4. 不新增 prototype-distance response features。
5. 不改 Step 2 response predictor。
6. 不重做 architecture / latent_size / encoder_dims 大掃。
```

只做：

```text
target per-cancer prototype → source EMA anchor
```

## 3. 方法定義

### 3.1 Source-anchor EMA prototype

```text
P_source_ema[c] ← m · P_source_ema[c] + (1 - m) · mean(z_source | cancer=c)
```

Round 12 主線：`proto_ema_momentum = 0.95`。

限制：

```text
1. 只由 source / CCLE latent 更新。
2. 不進 optimizer。
3. 不反向傳梯度。
4. 作為 stop-gradient target。
```

### 3.2 Prototype alignment loss

```text
L_proto_align =
  mean_c distance(
    mean(z_target | cancer=c),
    stopgrad(P_source_ema[c])
  )
```

主線 metric：`cosine`；小型對照：`euclidean`。

## 4. Branch 設計

```text
12A：Round 11 baseline prototype-gap diagnostics
12B：Source-anchor EMA Prototype Alignment main sweep
12C：Prototype Alignment + reconstruction small branch
12D：Ablation controls
```

### 4.1 Round 12A baseline diagnostics

```bash
python tools/analyze_round12_baseline_prototype_gaps.py \
  --round11-root result/optimization_runs/round11_stability_recon \
  --outdir result/optimization_runs/round12_proto_alignment/round12a_baseline_qc
```

### 4.2 Round 12B main

- baseline = `exp_035`
- `reconstruction_loss_type = mse`
- `lambda_proto_align` = 0.0001 / 0.0003 / 0.001 / 0.003
- schedules = 20→60 / 40→90 / 60→120
- seeds = 101 / 202 / 303
- jobs = 36 + no-proto control 3 = 39

### 4.3 Round 12C reconstruction branch

- recon = `hybrid_mse_smooth_l1` / `smooth_l1`
- beta = 0.5 / 1.0
- `lambda_proto_align` = 0.0003 / 0.001
- schedule = 40→90
- seeds = 101 / 202 / 303
- jobs = 24

### 4.4 Round 12D controls

- euclidean control：3 jobs (`lambda_proto_align=0.0003`, schedule 40→90)

### 4.5 Total

```text
Pretrain total = 66
Finetune total = Top-30 × 4 = 120
```

## 5. 主要檔案

```text
config/round12_proto_alignment_settings.json
tools/source_anchor_prototypes.py
tools/round12_config_builder.py
tools/analyze_round12_baseline_prototype_gaps.py
tools/analyze_round12_proto_alignment.py
tools/round12_selection.py
tools/run_round12_proto_alignment_pipeline.sh
docs/round12_proto_alignment_manual.md
```

## 6. 必須確認的 pretrain 接線

`pretrain_VAEwC.py` 必須確認：

1. 解析 `resolve_source_anchor_proto_training_params(param)`
2. 初始化 `SourceAnchorEMAPrototypes`
3. 每個 epoch 計算 `get_proto_align_lambda_eff(...)`
4. `train_d_ae(...)` 傳入 prototype 相關參數
5. `total_loss = total_loss + lambda_proto_align_eff * proto_align_loss`

## 7. 執行前測試

```bash
python -m compileall .

pytest tests/test_source_anchor_prototypes.py \
  tests/test_round12_config_builder.py \
  tests/test_round12_proto_alignment_training_flags.py \
  tests/test_round12_selection.py \
  tests/test_analyze_round12_proto_alignment.py \
  tests/test_round11_config_builder.py \
  tests/test_round11_selection.py \
  tests/test_reconstruction_losses.py -q
```

## 8. 執行順序（IDE）

```text
1. 確認 pretrain_VAEwC.py 主流程已接上 source-anchor prototype alignment。
2. 跑 compileall。
3. 跑 Round 12 tests。
4. 跑 config builder smoke，確認 66 jobs。
5. 跑 1–3 個 pretrain smoke。
6. 檢查 run_summary / gan_metrics 的 proto metadata。
7. 執行完整 Round 12 pipeline。
8. 檢查 final_report。
9. 更新 docs/pipeline_summary.md 與 docs/round12_final_report.md。
```

## 9. 一句話總結

```text
Round 12 的核心是：
在 Round 11 best exp_035 的 stable 10C weak global guard 架構上，
加入 source-anchor EMA prototype alignment，
用低權重、晚啟動、source stop-gradient 的方式，
降低 same-cancer source/target prototype gap，
同時保留 cancer biology、conditional deconfounding 與 downstream drug response utility。
```
