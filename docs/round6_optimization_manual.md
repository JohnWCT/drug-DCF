# drug-DCF Round 6 多分支優化操作手冊

## Tumor-topology-aware latent representation optimization

> 方法主題：**Tumor-topology-aware latent representation learning**  
> 設計動機與 Round 5 結論見 `docs/pipeline_summary.md` §14.1。

## 快速索引

| 分支 | Sweep | Jobs | Run ID |
|------|-------|------|--------|
| **6A** Topology | `config/pretrain_sweeps/vaewc_round6A_tumor_topology.json` | 16 | `vaewc_round6A_tumor_topology` |
| **6B** Topology + class-gap | `vaewc_round6B_topology_classgap_combo.json` | 18 | `vaewc_round6B_topology_classgap_combo` |
| **6C** Subspace split | `vaewc_round6C_tumor_transfer_subspace.json` | 24 | `vaewc_round6C_tumor_transfer_subspace` |
| **6D** Within-domain SupCon | `vaewc_round6D_within_domain_tumor_supcon.json` | 32 | `vaewc_round6D_within_domain_tumor_supcon` |
| **6E** VICReg stabilizer | `vaewc_round6E_tumor_vicreg_stabilizer.json` | 12 | `vaewc_round6E_tumor_vicreg_stabilizer` |
| **6S** Selection | `round6_sweetspot` | — | `round6_combined` |

Baseline：`config/params_proto_base_exp001_vaewc.json`（R5 下游最佳 **exp_001**）。

## 核心原則

1. 不提高 `lambda_cls` 作為主要方法；不以 cross-domain InfoNCE 為主線。
2. 所有新 loss 支援 `lambda=0` 回到 baseline；僅接入 **GAN generator / encoder**，不接入 discriminator。
3. **不修改** `config/params_finetune_mini.json`。
4. Selection 使用 **`round6_sweetspot`**（ideal kmeans 0.65–0.78、ideal wasserstein 0.55–0.70、偏好 `latent_size=32`）。

## 新增程式

```text
tools/tumor_geometry.py      # compute_tumor_topology_loss
tools/tumor_subspace.py      # split_tumor_transfer_latent, orthogonality
tools/tumor_supcon.py        # within-domain SupCon
tools/tumor_vicreg.py        # variance + covariance
tools/round6_selection.py    # sweetspot_score
tools/analyze_round6_pretrain.py
```

## 一鍵 Pretrain（6A–6E）

```bash
bash tools/run_round6_pretrain.sh
# 或全流程（pretrain → aggregate）：
bash tools/run_round6_full_pipeline.sh
# 環境變數：DEVICE=cuda PRETRAIN_PARALLEL=20 FINETUNE_MAX_PARALLEL=26
```

若 pretrain 已完成、僅需 selection 後續：

```bash
bash tools/run_round6_post_pretrain.sh
```

## 診斷

```bash
python tools/analyze_round6_pretrain.py \
  --run-dirs \
    result/optimization_runs/vaewc_round6A_tumor_topology \
    result/optimization_runs/vaewc_round6B_topology_classgap_combo \
    result/optimization_runs/vaewc_round6C_tumor_transfer_subspace \
    result/optimization_runs/vaewc_round6D_within_domain_tumor_supcon \
    result/optimization_runs/vaewc_round6E_tumor_vicreg_stabilizer \
  --out-dir result/optimization_runs/round6_combined/reports
```

## Combined selection

```bash
python tools/optimization_runner.py select \
  --run-dir result/optimization_runs/round6_combined \
  --result-dirs \
    result/optimization_runs/vaewc_round6A_tumor_topology/pretrain,\
    result/optimization_runs/vaewc_round6B_topology_classgap_combo/pretrain,\
    result/optimization_runs/vaewc_round6C_tumor_transfer_subspace/pretrain,\
    result/optimization_runs/vaewc_round6D_within_domain_tumor_supcon/pretrain,\
    result/optimization_runs/vaewc_round6E_tumor_vicreg_stabilizer/pretrain \
  --selection-mode round6_sweetspot \
  --exclude-proto-ineffective \
  --force-baseline-models exp_001,exp_005,exp_746 \
  --top-k 30
```

## Finetune + aggregate

```bash
# 首輪或完整重跑
python tools/optimization_runner.py finetune \
  --run-dir result/optimization_runs/round6_combined \
  --top10 result/optimization_runs/round6_combined/selection/pretrain_top10.csv \
  --epochs 1000 --batch-size 4096 --mini-batch-size 1024 --max-parallel 42

# 僅重跑 failed / running jobs（推薦）
bash tools/run_round6_finetune_retry.sh

python tools/optimization_runner.py aggregate --run-dir result/optimization_runs/round6_combined
```

## 執行結果摘要（定案 2026-06-13）

| 階段 | 結果 |
|------|------|
| Pretrain 6A–6E | **102/102** |
| Selection | **16** 模型（top-k=30 因 sweetspot gate 不足） |
| Finetune | 首輪 41/64 → 重跑後 **64/64 success** |
| 重跑設定 | `batch=12288`, `mini=3072`, `parallel=42` |
| 下游最佳 | **exp_010** Avg TCGA **0.5569**（4/4，λ=0，6E） |
| R5 基準 | exp_001 **0.5403**（已被 R6 超越 +0.0166） |
| Active-loss Top-5 | **exp_012** #4（VICReg λ=0.0003，Integrated 0.562） |

完整分析：`docs/pipeline_summary.md` §14.6–§14.10。  
Aggregate：`result/optimization_runs/round6_combined/aggregate/aggregate_scores.csv`  
Checkpoint 建議：**exp_010**（主線）、**exp_012**（active VICReg 對照）。

## 測試

```bash
pytest tests/test_tumor_geometry.py tests/test_tumor_subspace.py \
  tests/test_tumor_supcon.py tests/test_tumor_vicreg.py \
  tests/test_round6_selection.py tests/test_round6_config_generation.py \
  tests/test_analyze_round6_pretrain.py -q
```

## 成功標準

1. `Average_TCGA_AUC_mean` **> 0.5403**（R5 exp_001）
2. `kmeans_ari` ≥ 0.65；無 alignment collapse
3. 至少一個 **active tumor loss** 模型進 downstream Top-5
4. sweetspot selection 較 wasserstein-first 更接近下游最佳

## Round 6F（可選）

Tumor-conditioned domain discriminator（`domain_discriminator_mode=tumor_conditioned`）**尚未實作**；待 6A–6E pretrain 結果後再評估。
