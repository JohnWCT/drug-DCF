# drug-DCF Round 8 多分支優化操作手冊

## Broad Architecture / Hyperparameter Confirmation without New Methods

> 設計動機與 Round 7 定案見 `docs/pipeline_summary.md` §16.1。  
> 主成功標準：**Average_TCGA_AUC_mean > 0.5918**（超越 R7 **exp_048**）。

## 快速索引

| 分支 | Sweep / 工具 | Jobs | Run ID |
|------|--------------|------|--------|
| **8A** Control architecture | `config/pretrain_sweeps/vaewc_round8A_control_arch_broad.json` | 288 | `vaewc_round8A_control_arch_broad` |
| **8B** VICReg architecture | `config/pretrain_sweeps/vaewc_round8B_vicreg_arch_broad.json` | 224 | `vaewc_round8B_vicreg_arch_broad` |
| **8C** Selection | `--selection-mode round8_architecture_broad_probe` | top_k=50 | `round8_combined` |
| **8D** Two-stage finetune | mini + `round8_finetune_sensitivity_broad.json` | 200 + 288 | `round8_finetune_sensitivity` |

**GPU 平行度：** `config/gpu_parallel_profile.json` + `tools/gpu_parallel_env.sh`（pretrain **33**、finetune **26**）。

Baseline configs：

- **8A control：** `config/params_proto_base_exp048_context_broad_control.json`
- **8B VICReg：** `config/params_proto_base_exp048_exp021_vicreg_broad.json`

## 核心原則

**要做：**

1. 擴大 control-like 與 VICReg-only 架構搜尋（latent / encoder / dropout / schedule）。
2. Selection 使用 **`round8_architecture_broad_probe`**（architecture diversity + downstream probe）。
3. Forced baselines：**exp_048, exp_021, exp_010, exp_012, exp_005, exp_746**。
4. Two-stage finetune：first-pass mini（50×4）→ sensitivity（12×24）。
5. 主指標仍為 **Average_TCGA_AUC_mean**。

**不要做：** 新 loss、topology、class-gap、SupCon、subspace、InfoNCE、改 evaluation protocol。

## 新增程式

```text
tools/round8_selection.py
tools/analyze_round8_pretrain.py
tools/build_round8_finetune_sensitivity_select.py
tools/run_round8_full_pipeline.sh
tools/run_round8_pretrain_retry.sh
```

## 一鍵全流程

```bash
bash tools/run_round8_full_pipeline.sh
# OOM 調整：
PRETRAIN_PARALLEL=28 bash tools/run_round8_full_pipeline.sh
FINETUNE_PARALLEL=20 bash tools/run_round8_full_pipeline.sh
# 僅補跑 failed pretrain（建議 parallel=12；含 manifest repair）
bash tools/run_round8_pretrain_retry.sh
PRETRAIN_RETRY_PARALLEL=8 bash tools/run_round8_pretrain_retry.sh
```

## Generate + Pretrain（分段）

```bash
python tools/optimization_runner.py generate \
  --sweep-spec config/pretrain_sweeps/vaewc_round8A_control_arch_broad.json \
  --run-dir result/optimization_runs/vaewc_round8A_control_arch_broad --force

python tools/optimization_runner.py generate \
  --sweep-spec config/pretrain_sweeps/vaewc_round8B_vicreg_arch_broad.json \
  --run-dir result/optimization_runs/vaewc_round8B_vicreg_arch_broad --force

python tools/optimization_runner.py pretrain \
  --manifest result/optimization_runs/vaewc_round8A_control_arch_broad/manifests/pretrain_sweep_manifest.csv \
  --run-dir result/optimization_runs/vaewc_round8A_control_arch_broad \
  --max-parallel 33

python tools/optimization_runner.py pretrain \
  --manifest result/optimization_runs/vaewc_round8B_vicreg_arch_broad/manifests/pretrain_sweep_manifest.csv \
  --run-dir result/optimization_runs/vaewc_round8B_vicreg_arch_broad \
  --max-parallel 33
```

## 診斷 + Selection（8C）

```bash
python tools/analyze_round8_pretrain.py \
  --run-dirs \
    result/optimization_runs/vaewc_round8A_control_arch_broad \
    result/optimization_runs/vaewc_round8B_vicreg_arch_broad \
  --outdir result/optimization_runs/round8_combined/reports

python tools/optimization_runner.py select \
  --run-dir result/optimization_runs/round8_combined \
  --result-dir result/optimization_runs/vaewc_round8A_control_arch_broad/pretrain \
  --result-dirs result/optimization_runs/vaewc_round8B_vicreg_arch_broad/pretrain \
  --filter-config config/visualize_vaewc_filter.json \
  --selection-mode round8_architecture_broad_probe \
  --exclude-proto-ineffective \
  --force-baseline-models exp_048,exp_021,exp_010,exp_012,exp_005,exp_746 \
  --top-k 50 \
  --min-passing 1 \
  --require-controls 0
```

Selection groups：G1_vicreg_active_best … G8_best_kmeans、G9_forced_baseline、G10_fill_ranked。

## First-pass finetune + aggregate

```bash
python tools/optimization_runner.py finetune \
  --run-dir result/optimization_runs/round8_combined \
  --top10 result/optimization_runs/round8_combined/selection/pretrain_top10.csv \
  --finetune-config config/params_finetune_mini.json \
  --epochs 1000 --batch-size 12288 --mini-batch-size 3072 --max-parallel 26 \
  --force-manifest

python tools/optimization_runner.py aggregate --run-dir result/optimization_runs/round8_combined
python tools/optimization_runner.py report --run-dir result/optimization_runs/round8_combined
```

## Second-pass finetune sensitivity（8D）

```bash
python tools/build_round8_finetune_sensitivity_select.py \
  --aggregate result/optimization_runs/round8_combined/aggregate/aggregate_scores.csv \
  --selection result/optimization_runs/round8_combined/selection/pretrain_top10.csv \
  --outdir result/optimization_runs/round8_finetune_sensitivity/selection \
  --max-models 12 \
  --force-models exp_048,exp_021,exp_746

python tools/optimization_runner.py finetune \
  --run-dir result/optimization_runs/round8_finetune_sensitivity \
  --top10 result/optimization_runs/round8_finetune_sensitivity/selection/model_select.csv \
  --finetune-config config/finetune_sweeps/round8_finetune_sensitivity_broad.json \
  --epochs 1000 --batch-size 12288 --mini-batch-size 3072 --max-parallel 26 \
  --force-manifest

python tools/optimization_runner.py aggregate --run-dir result/optimization_runs/round8_finetune_sensitivity
python tools/optimization_runner.py report --run-dir result/optimization_runs/round8_finetune_sensitivity
```

Grid：2 ftlr × 2 loss × 3 hidden_dims × 2 dropout = **24 combos/checkpoint**。

## 測試

```bash
python -m compileall .
pytest tests/test_round8_config_generation.py \
  tests/test_round8_selection.py \
  tests/test_analyze_round8_pretrain.py \
  tests/test_build_round8_finetune_sensitivity_select.py -q
```

## 成功 / 失敗判讀

| 結果 | 建議 |
|------|------|
| Avg TCGA > 0.5918 | 採用 Round 8 最佳 checkpoint |
| VICReg 仍主導 Top-5 | 固定 VICReg 主線，縮小 control sweep |
| Control 接近 0.5723 | 保留 control 作 robustness 對照 |
| Sensitivity 提升 ≥ 0.005 | 下一輪固定 pretrain，優化 classifier head |
| 全未超越 exp_048 | pretrain 可能近上限；檢視 finetune / 資料協議 |

## 執行結果摘要（2026-06-14 完成；pretrain 512/512 於同日補完）

| 階段 | 結果 |
|------|------|
| **Pretrain 8A** | **288/288 success** |
| **Pretrain 8B** | **224/224 success** |
| **Pretrain 合計** | **512/512 success** |
| **Selection（8C）** | **50 模型**（基於初跑 506 checkpoint） |
| **First-pass finetune** | **200/200 success** |
| **Second-pass sensitivity** | **216/216 success**（9 模型 × 24） |
| **執行時間** | ~18.4 h（pipeline）+ ~3 min（pretrain retry） |
| **下游最佳** | **exp_188** Avg TCGA **0.5777**（8A control，latent64，wide_768） |
| vs R7 exp_048（0.5918） | **0/50 超越**；R7 定案仍有效 |
| vs R6 exp_010（0.5569） | **5/50 超越** |

**Pretrain retry：** 初跑 6 failed（8A×4 OOM/CUBLAS、8B×2 OOM/EmptyDataError）→ `tools/run_round8_pretrain_retry.sh` 全數補完；下游 finetune 未重跑。

**First-pass Top-5：** exp_188（0.5777）> exp_021（0.5723）> exp_010（0.5644）> exp_048（0.5630）> exp_155（0.5610）。

**Second-pass：** 最佳 exp_188 **0.5479**（低於 first-pass，sensitivity grid 無增益）。

**Aggregate 路徑：**
- `result/optimization_runs/round8_combined/aggregate/aggregate_scores.csv`
- `result/optimization_runs/round8_finetune_sensitivity/aggregate/aggregate_scores.csv`

**定案：** 全專案主線仍為 **R7 exp_048**；R8 **exp_188** 為架構掃描最佳候選。詳見 `docs/pipeline_summary.md` §16.6–16.7。
