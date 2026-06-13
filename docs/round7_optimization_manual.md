# drug-DCF Round 7 多分支優化操作手冊

## exp_010 Neighborhood Refinement + VICReg Focused Ablation + Downstream-aware Selection

> 設計動機與 Round 6 定案結論見 `docs/pipeline_summary.md` §15.1。  
> 主成功標準：**Average_TCGA_AUC_mean > 0.5569**（超越 R6 **exp_010**）。

## 快速索引

| 分支 | Sweep / 工具 | Jobs | Run ID |
|------|--------------|------|--------|
| **7A** Control refinement | `config/pretrain_sweeps/vaewc_round7A_exp010_control_refinement.json` | 108 | `vaewc_round7A_exp010_control_refinement` |
| **7B** VICReg ablation | `config/pretrain_sweeps/vaewc_round7B_vicreg_focused_ablation.json` | 56 | `vaewc_round7B_vicreg_focused_ablation` |
| **7C** Selection | `--selection-mode round7_diverse_downstream_probe` | — | `round7_combined` |
| **7D** Finetune sensitivity | `config/finetune_sweeps/round7_finetune_sensitivity.json` | 8/checkpoint | `round7_finetune_sensitivity` |

Baseline configs：

- **exp_010-like**：`config/params_proto_base_exp010_like_vaewc.json`（latent=64，encoder `[512,256,128]`，active tumor λ=0）
- **exp_012-like**：`config/params_proto_base_exp012_like_vaewc.json`（同上 + VICReg λ=0.0003）

## 核心原則

**要做：**

1. 以 exp_010-like 設定作主線；固定 `latent_size=64`、`encoder_dims=[512,256,128]`。
2. 7A **不加入** active tumor loss；7B **僅 VICReg**（var/cov）。
3. 不再擴大 topology / class-gap / SupCon / subspace。
4. Selection 使用 **`round7_diverse_downstream_probe`**（downstream-aware diverse Top-K）。
5. 所有 branch 保留 λ=0 control；主指標仍為 **Average_TCGA_AUC_mean**。

**不要做：** 新 git branch、新 tumor loss、symmetric InfoNCE 重啟、subspace 擴大、大 grid finetune。

## 新增程式

```text
tools/round7_selection.py           # round7_diverse_downstream_probe
tools/analyze_round7_pretrain.py    # 7A/7B diagnostics
tools/run_round7_post_pretrain.sh   # diagnostics → selection → finetune → aggregate
tools/run_round7_pretrain.sh        # 7A+7B generate + pretrain + diagnostics
```

## 一鍵 Pretrain（7A + 7B）

```bash
bash tools/run_round7_pretrain.sh
# 環境變數：DEVICE=cuda PARALLEL=20（同 Round 6，目標 ~80%+ GPU）
```

## Generate configs

```bash
python tools/optimization_runner.py generate \
  --sweep-spec config/pretrain_sweeps/vaewc_round7A_exp010_control_refinement.json \
  --run-dir result/optimization_runs/vaewc_round7A_exp010_control_refinement \
  --force

python tools/optimization_runner.py generate \
  --sweep-spec config/pretrain_sweeps/vaewc_round7B_vicreg_focused_ablation.json \
  --run-dir result/optimization_runs/vaewc_round7B_vicreg_focused_ablation \
  --force
```

預期 manifest：**108**（7A）、**56**（7B）。

## Pretrain

```bash
python tools/optimization_runner.py pretrain \
  --run-dir result/optimization_runs/vaewc_round7A_exp010_control_refinement \
  --device cuda --max-parallel 20

python tools/optimization_runner.py pretrain \
  --run-dir result/optimization_runs/vaewc_round7B_vicreg_focused_ablation \
  --device cuda --max-parallel 20
```

## 診斷

```bash
python tools/analyze_round7_pretrain.py \
  --run-dirs \
    result/optimization_runs/vaewc_round7A_exp010_control_refinement \
    result/optimization_runs/vaewc_round7B_vicreg_focused_ablation \
  --outdir result/optimization_runs/round7_combined/reports
```

輸出：`round7_pretrain_diagnostics.csv` / `.md`

## Combined selection（7C）

> Runner 輸出目錄為 **`pretrain`**（非 `pretrain_results`）。Finetune 使用 **`--top10`**（非 `--model-select-path`）。

```bash
python tools/optimization_runner.py select \
  --run-dir result/optimization_runs/round7_combined \
  --result-dir result/optimization_runs/vaewc_round7A_exp010_control_refinement/pretrain \
  --result-dirs result/optimization_runs/vaewc_round7B_vicreg_focused_ablation/pretrain \
  --filter-config config/visualize_vaewc_filter.json \
  --selection-mode round7_diverse_downstream_probe \
  --exclude-proto-ineffective \
  --force-baseline-models exp_010,exp_012,exp_001,exp_005,exp_746 \
  --top-k 30 \
  --min-passing 5
```

Selection groups：`G1_exp010_like_control` … `G7_historical_baseline`（forced baselines 缺失時 report 警告，不 silent skip）。

## Finetune 首輪 + aggregate

```bash
bash tools/run_round7_post_pretrain.sh
# 或手動：
python tools/optimization_runner.py finetune \
  --run-dir result/optimization_runs/round7_combined \
  --top10 result/optimization_runs/round7_combined/selection/pretrain_top10.csv \
  --epochs 1000 --batch-size 12288 --mini-batch-size 3072 --max-parallel 42

python tools/optimization_runner.py aggregate --run-dir result/optimization_runs/round7_combined
```

## Finetune sensitivity（7D，第二輪）

從首輪 aggregate 挑 4–12 個 checkpoint，寫入  
`result/optimization_runs/round7_finetune_sensitivity/selection/model_select.csv`，然後：

```bash
python tools/optimization_runner.py finetune \
  --manifest result/optimization_runs/round7_finetune_sensitivity/manifests/finetune_dispatch_manifest.csv \
  --run-dir result/optimization_runs/round7_finetune_sensitivity \
  --top10 result/optimization_runs/round7_finetune_sensitivity/selection/model_select.csv \
  --finetune-config config/finetune_sweeps/round7_finetune_sensitivity.json \
  --epochs 1000 \
  --batch-size 12288 \
  --mini-batch-size 3072 \
  --max-parallel 42 \
  --force-manifest

python tools/optimization_runner.py aggregate \
  --run-dir result/optimization_runs/round7_finetune_sensitivity
```

Grid：2 loss × 2 hidden_dims × 2 dropout = **8 combos/checkpoint**。

## 測試

```bash
python -m compileall .
pytest tests/test_round7_config_generation.py \
  tests/test_round7_selection.py \
  tests/test_analyze_round7_pretrain.py \
  tests/test_round6_selection.py \
  tests/test_tumor_vicreg.py \
  tests/test_optimization_selection_round5.py -q
```

Docker：

```bash
docker exec -w /workspace/DAPL DAPL python3 -m pytest tests/test_round7_*.py -q
```

## 7A / 7B 觀察重點

**7A：** `lambda_cls` 15/20/25；cls schedule 30/80、40/90、50/110；`gan_patience` 30/50/70；`gan_gen_update_interval` 5/10。

**7B：** paired vs asymmetric var/cov；λ 0.0001–0.0005；VICReg schedule；Integrated Avg vs Avg TCGA。

## 成功 / 失敗判讀

| 結果 | 建議 |
|------|------|
| Avg TCGA > 0.5569 | 採用最佳 7A 或 7B checkpoint |
| VICReg 進 Top-5 但未超 exp_010 | 保留 VICReg 作 robustness ablation |
| 7A/7B 均未超越 exp_010 | pretrain 可能近上限；轉向下游 finetune / 資料協議 |
| Finetune sensitivity 有提升 | 下一輪固定 exp_010-like pretrain，優化 classifier |
