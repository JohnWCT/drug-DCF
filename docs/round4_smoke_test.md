# Round 4 Smoke Test

最小驗收流程：確認 cross-domain InfoNCE、cMMD 支線、K-means-aware selection 可跑通，無需完整 48+ jobs。

## 前置

```bash
docker exec -w /workspace/DAPL DAPL python3 -m compileall .
docker exec -w /workspace/DAPL DAPL pytest tests/ -q
```

若容器內無 `pytest`：`pip install -r tests/requirements-dev.txt`

`ruff` / `black` 在 DAPL 容器內可能未安裝；以 `compileall + pytest` 為主要閘門。

---

## 1. 產生小型 manifest（2–4 jobs）

可手動複製 sweep spec 並將各軸縮為 1–2 值，或使用現有 spec + `--force` 後只跑 manifest 前 2 列。

```bash
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_config_generator.py \
  --sweep-spec config/pretrain_sweeps/vaewc_round4_cross_domain_infonce.json \
  --manifest-dir result/optimization_runs/vaewc_round4_smoke/manifests \
  --force
```

---

## 2. Smoke pretrain（極小 epoch）

```bash
# 編輯 generated config：pretrain_num_epochs=2, train_num_epochs=5, gan_patience=3
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py pretrain \
  --manifest result/optimization_runs/vaewc_round4_smoke/manifests/pretrain_sweep_manifest.csv \
  --run-dir result/optimization_runs/vaewc_round4_smoke \
  --batch-size 128 --max-parallel 2 --smoke-test
```

**檢查：** `pretrain/exp_*/g_loss.csv` 含 `proto_mode`, `proto_t2s_loss`, `lambda_cmmd_eff` 等欄位。

---

## 3. Selection（Round 4 模式）

```bash
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py select \
  --run-dir result/optimization_runs/vaewc_round4_smoke \
  --no-filter \
  --selection-mode round4_kmeans_first \
  --exclude-proto-ineffective \
  --min-passing 2 --require-controls 1
```

**檢查：** `selection/pretrain_top10.csv`、`selection/model_select.csv` 存在；report 含 `selection_mode`。

---

## 4. 可選：1 個 finetune mini job

```bash
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py finetune \
  --manifest result/optimization_runs/vaewc_round4_smoke/manifests/finetune_dispatch_manifest.csv \
  --run-dir result/optimization_runs/vaewc_round4_smoke \
  --top10 result/optimization_runs/vaewc_round4_smoke/selection/pretrain_top10.csv \
  --epochs 5 --batch-size 512 --mini-batch-size 128 --max-parallel 1
```

---

## 5. Aggregate

```bash
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py aggregate \
  --run-dir result/optimization_runs/vaewc_round4_smoke
```

---

## 正式 Round 4 執行模板

### 產生 configs

```bash
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_config_generator.py \
  --sweep-spec config/pretrain_sweeps/vaewc_round4_cross_domain_infonce.json \
  --manifest-dir result/optimization_runs/vaewc_round4_cross_domain_infonce/manifests \
  --force
```

### Pretrain（~48 jobs）

```bash
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py pretrain \
  --manifest result/optimization_runs/vaewc_round4_cross_domain_infonce/manifests/pretrain_sweep_manifest.csv \
  --run-dir result/optimization_runs/vaewc_round4_cross_domain_infonce \
  --batch-size 128 --max-parallel 20
```

### Selection

```bash
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py select \
  --run-dir result/optimization_runs/vaewc_round4_cross_domain_infonce \
  --filter-config config/visualize_vaewc_filter.json \
  --selection-mode round4_kmeans_first \
  --exclude-proto-ineffective \
  --min-passing 10 --require-controls 2 \
  --run-tag vaewc_round4_cross_domain_infonce
```

### Finetune

```bash
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py finetune \
  --manifest result/optimization_runs/vaewc_round4_cross_domain_infonce/manifests/finetune_dispatch_manifest.csv \
  --run-dir result/optimization_runs/vaewc_round4_cross_domain_infonce \
  --top10 result/optimization_runs/vaewc_round4_cross_domain_infonce/selection/pretrain_top10.csv \
  --epochs 1000 --batch-size 4096 --mini-batch-size 1024 --max-parallel 26
```

### Aggregate + Report

```bash
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py aggregate \
  --run-dir result/optimization_runs/vaewc_round4_cross_domain_infonce
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py report \
  --run-dir result/optimization_runs/vaewc_round4_cross_domain_infonce
```

### 一鍵腳本

```bash
docker exec -d DAPL bash /workspace/DAPL/tools/run_round4_cross_domain_infonce.sh
```

---

## 支線

| Run ID | Sweep spec |
|--------|------------|
| `vaewc_round4_cmmd_branch` | `config/pretrain_sweeps/vaewc_round4_cmmd_branch.json` |
| `vaewc_round4_latent_ablation` | `config/pretrain_sweeps/vaewc_round4_latent_ablation.json` |

與 InfoNCE 主線**分開跑**，避免組合爆炸。

---

## Round 4.1 執行狀態（2026-06-10）

| 階段 | 狀態 |
|------|------|
| 主線 pretrain `vaewc_round4_1_t2s_infonce_collapse_guard` | **完成** 60/60 |
| 診斷報告 `analyze_round4_pretrain.py` | **已產出** → `.../reports/round4_1_pretrain_diagnostics.md` |
| Selection `round4_1_structure_first` | **待執行** |
| Finetune / 下游 TCGA | **完成** 36/36；最佳 exp_035 Avg=0.534 |
| `vaewc_round4_1_cmmd_branch` | **未啟動** |
| `vaewc_round4_1_latent_ablation` | **未啟動** |

**一句話結論：** t2s InfoNCE 已解決 Round 4 symmetric 的結構崩潰（mean kmeans_ari 0.22→0.52），但在 structure-first 硬篩下仍無 InfoNCE 通過 stage-1；下一步以 selection + finetune 驗證下游，支線 sweep 可並行補跑。

---

## Round 4.1 補充驗收（collapse-aware t2s InfoNCE）

### 單元測試（無 GPU）

```bash
docker exec -w /workspace/DAPL DAPL pytest tests/test_round41_collapse.py tests/test_proto_structure_metrics.py -q
```

涵蓋：

1. `proto_direction=target_to_source` 僅計算 t2s loss（`proto_s2t_loss=0`）
2. `detach_prototypes=True` 時 source prototype 無梯度、target anchor 有梯度
3. `best_gan_epoch < proto_start_epoch` → `proto_not_effective_checkpoint=true`
4. collapse selection：wasserstein 好但 kmeans_ari 崩潰的模型不可入選

### Round 4.1 selection

```bash
docker exec -w /workspace/DAPL DAPL python3 tools/optimization_runner.py select \
  --run-dir result/optimization_runs/vaewc_round4_1_t2s_infonce_collapse_guard \
  --selection-mode round4_1_structure_first \
  --exclude-proto-ineffective \
  --no-filter
```

（`round4_1_structure_first` 使用 Stage 1 structure + wasserstein 硬篩，不依賴 visualize 嚴格 FID 門檻。）

### 診斷報告

```bash
docker exec -w /workspace/DAPL DAPL python3 tools/analyze_round4_pretrain.py \
  --result-dir result/optimization_runs/vaewc_round4_1_t2s_infonce_collapse_guard/pretrain \
  --out-dir result/optimization_runs/vaewc_round4_1_t2s_infonce_collapse_guard/reports
```

### Round 4.1 sweep specs

| Run ID | Sweep spec |
|--------|------------|
| `vaewc_round4_1_t2s_infonce_collapse_guard` | `config/pretrain_sweeps/vaewc_round4_1_t2s_infonce_collapse_guard.json` |
| `vaewc_round4_1_cmmd_branch` | `config/pretrain_sweeps/vaewc_round4_1_cmmd_branch.json` |
| `vaewc_round4_1_latent_ablation` | `config/pretrain_sweeps/vaewc_round4_1_latent_ablation.json` |
