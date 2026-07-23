# Round 24 — IDE 操作手冊

**狀態：** IMPLEMENTATION_SPEC · 尚未開始正式訓練  
**解題計畫：** [`round24_solution_plan.md`](round24_solution_plan.md)  
**問題定義：** [`round24_problem_definition_plan.md`](round24_problem_definition_plan.md)

> 本手冊同時列出「目前可用」與「待實作」命令。標示 **[PLANNED]** 的 CLI 在程式完成前不可執行或宣稱已驗證。

---

## 0. 給 IDE Agent 的主指令

可將以下區塊直接貼給 IDE Agent：

```text
執行 Round 24 TCGA recovery，嚴格遵守：

1. 所有命令在 Docker 容器 DAPL、/workspace/DAPL 執行。
2. 最終必須是單一 unified model；禁止 per-target champion。
3. 五個 TCGA target 的 5-fold mean DrugMacro AUROC 必須全部超越基準。
4. 5:4:3:2:1 只在全數通過後排序，不能掩蓋任何 target failure。
5. eval3 = Round 18 的 5 source-fold + Stage 18E TCGA 評估。
6. TCGA 不得進入 loss、early stopping、checkpoint selection 或超參數搜尋。
7. GDSC development / validation / test 僅供診斷，不參與選模。
8. Stage 24A 必須先解釋 gdsc_intersect13 906 raw vs 886 eligible rows。
9. formal gate 前鎖定 candidate manifest；完成後不得追加候選。
10. 若沒有單一候選五 target 全過，寫 NO_LOCK，不得包裝成加權成功。
11. 每 stage 先 smoke、再 formal、再 analyze；失敗時保留 artifact 並 resume。
12. 使用 Telegram 發送 stage start/done/fail，但通知失敗不得中斷實驗。
```

---

## 1. 執行環境

| 項目 | 固定值 |
|------|--------|
| Host repo | `/home/wasijk/Workspace/Drug/DAPL` |
| Container | `DAPL` |
| Container repo | `/workspace/DAPL` |
| Python path | `/workspace/DAPL` |
| GPU | CUDA required for training |
| 主輸出 | `reports/round24/`、`result/optimization_runs/round24_tcga_recovery/` |

所有 Python 命令使用：

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && PYTHONPATH=/workspace/DAPL python3 <command>'
```

Shell stage 使用：

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && bash <script>'
```

---

## 2. 程式狀態與入口

### 2.1 目前可用

| 功能 | 路徑 / 命令 |
|------|-------------|
| Round 18 5CV pipeline | `step1_finetune_latent_pipeline_round18_cv.py` |
| Round 18 formal manifest | `tools/round18_config_builder.py` |
| Round 18 OOM dispatcher | `tools/round18_oom_runner.py` |
| Round 18 Stage 18E | `tools/run_round18_stage18e_locked_eval.sh` |
| 5-fold external analyzer | `tools/analyze_round18_external_eval.py` |
| DrugMacro metrics | `tools/round18_cv_metrics.py` |
| BioCDA 3-seed TCGA | `scripts/compare_biocda_tcga.py` |
| R23 TCGA ranking | `scripts/select_biocda_architecture_tcga.py` |
| R23 GDSC diagnostic | `scripts/evaluate_xa_candidates.py` |

### 2.2 待實作契約 [PLANNED]

```text
configs/round24/eval3.yaml
scripts/round24/run_round24.py
biocda/validation/round24_protocol.py
biocda/validation/round24_gate.py
scripts/round24/analyze_features.py
scripts/round24/diagnose_gdsc_intersect13.py
scripts/round24/analyze_objective_alignment.py
scripts/round24/lock_round24_model.py
tests/test_round24_protocol.py
tests/test_round24_gate.py
```

統一 CLI：

```bash
# [PLANNED]
python3 scripts/round24/run_round24.py <subcommand> --config configs/round24/eval3.yaml
```

子命令：

```text
preflight  protocol  baseline  features  diagnose
train      evaluate  select    lock      all
```

---

## 3. Stage 24A — Preflight 與 eval3 鎖定

### 3.1 Host / container 基本檢查

```bash
docker ps --filter name=DAPL
docker exec DAPL bash -lc 'cd /workspace/DAPL && pwd && python3 --version'
docker exec DAPL bash -lc 'nvidia-smi'
```

### 3.2 必要資產

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
test -f config/round18_architecture_settings.json && \
test -f tools/run_round18_stage18e_locked_eval.sh && \
test -f tools/round18_cv_metrics.py && \
test -d result/optimization_runs/round18_architecture && \
echo PRECHECK_OK'
```

五個 TCGA CSV：

```text
data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_gdsc_intersect13.csv
data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain_tcga_only3.csv
data/TCGA/TCGA_drug_response_from_DAPL.csv
data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_gdsc_intersect.csv
data/TCGA/TCGA_AACDR_response_final_with_smiles_intersect_pretrain_tcga_only.csv
```

### 3.3 Protocol preflight [PLANNED]

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py preflight \
  --config configs/round24/eval3.yaml \
  --strict'
```

預期輸出：

```text
reports/round24/stage24a/eval3_manifest.json
reports/round24/stage24a/cohort_coverage.csv
reports/round24/stage24a/dropped_rows.csv
```

### Checkpoint 24A-1

- 五個 CSV hash 已寫入 manifest。
- source folds = 5。
- DrugMacro support = 10/2/2。
- `gdsc_intersect13` 906 raw rows 與 886 eligible rows逐筆可追溯。
- 任一項失敗：停止，不可跑 baseline。

### 3.4 Baseline smoke / formal [PLANNED]

```bash
# Smoke：一 fold、一 target、少量 batch
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py baseline \
  --config configs/round24/eval3.yaml \
  --smoke'

# Formal：Round 18 pooled_mlp × own_plus_summary，5 folds × 5 targets
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py baseline \
  --config configs/round24/eval3.yaml'
```

### Checkpoint 24A-2

```text
5/5 folds complete
25/25 fold-target prediction sets complete
每 target fold mean/std + ensemble 可重算
baseline_summary.json 與原始 predictions 一致
```

---

## 4. Stage 24B — B0/P0/X0 同協議重建

### 4.1 Smoke [PLANNED]

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py train \
  --config configs/round24/eval3.yaml \
  --stage 24b --smoke'
```

Smoke 必須驗證：

- B0/P0/X0 forward、loss、backward。
- feature dimensions 與 model input contract。
- 一個 checkpoint 可完成一個 TCGA target inference。
- prediction 欄位含 `row_id`、`fold_id`、`DRUG_NAME`、`Label`、`probability`。

### 4.2 Formal train / evaluate [PLANNED]

```bash
docker exec \
  -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 \
  DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py train \
  --config configs/round24/eval3.yaml \
  --stage 24b --max-jobs-per-gpu 3'

docker exec DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py evaluate \
  --config configs/round24/eval3.yaml \
  --stage 24b'
```

### Checkpoint 24B

- 每候選 5/5 fold checkpoint。
- 每候選 25/25 fold-target predictions。
- 三種協議差值（eval3 / R23 3-seed / R20 15-fold）已輸出。
- 若 P0 或 X0 五 target 全過，跳至 Stage 24F。

---

## 5. Stage 24C — Feature attribution

候選：

```text
F0 own_plus_summary
F1 z_plus_summary
F2 z_plus_context16
F3 z_plus_context32
F4 z_plus_summary_context16
```

### 5.1 Feature coverage smoke [PLANNED]

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py features \
  --config configs/round24/eval3.yaml \
  --coverage-only'
```

檢查：

- feature dimensions 為 86/75/80/96/91。
- CCLE/source 與 TCGA patient coverage。
- feature names、source path、projection hash。
- 無 target label 欄位混入 feature。

### 5.2 Formal attribution [PLANNED]

```bash
docker exec \
  -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 \
  DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py features \
  --config configs/round24/eval3.yaml \
  --max-jobs-per-gpu 3'
```

輸出必須包含：

```text
reports/round24/stage24c/feature_attribution_long.csv
reports/round24/stage24c/feature_attribution_summary.json
reports/round24/stage24c/feature_selection_decision.json
```

只保留前兩名 feature recipe；全數通過者直接進 Stage 24F。

---

## 6. Stage 24D — gdsc_intersect13 診斷

### 執行 [PLANNED]

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py diagnose \
  --config configs/round24/eval3.yaml \
  --target gdsc_intersect13'
```

### 檢查

- `per_drug.csv` 包含 n/positive/negative/AUROC/AUPRC/valid/fold std。
- bottom drugs 可追溯到 predictions。
- Global vs DrugMacro 分解無使用 threshold-based accuracy 代替 AUROC。
- calibration 僅診斷，不改正式 prediction。
- 診斷完成後封存 Stage 24E 候選矩陣。

---

## 7. Stage 24E — 受限模型優化

### 7.1 候選預登記 [PLANNED]

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py select \
  --config configs/round24/eval3.yaml \
  --preregister-only'
```

必須產生：

```text
reports/round24/stage24e/candidate_manifest.json
reports/round24/stage24e/candidate_manifest.sha256
```

manifest 封存後禁止修改。

### 7.2 Train [PLANNED]

```bash
# Smoke
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py train \
  --config configs/round24/eval3.yaml \
  --stage 24e --smoke'

# Formal
docker exec \
  -e OMP_NUM_THREADS=2 \
  -e MKL_NUM_THREADS=2 \
  DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py train \
  --config configs/round24/eval3.yaml \
  --stage 24e --max-jobs-per-gpu 3'
```

Early stopping 只能讀 source validation metrics；run log 若出現 TCGA metric 參與 epoch selection，該候選立即 `INVALID`。

---

## 8. Stage 24F — 一次性正式 gate

### 8.1 Formal evaluate [PLANNED]

執行前確認：

```text
candidate_manifest hash locked
所有候選 5/5 source folds complete
TCGA formal output 尚不存在
git status 與 config hash 已記錄
```

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py evaluate \
  --config configs/round24/eval3.yaml \
  --stage 24f --formal'
```

### 8.2 Select / lock [PLANNED]

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py select \
  --config configs/round24/eval3.yaml \
  --strict-all-targets'

docker exec DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py lock \
  --config configs/round24/eval3.yaml'
```

### 最終 gate checklist

| 項目 | 必須 |
|------|------|
| 單一模型 | 是 |
| 5/5 folds | 是 |
| 五 target 均高於基準 | 是 |
| TCGA 未用於 early stop | 是 |
| GDSC selection role | `none` |
| candidate manifest hash 一致 | 是 |
| lock hashes 完整 | 是 |

若任何項目否，`status=NO_LOCK`。

---

## 9. Stage 24G — 分析與文件

```bash
# [PLANNED]
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
python3 scripts/round24/run_round24.py all \
  --config configs/round24/eval3.yaml \
  --analysis-only'
```

驗收：

- P1–P9 各有量測結果或明確限制。
- GDSC–TCGA Spearman/Pareto 標示為關聯而非因果。
- `round24_final_report.md`、`RESULTS_SUMMARY.md`、架構文件與 lock 一致。
- 舊 R20/R23 lock 不被改寫，只由 `supersedes` 引用。

---

## 10. GPU、OOM 與平行度

Round 18 dispatcher 的 OOM exit code 為 **42**，micro-batch ladder：

```text
512 → 256 → 128 → 64 → 32
```

建議起始值：

| 工作 | `max-jobs-per-gpu` |
|------|-------------------:|
| Smoke | 1 |
| Training | 3 |
| TCGA inference | 8（若記憶體允許） |

監控：

```bash
docker exec DAPL bash -lc 'nvidia-smi'
docker exec DAPL bash -lc 'ps -ef | rg "round24|round18"'
```

若 utilization 低但 VRAM 未滿，先確認 CPU feature/graph 建構瓶頸，不可只提高 batch 造成 OOM。

---

## 11. Resume 與失敗恢復

### Complete 契約

一個 training job 僅在以下三者齊全時視為 complete：

```text
status.json（status=complete）
metrics.json
best_checkpoint.pt
```

Inference job 需有：

```text
status.json（status=complete）
predictions.csv
metrics.json
```

### 恢復流程

1. 查看 stage dispatch summary。
2. 找出 `failed` / `pending` job。
3. 讀該 job `run.log` 與 OOM history。
4. 保留 complete job；預設 resume 只重跑未完成項目。
5. 只在 artifact hash 已變更時使用 `--no-resume` 全重跑。

禁止直接刪除整個 formal output；先備份 manifest 與失敗 log。

---

## 12. Telegram

一次性設定：

```bash
export TELEGRAM_BOT_TOKEN='...'
export TELEGRAM_CHAT_ID='...'
docker exec \
  -e TELEGRAM_BOT_TOKEN \
  -e TELEGRAM_CHAT_ID \
  DAPL bash /workspace/DAPL/scripts/telegram_secure_setup.sh
```

Stage 通知至少包含：

```text
Round24 stage
event=start|done|fail
complete_jobs/total_jobs
current candidate/fold
output path
```

通知失敗必須 fail silently，不可中止訓練。

---

## 13. Tests

現有 smoke：

```bash
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
pytest -q test_biocda_xa_v2_contracts.py'
```

Round 24 實作後：

```bash
# [PLANNED]
docker exec DAPL bash -lc 'cd /workspace/DAPL && \
pytest -q tests/test_round24_protocol.py tests/test_round24_gate.py'
```

最低測試範圍：

- target priority 與基準常數。
- all-target pass / one-target fail → `NO_LOCK`。
- 加權 winner 不可越過 hard gate。
- 5 folds、row uniqueness、candidate manifest hash。
- TCGA selection leakage guard。
- GDSC `selection_role=none`。
- lock schema 與 supersedes。

---

## 14. 故障排除

| 症狀 | 原因 / 處理 |
|------|-------------|
| `No model checkpoints found` | 檢查 fold checkpoint 路徑與 manifest |
| 906 raw 只剩 886 eligible | 讀 `dropped_rows.csv`；缺唯一 drop reason 即 Stage 24A BLOCKED |
| DrugMacro 為 null | 該 drug 未通過 10/2/2；檢查 support，不可改門檻補分 |
| fold 數不足 | resume missing fold；禁止用 best fold 替代 |
| OOM exit 42 | dispatcher 自動降 micro-batch；耗盡後降低平行度 |
| P0/X0 分數與 R23 不同 | 預期：協議已改為 source 5-fold；先查 cohort/hash |
| 加權第一但一個 target 未過 | 正常結果，狀態必須 `NO_LOCK` |
| GDSC 高但 TCGA 低 | 只記 diagnostic；不可因此 lock |
| candidate manifest formal 後變更 | 整個 Stage 24F invalid，需新 round |
| Telegram 無通知 | 檢查 `.env`；不得阻斷主流程 |

---

## 15. 一頁式執行順序

```text
24A preflight
  └─ cohort/hash/906-vs-886 對齊
     └─ baseline smoke → baseline formal
        └─ 24B B0/P0/X0 同協議
           ├─ 已五 target 全過 → 24F
           └─ 未通過 → 24C feature attribution
                        → 24D per-drug diagnose
                        → 24E preregister/train
                        → 24F one-shot TCGA gate
                           ├─ PASS → LOCKED
                           └─ FAIL → NO_LOCK
                              → 24G scientific report
```

