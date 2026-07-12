# drug-DCF Round 17R IDE 操作手冊

## 18-class-clean Focused Rerun + Confirmation

## 0. Round 17R 定位

Round 17 pre-18class pipeline 已完成；18-class-clean 確認改由 Round 17R 進行，目前不直接進入新方法 Round 18。

### 執行狀態（2026-07-12）

| Stage | 狀態 |
|-------|------|
| 17R-A | ✅ 完成（20 features，QC 全通過） |
| 17R-B | ✅ 完成（126/126；peak AUC **0.6074**） |
| 17R-C | ✅ 完成（180/180） |
| 17R-D | ✅ 完成（50/50；10-seed best **0.5915**，`own_plus_summary`） |
| 17R-F | ✅ 完成（`r13_exp_008` tSNE） |

**ALL_DONE。** 彙整報告（含各資料集 Top-5 + 策略）：`docs/round17r_18class_final_report.md`

Round 17R 的定位是：

```text
Round 17R:
18-class-clean focused rerun and confirmation
```

目的：

```text
1. 修正 prototype feature extraction 的 cancer class universe。
2. 讓 feature extraction、tSNE、checkpoint metadata 全部使用同一套 18 類 cancer type。
3. 只重跑 Round 17 中最有價值的候選，不重新跑完整 1440 jobs。
4. 確認 own_plus_summary / direct prototype / minimal source geometry 在 18-class-clean 後的排序是否改變。
5. 決定是否進入 final validation，或是否仍值得做小範圍 hyperparameter optimization。
```

Round 17R 不是新方法探索；是 Round 17 的 clean rerun / focused confirmation。

---

## 1. Round 17R 核心問題

```text
Q1. 18-class-clean feature extraction 後，Round 17 ranking 是否改變？
Q2. own_plus_summary 是否仍是最穩定的主線？
Q3. direct prototype candidates（context_16 / delta_8）是否仍接近 own_plus_summary？
Q4. minimal_source_only_min_margin 在 tcga_only3 / dapl 上的優勢是否保留？
Q5. 5-target macro ranking 是否與 historical gdsc_intersect13 ranking 一致？
Q6. 是否需要再進一輪 focused hyperparameter optimization？
```

---

## 2. 不做什麼

```text
1. 不新增新的 loss / pretrain branch / feature family
2. 不全面重跑 Round 17A 的 1440 jobs
3. 不先實作 two_tower_proto / proto_film
4. 不把 single seed best 當作結論
```

只做：`18-class-clean feature extraction` + focused rerun + seed confirmation。

---

## 3. Stage 結構

```text
17R-A: 18-class-clean feature extraction smoke
17R-B: focused 3-seed rerun on selected candidates
17R-C: focused hyperparameter refinement (if 17R-B shows signal)
17R-D: 10-seed confirmation and final model selection
17R-F: final 18-class prototype tSNE (optional)
```

輸出根目錄：

```text
result/optimization_runs/round17r_18class/
```

---

## 4. 新增 / 修改檔案

### configs

- `config/round17r_18class_focused_settings.json`
- `config/params_finetune_round17r_focused.json`

### tools

- `tools/round17r_18class_config_builder.py`
- `tools/analyze_round17r_18class.py`
- `tools/run_round17r_stage17r_a_feature_smoke.sh`
- `tools/run_round17r_stage17r_b_focused.sh`
- `tools/run_round17r_stage17r_c_refine.sh`
- `tools/run_round17r_stage17r_d_confirm.sh`
- `tools/run_round17r_stage17r_f_tsne.sh`

### QC 修改

- `tools/extract_round13_proto_features.py`：強制 checkpoint metadata 18 類、禁止 legacy 28-class cache、寫入 QC metadata
- `tools/visualize_round17_prototype_tsne.py`：支援 Round 17R model specs / checkpoint 解析

---

## 5. 18-class-clean QC

每個 feature folder 必須有：

```text
feature_metadata.json
prototype_coverage.csv
cancer_type_mapping.json
feature_names.json
```

`feature_metadata.json` 必須包含：

```json
{
  "prototype_class_source": "checkpoint_metadata",
  "n_trainable_cancer_types": 18,
  "source_prototypes_used": 18,
  "target_prototypes_used": 18,
  "uses_legacy_28class_cache": false
}
```

若 `n_trainable_cancer_types != 18`，該 job 不可進入 finetune。

---

## 6. Primary candidates（17R-B）

```text
1. r13_exp_008_control / own_plus_summary
2. r13_exp_008 / own_plus_summary
3. r15c_exp_005 / own_plus_summary
4. r15c_exp_024 / own_plus_summary
5. r13_exp_008 / own_proto_context_projected_16
6. r13_exp_008 / own_proto_delta_projected_8
7. r13_exp_008 / minimal_source_only_min_margin
```

規模：`7 candidates × 6 combos × 3 seeds = 126 jobs`

---

## 7. Finetune combos（縮圈）

使用 `config/params_finetune_round17r_focused.json`：

- `lr ∈ {1e-4, 2e-4, 3e-4}`
- `weight_decay ∈ {3e-5, 1e-4, 3e-4}`
- `dropout ∈ {0.15, 0.20, 0.25}`
- default head only
- `batch_size=24576`, `mini_batch_size=6144`

不使用：`512,256` / `256,128` head、`8192/2048`、`lr>5e-4`。

---

## 8. 執行順序

### 啟動前 checklist

```bash
docker exec -w /workspace/DAPL DAPL python3 -m py_compile \
  tools/prototype_response_features.py \
  tools/extract_round13_proto_features.py \
  tools/round17r_18class_config_builder.py \
  tools/analyze_round17r_18class.py \
  tools/visualize_round17_prototype_tsne.py

docker exec -w /workspace/DAPL DAPL pytest tests/test_round17r_*.py -q
```

### 17R-A feature smoke

```bash
docker exec -w /workspace/DAPL DAPL bash tools/run_round17r_stage17r_a_feature_smoke.sh
```

QC：

```bash
docker exec -w /workspace/DAPL DAPL python3 - <<'PY'
import json
from pathlib import Path
paths = list(Path("result/optimization_runs/round17r_18class/features").rglob("feature_metadata.json"))
assert paths
for p in paths:
    meta = json.loads(p.read_text())
    assert meta["n_trainable_cancer_types"] == 18
    assert meta["uses_legacy_28class_cache"] is False
print("OK: ready for Stage 17R-B")
PY
```

### 17R-B focused 3-seed

```bash
docker exec -w /workspace/DAPL DAPL bash -lc \
  'FINETUNE_PARALLEL=12 bash tools/run_round17r_stage17r_b_focused.sh'
```

報告：

```text
result/optimization_runs/round17r_18class/reports_stage17r_b/round17r_final_report.md
result/optimization_runs/round17r_18class/reports_stage17r_b/round17r_top_candidates.csv
```

### 17R-C / 17R-D / 17R-F

```bash
docker exec -w /workspace/DAPL DAPL bash -lc \
  'FINETUNE_PARALLEL=12 bash tools/run_round17r_stage17r_c_refine.sh'

docker exec -w /workspace/DAPL DAPL bash -lc \
  'FINETUNE_PARALLEL=12 bash tools/run_round17r_stage17r_d_confirm.sh'

docker exec -w /workspace/DAPL DAPL bash tools/run_round17r_stage17r_f_tsne.sh
```

---

## 9. Stage gate

### 進入 17R-C

任一 candidate：

1. historical `Average_TCGA_AUC >= 0.595`
2. 或 Integrated5 高於 Round17C best own_plus_summary
3. 或 direct prototype 與 own_plus_summary gap `<= 0.003`
4. 或 minimal_source 在 tcga_only3 / dapl 保持 top-tier

### 進入 17R-D

任一 candidate：

1. 5-seed mean `>= 0.600` historical
2. 或 Integrated5 明顯優於 Round17C top own_plus_summary
3. 或 direct prototype 5-seed std 低於 own_plus_summary
4. 或 best `>= 0.6112` 且非孤立 single seed

### Final recommendation

| 條件 | 決策 |
|------|------|
| 10-seed historical mean `>= 0.6112` | 進入 final validation |
| own_plus_summary 仍最佳但不超 Round 13 | 停搜尋；own_plus_summary 作 primary |
| direct prototype 追平（±0.003）且更穩 | co-primary / small head search |
| minimal_source 只在 target-specific 強 | ablation / insight，不作 primary |

---

## 10. 一句話總結

Round 17R 不是再探索新方法，而是把 Round 17 的關鍵候選在 18-class-clean prototype universe 下重跑，用 focused hyperparameter 與 seed confirmation 決定是否進入 final validation。

---

## 附錄：樣本數實況

實際保留/剔除樣本統計已整理於：

- `docs/round17r_18class_dataset_sample_usage.md`
