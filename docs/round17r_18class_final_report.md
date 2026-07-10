# Round 17R Final Report（18-class-clean）

**Run:** `result/optimization_runs/round17r_18class`  
**Pipeline:** Stage 17R-A → 17R-B →（待跑）17R-C → 17R-D → 17R-F  
**As of:** 2026-07-11

## 整體完成度

| Stage | 計畫 | 完成 | 狀態 | 完成時間（UTC） |
|-------|------|------|------|----------------|
| 17R-A feature smoke | 20 features | **20/20** | ✅ 完成 | 2026-07-08 |
| 17R-B focused finetune | 126 jobs（7 candidates × 6 combos × 3 seeds） | **126/126** | ✅ 完成 | 2026-07-09 |
| 17R-C hyperparameter refine | 依 17R-B top-k | — | ⏳ 待跑 | — |
| 17R-D 10-seed confirm | top-5 × 10 seeds | — | ⏳ 待跑 | — |
| 17R-F prototype tSNE | 2 models | — | ⏳ 待跑 | — |

**GPU 參數（17R-B）：** `FINETUNE_PARALLEL=20`，`FINETUNE_BATCH_SIZE=24576`，`FINETUNE_MINI_BATCH_SIZE=6144`，`FINETUNE_EPOCHS=1500`

**報表路徑：**

- `reports_stage17r_a/round17r_final_report.md`
- `reports_stage17r_b/round17r_top_candidates.csv`
- `reports_stage17r_b/round17r_vs_round17_pre18class_comparison.csv`
- `manifests/stage17r_b_finetune_dispatch_manifest.csv`

---

## 18-class-clean QC

| 項目 | 結果 |
|------|------|
| Feature folders | 20 |
| `n_trainable_cancer_types` | 18（全部） |
| `uses_legacy_28class_cache` | false（全部） |
| `prototype_class_source` | `checkpoint_metadata` |

Latent 在 feature extraction 階段即過濾至 checkpoint 18 類；GDSC / TCGA 無 latent 的樣本於 finetune / eval 階段 skip（詳見 `docs/round17r_18class_dataset_sample_usage.md`）。

---

## 17R-B 排名（`Average_TCGA_AUC`，best combo per candidate）

| Rank | Model | feature_mode | AUC | Integrated5 TargetMacro | Integrated5 DrugMacro |
|------|-------|--------------|-----|-------------------------|----------------------|
| 1 | `r15c_exp_024` | `own_plus_summary` | **0.6074** | 0.5637 | 0.5742 |
| 2 | `r15c_exp_005` | `own_plus_summary` | 0.6071 | 0.5625 | 0.5728 |
| 3 | `r13_exp_008` | `own_plus_summary` | 0.6059 | 0.5657 | 0.5757 |
| 4 | `r13_exp_008_control` | `own_plus_summary` | 0.6043 | 0.5614 | 0.5721 |
| 5 | `r13_exp_008` | `own_proto_context_projected_16` | 0.5792 | 0.5487 | 0.5535 |
| 6 | `r13_exp_008` | `minimal_source_only_min_margin` | 0.5678 | 0.5439 | 0.5487 |
| 7 | `r13_exp_008` | `own_proto_delta_projected_8` | 0.5665 | 0.5367 | 0.5439 |

### vs 基準

| Benchmark | AUC | Gap（17R-B best） |
|-----------|-----|-------------------|
| Round 13 best `r13_exp_008_own_plus_summary` | **0.6112** | **−0.0039** |
| Round 17C best（pre-18class, 10-seed） | 0.5892（`context_16`） | +0.0182 |
| Round 17C best `own_plus_summary`（10-seed） | 0.5868（`r15c_exp_005`） | +0.0206 |

18-class-clean 後，`own_plus_summary` 明顯回升，與 Round 13 差距從 pre-18class 的 **−0.022** 縮小至 **−0.004**。

---

## vs Pre-18class Round 17（可比對候選）

| Model | feature_mode | 17R-B | Pre-18 | Δ |
|-------|--------------|-------|--------|---|
| `r15c_exp_024` | `own_plus_summary` | 0.6074 | 0.5821 | **+0.0253** |
| `r15c_exp_005` | `own_plus_summary` | 0.6071 | 0.5868 | **+0.0202** |
| `r13_exp_008` | `own_proto_context_projected_16` | 0.5792 | 0.5892 | −0.0101 |
| `r13_exp_008` | `own_proto_delta_projected_8` | 0.5665 | 0.5840 | −0.0175 |

`own_plus_summary` 在 18-class-clean 後顯著提升；direct prototype 候選則略降。

---

## 核心問題答覆（基於 17R-A + 17R-B）

| 問題 | 結果 |
|------|------|
| Q1. 18-class-clean 後 ranking 是否改變？ | **是**；`own_plus_summary` 大幅回升，direct prototype 相對落後 |
| Q2. `own_plus_summary` 是否仍是最穩定主線？ | **是**；Top-4 全為 `own_plus_summary` |
| Q3. `context_16` / `delta_8` 是否接近 `own_plus_summary`？ | **否**；gap 約 **0.026–0.039**（遠大於 ±0.003） |
| Q4. `minimal_source` 在 tcga_only3 / dapl 優勢是否保留？ | **待 17R-C/D 細查**；整體 historical 排名第 6 |
| Q5. 5-target ranking 是否與 gdsc_intersect13 一致？ | 大方向一致（`own_plus_summary` 居前） |
| Q6. 是否需要 hyperparameter refinement？ | **建議進入 17R-C**（見 Stage gate） |
| 是否重現 Round 13（0.6112）？ | **尚未**；best gap **−0.0039**（已非常接近） |

---

## Stage gate 判定（17R-B → 17R-C）

依 `docs/round17r_18class_ide_manual.md` §9：

| 條件 | 判定 |
|------|------|
| historical `Average_TCGA_AUC >= 0.595` | ✅ best = 0.6074 |
| Integrated5 高於 Round17C best own_plus（0.5868） | ❌ best Integrated5 = 0.5657 |
| direct prototype 與 own_plus gap `<= 0.003` | ❌ gap ≈ 0.028 |
| minimal_source 在 tcga_only3 / dapl top-tier | ⏳ 待細查 |

**結論：符合 gate 條件 1，建議執行 17R-C。**

---

## 與原始 Round 17 的關係

| 路線 | 狀態 |
|------|------|
| Round 17 pre-18class（17A/B/C + 17F） | ✅ 已完成；報告見 `docs/round17_final_report.md` |
| Round 17 完整 18-class 重跑（1440 jobs） | ❌ 暫停（manifest 多數 pending） |
| **Round 17R**（focused 18-class） | **進行中**；17R-A/B 完成，17R-C/D/F 待跑 |

Round 17R 是 Round 17 在 18-class-clean universe 下的 **focused confirmation**，不取代原始 Round 17 的完整 sweep 結論，但提供更可信的 headline 數字。

---

## 下一步

```bash
# 17R-C: hyperparameter refinement
docker exec -w /workspace/DAPL DAPL bash -lc \
  'FINETUNE_PARALLEL=20 bash tools/run_round17r_stage17r_c_refine.sh'

# 17R-D: 10-seed confirmation
docker exec -w /workspace/DAPL DAPL bash -lc \
  'FINETUNE_PARALLEL=20 bash tools/run_round17r_stage17r_d_confirm.sh'

# 17R-F: 18-class tSNE
docker exec -w /workspace/DAPL DAPL bash tools/run_round17r_stage17r_f_tsne.sh
```

---

## 參考文件

- IDE 操作手冊：`docs/round17r_18class_ide_manual.md`
- 樣本使用統計：`docs/round17r_18class_dataset_sample_usage.md`
- Round 17 總報告：`docs/round17_final_report.md`

---

*Generated from `reports_stage17r_b/` + manifest status on 2026-07-11.*
