# Round 18 Stage 18E Report — Locked External Evaluation

**Date:** 2026-07-14  
**Status:** **ALL_DONE — internal 25/25 + TCGA 125/125 + analyze**  
**Verdict:** `cross_attention_external_success = false`（TCGA non-worse **2/5**；門檻 ≥3/5）  
**Full Round:** 18A–18E done；**18F pending** — see [`docs/round18_final_report.md`](round18_final_report.md)  
**Root:** `result/optimization_runs/round18_architecture/`

---

## 1. Scope

Stage 18E 在 **選模完成後** 評估 18D lock 的 5 candidates：

1. held-out **internal test** ensemble  
2. **5-target TCGA** fold inference + probability ensemble  
3. Integrated5 + paired bootstrap（不得回頭改 architecture selection）

| Item | Value |
|------|-------|
| Candidates | lock `formal_candidates` only（5） |
| Checkpoints | `stage18d/{arch}/fold_k/checkpoint.pt` |
| Ensemble | 5-fold probability **mean**（禁止 best-fold） |
| Primary metric | DrugMacro AUC |
| Bootstrap | paired over Patient_id／ModelID；2000 reps；`--n-jobs` 平行 |
| Docker | `ROUND18_NUM_WORKERS=0`；`MAX_JOBS_PER_GPU=8` |

### Locked candidates

| Role | Architecture |
|------|--------------|
| Anchor | `pooled_mlp__own_plus_summary` |
| Best pooled Transformer | `pooled_transformer__P3_deeper128__own_proto_context_projected_16` |
| Efficient Transformer | `pooled_transformer__P1_compact64__own_proto_context_projected_16` |
| Best cross-attn pure | `cross_attn__X3__pure__own_proto_context_projected_16` |
| Best cross-attn residual | `cross_attn__X3__pooled_residual__own_proto_context_projected_16` |

---

## 2. Jobs & completion

| Block | Jobs | Status |
|-------|------|--------|
| Internal infer | 5 arch × 5 folds = **25** | DONE |
| TCGA infer | 5 arch × 5 targets × 5 folds = **125** | DONE |
| Analyze / bootstrap | ensemble + metrics + 96 bootstrap jobs | DONE |

Runner: `SMOKE_ONLY=0 MAX_JOBS_PER_GPU=8 ROUND18_NUM_WORKERS=0 bash tools/run_round18_stage18e_locked_eval.sh`  
Analyze: `python tools/analyze_round18_external_eval.py --outdir result/optimization_runs/round18_architecture --n-bootstrap 2000 --n-jobs 16`

---

## 3. Internal held-out test（ensemble）

Source: `reports/round18_internal_test_summary.csv`

| Rank | Architecture | DrugMacro AUC | Global AUC |
|------|--------------|---------------|------------|
| 1 | P3 × context16 | **0.6131** | 0.8586 |
| 2 | X3 residual × context16 | 0.6110 | 0.8579 |
| 3 | X3 pure × context16 | 0.6056 | **0.8632** |
| 4 | P1 × context16 | 0.5905 | 0.8244 |
| 5 | MLP × own_plus_summary | 0.5358 | 0.7705 |

- X3 pure ≫ MLP（DrugMacro ≈ +0.070；paired bootstrap P(Δ>0)≈0.9995）。  
- P3 在此切面 DrugMacro 最高；X3 pure 相對 P3／residual 無穩定優勢（CI 含 0）。

---

## 4. TCGA five-target DrugMacro AUC

Source: `reports/round18_five_target_tcga_summary.csv`

| Target | MLP | P3 | P1 | X3 pure | X3 res | X3 vs MLP |
|--------|-----|----|----|---------|--------|-----------|
| gdsc_intersect13 | **0.5415** | 0.4902 | 0.4624 | 0.4593 | 0.4951 | worse |
| tcga_only3 | **0.5508** | 0.4023 | 0.4321 | 0.3584 | 0.4059 | worse |
| dapl | **0.5154** | 0.4628 | 0.4181 | 0.4620 | 0.4095 | worse |
| aacdr_tcga_only | 0.5183 | 0.4450 | 0.5018 | **0.5340** | 0.4492 | non-worse |
| aacdr_gdsc_intersect | 0.5178 | 0.5667 | 0.5267 | **0.5601** | 0.5559 | non-worse |

X3 pure vs MLP：**2/5** non-worse（未達成功門檻 ≥3/5）。

---

## 5. Integrated5（5 TCGA 目標平均）

Source: `reports/round18_integrated5_summary.csv`

| Rank | Architecture | Integrated5 DrugMacro |
|------|--------------|------------------------|
| 1 | **MLP × own_plus_summary** | **0.5288** |
| 2 | X3 pure × context16 | 0.4748 |
| 3 | P3 × context16 | 0.4734 |
| 4 | P1 × context16 | 0.4682 |
| 5 | X3 residual × context16 | 0.4631 |

TCGA 宏觀平均上 **MLP 最佳**；formal／internal 的 cross-attn／Transformer 優勢未外推。

---

## 6. Paired bootstrap（X3 pure − comparator，DrugMacro）

Source: `reports/round18e_paired_bootstrap_deltas.csv`（n=2000）

| Contrast | internal mean Δ [95% CI] | P(Δ>0) |
|----------|--------------------------|--------|
| vs MLP | **+0.065** [+0.024, +0.108] | ≈0.9995 |
| vs P3 | −0.006 [−0.031, +0.019] | ≈0.31 |
| vs P1 | +0.013 [−0.024, +0.052] | ≈0.75 |
| vs residual | −0.005 [−0.032, +0.021] | ≈0.35 |

TCGA：相對 MLP 多數目標 mean Δ 為負（尤其 `tcga_only3`、`gdsc_intersect13`）；AACDR 子集傾向非負。

---

## 7. Success verdict

Source: `reports/round18e_success_verdict.json`

```json
{
  "cross_attention_external_success": false,
  "notes": [
    "internal X3_pure=0.6056 vs MLP=0.5358; TCGA non-worse=2/5",
    "internal residual=0.6110 (delta vs pure=0.0054)"
  ],
  "prefer_pure_for_18F": true
}
```

**固定規則（選模後評估，不污染 lock）：**  
internal X3_pure ≥ MLP **且** TCGA DrugMacro non-worse ≥ 3/5 → success。  
實際：internal 通過、TCGA **2/5** → **false**。

若做 18F：鎖定 formal champion **`cross_attn__X3__pure__own_proto_context_projected_16`**。

---

## 8. Takeaways

1. In-domain（internal）：cross-attn／P3+context16 遠勝 MLP。  
2. External（TCGA）：固定規則下 **未通過**；Integrated5 以 MLP 領先。  
3. Residual 無外部加分（pure≈residual）。  
4. 不得用 18E 結果回頭改 18D selection。

---

## 9. Key paths

```text
result/optimization_runs/round18_architecture/
  manifests/
    stage18e_internal_test_manifest.csv
    stage18e_tcga_manifest.csv
  reports/
    round18_internal_test_summary.csv
    round18_five_target_tcga_summary.csv
    round18_integrated5_summary.csv
    round18_external_eval_summary.csv
    round18e_paired_bootstrap_deltas.csv
    round18e_success_verdict.json
    round18e_internal_*_ensemble_predictions.csv
    round18e_tcga_*_*_ensemble_predictions.csv
  stage18e_internal/
  stage18e_tcga/
```

Code:

- `tools/run_round18_stage18e_locked_eval.sh`  
- `tools/analyze_round18_external_eval.py`  
- `tools/round18_tcga_dataset.py`  
- `tools/round18_prediction_ensemble.py`  
- `tools/round18_config_builder.py`（`build_stage18e_manifest`）  
- `step1_finetune_latent_pipeline_round18_cv.py`（`infer_internal_test` / `infer_tcga`）
